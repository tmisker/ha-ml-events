"""Machine Learning Service for ML Events."""
import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import asyncio
import json
import os

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from homeassistant.core import HomeAssistant
from homeassistant.helpers.recorder import get_instance, history
from homeassistant.util import dt as dt_util

from .const import (
    DOMAIN,
    DATA_MODELS,
    DATA_DETECTOR_CONFIG,
    CONF_INCLUDED_SENSORS,
    CONF_AUTO_SELECT_SENSORS,
    CONF_TRAINING_WINDOW,
    CONF_MODEL_TYPE,
    DEFAULT_TRAINING_WINDOW,
)

_LOGGER = logging.getLogger(__name__)

class MLService:
    """Machine Learning Service for ML Events."""

    def __init__(self, hass: HomeAssistant):
        """Initialize the ML Service."""
        self.hass = hass
        self.models = {}
        self.model_info = {}
        
        # Create models directory if it doesn't exist
        config_dir = self.hass.config.path("ml_events")
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
            
    async def train_model(self, detector_name: str, labels: List[datetime]) -> bool:
        """Train a model for the detector using the provided labels."""
        # Find config for this detector
        detector_config = None
        for entry_id, config in self.hass.data[DOMAIN][DATA_DETECTOR_CONFIG].items():
            if config.get("detector_name") == detector_name:
                detector_config = config
                break
                
        if not detector_config:
            _LOGGER.error(f"No configuration found for detector {detector_name}")
            return False
            
        # Get configuration values
        auto_select = detector_config.get(CONF_AUTO_SELECT_SENSORS, True)
        included_sensors = detector_config.get(CONF_INCLUDED_SENSORS, [])
        window_minutes = detector_config.get(CONF_TRAINING_WINDOW, DEFAULT_TRAINING_WINDOW)
        model_type = detector_config.get(CONF_MODEL_TYPE, "random_forest")
        
        # Collect training data
        try:
            # We'll run this in an executor since it's CPU-bound
            training_data = await self.hass.async_add_executor_job(
                self._collect_training_data, 
                labels, 
                included_sensors, 
                auto_select,
                window_minutes
            )
            
            if not training_data or training_data.empty:
                _LOGGER.error(f"Failed to collect training data for {detector_name}")
                return False
                
            # Train the model
            model, model_info = await self.hass.async_add_executor_job(
                self._train_model_internal,
                training_data,
                model_type
            )
            
            if not model:
                _LOGGER.error(f"Failed to train model for {detector_name}")
                return False
                
            # Store the model and info
            self.models[detector_name] = model
            self.model_info[detector_name] = model_info
            
            # Save model to disk
            await self.hass.async_add_executor_job(
                self._save_model_to_disk,
                detector_name,
                model,
                model_info
            )
            
            # Store model in hass.data for persistence
            self.hass.data[DOMAIN][DATA_MODELS][detector_name] = {
                "model": model,
                "info": model_info
            }
            
            return True
            
        except Exception as e:
            _LOGGER.error(f"Error training model for {detector_name}: {e}")
            return False
    
    def _collect_training_data(
        self, 
        labels: List[datetime], 
        included_sensors: List[str], 
        auto_select: bool,
        window_minutes: int
    ) -> pd.DataFrame:
        """Collect training data from Home Assistant history."""
        all_data = []
        
        # If auto_select is True and no sensors are included, try to find relevant ones
        if auto_select and not included_sensors:
            # We'll query all entity states
            # As a first step, we can use common domains like binary_sensor, sensor, etc.
            entity_filter = None
        else:
            # Use only the specified sensors
            entity_filter = included_sensors
            
        # For each label (positive example)
        for label_time in labels:
            # Define window around the label
            start_time = label_time - timedelta(minutes=window_minutes//2)
            end_time = label_time + timedelta(minutes=window_minutes//2)
            
            # Get data from that time period
            history_data = history.get_significant_states(
                self.hass,
                start_time,
                end_time,
                entity_filter,
                include_start_time_state=True,
            )
            
            if not history_data:
                continue
                
            # Process this data into a row for our dataset
            row_data = self._process_history_to_features(history_data, label_time, True)
            if row_data:
                all_data.append(row_data)
                
        # Now collect some negative examples (times when the event was not happening)
        # We'll take random times that are not close to any labeled events
        negative_timestamps = self._generate_negative_timestamps(labels, window_minutes)
        
        for neg_time in negative_timestamps:
            start_time = neg_time - timedelta(minutes=window_minutes//2)
            end_time = neg_time + timedelta(minutes=window_minutes//2)
            
            history_data = history.get_significant_states(
                self.hass,
                start_time,
                end_time,
                entity_filter,
                include_start_time_state=True,
            )
            
            if not history_data:
                continue
                
            row_data = self._process_history_to_features(history_data, neg_time, False)
            if row_data:
                all_data.append(row_data)
                
        # Convert to DataFrame
        if not all_data:
            return pd.DataFrame()
            
        df = pd.DataFrame(all_data)
        
        # Clean up data (handle NaNs, etc.)
        df = df.fillna(0)
        
        # If auto-select is enabled, do feature selection
        if auto_select and len(df.columns) > 10:  # Only if we have many columns
            # Simple feature selection: remove constant columns and highly correlated features
            # Keep at least 10 most varying features, plus the target
            df = self._select_features(df)
            
        return df
    
    def _process_history_to_features(
        self, 
        history_data: Dict[str, List], 
        center_time: datetime,
        is_positive: bool
    ) -> Dict[str, Any]:
        """Process historical state data into features for ML."""
        features = {"event_active": 1 if is_positive else 0}
        
        # For each entity
        for entity_id, states in history_data.items():
            if not states:
                continue
                
            # Extract the entity domain and object_id
            domain, object_id = entity_id.split(".")
            
            # Skip entities we don't want to use
            if domain in ["automation", "script", "input_boolean", "device_tracker"]:
                continue
                
            # Get state closest to the center time
            closest_state = None
            min_time_diff = float('inf')
            
            for state in states:
                time_diff = abs((state.last_updated - center_time).total_seconds())
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_state = state
            
            if closest_state:
                # Try to convert the state to a number
                try:
                    # For binary sensors we'll use 1/0
                    if domain == "binary_sensor":
                        state_value = 1 if closest_state.state == "on" else 0
                    # For regular sensors, try to convert to float
                    else:
                        state_value = float(closest_state.state)
                    
                    # Store the feature
                    features[f"{entity_id}"] = state_value
                    
                    # For sensors, also include rate of change if we have multiple states
                    if domain == "sensor" and len(states) > 1:
                        # Sort states by time
                        sorted_states = sorted(states, key=lambda x: x.last_updated)
                        
                        # Calculate rate of change (per minute)
                        try:
                            first_state = float(sorted_states[0].state)
                            last_state = float(sorted_states[-1].state)
                            time_diff_minutes = (sorted_states[-1].last_updated - 
                                                sorted_states[0].last_updated).total_seconds() / 60
                            
                            if time_diff_minutes > 0:
                                rate = (last_state - first_state) / time_diff_minutes
                                features[f"{entity_id}_rate"] = rate
                        except (ValueError, ZeroDivisionError):
                            pass
                            
                except (ValueError, TypeError):
                    # If we can't convert to number, we'll skip this feature
                    pass
        
        return features
    
    def _generate_negative_timestamps(
        self, 
        positive_times: List[datetime], 
        window_minutes: int
    ) -> List[datetime]:
        """Generate timestamps for negative examples, avoiding positive examples."""
        negative_times = []
        
        # We want about 2x as many negative examples as positive
        num_negatives = min(len(positive_times) * 2, 50)  # Cap at 50 to avoid too much data
        
        # Start from 7 days ago
        start_time = dt_util.now() - timedelta(days=7)
        end_time = dt_util.now()
        
        # Calculate the total seconds in our time range
        total_seconds = (end_time - start_time).total_seconds()
        
        # To avoid collisions with positive examples, we'll keep a set of excluded times
        excluded_ranges = []
        for pos_time in positive_times:
            # Extend the exclusion window to avoid partial overlaps
            window_seconds = (window_minutes + 5) * 60  # Add 5 min buffer
            excluded_start = pos_time - timedelta(seconds=window_seconds//2)
            excluded_end = pos_time + timedelta(seconds=window_seconds//2)
            excluded_ranges.append((excluded_start, excluded_end))
        
        # Try to generate our target number of negatives
        attempts = 0
        while len(negative_times) < num_negatives and attempts < 100:
            # Pick a random time in our range
            random_offset = np.random.randint(0, int(total_seconds))
            candidate_time = start_time + timedelta(seconds=random_offset)
            
            # Check if this time overlaps with any positive example
            is_excluded = False
            for ex_start, ex_end in excluded_ranges:
                if ex_start <= candidate_time <= ex_end:
                    is_excluded = True
                    break
            
            if not is_excluded:
                negative_times.append(candidate_time)
            
            attempts += 1
        
        return negative_times
    
    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select the most relevant features from the dataset."""
        # Remove the target column temporarily
        target = df["event_active"]
        features_df = df.drop(columns=["event_active"])
        
        # Remove constant columns
        non_constant_cols = [col for col in features_df.columns 
                            if features_df[col].nunique() > 1]
        features_df = features_df[non_constant_cols]
        
        # Get standard deviation for each column
        std_devs = features_df.std()
        
        # Sort columns by standard deviation (highest first)
        sorted_cols = std_devs.sort_values(ascending=False).index.tolist()
        
        # Take top 20 columns
        selected_features = sorted_cols[:20]
        
        # Return dataset with selected features and target
        result = features_df[selected_features].copy()
        result["event_active"] = target
        
        return result
    
    def _train_model_internal(
        self, 
        data: pd.DataFrame,
        model_type: str
    ) -> Tuple[Any, Dict[str, Any]]:
        """Train a machine learning model on the collected data."""
        # Split features and target
        X = data.drop(columns=["event_active"])
        y = data["event_active"]
        
        # Save column names for later
        feature_names = X.columns.tolist()
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.25, random_state=42
        )
        
        # Select model based on type
        if model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(random_state=42)
        elif model_type == "support_vector":
            model = SVC(probability=True, random_state=42)
        elif model_type == "neural_network":
            model = MLPClassifier(hidden_layer_sizes=(10, 5), random_state=42, max_iter=1000)
        else:
            # Default to random forest
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        accuracy = model.score(X_test, y_test)
        
        # Get feature importance if available
        top_features = []
        if hasattr(model, 'feature_importances_'):
            # Get feature importances
            importances = model.feature_importances_
            # Sort feature importances in descending order
            indices = np.argsort(importances)[::-1]
            
            # Get top 5 features
            for i in range(min(5, len(feature_names))):
                idx = indices[i]
                top_features.append({
                    "name": feature_names[idx],
                    "importance": float(importances[idx])
                })
        
        # Create a model package with all necessary components
        model_package = {
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names
        }
        
        # Create model info
        model_info = {
            "accuracy": float(accuracy),
            "examples_count": len(data),
            "positive_examples": int(y.sum()),
            "negative_examples": int(len(y) - y.sum()),
            "top_features": top_features,
            "last_trained": dt_util.now().isoformat(),
            "model_type": model_type,
            "feature_names": feature_names
        }
        
        return model_package, model_info
    
    def _save_model_to_disk(
        self, 
        detector_name: str, 
        model_package: Dict[str, Any],
        model_info: Dict[str, Any]
    ) -> None:
        """Save the trained model to disk."""
        # Get path
        config_dir = self.hass.config.path("ml_events")
        model_path = os.path.join(config_dir, f"{detector_name}_model.pkl")
        info_path = os.path.join(config_dir, f"{detector_name}_info.json")
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model_package, f)
            
        # Save info
        with open(info_path, 'w') as f:
            json.dump(model_info, f)
            
    async def predict(self, detector_name: str) -> Tuple[bool, float]:
        """Make a prediction using the trained model."""
        # Check if model exists
        if detector_name not in self.models:
            # Try to load from disk
            try:
                model_package, model_info = await self.hass.async_add_executor_job(
                    self._load_model_from_disk,
                    detector_name
                )
                
                if model_package:
                    self.models[detector_name] = model_package
                    self.model_info[detector_name] = model_info
                else:
                    return False, 0.0
            except Exception as e:
                _LOGGER.error(f"Error loading model for {detector_name}: {e}")
                return False, 0.0
        
        # Get model
        model_package = self.models[detector_name]
        model = model_package["model"]
        scaler = model_package["scaler"]
        feature_names = model_package["feature_names"]
        
        # Collect current sensor states
        try:
            # Get current states for all the features we need
            current_data = {}
            
            for feature in feature_names:
                # Skip rate features as we can't compute them at prediction time
                if feature.endswith("_rate"):
                    current_data[feature] = 0.0
                    continue
                    
                # Check if this is a valid entity ID
                if "." in feature:
                    entity_id = feature
                    state = self.hass.states.get(entity_id)
                    
                    if state:
                        try:
                            # For binary sensors use 1/0
                            domain = entity_id.split('.')[0]
                            if domain == "binary_sensor":
                                state_value = 1 if state.state == "on" else 0
                            else:
                                # Try to convert to float
                                state_value = float(state.state)
                                
                            current_data[feature] = state_value
                        except (ValueError, TypeError):
                            current_data[feature] = 0.0
                    else:
                        current_data[feature] = 0.0
            
            # Convert to DataFrame
            X = pd.DataFrame([current_data])
            
            # Make sure we have all required features in the right order
            missing_features = set(feature_names) - set(X.columns)
            for feature in missing_features:
                X[feature] = 0.0
            
            X = X[feature_names]  # Reorder columns to match training data
            
            # Scale features
            X_scaled = scaler.transform(X)
            
            # Make prediction
            if hasattr(model, 'predict_proba'):
                # Get probability of positive class
                y_proba = model.predict_proba(X_scaled)[0]
                confidence = y_proba[1]  # Probability of positive class
                prediction = confidence >= 0.5
            else:
                # Just make a prediction without probability
                prediction = model.predict(X_scaled)[0]
                confidence = 1.0 if prediction else 0.0
                
            return bool(prediction), float(confidence)
            
        except Exception as e:
            _LOGGER.error(f"Error making prediction for {detector_name}: {e}")
            return False, 0.0
    
    def _load_model_from_disk(self, detector_name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load a saved model from disk."""
        config_dir = self.hass.config.path("ml_events")
        model_path = os.path.join(config_dir, f"{detector_name}_model.pkl")
        info_path = os.path.join(config_dir, f"{detector_name}_info.json")
        
        # Check if files exist
        if not os.path.exists(model_path) or not os.path.exists(info_path):
            return None, None
            
        # Load model
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)
            
        # Load info
        with open(info_path, 'r') as f:
            model_info = json.load(f)
            
        return model_package, model_info
    
    def get_model_info(self, detector_name: str) -> Dict[str, Any]:
        """Get information about a trained model."""
        if detector_name in self.model_info:
            return self.model_info[detector_name]
        return None