"""The ML Events integration."""
import logging
import os
import asyncio
import voluptuous as vol
from datetime import datetime

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers.typing import ConfigType
import homeassistant.helpers.config_validation as cv

from .const import (
    DOMAIN,
    SERVICE_LABEL_EVENT,
    SERVICE_TRAIN_MODEL,
    SERVICE_CLEAR_LABELS,
    CONF_DETECTOR_NAME,
    DATA_LABELS,
    DATA_MODELS,
    DATA_DETECTOR_CONFIG,
    DATA_TRAINED_DETECTORS,
    DATA_ML_SERVICE,
)
from .ml_service import MLService

_LOGGER = logging.getLogger(__name__)

PLATFORMS = ["binary_sensor"]

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the ML Events component."""
    hass.data[DOMAIN] = {
        DATA_MODELS: {},
        DATA_LABELS: {},
        DATA_DETECTOR_CONFIG: {},
        DATA_TRAINED_DETECTORS: set(),
    }
    
    # Create ML service
    ml_service = MLService(hass)
    hass.data[DOMAIN][DATA_ML_SERVICE] = ml_service
    
    # Register services
    register_services(hass)
    
    return True

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up ML Events from a config entry."""
    # Store the config
    detector_name = entry.data.get(CONF_DETECTOR_NAME)
    hass.data[DOMAIN][DATA_DETECTOR_CONFIG][entry.entry_id] = entry.data
    
    # Init labels storage for this detector if it doesn't exist
    if detector_name not in hass.data[DOMAIN][DATA_LABELS]:
        hass.data[DOMAIN][DATA_LABELS][detector_name] = []
    
    # Forward to platforms
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    
    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    # Remove model data
    detector_name = entry.data.get(CONF_DETECTOR_NAME)
    if detector_name in hass.data[DOMAIN][DATA_MODELS]:
        del hass.data[DOMAIN][DATA_MODELS][detector_name]
    
    # Unload platforms
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    
    if unload_ok:
        # Remove entry-specific data
        hass.data[DOMAIN][DATA_DETECTOR_CONFIG].pop(entry.entry_id)
        if detector_name in hass.data[DOMAIN][DATA_TRAINED_DETECTORS]:
            hass.data[DOMAIN][DATA_TRAINED_DETECTORS].remove(detector_name)
    
    return unload_ok

def register_services(hass: HomeAssistant) -> None:
    """Register services for the ML Events integration."""
    
    async def handle_label_event(call: ServiceCall) -> None:
        """Handle the label_event service call."""
        detector_name = call.data.get(CONF_DETECTOR_NAME)
        if not detector_name:
            _LOGGER.error("No detector name provided for label_event service")
            return
            
        # Add timestamp to labels
        if detector_name not in hass.data[DOMAIN][DATA_LABELS]:
            hass.data[DOMAIN][DATA_LABELS][detector_name] = []
            
        hass.data[DOMAIN][DATA_LABELS][detector_name].append(datetime.now())
        _LOGGER.info(f"Added label for {detector_name} at {datetime.now()}")
    
    async def handle_train_model(call: ServiceCall) -> None:
        """Handle the train_model service call."""
        detector_name = call.data.get(CONF_DETECTOR_NAME)
        if not detector_name:
            _LOGGER.error("No detector name provided for train_model service")
            return
            
        # Get labels for this detector
        if detector_name not in hass.data[DOMAIN][DATA_LABELS]:
            _LOGGER.error(f"No labels found for detector {detector_name}")
            return
            
        labels = hass.data[DOMAIN][DATA_LABELS][detector_name]
        if len(labels) < 3:
            _LOGGER.error(f"Not enough labels for {detector_name}. Need at least 3, got {len(labels)}")
            return
        
        # Train model using ML service
        ml_service = hass.data[DOMAIN][DATA_ML_SERVICE]
        try:
            success = await ml_service.train_model(detector_name, labels)
            if success:
                hass.data[DOMAIN][DATA_TRAINED_DETECTORS].add(detector_name)
                _LOGGER.info(f"Successfully trained model for {detector_name}")
            else:
                _LOGGER.error(f"Failed to train model for {detector_name}")
        except Exception as e:
            _LOGGER.error(f"Error training model for {detector_name}: {e}")
    
    async def handle_clear_labels(call: ServiceCall) -> None:
        """Handle the clear_labels service call."""
        detector_name = call.data.get(CONF_DETECTOR_NAME)
        if not detector_name:
            _LOGGER.error("No detector name provided for clear_labels service")
            return
            
        # Clear labels for this detector
        if detector_name in hass.data[DOMAIN][DATA_LABELS]:
            hass.data[DOMAIN][DATA_LABELS][detector_name] = []
            _LOGGER.info(f"Cleared labels for {detector_name}")
    
    # Register the services
    hass.services.async_register(
        DOMAIN, 
        SERVICE_LABEL_EVENT, 
        handle_label_event,
        schema=vol.Schema({vol.Required(CONF_DETECTOR_NAME): cv.string})
    )
    
    hass.services.async_register(
        DOMAIN, 
        SERVICE_TRAIN_MODEL, 
        handle_train_model,
        schema=vol.Schema({vol.Required(CONF_DETECTOR_NAME): cv.string})
    )
    
    hass.services.async_register(
        DOMAIN, 
        SERVICE_CLEAR_LABELS, 
        handle_clear_labels,
        schema=vol.Schema({vol.Required(CONF_DETECTOR_NAME): cv.string})
    )