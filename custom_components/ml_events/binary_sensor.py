"""Binary sensor platform for ML Events."""
import logging
from datetime import datetime, timedelta
import voluptuous as vol

from homeassistant.components.binary_sensor import (
    BinarySensorEntity,
    BinarySensorDeviceClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.util import dt as dt_util

from .const import (
    DOMAIN,
    CONF_DETECTOR_NAME,
    DATA_ML_SERVICE,
    DATA_TRAINED_DETECTORS,
    ATTR_CONFIDENCE,
    ATTR_LAST_TRAINED,
    ATTR_EXAMPLES_COUNT,
    ATTR_TOP_FEATURES,
    ATTR_MODEL_ACCURACY,
    DEFAULT_POLLING_INTERVAL,
)

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up ML Events binary sensors from a config entry."""
    detector_name = config_entry.data.get(CONF_DETECTOR_NAME)
    
    # Add the sensor entity
    sensor = MLEventsBinarySensor(hass, config_entry)
    async_add_entities([sensor], True)

class MLEventsBinarySensor(BinarySensorEntity):
    """Binary sensor for detecting ML Events."""

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry):
        """Initialize the sensor."""
        self.hass = hass
        self.config_entry = config_entry
        self._detector_name = config_entry.data.get(CONF_DETECTOR_NAME)
        self._attr_unique_id = f"ml_event_{self._detector_name}"
        self._attr_name = f"ML Event: {self._detector_name}"
        self._state = False
        self._confidence = 0.0
        self._last_trained = None
        self._examples_count = 0
        self._top_features = []
        self._model_accuracy = 0.0
        self._ml_service = hass.data[DOMAIN][DATA_ML_SERVICE]
        
        # Set device info
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, f"ml_events_device_{self._detector_name}")},
            name=f"ML Events: {self._detector_name}",
            manufacturer="Home Assistant Community",
            model="ML Detector",
            sw_version="0.1.0",
        )
        
        # Make it clear what this sensor is
        self._attr_has_entity_name = True
        self._attr_device_class = BinarySensorDeviceClass.OCCUPANCY
        
    @property
    def is_on(self) -> bool:
        """Return true if the binary sensor is on."""
        return self._state
        
    @property
    def available(self) -> bool:
        """Return True if entity is available."""
        # Only available if the detector has been trained
        return self._detector_name in self.hass.data[DOMAIN][DATA_TRAINED_DETECTORS]
    
    @property
    def extra_state_attributes(self) -> dict:
        """Return the state attributes."""
        return {
            ATTR_CONFIDENCE: f"{self._confidence:.2f}",
            ATTR_LAST_TRAINED: self._last_trained,
            ATTR_EXAMPLES_COUNT: self._examples_count,
            ATTR_TOP_FEATURES: self._top_features,
            ATTR_MODEL_ACCURACY: f"{self._model_accuracy:.2f}",
        }
    
    async def async_update(self) -> None:
        """Update the sensor state."""
        if not self.available:
            return
            
        # Get prediction from ML service
        try:
            prediction, confidence = await self._ml_service.predict(self._detector_name)
            
            # Update state
            self._state = prediction
            self._confidence = confidence
            
            # Update other attributes
            model_info = self._ml_service.get_model_info(self._detector_name)
            if model_info:
                self._last_trained = model_info.get("last_trained")
                self._examples_count = model_info.get("examples_count", 0)
                self._top_features = model_info.get("top_features", [])
                self._model_accuracy = model_info.get("accuracy", 0.0)
                
        except Exception as e:
            _LOGGER.error(f"Error predicting for {self._detector_name}: {e}")