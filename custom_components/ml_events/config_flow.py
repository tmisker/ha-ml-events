"""Config flow for ML Events integration."""
import logging
import voluptuous as vol
from typing import Any, Dict, Optional

from homeassistant import config_entries
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import selector
from homeassistant.components.sensor import DOMAIN as SENSOR_DOMAIN

from .const import (
    DOMAIN,
    CONF_DETECTOR_NAME,
    CONF_INCLUDED_SENSORS,
    CONF_AUTO_SELECT_SENSORS,
    CONF_TRAINING_WINDOW,
    CONF_MODEL_TYPE,
    DEFAULT_TRAINING_WINDOW,
    DEFAULT_AUTO_SELECT_SENSORS,
    DEFAULT_MODEL_TYPE,
)

_LOGGER = logging.getLogger(__name__)

# Model type options
MODEL_TYPES = [
    "random_forest",
    "gradient_boosting",
    "support_vector",
    "neural_network",
]

class MLEventsConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for ML Events."""

    VERSION = 1
    
    async def async_step_user(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Handle the initial step."""
        errors = {}
        
        if user_input is not None:
            # Validate detector name uniqueness
            detector_name = user_input[CONF_DETECTOR_NAME]
            await self.async_set_unique_id(detector_name)
            self._abort_if_unique_id_configured()
            
            return self.async_create_entry(
                title=detector_name, 
                data=user_input
            )
        
        # Get all available sensors
        entity_reg = er.async_get(self.hass)
        all_sensors = [
            entity_id for entity_id in entity_reg.entities
            if entity_id.split('.')[0] in ('sensor', 'binary_sensor')
        ]
        
        # Define the schema for the form
        data_schema = vol.Schema({
            vol.Required(CONF_DETECTOR_NAME): str,
            vol.Optional(CONF_AUTO_SELECT_SENSORS, default=DEFAULT_AUTO_SELECT_SENSORS): bool,
            vol.Optional(CONF_INCLUDED_SENSORS, default=[]): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=all_sensors,
                    multiple=True,
                    mode=selector.SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(CONF_TRAINING_WINDOW, default=DEFAULT_TRAINING_WINDOW): int,
            vol.Optional(CONF_MODEL_TYPE, default=DEFAULT_MODEL_TYPE): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=MODEL_TYPES,
                    multiple=False,
                    mode=selector.SelectSelectorMode.DROPDOWN,
                )
            ),
        })
        
        return self.async_show_form(
            step_id="user", 
            data_schema=data_schema, 
            errors=errors,
        )
    
    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        """Get options flow for this handler."""
        return MLEventsOptionsFlow(config_entry)


class MLEventsOptionsFlow(config_entries.OptionsFlow):
    """Handle options for the ML Events integration."""

    def __init__(self, config_entry):
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(self, user_input=None):
        """Manage the options."""
        errors = {}
        
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)
        
        # Get current values or defaults
        data = self.config_entry.data
        options = self.config_entry.options
        
        current_sensors = options.get(
            CONF_INCLUDED_SENSORS, 
            data.get(CONF_INCLUDED_SENSORS, [])
        )
        current_auto_select = options.get(
            CONF_AUTO_SELECT_SENSORS, 
            data.get(CONF_AUTO_SELECT_SENSORS, DEFAULT_AUTO_SELECT_SENSORS)
        )
        current_window = options.get(
            CONF_TRAINING_WINDOW, 
            data.get(CONF_TRAINING_WINDOW, DEFAULT_TRAINING_WINDOW)
        )
        current_model_type = options.get(
            CONF_MODEL_TYPE, 
            data.get(CONF_MODEL_TYPE, DEFAULT_MODEL_TYPE)
        )
        
        # Get all available sensors
        entity_reg = er.async_get(self.hass)
        all_sensors = [
            entity_id for entity_id in entity_reg.entities
            if entity_id.split('.')[0] in ('sensor', 'binary_sensor')
        ]
        
        # Define the schema for the form
        data_schema = vol.Schema({
            vol.Optional(CONF_AUTO_SELECT_SENSORS, default=current_auto_select): bool,
            vol.Optional(CONF_INCLUDED_SENSORS, default=current_sensors): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=all_sensors,
                    multiple=True,
                    mode=selector.SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(CONF_TRAINING_WINDOW, default=current_window): int,
            vol.Optional(CONF_MODEL_TYPE, default=current_model_type): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=MODEL_TYPES,
                    multiple=False,
                    mode=selector.SelectSelectorMode.DROPDOWN,
                )
            ),
        })
        
        return self.async_show_form(
            step_id="init", 
            data_schema=data_schema, 
            errors=errors,
        )