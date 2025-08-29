"""Config flow for LiteLLM Conversation integration."""
from __future__ import annotations

import logging
from typing import Any

import aiohttp
import async_timeout
import voluptuous as vol

from homeassistant import config_entries, core
from homeassistant.const import CONF_API_KEY, CONF_NAME
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers.aiohttp_client import async_get_clientsession
import homeassistant.helpers.config_validation as cv

from .const import (
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_FREQUENCY_PENALTY,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_PRESENCE_PENALTY,
    CONF_PROMPT,
    CONF_SERVICE_NAME,
    CONF_SERVICE_TYPE,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DEFAULT_BASE_URL,
    DEFAULT_CONF_PROMPT,
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_PRESENCE_PENALTY,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DOMAIN,
    SERVICE_MODEL_MAP,
    SERVICE_TYPE_AI_TASK,
    SERVICE_TYPE_CONVERSATION,
    SERVICE_TYPE_NAMES,
    SERVICE_TYPE_STT,
    SERVICE_TYPE_TTS,
    SERVICE_TYPES,
)

_LOGGER = logging.getLogger(__name__)


async def validate_input(hass: core.HomeAssistant, data: dict[str, Any]) -> dict[str, Any]:
    """Validate the user input allows us to connect."""
    session = async_get_clientsession(hass)
    
    # Test connection to the LiteLLM endpoint
    headers = {
        "Authorization": f"Bearer {data[CONF_API_KEY]}",
        "Content-Type": "application/json",
    }
    
    base_url = data[CONF_BASE_URL].rstrip("/")
    models_url = f"{base_url}/models"
    
    try:
        async with async_timeout.timeout(10):
            async with session.get(models_url, headers=headers) as response:
                if response.status == 200:
                    models_data = await response.json()
                    available_models = [model["id"] for model in models_data.get("data", [])]
                    return {"title": "LiteLLM Conversation", "models": available_models}
                else:
                    raise InvalidAuth
    except aiohttp.ClientError:
        raise CannotConnect
    except Exception:
        raise CannotConnect


class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for LiteLLM Conversation."""

    VERSION = 1

    def __init__(self) -> None:
        """Initialize the config flow."""
        self._base_url: str | None = None
        self._api_key: str | None = None
        self._available_models: list[str] = []
        self._service_type: str | None = None
        self._main_entry_id: str | None = None

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        # Check if this is a service sub-entry creation
        if self.context.get("source") == "options" and user_input and CONF_SERVICE_TYPE in user_input:
            return await self._async_create_service_entry(user_input)
            
        errors: dict[str, str] = {}
        
        if user_input is not None:
            try:
                info = await validate_input(self.hass, user_input)
            except CannotConnect:
                errors["base"] = "cannot_connect"
            except InvalidAuth:
                errors["base"] = "invalid_auth"
            except Exception:  # pylint: disable=broad-except
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"
            else:
                # Store connection info and available models
                self._base_url = user_input[CONF_BASE_URL]
                self._api_key = user_input[CONF_API_KEY]
                self._available_models = info["models"]
                
                # Create the main integration entry
                return self.async_create_entry(
                    title=info["title"],
                    data={
                        CONF_BASE_URL: self._base_url,
                        CONF_API_KEY: self._api_key,
                        "available_models": self._available_models,
                    }
                )

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_BASE_URL, default=DEFAULT_BASE_URL): str,
                    vol.Required(CONF_API_KEY): str,
                }
            ),
            errors=errors,
        )

    async def _async_create_service_entry(self, service_data: dict[str, Any]) -> FlowResult:
        """Create a service sub-entry."""
        service_name = service_data.get(CONF_SERVICE_NAME, f"LiteLLM {service_data[CONF_SERVICE_TYPE].title()}")
        
        return self.async_create_entry(
            title=service_name,
            data=service_data,
        )


    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle reconfiguration of the integration."""
        config_entry = self._get_reconfigure_entry()
        
        # Check if this is a service sub-entry
        if config_entry.data.get(CONF_SERVICE_TYPE):
            return await self._async_step_reconfigure_service(user_input, config_entry)
        else:
            return await self._async_step_reconfigure_main(user_input, config_entry)

    async def _async_step_reconfigure_main(
        self, user_input: dict[str, Any] | None = None, config_entry=None
    ) -> FlowResult:
        """Handle reconfiguration of the main integration entry."""
        errors: dict[str, str] = {}
        
        if user_input is not None:
            try:
                info = await validate_input(self.hass, user_input)
            except CannotConnect:
                errors["base"] = "cannot_connect"
            except InvalidAuth:
                errors["base"] = "invalid_auth"
            except Exception:  # pylint: disable=broad-except
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"
            else:
                # Update the main entry
                self.hass.config_entries.async_update_entry(
                    config_entry,
                    data={
                        CONF_BASE_URL: user_input[CONF_BASE_URL],
                        CONF_API_KEY: user_input[CONF_API_KEY],
                        "available_models": info["models"],
                    }
                )
                return self.async_abort(reason="reconfigure_successful")

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_BASE_URL,
                        default=config_entry.data.get(CONF_BASE_URL, DEFAULT_BASE_URL),
                    ): str,
                    vol.Required(
                        CONF_API_KEY,
                        default=config_entry.data.get(CONF_API_KEY, ""),
                    ): str,
                }
            ),
            errors=errors,
        )

    async def _async_step_reconfigure_service(
        self, user_input: dict[str, Any] | None = None, config_entry=None
    ) -> FlowResult:
        """Handle reconfiguration of a service sub-entry."""
        service_type = config_entry.data[CONF_SERVICE_TYPE]
        
        # Get available models for this service type
        available_models = SERVICE_MODEL_MAP.get(service_type, [])
        
        if user_input is not None:
            # Update the service entry
            updated_data = dict(config_entry.data)
            updated_data.update(user_input)
            
            self.hass.config_entries.async_update_entry(
                config_entry,
                data=updated_data,
            )
            return self.async_abort(reason="reconfigure_successful")

        # Build schema based on service type
        schema_fields = {
            vol.Required(
                CONF_SERVICE_NAME,
                default=config_entry.data.get(CONF_SERVICE_NAME, ""),
            ): str,
            vol.Required(
                CONF_MODEL,
                default=config_entry.data.get(CONF_MODEL, available_models[0] if available_models else ""),
            ): str,
        }
        
        # Add service-specific fields
        if service_type == SERVICE_TYPE_CONVERSATION:
            schema_fields.update({
                vol.Optional(
                    CONF_PROMPT,
                    default=config_entry.data.get(CONF_PROMPT, DEFAULT_PROMPT),
                ): cv.template,
                vol.Optional(
                    CONF_MAX_TOKENS,
                    default=config_entry.data.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
                ): int,
                vol.Optional(
                    CONF_TEMPERATURE,
                    default=config_entry.data.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
                ): vol.All(vol.Coerce(float), vol.Range(min=0, max=2)),
                vol.Optional(
                    CONF_TOP_P,
                    default=config_entry.data.get(CONF_TOP_P, DEFAULT_TOP_P),
                ): vol.All(vol.Coerce(float), vol.Range(min=0, max=1)),
                vol.Optional(
                    CONF_PRESENCE_PENALTY,
                    default=config_entry.data.get(CONF_PRESENCE_PENALTY, DEFAULT_PRESENCE_PENALTY),
                ): vol.All(vol.Coerce(float), vol.Range(min=-2, max=2)),
                vol.Optional(
                    CONF_FREQUENCY_PENALTY,
                    default=config_entry.data.get(CONF_FREQUENCY_PENALTY, DEFAULT_FREQUENCY_PENALTY),
                ): vol.All(vol.Coerce(float), vol.Range(min=-2, max=2)),
            })

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=vol.Schema(schema_fields),
        )

    @staticmethod
    @core.callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> OptionsFlowHandler:
        """Create the options flow."""
        return OptionsFlowHandler(config_entry)


class OptionsFlowHandler(config_entries.OptionsFlow):
    """LiteLLM Conversation config flow options handler."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize LiteLLM Conversation options flow."""
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        # Check if this is the main integration entry
        if not self.config_entry.data.get(CONF_SERVICE_TYPE):
            return await self.async_step_add_service()
        else:
            # This is a service entry, should not have options flow
            return self.async_abort(reason="not_supported")

    async def async_step_add_service(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle adding a new service."""
        if user_input is not None:
            service_type = user_input[CONF_SERVICE_TYPE]
            
            # Create a new config entry for the service
            service_data = {
                CONF_SERVICE_TYPE: service_type,
                CONF_BASE_URL: self.config_entry.data[CONF_BASE_URL],
                CONF_API_KEY: self.config_entry.data[CONF_API_KEY],
                "available_models": self.config_entry.data["available_models"],
                "parent_entry_id": self.config_entry.entry_id,
            }
            
            # Start the service configuration flow
            if service_type == SERVICE_TYPE_CONVERSATION:
                return await self.async_step_conversation_config(service_data)
            elif service_type == SERVICE_TYPE_STT:
                return await self.async_step_stt_config(service_data)
            elif service_type == SERVICE_TYPE_TTS:
                return await self.async_step_tts_config(service_data)
            elif service_type == SERVICE_TYPE_AI_TASK:
                return await self.async_step_ai_task_config(service_data)

        return self.async_show_form(
            step_id="add_service",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_SERVICE_TYPE): vol.In(
                        {service_type: SERVICE_TYPE_NAMES[service_type] for service_type in SERVICE_TYPES}
                    ),
                }
            ),
        )

    async def async_step_conversation_config(
        self, service_data: dict[str, Any], user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Configure a conversation agent."""
        # Use models from API instead of hardcoded list
        available_models = service_data.get("available_models", [])
        # Filter for conversation models if we have the full list
        if available_models:
            # Use all available models for conversation
            conversation_models = available_models
        else:
            # Fallback to hardcoded models
            conversation_models = SERVICE_MODEL_MAP[SERVICE_TYPE_CONVERSATION]
        
        if user_input is not None:
            service_data.update(user_input)
            
            # Create the service config entry directly
            return self.async_create_entry(
                title=user_input.get(CONF_SERVICE_NAME, "LiteLLM Conversation"),
                data=service_data,
            )

        return self.async_show_form(
            step_id="conversation_config",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_SERVICE_NAME, default="LiteLLM Conversation"): str,
                    vol.Required(CONF_MODEL, default=conversation_models[0] if conversation_models else DEFAULT_MODEL, description="Model name (select from dropdown or enter custom model name)"): str,
                    vol.Optional(CONF_PROMPT, default=DEFAULT_CONF_PROMPT): cv.template,
                    vol.Optional(CONF_MAX_TOKENS, default=DEFAULT_MAX_TOKENS): int,
                    vol.Optional(CONF_TEMPERATURE, default=DEFAULT_TEMPERATURE): vol.All(vol.Coerce(float), vol.Range(min=0, max=2)),
                    vol.Optional(CONF_TOP_P, default=DEFAULT_TOP_P): vol.All(vol.Coerce(float), vol.Range(min=0, max=1)),
                    vol.Optional(CONF_PRESENCE_PENALTY, default=DEFAULT_PRESENCE_PENALTY): vol.All(vol.Coerce(float), vol.Range(min=-2, max=2)),
                    vol.Optional(CONF_FREQUENCY_PENALTY, default=DEFAULT_FREQUENCY_PENALTY): vol.All(vol.Coerce(float), vol.Range(min=-2, max=2)),
                }
            ),
        )

    async def async_step_stt_config(
        self, service_data: dict[str, Any], user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Configure an STT service."""
        # Use models from API that support STT, fallback to hardcoded
        available_models = service_data.get("available_models", [])
        # Filter for STT models or use fallback
        stt_models = [model for model in available_models if "whisper" in model.lower()] if available_models else SERVICE_MODEL_MAP[SERVICE_TYPE_STT]
        if not stt_models:
            stt_models = SERVICE_MODEL_MAP[SERVICE_TYPE_STT]
        
        if user_input is not None:
            service_data.update(user_input)
            
            # Create the service config entry directly
            return self.async_create_entry(
                title=user_input.get(CONF_SERVICE_NAME, "LiteLLM STT"),
                data=service_data,
            )

        return self.async_show_form(
            step_id="stt_config",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_SERVICE_NAME, default="LiteLLM STT"): str,
                    vol.Required(CONF_MODEL, default=stt_models[0] if stt_models else "whisper-1", description="Model name (e.g., whisper-1 or custom model)"): str,
                }
            ),
        )

    async def async_step_tts_config(
        self, service_data: dict[str, Any], user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Configure a TTS service."""
        # Use models from API that support TTS, fallback to hardcoded
        available_models = service_data.get("available_models", [])
        # Filter for TTS models or use fallback
        tts_models = [model for model in available_models if "tts" in model.lower()] if available_models else SERVICE_MODEL_MAP[SERVICE_TYPE_TTS]
        if not tts_models:
            tts_models = SERVICE_MODEL_MAP[SERVICE_TYPE_TTS]
        
        if user_input is not None:
            service_data.update(user_input)
            
            # Create the service config entry directly
            return self.async_create_entry(
                title=user_input.get(CONF_SERVICE_NAME, "LiteLLM TTS"),
                data=service_data,
            )

        return self.async_show_form(
            step_id="tts_config",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_SERVICE_NAME, default="LiteLLM TTS"): str,
                    vol.Required(CONF_MODEL, default=tts_models[0] if tts_models else "tts-1", description="Model name (e.g., tts-1, tts-1-hd, or custom model)"): str,
                }
            ),
        )

    async def async_step_ai_task_config(
        self, service_data: dict[str, Any], user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Configure an AI Task service."""
        # Use models from API that support vision, fallback to hardcoded
        available_models = service_data.get("available_models", [])
        # For AI Task, prefer vision models or use all models as fallback
        vision_models = available_models if available_models else SERVICE_MODEL_MAP[SERVICE_TYPE_AI_TASK]
        
        if user_input is not None:
            service_data.update(user_input)
            
            # Create the service config entry directly
            return self.async_create_entry(
                title=user_input.get(CONF_SERVICE_NAME, "LiteLLM AI Task"),
                data=service_data,
            )

        return self.async_show_form(
            step_id="ai_task_config",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_SERVICE_NAME, default="LiteLLM AI Task"): str,
                    vol.Required(CONF_MODEL, default=vision_models[0] if vision_models else "gpt-4o", description="Vision model name (e.g., gpt-4o, claude-3-5-sonnet, or custom model)"): str,
                }
            ),
        )


class CannotConnect(Exception):
    """Error to indicate we cannot connect."""


class InvalidAuth(Exception):
    """Error to indicate there is invalid auth."""