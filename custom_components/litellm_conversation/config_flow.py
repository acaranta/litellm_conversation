"""Config flow for LiteLLM Conversation integration."""
from __future__ import annotations

import logging
from typing import Any

import aiohttp
import async_timeout
import voluptuous as vol

from homeassistant import config_entries, core
from homeassistant.const import CONF_API_KEY
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers.aiohttp_client import async_get_clientsession
import homeassistant.helpers.config_validation as cv
from homeassistant.helpers.selector import (
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from .const import (
    CONF_BASE_URL,
    CONF_FREQUENCY_PENALTY,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_PRESENCE_PENALTY,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DEFAULT_BASE_URL,
    DEFAULT_CONF_PROMPT,
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_PRESENCE_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DOMAIN,
    SERVICE_TYPE_AI_TASK,
    SERVICE_TYPE_CONVERSATION,
    SERVICE_TYPE_STT,
    SERVICE_TYPE_TTS,
    SERVICE_TYPE_NAMES,
)

_LOGGER = logging.getLogger(__name__)

SUPPORTED_SUBENTRY_TYPES = [
    SERVICE_TYPE_CONVERSATION,
    SERVICE_TYPE_STT,
    SERVICE_TYPE_TTS,
    SERVICE_TYPE_AI_TASK,
]


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

    @staticmethod
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> OptionsFlowHandler:
        """Create the options flow."""
        return OptionsFlowHandler(config_entry)

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
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

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle reconfiguration of the integration."""
        config_entry = self._get_reconfigure_entry()
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


class OptionsFlowHandler(config_entries.OptionsFlow):
    """LiteLLM Conversation config flow options handler."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize LiteLLM Conversation options flow."""
        super().__init__()

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        return self.async_show_menu(
            step_id="init",
            menu_options={
                "add_conversation": "Add Conversation Agent",
                "add_stt": "Add Speech-to-Text Service", 
                "add_tts": "Add Text-to-Speech Service",
                "add_ai_task": "Add AI Task Service",
            },
        )

    async def async_step_add_conversation(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Add a conversation service."""
        return await self._add_subentry(SERVICE_TYPE_CONVERSATION, user_input)

    async def async_step_add_stt(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Add an STT service.""" 
        return await self._add_subentry(SERVICE_TYPE_STT, user_input)

    async def async_step_add_tts(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Add a TTS service."""
        return await self._add_subentry(SERVICE_TYPE_TTS, user_input)

    async def async_step_add_ai_task(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Add an AI Task service."""
        return await self._add_subentry(SERVICE_TYPE_AI_TASK, user_input)

    async def _add_subentry(
        self, service_type: str, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Add a service subentry."""
        available_models = self.config_entry.data.get("available_models", [])
        
        # Get default model based on service type
        if service_type == SERVICE_TYPE_STT:
            default_model = "whisper-1"
            if available_models:
                default_model = next((m for m in available_models if "whisper" in m.lower()), default_model)
        elif service_type == SERVICE_TYPE_TTS:
            default_model = "tts-1"
            if available_models:
                default_model = next((m for m in available_models if "tts" in m.lower()), default_model)
        elif service_type == SERVICE_TYPE_AI_TASK:
            default_model = "gpt-4o"
            # Use first available model or fallback
            if available_models:
                default_model = available_models[0]
        else:  # conversation
            default_model = DEFAULT_MODEL
            if available_models:
                default_model = available_models[0]

        if user_input is not None:
            # Create subentry data
            subentry_data = {
                CONF_MODEL: user_input[CONF_MODEL],
            }
            
            # Add service-specific fields
            if service_type == SERVICE_TYPE_CONVERSATION:
                subentry_data.update({
                    CONF_PROMPT: user_input.get(CONF_PROMPT, DEFAULT_CONF_PROMPT),
                    CONF_MAX_TOKENS: user_input.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
                    CONF_TEMPERATURE: user_input.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
                    CONF_TOP_P: user_input.get(CONF_TOP_P, DEFAULT_TOP_P),
                    CONF_PRESENCE_PENALTY: user_input.get(CONF_PRESENCE_PENALTY, DEFAULT_PRESENCE_PENALTY),
                    CONF_FREQUENCY_PENALTY: user_input.get(CONF_FREQUENCY_PENALTY, DEFAULT_FREQUENCY_PENALTY),
                })

            # Create a new config entry for the service directly
            entry_data = {
                "service_type": service_type,
                CONF_BASE_URL: self.config_entry.data[CONF_BASE_URL],
                CONF_API_KEY: self.config_entry.data[CONF_API_KEY],
                "parent_entry_id": self.config_entry.entry_id,
                **subentry_data,
            }

            await self.hass.config_entries.async_add(
                domain=DOMAIN,
                title=f"LiteLLM {SERVICE_TYPE_NAMES[service_type]}",
                data=entry_data,
            )
            
            return self.async_create_entry(
                title="",
                data={},
            )

        # Build form schema
        schema_fields = {
            vol.Required(CONF_MODEL, default=default_model): TextSelector(),
        }
        
        # Add service-specific fields
        if service_type == SERVICE_TYPE_CONVERSATION:
            schema_fields.update({
                vol.Optional(CONF_PROMPT, default=DEFAULT_CONF_PROMPT): TextSelector(
                    TextSelectorConfig(multiline=True)
                ),
                vol.Optional(CONF_MAX_TOKENS, default=DEFAULT_MAX_TOKENS): NumberSelector(
                    NumberSelectorConfig(min=1, max=4096, mode=NumberSelectorMode.BOX)
                ),
                vol.Optional(CONF_TEMPERATURE, default=DEFAULT_TEMPERATURE): NumberSelector(
                    NumberSelectorConfig(min=0.0, max=2.0, step=0.1, mode=NumberSelectorMode.BOX)
                ),
                vol.Optional(CONF_TOP_P, default=DEFAULT_TOP_P): NumberSelector(
                    NumberSelectorConfig(min=0.0, max=1.0, step=0.1, mode=NumberSelectorMode.BOX)
                ),
                vol.Optional(CONF_PRESENCE_PENALTY, default=DEFAULT_PRESENCE_PENALTY): NumberSelector(
                    NumberSelectorConfig(min=-2.0, max=2.0, step=0.1, mode=NumberSelectorMode.BOX)
                ),
                vol.Optional(CONF_FREQUENCY_PENALTY, default=DEFAULT_FREQUENCY_PENALTY): NumberSelector(
                    NumberSelectorConfig(min=-2.0, max=2.0, step=0.1, mode=NumberSelectorMode.BOX)
                ),
            })

        return self.async_show_form(
            step_id=f"add_{service_type}",
            data_schema=vol.Schema(schema_fields),
        )



class CannotConnect(Exception):
    """Error to indicate we cannot connect."""


class InvalidAuth(Exception):
    """Error to indicate there is invalid auth."""