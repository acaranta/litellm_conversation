"""Config flow for LiteLLM Conversation integration."""
from __future__ import annotations

import logging
from typing import Any
import aiohttp
import async_timeout

import voluptuous as vol

from homeassistant import config_entries, core
from homeassistant.const import CONF_API_KEY
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers.aiohttp_client import async_get_clientsession
import homeassistant.helpers.config_validation as cv

from .const import (
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PRESENCE_PENALTY,
    CONF_FREQUENCY_PENALTY,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DEFAULT_BASE_URL,
    DEFAULT_CONF_PROMPT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_PRESENCE_PENALTY,
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DOMAIN,
    RECOMMENDED_MODELS,
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
                # Store available models for options flow
                user_input["available_models"] = info["models"]
                return self.async_create_entry(title=info["title"], data=user_input)

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_BASE_URL, default=DEFAULT_BASE_URL): str,
                    vol.Required(CONF_API_KEY): str,
                    vol.Optional(CONF_CHAT_MODEL, default=DEFAULT_MODEL): vol.In(RECOMMENDED_MODELS),
                    vol.Optional(CONF_PROMPT, default=DEFAULT_CONF_PROMPT): str,
                }
            ),
            errors=errors,
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
        if user_input is not None:
            return self.async_create_entry(
                title="",
                data=user_input,
            )
        
        # Get available models from stored config data
        available_models = self.config_entry.data.get("available_models", RECOMMENDED_MODELS)
        current_model = self.config_entry.options.get(CONF_CHAT_MODEL, self.config_entry.data.get(CONF_CHAT_MODEL, DEFAULT_MODEL))
        
        schema = vol.Schema(
            {
                vol.Optional(
                    CONF_PROMPT,
                    description={
                        "suggested_value": self.config_entry.options.get(
                            CONF_PROMPT, DEFAULT_PROMPT
                        )
                    },
                ): cv.template,
                vol.Optional(
                    CONF_CHAT_MODEL,
                    description={"suggested_value": current_model},
                ): vol.In(available_models),
                vol.Optional(
                    CONF_MAX_TOKENS,
                    description={
                        "suggested_value": self.config_entry.options.get(
                            CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS
                        )
                    },
                ): int,
                vol.Optional(
                    CONF_TOP_P,
                    description={
                        "suggested_value": self.config_entry.options.get(
                            CONF_TOP_P, DEFAULT_TOP_P
                        )
                    },
                ): vol.All(vol.Coerce(float), vol.Range(min=0, max=1)),
                vol.Optional(
                    CONF_TEMPERATURE,
                    description={
                        "suggested_value": self.config_entry.options.get(
                            CONF_TEMPERATURE, DEFAULT_TEMPERATURE
                        )
                    },
                ): vol.All(vol.Coerce(float), vol.Range(min=0, max=2)),
                vol.Optional(
                    CONF_PRESENCE_PENALTY,
                    description={
                        "suggested_value": self.config_entry.options.get(
                            CONF_PRESENCE_PENALTY, DEFAULT_PRESENCE_PENALTY
                        )
                    },
                ): vol.All(vol.Coerce(float), vol.Range(min=-2, max=2)),
                vol.Optional(
                    CONF_FREQUENCY_PENALTY,
                    description={
                        "suggested_value": self.config_entry.options.get(
                            CONF_FREQUENCY_PENALTY, DEFAULT_FREQUENCY_PENALTY
                        )
                    },
                ): vol.All(vol.Coerce(float), vol.Range(min=-2, max=2)),
            }
        )

        return self.async_show_form(
            step_id="init",
            data_schema=schema,
        )


class CannotConnect(Exception):
    """Error to indicate we cannot connect."""


class InvalidAuth(Exception):
    """Error to indicate there is invalid auth."""