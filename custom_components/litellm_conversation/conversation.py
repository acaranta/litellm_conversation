"""Conversation support for LiteLLM."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Literal

import aiohttp
import async_timeout

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady, TemplateError
from homeassistant.helpers import config_validation as cv, intent, template
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.util import ulid

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
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_PRESENCE_PENALTY,
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DOMAIN,
    SERVICE_TYPE_CONVERSATION,
)

_LOGGER = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = [
    "af",
    "ar",
    "bg",
    "bn",
    "ca",
    "cs",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "fa",
    "fi",
    "fr",
    "he",
    "hi",
    "hr",
    "hu",
    "id",
    "is",
    "it",
    "ja",
    "ko",
    "lv",
    "ms",
    "nb",
    "nl",
    "nn",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sl",
    "sr",
    "sv",
    "th",
    "tr",
    "uk",
    "ur",
    "vi",
    "zh",
]


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: conversation.ConversationEntitySetupCallback,
) -> None:
    """Set up conversation entities."""
    _LOGGER.debug("Setting up conversation entities for entry %s", config_entry.entry_id)
    _LOGGER.debug("Found %d subentries", len(config_entry.subentries))
    
    # Set up conversation agents from subentries
    conversation_count = 0
    for subentry in config_entry.subentries.values():
        _LOGGER.debug("Processing subentry %s of type %s", subentry.subentry_id, subentry.subentry_type)
        if subentry.subentry_type != SERVICE_TYPE_CONVERSATION:
            continue

        conversation_count += 1
        _LOGGER.debug("Adding conversation entity %d", conversation_count)
        async_add_entities(
            [LiteLLMConversationEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )
    
    _LOGGER.debug("Added %d conversation entities", conversation_count)


class LiteLLMConversationEntity(conversation.ConversationEntity):
    """LiteLLM conversation agent."""

    def __init__(self, entry: ConfigEntry, subentry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.entry = entry
        self.subentry = subentry
        self.history: dict[str, list[dict]] = {}

    @property
    def name(self) -> str:
        """Return the name of the entity."""
        return self.subentry.title

    @property
    def unique_id(self) -> str:
        """Return a unique ID to use for this entity."""
        return f"{self.entry.entry_id}_{self.subentry.subentry_id}"

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass."""
        await super().async_added_to_hass()
        self.entry.async_on_unload(
            self.entry.add_update_listener(self._async_entry_update_listener)
        )

    async def _async_entry_update_listener(
        self, hass: HomeAssistant, entry: ConfigEntry
    ) -> None:
        """Handle options update."""
        # The entry is already updated in the config registry
        self.async_write_ha_state()

    @property
    def supported_features(self) -> conversation.ConversationEntityFeature:
        """Return the supported features."""
        return conversation.ConversationEntityFeature.CONTROL

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process a sentence."""
        # Read configuration from sub-entry data
        raw_prompt = self.subentry.data.get(CONF_PROMPT, DEFAULT_PROMPT)
        model = self.subentry.data.get(CONF_MODEL, DEFAULT_MODEL)
        max_tokens = self.subentry.data.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        top_p = self.subentry.data.get(CONF_TOP_P, DEFAULT_TOP_P)
        temperature = self.subentry.data.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        presence_penalty = self.subentry.data.get(CONF_PRESENCE_PENALTY, DEFAULT_PRESENCE_PENALTY)
        frequency_penalty = self.subentry.data.get(CONF_FREQUENCY_PENALTY, DEFAULT_FREQUENCY_PENALTY)
        
        # Get connection data from parent entry
        base_url = self.entry.data[CONF_BASE_URL]
        api_key = self.entry.data[CONF_API_KEY]

        if user_input.conversation_id in self.history:
            conversation_id = user_input.conversation_id
            messages = self.history[conversation_id]
        else:
            conversation_id = ulid.ulid()
            try:
                prompt = self._async_generate_prompt(raw_prompt)
            except TemplateError as err:
                _LOGGER.error("Error rendering prompt: %s", err)
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    f"Sorry, I had a problem with my template: {err}",
                )
                return conversation.ConversationResult(
                    response=intent_response, conversation_id=conversation_id
                )
            messages = [{"role": "system", "content": prompt}]

        messages.append({"role": "user", "content": user_input.text})

        _LOGGER.debug("Prompt for %s: %s", model, messages)

        client_session = async_get_clientsession(self.hass)
        base_url = self.entry.data[CONF_BASE_URL].rstrip("/")
        
        headers = {
            "Authorization": f"Bearer {self.entry.data[CONF_API_KEY]}",
            "Content-Type": "application/json",
        }

        data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "temperature": temperature,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "user": conversation_id,
        }

        _LOGGER.debug("Request data: %s", data)

        try:
            async with async_timeout.timeout(30):
                async with client_session.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=data,
                ) as response:
                    response.raise_for_status()
                    result = await response.json()

        except aiohttp.ClientResponseError as err:
            _LOGGER.error("Request failed: %s", err)
            if err.status == 401:
                error_msg = "Invalid API key"
            elif err.status == 429:
                error_msg = "Rate limit exceeded"
            else:
                error_msg = f"Request failed with status {err.status}"

            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                error_msg,
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        except (aiohttp.ClientError, asyncio.TimeoutError) as err:
            _LOGGER.error("Request failed: %s", err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                "Sorry, I had a problem talking to LiteLLM",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        _LOGGER.debug("Response: %s", result)

        try:
            response_message = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            _LOGGER.error("Unexpected response format: %s", result)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                "Sorry, I got an unexpected response from LiteLLM",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        messages.append({"role": "assistant", "content": response_message})
        self.history[conversation_id] = messages

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(response_message)
        return conversation.ConversationResult(
            response=intent_response, conversation_id=conversation_id
        )

    def _async_generate_prompt(self, raw_prompt: str) -> str:
        """Generate a prompt for the user."""
        return template.Template(raw_prompt, self.hass).async_render(
            {
                "ha_name": self.hass.config.location_name,
            },
            parse_result=False,
        )