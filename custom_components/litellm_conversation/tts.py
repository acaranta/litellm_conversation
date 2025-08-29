"""Support for LiteLLM text-to-speech service."""
from __future__ import annotations

import asyncio
import logging

import aiohttp
import async_timeout

from homeassistant.components.tts import TextToSpeechEntity, TtsAudioType
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CONF_BASE_URL,
    CONF_MODEL,
    CONF_SERVICE_NAME,
    CONF_SERVICE_TYPE,
    DOMAIN,
    SERVICE_TYPE_TTS,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up LiteLLM TTS platform via config entry."""
    # Set up TTS entities from subentries
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != SERVICE_TYPE_TTS:
            continue

        async_add_entities(
            [LiteLLMTTSEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class LiteLLMTTSEntity(TextToSpeechEntity):
    """LiteLLM text-to-speech entity."""

    def __init__(self, config_entry: ConfigEntry, subentry: ConfigEntry) -> None:
        """Initialize LiteLLM TTS entity."""
        self._config_entry = config_entry
        self._subentry = subentry
        self._attr_name = subentry.title
        self._attr_unique_id = f"{config_entry.entry_id}_{subentry.subentry_id}"

    @property
    def supported_languages(self) -> list[str]:
        """Return list of supported languages."""
        return [
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

    @property
    def supported_options(self) -> list[str]:
        """Return list of supported options."""
        return ["voice"]

    @property
    def default_options(self) -> dict[str, str]:
        """Return default options."""
        return {"voice": "alloy"}

    async def async_get_tts_audio(
        self, message: str, language: str, options: dict[str, str]
    ) -> TtsAudioType:
        """Load TTS audio file from the LiteLLM API."""
        
        voice = options.get("voice", "alloy")
        model = self._subentry.data.get(CONF_MODEL, "tts-1")
        
        session = async_get_clientsession(self.hass)
        base_url = self._config_entry.data[CONF_BASE_URL].rstrip("/")
        
        headers = {
            "Authorization": f"Bearer {self._config_entry.data[CONF_API_KEY]}",
            "Content-Type": "application/json",
        }

        data = {
            "model": model,
            "input": message,
            "voice": voice,
            "response_format": "mp3",
        }

        try:
            async with async_timeout.timeout(60):
                async with session.post(
                    f"{base_url}/audio/speech",
                    headers=headers,
                    json=data,
                ) as response:
                    response.raise_for_status()
                    audio_data = await response.read()
                    
                    return ("mp3", audio_data)

        except aiohttp.ClientResponseError as err:
            _LOGGER.error("TTS request failed with status %s: %s", err.status, err)
            return None
        except (aiohttp.ClientError, asyncio.TimeoutError) as err:
            _LOGGER.error("TTS request failed: %s", err)
            return None
        except Exception as err:
            _LOGGER.exception("Unexpected error during TTS processing: %s", err)
            return None