"""Support for LiteLLM speech-to-text service."""
from __future__ import annotations

import asyncio
import io
import logging
from collections.abc import AsyncIterable
from typing import Any

import aiohttp
import async_timeout
from aiohttp import FormData

from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
    SpeechToTextEntity,
)
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
    SERVICE_TYPE_STT,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up LiteLLM STT platform via config entry."""
    # Only set up if this is an STT service entry
    if config_entry.data.get(CONF_SERVICE_TYPE) == SERVICE_TYPE_STT:
        async_add_entities([LiteLLMSTTEntity(config_entry)])


class LiteLLMSTTEntity(SpeechToTextEntity):
    """LiteLLM speech-to-text entity."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize LiteLLM STT entity."""
        self._config_entry = config_entry
        self._attr_name = config_entry.data.get(CONF_SERVICE_NAME, "LiteLLM STT")
        self._attr_unique_id = config_entry.entry_id

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
    def supported_formats(self) -> list[AudioFormats]:
        """Return list of supported formats."""
        return [AudioFormats.WAV, AudioFormats.OGG, AudioFormats.FLAC]

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        """Return list of supported codecs."""
        return [AudioCodecs.PCM, AudioCodecs.OPUS, AudioCodecs.FLAC]

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        """Return list of supported bit rates."""
        return [AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        """Return list of supported sample rates."""
        return [AudioSampleRates.SAMPLERATE_16000]

    @property
    def supported_channels(self) -> list[AudioChannels]:
        """Return list of supported channels."""
        return [AudioChannels.CHANNEL_MONO]

    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> SpeechResult:
        """Process an audio stream to STT service."""
        
        # Collect audio data
        audio_data = b""
        async for chunk in stream:
            audio_data += chunk
        
        if not audio_data:
            return SpeechResult("", SpeechResultState.ERROR)

        # Create form data for the API request
        model = self._config_entry.data.get(CONF_MODEL, "whisper-1")
        data = FormData()
        data.add_field("file", io.BytesIO(audio_data), filename="audio.wav", content_type="audio/wav")
        data.add_field("model", model)
        if metadata.language:
            data.add_field("language", metadata.language)

        session = async_get_clientsession(self.hass)
        base_url = self._config_entry.data[CONF_BASE_URL].rstrip("/")
        
        headers = {
            "Authorization": f"Bearer {self._config_entry.data[CONF_API_KEY]}",
        }

        try:
            async with async_timeout.timeout(60):
                async with session.post(
                    f"{base_url}/audio/transcriptions",
                    headers=headers,
                    data=data,
                ) as response:
                    response.raise_for_status()
                    result = await response.json()

            if "text" in result:
                return SpeechResult(result["text"], SpeechResultState.SUCCESS)
            else:
                _LOGGER.error("No text found in STT response: %s", result)
                return SpeechResult("", SpeechResultState.ERROR)

        except aiohttp.ClientResponseError as err:
            _LOGGER.error("STT request failed with status %s: %s", err.status, err)
            return SpeechResult("", SpeechResultState.ERROR)
        except (aiohttp.ClientError, asyncio.TimeoutError) as err:
            _LOGGER.error("STT request failed: %s", err)
            return SpeechResult("", SpeechResultState.ERROR)
        except Exception as err:
            _LOGGER.exception("Unexpected error during STT processing: %s", err)
            return SpeechResult("", SpeechResultState.ERROR)