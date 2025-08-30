"""The LiteLLM Conversation integration."""
from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from .const import (
    CONF_SERVICE_TYPE,
    DOMAIN,
    SERVICE_TYPE_AI_TASK,
    SERVICE_TYPE_CONVERSATION,
    SERVICE_TYPE_STT,
    SERVICE_TYPE_TTS,
)

PLATFORMS = (
    Platform.AI_TASK,
    Platform.CONVERSATION,
    Platform.STT,
    Platform.TTS,
)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up LiteLLM Conversation from a config entry."""
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = entry.data

    # Forward to all platforms - they will handle subentry filtering
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id, None)

    return unload_ok