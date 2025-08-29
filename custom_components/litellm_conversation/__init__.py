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

# Platform mapping for service types
SERVICE_PLATFORM_MAP = {
    SERVICE_TYPE_CONVERSATION: Platform.CONVERSATION,
    SERVICE_TYPE_STT: Platform.STT,
    SERVICE_TYPE_TTS: Platform.TTS,
    SERVICE_TYPE_AI_TASK: Platform.AI_TASK,
}


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up LiteLLM Conversation from a config entry."""
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = entry.data

    # Check if this is a main integration entry or a service sub-entry
    service_type = entry.data.get(CONF_SERVICE_TYPE)
    
    if service_type:
        # This is a service sub-entry, forward to appropriate platform
        platform = SERVICE_PLATFORM_MAP.get(service_type)
        if platform:
            await hass.config_entries.async_forward_entry_setups(entry, [platform])
    else:
        # This is the main integration entry, no platforms to forward
        # Sub-entries will be created through the config flow
        pass

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    service_type = entry.data.get(CONF_SERVICE_TYPE)
    
    if service_type:
        # This is a service sub-entry, unload the appropriate platform
        platform = SERVICE_PLATFORM_MAP.get(service_type)
        if platform:
            unload_ok = await hass.config_entries.async_unload_platforms(entry, [platform])
        else:
            unload_ok = True
    else:
        # This is the main integration entry, no platforms to unload
        unload_ok = True
    
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id, None)

    return unload_ok