"""Constants for LiteLLM Conversation."""
from __future__ import annotations

from homeassistant.components.conversation import DOMAIN as CONVERSATION_DOMAIN

DOMAIN = "litellm_conversation"

CONF_BASE_URL = "base_url"
CONF_API_KEY = "api_key"
CONF_MODEL = "model"
CONF_MAX_TOKENS = "max_tokens"
CONF_TEMPERATURE = "temperature"
CONF_TOP_P = "top_p"
CONF_PRESENCE_PENALTY = "presence_penalty"
CONF_FREQUENCY_PENALTY = "frequency_penalty"
CONF_PROMPT = "prompt"

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_MAX_TOKENS = 150
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_PRESENCE_PENALTY = 0.0
DEFAULT_FREQUENCY_PENALTY = 0.0
DEFAULT_PROMPT = """This smart home is controlled by Home Assistant.

An overview of the areas and the devices in this smart home:
{%- for area in areas() %}
  {%- set area_info = namespace(printed=false) %}
  {%- for device in area_devices(area) -%}
    {%- if not device_attr(device, "disabled_by") and not device_attr(device, "entry_type") and device_attr(device, "name") %}
      {%- if not area_info.printed %}

{{ area_name(area) }}:
        {%- set area_info.printed = true %}
      {%- endif %}
- {{ device_attr(device, "name") }}{% if device_attr(device, "model") and (device_attr(device, "model") | string) not in (device_attr(device, "name") | string) %} ({{ device_attr(device, "model") }}){% endif %}
    {%- endif %}
  {%- endfor %}
{%- endfor %}

Answer the user's questions about the world truthfully. If the user wants to control a device, reject the request and suggest using the Home Assistant app.
"""

DEFAULT_CONF_PROMPT = "Be helpful and friendly. Answer as a personal assistant."

RECOMMENDED_MODELS = [
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4-turbo-preview",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-haiku-20240307",
    "claude-3-sonnet-20240229", 
    "claude-3-opus-20240229",
    "claude-3-5-sonnet-20241022",
    "gemini-pro",
    "gemini-1.5-pro",
    "llama-2-70b-chat",
    "llama-3-70b-chat",
    "mixtral-8x7b-instruct",
]

EVENT_CONVERSATION_PROCESS = f"{DOMAIN}_process"

CONF_CHAT_MODEL = CONF_MODEL