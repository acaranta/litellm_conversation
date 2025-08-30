"""Constants for LiteLLM Conversation."""
from __future__ import annotations

from homeassistant.components.conversation import DOMAIN as CONVERSATION_DOMAIN
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.helpers import llm

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

DEFAULT_CONF_PROMPT = """You are a voice assistant for Home Assistant.

Current Home: {{ ha_name }}
User: {{ user_name }}
{% if device_id %}Device: {{ device_id }}{% endif %}
{% if language %}Language: {{ language }}{% endif %}

{% if llm_context.assistant == 'assist' %}
You can help control Home Assistant devices and answer questions about the home.

An overview of the areas and devices in this smart home:
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

Use the tools available to help control devices and answer questions about the smart home.
{% else %}
You are a helpful assistant. Answer questions and provide information, but you cannot control any devices.
{% endif %}

Be helpful, friendly, and concise in your responses."""
DEFAULT_LLM_HASS_API = [llm.LLM_API_ASSIST]

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

VISION_MODELS = [
    "gpt-4-vision-preview",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-haiku-20240307",
    "claude-3-sonnet-20240229",
    "claude-3-opus-20240229",
    "claude-3-5-sonnet-20241022",
    "gemini-pro-vision",
    "gemini-1.5-pro",
]

EVENT_CONVERSATION_PROCESS = f"{DOMAIN}_process"

CONF_CHAT_MODEL = CONF_MODEL
CONF_VISION_MODEL = "vision_model"
CONF_SERVICE_TYPE = "service_type"
CONF_SERVICE_NAME = "service_name"

DEFAULT_VISION_MODEL = "gpt-4o"

# Service types for multi-instance setup
SERVICE_TYPE_CONVERSATION = "conversation"
SERVICE_TYPE_STT = "stt"
SERVICE_TYPE_TTS = "tts"
SERVICE_TYPE_AI_TASK = "ai_task"

SERVICE_TYPES = [
    SERVICE_TYPE_CONVERSATION,
    SERVICE_TYPE_STT,
    SERVICE_TYPE_TTS,
    SERVICE_TYPE_AI_TASK,
]

SERVICE_TYPE_NAMES = {
    SERVICE_TYPE_CONVERSATION: "Conversation Agent",
    SERVICE_TYPE_STT: "Speech-to-Text Service", 
    SERVICE_TYPE_TTS: "Text-to-Speech Service",
    SERVICE_TYPE_AI_TASK: "AI Task Service",
}

# Model categories for different services
CONVERSATION_MODELS = RECOMMENDED_MODELS
STT_MODELS = ["whisper-1"]
TTS_MODELS = ["tts-1", "tts-1-hd"]
AI_TASK_MODELS = VISION_MODELS

SERVICE_MODEL_MAP = {
    SERVICE_TYPE_CONVERSATION: CONVERSATION_MODELS,
    SERVICE_TYPE_STT: STT_MODELS,
    SERVICE_TYPE_TTS: TTS_MODELS,
    SERVICE_TYPE_AI_TASK: AI_TASK_MODELS,
}