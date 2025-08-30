"""Support for LiteLLM AI Task service."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any

import aiohttp
import async_timeout

from homeassistant.components.ai_task import (
    AITaskEntity,
    AITaskEntityFeature,
    GenDataTask,
    GenDataTaskResult,
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
    CONF_VISION_MODEL,
    DEFAULT_VISION_MODEL,
    DOMAIN,
    SERVICE_TYPE_AI_TASK,
    VISION_MODELS,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up LiteLLM AI Task platform via config entry."""
    # Set up AI Task entities from subentries
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != SERVICE_TYPE_AI_TASK:
            continue

        async_add_entities(
            [LiteLLMAITaskEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class LiteLLMAITaskEntity(AITaskEntity):
    """LiteLLM AI Task entity."""

    def __init__(self, config_entry: ConfigEntry, subentry: ConfigEntry) -> None:
        """Initialize LiteLLM AI Task entity."""
        self._config_entry = config_entry
        self._subentry = subentry
        self._attr_name = subentry.title
        self._attr_unique_id = f"{config_entry.entry_id}_{subentry.subentry_id}"

    @property
    def supported_features(self) -> AITaskEntityFeature:
        """Return the supported features."""
        return AITaskEntityFeature.GENERATE_DATA

    async def _async_generate_data(
        self, task: GenDataTask, **kwargs: Any
    ) -> GenDataTaskResult:
        """Generate data based on the task instructions."""
        
        # Get the model from the service configuration
        vision_model = self._subentry.data.get(CONF_MODEL, DEFAULT_VISION_MODEL)
        
        # Prepare messages for the API call
        messages = [
            {"role": "user", "content": self._prepare_content(task)}
        ]

        session = async_get_clientsession(self.hass)
        base_url = self._config_entry.data[CONF_BASE_URL].rstrip("/")
        
        headers = {
            "Authorization": f"Bearer {self._config_entry.data[CONF_API_KEY]}",
            "Content-Type": "application/json",
        }

        data = {
            "model": vision_model,
            "messages": messages,
            "max_tokens": 1000,  # Increased for AI tasks
            "temperature": 0.1,  # Lower for more consistent results
        }

        _LOGGER.debug("AI Task request data: %s", data)

        try:
            async with async_timeout.timeout(60):  # Longer timeout for AI tasks
                async with session.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=data,
                ) as response:
                    response.raise_for_status()
                    result = await response.json()

        except aiohttp.ClientResponseError as err:
            _LOGGER.error("AI Task request failed with status %s: %s", err.status, err)
            return GenDataTaskResult(
                error_code="request_failed",
                error_message=f"Request failed with status {err.status}",
            )
        except (aiohttp.ClientError, asyncio.TimeoutError) as err:
            _LOGGER.error("AI Task request failed: %s", err)
            return GenDataTaskResult(
                error_code="connection_error",
                error_message="Failed to connect to LiteLLM service",
            )

        _LOGGER.debug("AI Task response: %s", result)

        try:
            response_text = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            _LOGGER.error("Unexpected AI Task response format: %s", result)
            return GenDataTaskResult(
                error_code="invalid_response",
                error_message="Unexpected response format from LiteLLM",
            )

        # If the task expects structured output, try to parse it
        if task.structure:
            try:
                # Try to extract JSON from the response
                parsed_data = self._parse_structured_response(response_text, task.structure)
                return GenDataTaskResult(data=parsed_data)
            except Exception as err:
                _LOGGER.error("Failed to parse structured response: %s", err)
                # Fall back to returning the raw text
                return GenDataTaskResult(data={"text": response_text})
        else:
            # Return as plain text
            return GenDataTaskResult(data=response_text)

    def _prepare_content(self, task: GenDataTask) -> list[dict[str, Any]]:
        """Prepare the content for the API call, including attachments."""
        content_parts = []
        
        # Add the main instruction
        content_parts.append({
            "type": "text",
            "text": task.instructions
        })
        
        # Add structured output instructions if needed
        if task.structure:
            structure_text = (
                f"\n\nPlease format your response as JSON according to this structure: "
                f"{json.dumps(task.structure, indent=2)}"
            )
            content_parts.append({
                "type": "text", 
                "text": structure_text
            })

        # Process attachments
        if task.attachments:
            for attachment in task.attachments:
                if attachment.media_type.startswith("image/"):
                    # Handle image attachments
                    image_data = base64.b64encode(attachment.data).decode()
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{attachment.media_type};base64,{image_data}",
                            "detail": "high"
                        }
                    })
                else:
                    # Handle other file types as text if possible
                    try:
                        text_content = attachment.data.decode('utf-8')
                        content_parts.append({
                            "type": "text",
                            "text": f"\n\nAttachment content ({attachment.filename}):\n{text_content}"
                        })
                    except UnicodeDecodeError:
                        _LOGGER.warning("Cannot process binary file: %s", attachment.filename)

        return content_parts

    def _parse_structured_response(self, response_text: str, structure: dict[str, Any]) -> dict[str, Any]:
        """Parse the response text to match the expected structure."""
        # First, try to find JSON in the response
        import re
        
        # Look for JSON blocks in the response
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        json_match = re.search(json_pattern, response_text, re.DOTALL)
        
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to parse the entire response as JSON
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
            
        # Look for inline JSON
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        json_matches = re.findall(json_pattern, response_text)
        
        for match in json_matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
                
        # If no valid JSON found, create a structured response based on the text
        if "count" in structure or "number" in structure:
            # Try to extract numbers for counting tasks
            numbers = re.findall(r'\b\d+\b', response_text)
            if numbers:
                return {"count": int(numbers[0]), "description": response_text}
        
        # Default fallback
        return {"result": response_text}