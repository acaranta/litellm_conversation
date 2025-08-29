# LiteLLM Conversation Integration for Home Assistant

[![GitHub Release][releases-shield]][releases]
[![GitHub Activity][commits-shield]][commits]
[![License][license-shield]](LICENSE)

[![hacs][hacsbadge]][hacs]
[![Project Maintenance][maintenance-shield]][user_profile]

_Integration to integrate with [LiteLLM](https://github.com/BerriAI/litellm)._

**This integration will set up the following platforms.**

Platform | Description
-- | --
`conversation` | Conversation agent powered by LiteLLM models
`stt` | Speech-to-text service using LiteLLM
`tts` | Text-to-speech service using LiteLLM

## Features

- **Flexible AI Model Support**: Works with 100+ LLM providers through LiteLLM proxy
- **Custom Base URL**: Configure your LiteLLM proxy endpoint 
- **Model Selection**: Choose from a wide variety of models including OpenAI, Claude, Gemini, and more
- **Voice Integration**: Full STT and TTS support for voice conversations
- **OpenAI Compatibility**: Drop-in replacement for OpenAI Conversation component
- **Rich Configuration**: Temperature, max tokens, presence penalty, and more
- **Template Support**: Customizable system prompts with Home Assistant context

## Installation

### HACS Installation (Recommended)

1. Open HACS in your Home Assistant instance
2. Go to "Integrations" 
3. Click the three dots menu in the top right
4. Select "Custom repositories"
5. Add this repository URL: `https://github.com/acaranta/litellm_conversation`
6. Select "Integration" as the category
7. Click "Add"
8. Search for "LiteLLM Conversation" in HACS
9. Click "Install"
10. Restart Home Assistant

### Manual Installation

1. Using the tool of choice open the directory (folder) for your HA configuration (where you find `configuration.yaml`)
2. If you do not have a `custom_components` directory (folder) there, you need to create it
3. In the `custom_components` directory (folder) create a new folder called `litellm_conversation`
4. Download _all_ the files from the `custom_components/litellm_conversation/` directory (folder) in this repository
5. Place the files you downloaded in the new directory (folder) you created
6. Restart Home Assistant

## Configuration

### Setting up LiteLLM Proxy

Before configuring this integration, you need a running LiteLLM proxy. You can:

1. **Use the official LiteLLM proxy**: Follow the [LiteLLM documentation](https://docs.litellm.ai/docs/simple_proxy) to set up your own proxy
2. **Use OpenAI directly**: Set base URL to `https://api.openai.com/v1` (default)
3. **Use any OpenAI-compatible endpoint**: Set your custom base URL

### Home Assistant Configuration

1. In the HA UI go to "Configuration" -> "Integrations" click "+" and search for "LiteLLM Conversation"
2. Enter your configuration:
   - **Base URL**: Your LiteLLM proxy endpoint (default: `https://api.openai.com/v1`)
   - **API Key**: Your API key for the service
   - **Model**: Choose from available models (will be fetched from your endpoint)
   - **System Prompt**: Customize how the AI assistant behaves

### Configuration Options

You can configure the following options in the integration options:

- **System Prompt**: Customize the AI assistant's behavior and context
- **Model**: Select from available models on your LiteLLM proxy
- **Max Tokens**: Maximum tokens in the response (default: 150)
- **Temperature**: Controls randomness (0.0 to 2.0, default: 1.0)
- **Top P**: Controls diversity via nucleus sampling (0.0 to 1.0, default: 1.0)  
- **Presence Penalty**: Penalizes repeated topics (-2.0 to 2.0, default: 0.0)
- **Frequency Penalty**: Penalizes repeated tokens (-2.0 to 2.0, default: 0.0)

## Usage

### As a Conversation Agent

Once configured, the integration creates a conversation agent that you can use in:

- Voice assistants
- Conversation intents  
- Automations
- Scripts

### Voice Integration

The integration provides both STT and TTS services:

1. **Speech-to-Text**: Configure in Home Assistant's STT settings
2. **Text-to-Speech**: Configure in Home Assistant's TTS settings

### Automation Example

```yaml
automation:
  - alias: "Ask AI Assistant"
    trigger:
      platform: event
      event_type: call_service
    action:
      service: conversation.process
      data:
        agent_id: conversation.litellm_conversation
        text: "What's the weather like?"
```

## Supported Models

The integration supports any model available through your LiteLLM proxy, including:

- **OpenAI**: gpt-3.5-turbo, gpt-4, gpt-4-turbo-preview, gpt-4o
- **Anthropic**: claude-3-haiku, claude-3-sonnet, claude-3-opus, claude-3.5-sonnet  
- **Google**: gemini-pro, gemini-1.5-pro
- **Meta**: llama-2-70b-chat, llama-3-70b-chat
- **Mistral**: mixtral-8x7b-instruct
- **And many more...**

## Troubleshooting

### Common Issues

1. **Cannot connect error**: 
   - Verify your base URL is correct and accessible
   - Check that your API key is valid
   - Ensure your LiteLLM proxy is running

2. **Model not found**:
   - Verify the model is available on your LiteLLM proxy
   - Check the `/models` endpoint of your proxy

3. **Permission denied**:
   - Verify your API key has the correct permissions
   - Check rate limits on your provider

### Debug Logging

To enable debug logging, add this to your `configuration.yaml`:

```yaml
logger:
  default: info
  logs:
    custom_components.litellm_conversation: debug
```

## Contributing

If you want to contribute to this please read the [Contribution guidelines](CONTRIBUTING.md)

## Credits

This project is based on the official Home Assistant OpenAI Conversation integration and inspired by the Google Generative AI Conversation integration.

---

[litellm_conversation]: https://github.com/acaranta/litellm_conversation
[commits-shield]: https://img.shields.io/github/commit-activity/y/acaranta/litellm_conversation.svg?style=for-the-badge
[commits]: https://github.com/acaranta/litellm_conversation/commits/main
[hacs]: https://github.com/hacs/integration
[hacsbadge]: https://img.shields.io/badge/HACS-Custom-orange.svg?style=for-the-badge
[license-shield]: https://img.shields.io/github/license/acaranta/litellm_conversation.svg?style=for-the-badge
[maintenance-shield]: https://img.shields.io/badge/maintainer-%40acaranta-blue.svg?style=for-the-badge
[releases-shield]: https://img.shields.io/github/release/acaranta/litellm_conversation.svg?style=for-the-badge
[releases]: https://github.com/acaranta/litellm_conversation/releases
[user_profile]: https://github.com/acaranta