# LiteLLM Conversation Integration for Home Assistant

[![GitHub Release][releases-shield]][releases]
[![GitHub Activity][commits-shield]][commits]
[![License][license-shield]](LICENSE)

[![hacs][hacsbadge]][hacs]
[![Project Maintenance][maintenance-shield]][user_profile]

## Foreword
This Integration :
- is it very early stages and does not currently works
- is VERY inspired byt [openai_conversation](https://github.com/home-assistant/core/tree/dev/homeassistant/components/openai_conversation) and [Google Generative AI](https://github.com/home-assistant/core/tree/dev/homeassistant/components/google_generative_ai_conversation) Core integrations
- is coded by a human partly (me) who is not skilled enough  and therefore uses Claude Code (AI) to try to make this integration exist. I am not denying in any way I am using AI, and I wish to make it clear here ^^
  
_Integration to integrate with [LiteLLM](https://github.com/BerriAI/litellm)._

**This integration will set up the following platforms.**

Platform | Description
-- | --
`conversation` | Conversation agent powered by LiteLLM models
`stt` | Speech-to-text service using LiteLLM
`tts` | Text-to-speech service using LiteLLM
`ai_task` | AI Task service for data generation and image analysis
## Features

- **Flexible AI Model Support**: Works with 100+ LLM providers through LiteLLM proxy
- **Custom Base URL**: Configure your LiteLLM proxy endpoint 
- **Model Selection**: Choose from a wide variety of models including OpenAI, Claude, Gemini, and more
- **Voice Integration**: Full STT and TTS support for voice conversations
- **AI Task Service**: Generate data and analyze images using vision models
- **Vision Model Support**: Process camera feeds and images with models like GPT-4 Vision, Claude 3 Vision, Gemini Pro Vision
- **File Processing**: Handle various file types including images, PDFs, and text files
- **Structured Output**: Generate data in specific formats for automation use
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

## Integration Icon

The LiteLLM Conversation integration uses the official LiteLLM rocket logo. For the icon to display properly in Home Assistant and HACS, it needs to be submitted to the [Home Assistant Brands Repository](https://github.com/home-assistant/brands).

### Current Icon Status: ⚠️ Pending Submission

The integration currently **does not display an icon** in the Home Assistant UI. This is normal for custom integrations that haven't been added to the brands repository yet.

### Icon Files Prepared

Icon files ready for brands repository submission are available in the `/brands/` directory:
- `icon.png` - Standard resolution (requires resizing to 256x256px)
- `icon@2x.png` - High DPI resolution (requires resizing to 512x512px)

**Note**: These files need to be properly resized and optimized before submission. See `/brands/README.md` for detailed instructions.

### How to Add the Icon

To enable the icon display:

1. **Fork the [Home Assistant Brands Repository](https://github.com/home-assistant/brands)**
2. **Properly resize the icons** (see `/brands/README.md` for instructions)
3. **Submit a Pull Request** adding the icons to `custom_integrations/litellm_conversation/`

After approval, the icon will automatically appear at:
- `https://brands.home-assistant.io/litellm_conversation/icon.png`

### Manual Icon Setup (Advanced)

If you want the icon to appear immediately:
1. Resize the icons in `/brands/custom_integrations/litellm_conversation/` to proper dimensions
2. Optimize the PNG files  
3. Host them at the expected URLs or create a local icon pack

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

### AI Task Service

The AI Task service allows you to generate data and analyze images using vision models:

#### Image Analysis Examples

```yaml
# Count objects in a camera feed
automation:
  - alias: "Count chickens in coop"
    trigger:
      platform: time
      at: "08:00:00"
    action:
      service: ai_task.generate_data
      data:
        instructions: "Count the number of chickens in this image"
        attachments:
          - entity_id: camera.chicken_coop
      response_variable: chicken_count

# Analyze security camera for activity
automation:  
  - alias: "Security analysis"
    trigger:
      platform: state
      entity_id: binary_sensor.front_door_motion
    action:
      service: ai_task.generate_data
      data:
        instructions: "Describe what activity you see in this security camera image"
        attachments:
          - entity_id: camera.front_door
```

#### Structured Data Generation

```yaml
# Generate structured weather summary
automation:
  - alias: "Weather summary"
    trigger:
      platform: time
      at: "07:00:00"
    action:
      service: ai_task.generate_data
      data:
        instructions: "Create a weather summary for today"
        structure:
          temperature: number
          condition: text
          recommendation: text
      response_variable: weather_data
```

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

### Text Models (Conversation, STT, TTS)
- **OpenAI**: gpt-3.5-turbo, gpt-4, gpt-4-turbo-preview, gpt-4o, gpt-4o-mini
- **Anthropic**: claude-3-haiku, claude-3-sonnet, claude-3-opus, claude-3.5-sonnet  
- **Google**: gemini-pro, gemini-1.5-pro
- **Meta**: llama-2-70b-chat, llama-3-70b-chat
- **Mistral**: mixtral-8x7b-instruct
- **And many more...**

### Vision Models (AI Task)
- **OpenAI**: gpt-4-vision-preview, gpt-4o, gpt-4o-mini
- **Anthropic**: claude-3-haiku, claude-3-sonnet, claude-3-opus, claude-3.5-sonnet
- **Google**: gemini-pro-vision, gemini-1.5-pro
- **Any vision-capable model available through your LiteLLM proxy**

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

## Versioning and Releases

This integration follows [Semantic Versioning](https://semver.org/) for HACS compatibility:

- **Major** (X.0.0): Breaking changes that require user intervention
- **Minor** (X.Y.0): New features, backward compatible  
- **Patch** (X.Y.Z): Bug fixes, backward compatible

### HACS Updates

HACS automatically detects new releases through GitHub releases. When a new version is available:

1. HACS will show an update notification
2. Click "Update" in HACS to install the new version
3. Restart Home Assistant to apply changes

### Release Process

Releases are automated through GitHub Actions:

1. **Automatic Release**: Push a git tag (e.g., `v1.1.0`) to trigger automatic release creation
2. **Manual Version Bump**: Use the "Version Bump" workflow in GitHub Actions to bump version and create release
3. **Version Consistency**: The workflow validates that git tag version matches `manifest.json` version

### Current Version

The current version is defined in [`custom_components/litellm_conversation/manifest.json`](custom_components/litellm_conversation/manifest.json) and should match the latest git tag.

### Development Builds

For development or beta testing:
- Pre-release versions use suffixes like `1.1.0-beta.1`
- Mark GitHub releases as "pre-release" for testing versions
- HACS can install pre-releases if configured to show beta versions

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