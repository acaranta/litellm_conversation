# LiteLLM Conversation - Home Assistant Brands Repository Files

This directory contains the icon files prepared for submission to the [Home Assistant Brands Repository](https://github.com/home-assistant/brands).

## Current Status: ⚠️ NEEDS OPTIMIZATION

The current icon files are based on the original LiteLLM favicon (160x160px) but need to be properly resized and optimized before submission.

## Required Optimizations

### 1. Resize Icons to Proper Dimensions
- **icon.png**: Must be exactly 256x256 pixels
- **icon@2x.png**: Must be exactly 512x512 pixels

### 2. Optimization Steps
Using ImageMagick or similar tool:
```bash
# Resize to 256x256 (standard resolution)
convert litellm_original.png -resize 256x256 -background transparent -gravity center -extent 256x256 icon.png

# Resize to 512x512 (high DPI)
convert litellm_original.png -resize 512x512 -background transparent -gravity center -extent 512x512 icon@2x.png

# Optimize PNG files
optipng -o7 icon.png icon@2x.png
```

### 3. Quality Requirements
- ✅ PNG format
- ✅ Transparent background
- ⚠️ Square aspect ratio (1:1) - **needs verification**
- ⚠️ Proper dimensions - **needs resizing**
- ⚠️ Optimized file size - **needs compression**

## Submission Process

### Step 1: Fork the Brands Repository
```bash
# Fork https://github.com/home-assistant/brands
git clone https://github.com/YOUR_USERNAME/brands.git
cd brands
```

### Step 2: Create Integration Directory
```bash
mkdir -p custom_integrations/litellm_conversation
```

### Step 3: Add Optimized Icons
Copy the properly sized and optimized icons:
- `custom_integrations/litellm_conversation/icon.png` (256x256px)
- `custom_integrations/litellm_conversation/icon@2x.png` (512x512px)

### Step 4: Submit Pull Request
1. Commit the changes
2. Push to your fork
3. Create PR to `https://github.com/home-assistant/brands`
4. Follow the PR template requirements

## Verification

After brands repository approval, icons will be served from:
- Standard: `https://brands.home-assistant.io/litellm_conversation/icon.png`
- High DPI: `https://brands.home-assistant.io/litellm_conversation/icon@2x.png`

## Integration Domain Match

The directory name `litellm_conversation` must exactly match:
- Integration domain in `manifest.json`: `"domain": "litellm_conversation"`
- Component directory name: `custom_components/litellm_conversation/`

## Alternative: Custom Icon Pack

If the brands repository submission is not accepted, an alternative is to create a custom icon pack:

1. Create SVG version of the icon
2. Package as HACS frontend resource
3. Reference in integration using icon identifiers

## Original Source

Icon source: [LiteLLM Official Favicon](https://raw.githubusercontent.com/BerriAI/litellm/main/docs/my-website/img/favicon.png)
- Original size: 160x160 pixels
- Format: PNG with transparency
- License: Use approved (open source project)