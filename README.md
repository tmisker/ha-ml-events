# Home Assistant ML Events

![GitHub release (latest by date)](https://img.shields.io/github/v/release/tmisker/ha-ml-events)
![GitHub](https://img.shields.io/github/license/tmisker/ha-ml-events)
[![hacs_badge](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://github.com/custom-components/hacs)

A machine learning integration that automatically detects events in your smart home by learning patterns from labeled examples.

## Overview

Home Assistant ML Events learns to detect situations in your home by analyzing patterns in your existing sensor data. Instead of creating complex automation rules yourself:

1. **Label events** when they occur (e.g., "shower active", "cooking dinner")
2. **Train the model** with a simple button press
3. **Automatic detection** of similar events in the future

All machine learning happens locally - your data never leaves your network!

![Overview Diagram](docs/images/overview.png)

## Features

- 🧠 Self-learning event detection using your existing sensors
- 🎯 Create multiple detectors for different situations
- 🔄 Continuous improvement through additional training
- 🔒 Privacy-first: all processing happens locally
- 📊 View model accuracy and influential sensors

## Installation

### HACS (Recommended)

1. Make sure [HACS](https://hacs.xyz/) is installed
2. Add this repository as a custom repository in HACS:
   - URL: `https://github.com/tmisker/ha-ml-events`
   - Category: Integration
3. Install the "ML Events" integration from HACS
4. Restart Home Assistant

### Manual Installation

1. Download the latest release
2. Copy the `custom_components/ml_events` folder to your `config/custom_components` directory
3. Restart Home Assistant

## Configuration

After installation, add the integration through the Home Assistant UI:

1. Go to **Configuration** → **Integrations**
2. Click the **+ Add Integration** button
3. Search for "ML Events" and select it

## Usage

### Creating Your First Event Detector

1. In the ML Events dashboard, click "Create New Detector"
2. Give it a name like "Shower Active" or "Cooking"
3. Select sensors that might be relevant (or let the system choose automatically)
4. Use the "Label Event" button in the UI when the event is happening
5. Collect 5-10 examples over a few days
6. Click "Train Model"
7. The detector is now active and will appear as a binary sensor

See the [detailed usage guide](docs/usage.md) for more information.

## Requirements

- Home Assistant 2023.3.0 or newer
- Python 3.9 or newer
- At least 3 labeled examples of your event

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.