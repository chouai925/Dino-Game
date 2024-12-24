# CV Final Project - Dino Game

This project implements a gesture recognition system to control the Chrome Dino game using hand gestures instead of keyboard inputs.

## Overview

The project consists of two main components:
- A gesture recognition model trained using PyTorch (`wc_gestureCNN_torch.py`)
- A detection script (`wc_dino_detect.py`) that connects to the Chrome Dino game and controls it using recognized gestures

## Prerequisites

- Python
- PyTorch
- Google Chrome browser
- Additional dependencies
## Quick Start

### Option 1: Using Pre-trained Model

1. Download the pre-trained model from:
   [best_model.pth](https://drive.google.com/file/d/1zeA9CPyU8yp0tcqpruLAcg3tSRVHV7x_/view?usp=drive_link)

2. Run the detection script:
   ```bash
   python wc_dino_detect.py
   ```

### Option 2: Training Your Own Model

1. Prepare your gesture dataset or use the provided `imagefolder_b` dataset

2. Train the model:
   ```bash
   python wc_gestureCNN_torch.py
   ```

3. Run the detection script:
   ```bash
   python wc_dino_detect.py
   ```

## Playing the Game

1. Open Google Chrome
2. Navigate to the Dino game (chrome://dino/)
3. Run the detection script
4. Use hand gestures to control the dinosaur:
   - Open Palm 👋: Jump
   - Closed Fist ✊: Duck/Crouch

## Dataset

The model can be trained using either:
- The provided `imagefolder_b` dataset
- Your own custom gesture dataset

## Model Architecture

The gesture recognition model is implemented using PyTorch and trained to recognize specific hand gestures for game control.

## Files Description

- `wc_gestureCNN_torch.py`: Model training script
- `wc_dino_detect.py`: Game control and gesture detection script
- `best_model.pth`: Pre-trained model weights (download separately)
