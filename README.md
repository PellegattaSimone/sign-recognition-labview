# ASL Recognition Project

## Overview
This project provides a LabVIEW and Python-based application for recognizing and translating American Sign Language (ASL) via real-time webcam capture or static files.

## Features & Usage (`front_panel.vi`)
The application interface includes the following controls:
* **Input Switch:** Toggles the active data source between static image files and the live camera feed.
* **Next Letter:** Click the `Next_letter` button to update the text field with the current ASL prediction.
* **Space Button:** Used to add a space between words. To operate, click the space button, click `Next_letter`, unclick the space button, and then continue with the subsequent letters.
* **Speak:** Click the `Speak` button to vocalize the translated text. This triggers the integrated `speak.py` script. Additionally, a "welcome" picture appears every time this button is pressed.
* **Hand Switch:** Configures the recognition model to expect either Left-Hand or Right-Hand ASL signs.

## System Architecture & Dependencies

To execute this project seamlessly, specific LabVIEW toolkits and a dual-Python architecture must be configured on the host system.

### 1. LabVIEW Modules (Install via VI Package Manager - VIPM)
The project utilizes open-source OpenCV wrappers to capture webcam data instead of proprietary NI-IMAQdx drivers.
* `OpenLvVision_OpenCv` (Core LabVIEW wrapper blocks)
* `OpenLvVision_OpenCv_Dependencies` (Required C++ OpenCV DLLs)
* `OpenLvVision_Image` (Memory allocation and array conversion tools)

### 2. National Instruments Drivers (Install via NI Package Manager - NIPM)
* **NI Vision Common Resources**: This is strictly required to render the OpenCV matrix data natively on the Front Panel. 
* *Note:* If you currently have `NI-IMAQdx` installed, it **does not** need to be uninstalled. The project will simply bypass it and utilize the free Common Resources base.

### 3. Dual-Python Environment Setup
Due to architectural constraints between LabVIEW and modern machine learning frameworks, this project relies on two separate Python installations.

#### Python 32-bit (Client-Side)
The LabVIEW Community Edition strictly operates on a 32-bit architecture. Therefore, a 32-bit Python environment is required to execute `asl_library.py` through the LabVIEW Python Node.
* **Required Libraries:**
  * `numpy` (Array processing)
  * `Pillow` (Image formatting)
  * `requests` (HTTP client for server communication)

#### Python 64-bit (Server-Side & Audio)
The PyTorch machine learning framework requires a 64-bit environment. The deep learning inference has been offloaded to `asl_server.py`, which must be executed in a 64-bit Python environment.
* **Required Libraries:**
  * `torch` (PyTorch)
  * `transformers` (Hugging Face vision models)
  * `flask` (Local server hosting)
  * `Pillow` (Image processing)
  * `pyttsx3` (Text-to-Speech execution for `speak.py`)
