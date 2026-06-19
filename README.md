# 🍄 PAMIF-Net: Cloud-Edge Collaborative Framework for Mushroom Freshness Monitoring

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8%2B-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Ready-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> Official code repository for the paper: **"Cloud-edge collaborative system for postharvest mushroom freshness monitoring using a patch-aligned spatial-spectral multimodal fusion network"** (Under Review at *Computers and Electronics in Agriculture*).

## 🌟 Overview
Postharvest button mushrooms (*Agaricus bisporus*) are highly perishable. Conventional visual inspections are subjective, while traditional hyperspectral models often lose spatial heterogeneity or lack physical interpretability. 

**PAMIF-Net (Patch-Aligned Multimodal Interaction Fusion Network)** dynamically integrates five complementary descriptors (Global spectral trends, Color-texture, Spatial-spectral tokens, Patch-level states, and MGAF interactions) onto a unified 10×10 spatial grid. This repository provides the complete pipeline from multi-dimensional feature extraction and model training to cloud-edge collaborative deployment.

### Key Highlights
- **100.00% Accuracy** achieved in offline 10-trial cross-validation for 5-class storage time recognition.
- **Interpretable AI (XAI)** using Input Gradient Saliency to pinpoint localized browning and moisture loss.
- **Cloud-Edge Deployability** validated on a fully independent dataset of 224 samples with a locally hostable real-time Streamlit diagnostic interface.

---

## 🚀 Quick Start & Interactive Demo

We have developed a complete interactive inference pipeline using Streamlit, allowing you to instantly visualize the spatial-spectral freshness diagnostics without writing any code.

### Step 1: Environment Setup
Clone this repository and install the required dependencies:
```bash
git clone [https://github.com/DavidsonGuo/PAMIF-Net.git](https://github.com/DavidsonGuo/PAMIF-Net.git)
cd PAMIF-Net
pip install -r requirements.txt

###Step 2: Launch the Diagnostic Interface
Execute the following command in your terminal to start the Streamlit web application locally:

Bash
streamlit run app.py
Upon execution, your default web browser will automatically open the application at http://localhost:8501/.

###Step 3: Run the Inference
Navigate to the sample_data/ folder provided in this repository.

Select and upload a representative .npy hyperspectral ROI sample (e.g., Day_9_Golden_Sample.npy) via the left sidebar in the web interface.

The platform will automatically execute the PAMIF-Net inference, outputting the predicted storage day, freshness grade, and rendering highly interpretable Input Gradient Saliency maps highlighting localized deterioration.
