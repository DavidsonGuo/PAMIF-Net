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
```

### Step 2: Launch the Diagnostic Interface
Execute the following command in your terminal to start the Streamlit web application locally:
```bash
streamlit run app.py
```
Upon execution, your default web browser will automatically open the application at `http://localhost:8501/`.

### Step 3: Run the Inference
1. Navigate to the `sample_data/` folder provided in this repository.
2. Select and upload a representative `.npy` hyperspectral ROI sample via the left sidebar in the web interface.
3. The platform will automatically execute the PAMIF-Net inference, outputting the predicted storage day, freshness grade, and rendering highly interpretable Input Gradient Saliency maps highlighting localized deterioration.

---

## 🛠️ Repository Structure
```text
PAMIF-Net/
│
├── app.py                   # Streamlit web application for real-time inference & visualization
├── train_model.py           # Core PAMIF-Net architecture and 10-trial cross-validation scripts
├── extract_features.py      # Multi-dimensional feature extraction (PCA, MGAF, Color-Texture)
├── generate_dataset.py      # Automated script for generating 150x150 ROI .npy tensors
├── requirements.txt         # Python dependency list
│
├── weights/                 # Directory containing the pre-trained models
│   ├── pamif_net.h5         # Final trained Keras model
│   └── pca_model.pkl        # Fitted PCA transformation matrix
│
└── sample_data/             # Subset of raw .npy tensors for testing the interactive app
```

## ⚙️ Model Training & Feature Extraction
If you wish to retrain the PAMIF-Net model or extract features from your own hyperspectral data, you can utilize the provided backend scripts:

1. **Automated ROI Generation:** Crop standard 150x150 regions of interest from raw hyperspectral images.
   ```bash
   python generate_dataset.py
   ```
2. **Feature Extraction:** Extract PCA, MGAF, and color-texture features for the multimodal architecture.
   ```bash
   python extract_features.py
   ```
3. **Model Training:** Run the 10-trial cross-validation and ablation studies.
   ```bash
   python train_model.py
   ```

---

## 📊 Dataset Availability
The complete 4.2GB visible-near-infrared (Vis-NIR) hyperspectral dataset of *Agaricus bisporus* acquired over a 9-day storage period is available upon reasonable request to the corresponding authors due to storage limitations. A representative subset is provided in the `sample_data/` directory for immediate algorithmic testing.

---

## 📩 Contact
For any inquiries regarding the dataset, code, or paper, please feel free to reach out:
**Zhen Guo**, School of Pharmaceutical Sciences and Food Engineering, **Liaocheng University**, China.  
📧 Email: [guozhen@lcu.edu.cn](mailto:guozhen@lcu.edu.cn)

---

## 📝 Citation
If you find our work or this codebase useful for your research, please consider citing our paper once it is published:
```bibtex
@article{Guo2024PAMIFNet,
  title={Cloud-edge collaborative system for postharvest mushroom freshness monitoring using a patch-aligned spatial-spectral multimodal fusion network},
  author={Guo, Zhen and Wang, Yaru and Cao, Lele and Auat-Cheein, Fernando A. and Guo, Xingfeng},
  journal={Computers and Electronics in Agriculture},
  year={2024},
  note={Under Review}
}
```

## 📄 License
This project is licensed under the MIT License - see the `LICENSE` file for details.
