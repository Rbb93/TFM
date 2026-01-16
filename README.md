# TFM – Analysis of Content Diversity in News Recommendation Systems

This repository contains the code and experiments developed for my **Master’s Thesis (TFM)**, focused on the **analysis of ideological diversity, sentiment, and polarization in news recommendation systems**.

The project studies how modern recommender models distribute political content and how this affects **diversity of exposure** and **user influence**, combining entropy-based measures with sentiment-aware metrics.

---

## Project Scope

The main goals of this work are:
- To analyze ideological diversity in recommended news.
- To measure diversity using entropy-based indicators.
- To incorporate sentiment and polarization into diversity metrics.
- To evaluate recommendation models on both international and regional news datasets.

---

## Repository Structure

TFM/
├── 1_train_export/ # Training of neural news recommenders
├── 2_metrics_RADio/ # Diversity metrics
├── 3_3Cat/ # Regional news analysis using LLMs


---

## 1️⃣ Training and Export (`1_train_export`)

Neural news recommendation models are trained using the **Microsoft Recommenders** framework:
- LSTUR
- NRMS
- NAML
- NPA

Experiments are conducted on the **MIND Small** dataset.

### Environment Requirements

Due to strict compatibility constraints of TensorFlow and the `recommenders` library, the following versions are required:

- **Python 3.9**
- **TensorFlow 2.10.0**
- **CUDA 11.2**
- **cuDNN 8.2.0**
- **NumPy 1.26.4**

Binary-only installation is required for `blis`, `thinc`, and `spacy`.

### Environment Setup

```bash
conda create -n reco_env python=3.9 -y
conda activate reco_env

pip install blis==0.7.9 --only-binary :all:
pip install thinc==8.1.10 --only-binary :all:
pip install spacy==3.5.4 --only-binary :all:
pip install recommenders[gpu]
pip install numpy==1.26.4
pip install tensorflow==2.10.0

conda install -c conda-forge -c nvidia cudatoolkit=11.2 cudnn=8.2.0 -y
```

Required external files:

- glove.6B.300d.txt

- MIND Small dataset

## 2️⃣ Diversity Metrics (2_metrics_RADio)

This module implements diversity metrics inspired by the RADio framework.

## Dependencies
```bash
pip install bs4 community elasticsearch gensim lxml nano \
python-louvain stop_words textblob textstat minimock textblob_nl pathlib

pip install https://github.com/explosion/spacy-models/releases/download/\
en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz#egg=en_core_web_sm
```

3️⃣ Regional Analysis (3_3Cat)

Analysis of a regional Catalan news dataset using local large language models via Ollama and LangChain to process the data and subsequently compute entropy and activation metrics in order to obtain average sentiment and polarization, with the goal of calculating a diversity metric for the analyzed dataset.

## Requirements
pip install langchain langchain_community sparqlwrapper


Additionally:

- Install Ollama locally.

- Download the Qwen3:8b model.

- Download the [3Cat news dataset]([https://ejemplo.com](https://drive.google.com/file/d/1o3pmii7_P0vx-PBHQ0pUHnC7samyXKtG/view?usp=sharing))

## License

This project is released under the Apache License 2.0.
