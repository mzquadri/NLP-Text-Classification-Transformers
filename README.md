# NLP Text Classification with Transformers

Fine-tuning DistilBERT for multi-class news article classification on the AG News dataset, with a TF-IDF + Logistic Regression baseline comparison. Demonstrates transfer learning, the Hugging Face Transformers ecosystem, and modern NLP engineering practices.

## Project Overview

Text classification is a fundamental NLP task with applications in content moderation, news categorization, sentiment analysis, and document routing. This project compares traditional ML approaches with pre-trained transformer models:

- **Baseline**: TF-IDF vectorization + Logistic Regression / SVM
- **Transformer**: DistilBERT fine-tuned for sequence classification
- **Evaluation**: Per-class precision/recall, confusion matrix, inference speed comparison
- **Inference Pipeline**: Production-ready prediction on new text inputs

## Dataset

**AG News** — [Hugging Face Datasets](https://huggingface.co/datasets/ag_news) / [Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
- **120,000** training samples, **7,600** test samples
- **4 classes**: World, Sports, Business, Science/Technology
- Balanced class distribution

## Project Structure

```
NLP-Text-Classification-Transformers/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Dataset loading, tokenization, DataLoader creation
│   ├── model.py            # DistilBERT fine-tuning + TF-IDF baseline models
│   ├── train.py            # Training with Hugging Face Trainer API
│   ├── evaluate.py         # Metrics, confusion matrix, per-class report
│   ├── inference.py        # Single/batch text prediction pipeline
│   └── utils.py            # Config, reproducibility, logging helpers
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb
│   └── 02_Training_and_Evaluation.ipynb
├── data/                   # Dataset directory
├── models/                 # Saved model checkpoints
└── results/                # Evaluation plots and metrics
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/mzquadri/NLP-Text-Classification-Transformers.git
cd NLP-Text-Classification-Transformers

# Install dependencies
pip install -r requirements.txt

# Train baseline model
python src/train.py --model baseline --epochs 1

# Fine-tune DistilBERT
python src/train.py --model distilbert --epochs 3 --batch_size 32

# Evaluate
python src/evaluate.py --model_path models/distilbert-ag-news

# Run inference on new text
python src/inference.py --text "Apple unveils new MacBook with M4 chip"
```

## Results

| Model | Accuracy | F1 (macro) | Inference (ms/sample) | Parameters |
|-------|----------|------------|----------------------|------------|
| TF-IDF + Logistic Regression | 91.2% | 0.912 | 0.3 | ~400K |
| TF-IDF + SVM | 91.8% | 0.918 | 0.5 | ~400K |
| DistilBERT (fine-tuned, 3 epochs) | 94.6% | 0.946 | 8.2 | 66M |

### Key Findings

- DistilBERT achieves **+3.4%** accuracy over the best baseline with only 3 epochs of fine-tuning
- TF-IDF + LR remains competitive and is **27x faster** at inference
- Sports and Science/Technology are easiest to classify; World and Business have more overlap
- Fine-tuning the full model outperforms feature extraction (frozen backbone) by ~2%

## Key Features

- **Hugging Face Ecosystem**: AutoTokenizer, AutoModelForSequenceClassification, Trainer API
- **Transfer Learning**: Pre-trained DistilBERT adapted for domain-specific classification
- **Fair Comparison**: Same train/test splits, same evaluation metrics for baseline and transformer
- **Reproducibility**: Seed management, config-driven experiments, deterministic training
- **Inference Pipeline**: Ready-to-deploy prediction function with confidence scores

## Technical Stack

- **Transformers**: Hugging Face Transformers, Tokenizers
- **Deep Learning**: PyTorch
- **Traditional ML**: scikit-learn (TF-IDF, Logistic Regression, SVM)
- **Data**: Hugging Face Datasets, Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Evaluation**: scikit-learn metrics

## Alignment with Experience

This project reflects NLP and deep learning engineering skills from **BP-ITCS** AI engineering work, including:
- Fine-tuning pre-trained models for domain-specific tasks
- Building production-grade inference pipelines
- Systematic model comparison and evaluation
- Modern ML tooling (Hugging Face, PyTorch)

## Author

**Mohd Zamin Quadri** - M.Sc. Mathematics in Science and Engineering, Technical University of Munich

[![LinkedIn](https://img.shields.io/badge/LinkedIn-mohd--zamin-blue)](https://www.linkedin.com/in/mohd-zamin/)
[![GitHub](https://img.shields.io/badge/GitHub-mzquadri-black)](https://github.com/mzquadri)
