# NLP Text Classification with Transformers

An experimental comparison of TF-IDF classifiers and a DistilBERT sequence classifier for four-class AG News categorization. It is a learning and research project, not a deployed content-moderation, routing, or editorial decision system.

## Evidence and limitations

This repository versions source code and notebooks, but not downloaded AG News data, trained models, checkpoints, metrics, plots, or run metadata. It therefore makes no verified accuracy, F1, latency, parameter, or model-comparison claim.

By default, the pipeline requires the public [AG News dataset](https://huggingface.co/datasets/ag_news) through Hugging Face Datasets. If access fails, it stops rather than silently evaluating template-generated text. A deterministic synthetic fixture is available only through `--synthetic` for offline development; its results are not comparable to AG News or to real-world news classification.

## Included methods

- TF-IDF with logistic regression, linear SVM, or random forest baselines.
- DistilBERT fine-tuning through Hugging Face `Trainer`.
- A stratified validation split for transformer tuning and the published test split for final evaluation.
- Classification metrics, confusion matrices, and optional latency measurement.
- Baseline and transformer inference helpers for locally saved artifacts.

## Install

```bash
git clone https://github.com/mzquadri/NLP-Text-Classification-Transformers.git
cd NLP-Text-Classification-Transformers
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
```

## Run an experiment

```bash
# Downloads AG News and writes untracked artifacts under models/ and results/.
python src/train.py --model baseline --baseline_type logistic_regression

# Fine-tunes DistilBERT. This downloads the base model and can require substantial RAM/VRAM.
python src/train.py --model distilbert --epochs 3 --batch_size 32

# Evaluate a locally saved model.
python src/evaluate.py --model_type baseline --model_path models/baseline
python src/inference.py --model_type baseline --text "Apple unveils a new laptop"
```

For offline code development only, add `--synthetic` to `train.py` or `evaluate.py`. Synthetic examples have class-specific templates, so they should not be used to report model capability.

## Reproducibility notes

Record the dataset revision, package versions, hardware, random seed, training arguments, split policy, and generated metrics before comparing runs. The repository sets common random seeds, but GPU and library-level operations can still be nondeterministic. Do not tune model choices or thresholds against the held-out test split.

## Verify the checkout

```bash
python scripts/check_repository.py
python scripts/smoke_test.py
```

The smoke test generates a small synthetic fixture and trains a TF-IDF baseline. It does not download AG News, download a transformer, or establish benchmark performance.

## License

MIT. See [LICENSE](LICENSE).
