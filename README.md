# 🎬 BERT IMDB Sentiment Analysis

This project fine-tunes **BERT (bert-base-uncased)** on the [IMDB 50K Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) to classify movie reviews as **positive** or **negative**.

---

## 📌 Features
- Dataset: [IMDB 50K Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
- Fine-tunes **BERT-base-uncased** for binary classification.
- Evaluation metrics: Accuracy, Precision, Recall, F1 Score.
- Training with Hugging Face **Trainer API** + Early Stopping.
- Saves fine-tuned model and tokenizer.
- Easy inference using a pipeline.

---

## 📂 Project Structure


bert-imdb-sentiment/
├── notebooks/
│ └── bert_imdb_sentiment.ipynb # Original Colab notebook
├── src/
│ ├── dataset.py # IMDbDataset class
│ ├── train.py # Training script
│ ├── evaluate.py # Evaluation & inference
├── requirements.txt # Dependencies
├── README.md # Project description
└── .gitignore # Ignore unnecessary files



---

## ⚙️ Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/USERNAME/bert-imdb-sentiment.git
cd bert-imdb-sentiment
pip install -r requirements.txt


🚀 Training
Run the training script:
python src/train.py


The best model will be saved in:
results/model/

🔎 Evaluation
Evaluate the model or run inference:

python src/evaluate.py

💡 Inference Example

from transformers import BertTokenizer, BertForSequenceClassification, TextClassificationPipeline
import torch

model = BertForSequenceClassification.from_pretrained("results/model")
tokenizer = BertTokenizer.from_pretrained("results/model")

pipeline = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

print(pipeline("This movie was amazing and full of emotions!"))
📌 Expected output:

[{'label': 'POSITIVE', 'score': 0.98}]

📝 License
This project is open-source for academic and experimental use





