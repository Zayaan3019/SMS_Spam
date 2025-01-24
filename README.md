# SMS Spam Classifier

This project is an advanced SMS spam classifier that combines traditional ML models with state-of-the-art transformer embeddings to detect spam messages effectively. 

## Features
- **Hybrid Feature Extraction**: Combines TF-IDF, GloVe, and BERT embeddings.
- **Model Ensembling**: Combines predictions from Random Forest and Logistic Regression.
- **REST API**: Real-time predictions using FastAPI.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/SMS_Spam_Classifier.git
   cd SMS_Spam_Classifier

## Training
```bash
python src/train.py --data_path data/raw/sms_data.csv --glove_path data/glove.6B.100d.txt
```
## API
Run the FastAPI server:
```bash
uvicorn src.api.app:app --reload
