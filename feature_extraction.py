from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Load GloVe embeddings
def load_glove_embeddings(filepath):
    embeddings_index = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vectors = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vectors
    return embeddings_index

# Hybrid feature extraction
class FeatureExtractor:
    def __init__(self, glove_path, transformer_model="bert-base-uncased"):
        self.tfidf = TfidfVectorizer(max_features=5000)
        self.glove_embeddings = load_glove_embeddings(glove_path)
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model)
        self.transformer = AutoModel.from_pretrained(transformer_model)

    def extract_features(self, texts):
        # TF-IDF features
        tfidf_features = self.tfidf.fit_transform(texts).toarray()

        # Transformer embeddings (CLS token)
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        outputs = self.transformer(**inputs)
        transformer_features = outputs.last_hidden_state[:, 0, :].detach().numpy()

        # Combine features
        return np.concatenate((tfidf_features, transformer_features), axis=1)
