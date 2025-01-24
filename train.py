from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from feature_extraction import FeatureExtractor
import joblib

def train_model(data_path, glove_path):
    # Load and preprocess data
    import pandas as pd
    df = pd.read_csv(data_path)
    df['text_cleaned'] = df['text'].apply(clean_text)

    # Extract features
    extractor = FeatureExtractor(glove_path)
    X = extractor.extract_features(df['text_cleaned'].tolist())
    y = df['label'].values

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train multiple models
    rf = RandomForestClassifier(n_estimators=100)
    lr = LogisticRegression()
    rf.fit(X_train, y_train)
    lr.fit(X_train, y_train)

    # Evaluate
    rf_preds = rf.predict(X_val)
    lr_preds = lr.predict(X_val)
    ensemble_preds = (rf_preds + lr_preds) // 2

    print("Random Forest Accuracy:", accuracy_score(y_val, rf_preds))
    print("Logistic Regression Accuracy:", accuracy_score(y_val, lr_preds))
    print("Ensemble Accuracy:", accuracy_score(y_val, ensemble_preds))

    # Save models
    joblib.dump(rf, "models/random_forest.pkl")
    joblib.dump(lr, "models/logistic_regression.pkl")
    joblib.dump(extractor, "models/vectorizer.pkl")
