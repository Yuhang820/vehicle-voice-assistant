import joblib
import torch
from transformers import BertTokenizer, BertForSequenceClassification

class IntentPredictor:
    def __init__(self):
        self.model = joblib.load("models/intent_classifier.pkl")
        self.vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

    def predict(self, text: str):
        vec = self.vectorizer.transform([text])
        intent = self.model.predict(vec)[0]
        proba = self.model.predict_proba(vec).max()
        return intent, round(float(proba), 3)


class BertIntentPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained("models/bert_intent")
        self.model = BertForSequenceClassification.from_pretrained("models/bert_intent")
        self.model.to(self.device)
        self.model.eval()
        self.id2label = joblib.load("models/bert_id2label.pkl")

    def predict(self, text: str):
        inputs = self.tokenizer(
            text, return_tensors="pt",
            truncation=True, padding=True, max_length=64
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        confidence, pred_id = probs.max(dim=-1)
        intent = self.id2label[pred_id.item()]
        return intent, round(confidence.item(), 3)