import joblib

model = joblib.load("models/EmailSpam_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

def Model(email:str) -> str:
    sample = [email]
    X_sample = vectorizer.transform(sample)
    prediction = model.predict(X_sample)
    return "Spam" if prediction[0] == 1 else "Not Spam"

