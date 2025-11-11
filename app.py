import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
import os
from SVMBinaryClassifier import SVMBinaryClassifier # Crucial import

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Spam Analyzer", page_icon="📧")
MODEL_DIR = "models"

# --- Feature Engineering for Logistic Regression ---
# These functions MUST match the logic from your notebook

@st.cache_data
def clean_text(text):
    """Cleans text for feature extraction."""
    text = str(text).lower()
    text = re.sub(r"[^a-z'\s]", '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def count_spam_words(text):
    """Counts common spam words."""
    spam_words = ['free', 'win', 'winner', 'cash', 'prize', 'claim',
                  'call', 'text', 'txt', 'reply', 'urgent', 'now',
                  'offer', 'deal', 'cheap', 'guarantee']
    words = text.lower().split()
    return sum(1 for word in words if word in spam_words)

def count_repeated_words(text):
    """Counts words that appear more than once."""
    words = text.lower().split()
    if not words:
        return 0
    word_counts = pd.Series(words).value_counts()
    return sum(1 for count in word_counts if count > 1)

def create_numerical_features(message, scaler):
    """
    Re-creates the 17 numerical features from the notebook
    for the Logistic Regression model.
    """
    # 1. Clean text
    cleaned_message = clean_text(message)
    words = cleaned_message.split()
    word_count = len(words)
    text_length = len(cleaned_message)
    
    # 2. Basic text features
    features = {}
    features['message_length'] = len(message) # Original length
    features['word_count'] = word_count
    features['text_length'] = text_length
    
    # Avoid division by zero
    wc_plus_1 = word_count + 1
    
    # 3. Derived text features
    features['avg_word_length'] = text_length / wc_plus_1
    features['unique_words'] = len(set(words))
    features['word_diversity'] = features['unique_words'] / wc_plus_1
    
    features['short_word_count'] = sum(1 for w in words if len(w) <= 2)
    features['short_word_ratio'] = features['short_word_count'] / wc_plus_1
    
    features['long_word_count'] = sum(1 for w in words if len(w) >= 8)
    features['long_word_ratio'] = features['long_word_count'] / wc_plus_1
    
    # 4. Spam indicator features
    features['spam_word_count'] = count_spam_words(cleaned_message)
    features['spam_word_ratio'] = features['spam_word_count'] / wc_plus_1
    
    features['repeated_word_count'] = count_repeated_words(cleaned_message)
    features['repetition_ratio'] = features['repeated_word_count'] / wc_plus_1
    
    greetings = ['hi', 'hello', 'hey', 'dear', 'good', 'morning', 'afternoon', 'evening']
    features['has_greeting'] = 1 if any(cleaned_message.startswith(g) for g in greetings) else 0
    
    action_words = ['call', 'text', 'reply', 'click', 'visit', 'claim', 'get', 'buy', 'order']
    features['action_word_count'] = sum(1 for w in words if w in action_words)
    features['action_word_ratio'] = features['action_word_count'] / wc_plus_1
    
    # 5. Create DataFrame and Scale
    # Must be in the exact order the scaler was trained on
    feature_cols = [
        'message_length', 'word_count', 'text_length', 'avg_word_length', 
        'unique_words', 'word_diversity', 'short_word_count', 'short_word_ratio', 
        'long_word_count', 'long_word_ratio', 'spam_word_count', 'spam_word_ratio', 
        'repeated_word_count', 'repetition_ratio', 'has_greeting', 
        'action_word_count', 'action_word_ratio'
    ]
    
    df = pd.DataFrame([features], columns=feature_cols)
    df = df.fillna(0)
    
    # Apply the loaded scaler
    scaled_features = scaler.transform(df)
    return scaled_features


# --- Model Loading ---

@st.cache_resource
def load_models():
    """Loads all models and artifacts from the 'models' directory."""
    try:
        # 1. Naive Bayes (ComplementNB Pipeline + Threshold)
        nb_artifacts = joblib.load(os.path.join(MODEL_DIR, "naive_bayes_spam_artifacts.pkl"))
        nb_pipe = nb_artifacts['model']
        nb_threshold = nb_artifacts['threshold']
        
        # 2. SVM + its Vectorizer
        svm_model = joblib.load(os.path.join(MODEL_DIR, "svm_model.pkl"))
        tfidf_vec = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
        
        # 3. Logistic Regression + its Scaler
        logreg_model = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "feature_scaler.pkl"))
        
        return nb_pipe, nb_threshold, svm_model, tfidf_vec, logreg_model, scaler
    except FileNotFoundError as e:
        st.error(f"Error loading models: {e}. Make sure all .pkl files are in the 'models' folder.")
        return None, None, None, None, None, None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None, None, None, None, None, None

# --- Prediction Functions ---

def predict_nb(message, pipe, threshold):
    """Predicts using the Naive Bayes pipeline and custom threshold."""
    prob_spam = pipe.predict_proba([message])[0][1]
    prediction = 1 if prob_spam >= threshold else 0
    confidence = prob_spam if prediction == 1 else (1 - prob_spam)
    return "Spam" if prediction == 1 else "Not Spam", confidence, prob_spam

def predict_svm(message, model, vectorizer):
    """Predicts using the SVM and its TF-IDF vectorizer."""
    vec_text = vectorizer.transform([message]).toarray()
    prob_spam = model.predict_proba(vec_text)[0][1]
    prediction = model.predict(vec_text)[0] # Already returns 0 or 1
    confidence = prob_spam if prediction == 1 else (1 - prob_spam)
    return "Spam" if prediction == 1 else "Not Spam", confidence, prob_spam

def predict_logreg(message, model, scaler):
    """Predicts using the Logistic Regression model and feature engineering."""
    features = create_numerical_features(message, scaler)
    prob_spam = model.predict_proba(features)[0][1]
    prediction = model.predict(features)[0]
    confidence = prob_spam if prediction == 1 else (1 - prob_spam)
    return "Spam" if prediction == 1 else "Not Spam", confidence, prob_spam

# --- Main Application UI ---

def run_app():
    # Load all models at the start
    (nb_pipe, nb_threshold, svm_model, tfidf_vec, 
     logreg_model, scaler) = load_models()
    
    if not all([nb_pipe, svm_model, logreg_model]):
        st.stop() # Stop app if models failed to load

    st.title("📧 Email Spam Analyzer")
    st.markdown("Compose an email below and see if our models think it's spam.")

    # --- Sidebar for Model Selection ---
    st.sidebar.title("Configuration")
    model_choice = st.sidebar.radio(
        "Select a Model to Test",
        (
            "Automatic (Best Model)", 
            "Complement Naïve Bayes", 
            "SVM", 
            "Logistic Regression"
        ),
        help="""
        - **Automatic:** Uses Complement Naïve Bayes, the best-performing text pipeline from your notebook.
        - **Complement NB:** Trained on TF-IDF.
        - **SVM:** Your custom-built SVM, trained on TF-IDF.
        - **Logistic Regression:** Trained on 17 custom numerical features.
        """
    )

    # --- Email Composition UI ---
    col1, col2 = st.columns([1, 2])
    with col1:
        st.text_input("To:", "recipient@example.com")
        subject = st.text_input("Subject:", "RE: Important Matter")
    with col2:
        body = st.text_area(
            "Compose your message:", 
            "Hello,\n\nThis is a test message. Click here to claim your free prize!", 
            height=200
        )
    
    full_text = subject + " " + body
    analyze_button = st.button("Analyze Email", type="primary", use_container_width=True)

    # --- Analysis Output ---
    if analyze_button:
        if not body.strip():
            st.warning("Please enter a message body to analyze.")
            st.stop()

        # Run all models for comparison
        nb_pred, nb_conf, nb_prob_spam = predict_nb(full_text, nb_pipe, nb_threshold)
        svm_pred, svm_conf, svm_prob_spam = predict_svm(full_text, svm_model, tfidf_vec)
        lr_pred, lr_conf, lr_prob_spam = predict_logreg(full_text, logreg_model, scaler)

        # Determine the "primary" result based on user choice
        if model_choice == "Automatic (Best Model)":
            primary_model_name = "Automatic (Complement NB)"
            primary_pred = nb_pred
            primary_conf = nb_conf
        elif model_choice == "Complement Naïve Bayes":
            primary_model_name = "Complement Naïve Bayes"
            primary_pred = nb_pred
            primary_conf = nb_conf
        elif model_choice == "SVM":
            primary_model_name = "SVM"
            primary_pred = svm_pred
            primary_conf = svm_conf
        else: # Logistic Regression
            primary_model_name = "Logistic Regression"
            primary_pred = lr_pred
            primary_conf = lr_conf
            
        st.markdown("---")
        
        # 1. Display Primary Result
        st.subheader(f"Result for: {primary_model_name}")
        if primary_pred == "Spam":
            st.error(f"**Prediction: SPAM** (Confidence: {primary_conf*100:.1f}%)", icon="🚨")
        else:
            st.success(f"**Prediction: NOT SPAM** (Confidence: {primary_conf*100:.1f}%)", icon="✅")

        st.markdown("---")
        
        # 2. Display Comparison
        st.subheader("All Model Predictions")
        st.write("See how all three models classified your message. 'Prob.' is the model's raw probability of being 'Spam'.")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                label="Complement Naïve Bayes", 
                value=nb_pred, 
                delta=f"{nb_prob_spam*100:.1f}% Prob.",
                delta_color="off"
            )
        with c2:
            st.metric(
                label="SVM", 
                value=svm_pred,
                delta=f"{svm_prob_spam*100:.1f}% Prob.",
                delta_color="off"
            )
        with c3:
            st.metric(
                label="Logistic Regression", 
                value=lr_pred,
                delta=f"{lr_prob_spam*100:.1f}% Prob.",
                delta_color="off"
            )

# --- Entry Point ---
if __name__ == "__main__":
    run_app()