# 📧 Email Spam Analyzer

This Streamlit web application allows you to test and compare three different machine learning models (Logistic Regression, Naïve Bayes and Support Vector Machine) to detect if an email message is spam.

The application is styled like an email client for a user-friendly experience. You can compose a message, select a model, and see the prediction and confidence score in real-time.

## Features

-   **Email UI:** A simple interface to "compose" an email (subject and body).
-   **Model Selection:** Choose between three different models:
    -   Logistic Regression (trained on 17 numerical features)
    -   Complement Naïve Bayes (trained on TF-IDF text)
    -   Manual SVM (custom-built model trained on TF-IDF text)
-   **Automatic Mode:** Defaults to the best-performing text model (Complement Naïve Bayes).
-   **Prediction Output:** Clear "Spam" (🚨) or "Not Spam" (✅) result.
-   **Confidence Score:** Shows how confident the model is in its prediction.
-   **Model Comparison:** A dashboard that displays the predictions and spam probabilities from all three models simultaneously.

## Prerequisites

Before you begin, ensure you have the following installed on your system:
-   Python (3.8 or newer)
-   `pip` (Python package installer)

## Setup & Installation

1.  **Create Project Folder:**
    Create a main folder for your application (e.g., `spam_classifier_app`).

    ```bash
    mkdir spam_classifier_app
    cd spam_classifier_app
    ```

2.  **Create `models` Directory:**
    Inside `spam_classifier_app`, create a subfolder named `models`.

    ```bash
    mkdir models
    ```

3.  **Add Model Files:**
    Move your five pre-trained model files (`.pkl`) into the `models/` directory. The required files are:
    -   `naive_bayes_spam_artifacts.pkl`
    -   `manual_svm_model.pkl`
    -   `tfidf_vectorizer.pkl`
    -   `logistic_regression.pkl`
    -   `feature_scaler.pkl`

4.  **Create Python Files:**
    Inside the main `spam_classifier_app` folder (at the same level as the `models` folder), create the following three files and copy the code provided previously into them:

    -   `requirements.txt` (lists the dependencies)
    -   `SVMBinaryClassifier.py` (contains the custom SVM class definition)
    -   `app.py` (the main Streamlit application code)

    Your final folder structure should look like this:
    ```
    spam_classifier_app/
    ├── models/
    │   ├── naive_bayes_spam_artifacts.pkl
    │   ├── manual_svm_model.pkl
    │   ├── tfidf_vectorizer.pkl
    │   ├── logistic_regression.pkl
    │   └── feature_scaler.pkl
    │
    ├── SVMBinaryClassifier.py
    ├── app.py
    └── requirements.txt
    ```

5.  **Install Dependencies:**
    From your terminal (while inside the `spam_classifier_app` directory), install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

## Running the App

Once all the files are in place and the dependencies are installed, you can run the app with a single command:

```bash
streamlit run app.py