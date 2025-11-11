import numpy as np

# This class definition is necessary for joblib to load your custom model
class SVMBinaryClassifier:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        # Convert labels to -1 and 1 for SVM
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Gradient descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1

                if condition:
                    # Correctly classified - only update for regularization
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Misclassified - update weights and bias
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        # Return 0 or 1, matching the notebook's final output logic
        return np.where(np.sign(linear_output) == 1, 1, 0)

    def predict_proba(self, X):
        linear_output = np.dot(X, self.w) - self.b
        # Convert to probability-like scores using sigmoid
        proba = 1 / (1 + np.exp(-linear_output))
        # Return [prob_ham, prob_spam]
        return np.column_stack([1-proba, proba])