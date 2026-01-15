import numpy as np


class MyGaussianNB_Base:
    """Base class cho Gaussian Naive Bayes"""
    def __init__(self, name="GNB"):
        self.name = name
        self.classes = None
        self.class_prior = {}
        self.mean = {}
        self.var = {}
        self.epsilon = 0
        self.train_time = 0
        self.pred_time = 0
        
    def _calculate_log_pdf_vectorized(self, X, mean, var):
        """Tính log-PDF của Gaussian (vectorized)"""
        const = -0.5 * np.log(2 * np.pi * var)
        dist = -0.5 * ((X - mean) ** 2) / var
        return const + dist
    
    def predict(self, X):
        """Dự đoán nhãn"""
        X = np.array(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        log_posteriors = np.zeros((n_samples, n_classes))
        
        for idx, c in enumerate(self.classes):
            log_prior = np.log(self.class_prior[c])
            log_pdf = self._calculate_log_pdf_vectorized(X, self.mean[c], self.var[c])
            log_likelihood = np.sum(log_pdf, axis=1)
            log_posteriors[:, idx] = log_prior + log_likelihood
        
        return self.classes[np.argmax(log_posteriors, axis=1)]
    
    def score(self, X, y):
        """Tính accuracy"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)