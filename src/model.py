"""
Advanced Transaction Categorization Model
Implements ensemble learning with multiple algorithms and advanced feature engineering
"""

import yaml
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import re
import joblib


class AmountFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract numerical features from transaction amounts"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.Series):
            X = X.values
        # Extract amount-based features
        features = np.array([
            [
                float(x) if isinstance(x, (int, float)) else 0.0,  # raw amount
                np.log1p(float(x)) if isinstance(x, (int, float)) else 0.0,  # log amount
                1 if (isinstance(x, (int, float)) and float(x) > 100) else 0,  # high amount flag
                1 if (isinstance(x, (int, float)) and float(x) < 10) else 0,  # low amount flag
            ]
            for x in X
        ])
        return features


class MerchantFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features from merchant names"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for text in X:
            text_str = str(text).upper()
            features.append([
                1 if '#' in text_str else 0,  # has store number
                1 if '*' in text_str else 0,  # has transaction ID
                len(text_str.split()),  # number of words
                len(text_str),  # character length
                sum(c.isdigit() for c in text_str) / max(len(text_str), 1),  # digit ratio
                1 if any(c.isupper() for c in text) else 0,  # has uppercase
            ])
        return np.array(features)


class TransactionCategorizer:
    """
    Advanced transaction categorization system with ensemble learning
    """
    
    def __init__(self, config_path="config/categories.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.categories = [cat['name'] for cat in config['categories']]
        self.keyword_map = {cat['name']: cat['keywords'] for cat in config['categories']}
        self.category_descriptions = {
            cat['name']: cat.get('description', '') 
            for cat in config['categories']
        }
        
        self.pipeline = None
        self.feature_names = None
        
    def _build_pipeline(self):
        """Build advanced feature extraction and classification pipeline"""
        
        # Text-based features
        text_features = FeatureUnion([
            ('tfidf_word', TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=3000,
                min_df=2,
                sublinear_tf=True,
                analyzer='word'
            )),
            ('tfidf_char', TfidfVectorizer(
                ngram_range=(2, 5),
                max_features=2000,
                analyzer='char',
                sublinear_tf=True
            )),
        ])
        
        # Ensemble classifier with multiple algorithms
        ensemble = VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                )),
                ('gb', GradientBoostingClassifier(
                    n_estimators=150,
                    max_depth=10,
                    learning_rate=0.1,
                    random_state=42,
                    subsample=0.8
                )),
                ('lr', LogisticRegression(
                    max_iter=1000,
                    C=2.0,
                    random_state=42,
                    class_weight='balanced',
                    solver='lbfgs',
                    multi_class='multinomial'
                ))
            ],
            voting='soft',
            n_jobs=-1
        )
        
        pipeline = Pipeline([
            ('features', text_features),
            ('classifier', ensemble)
        ])
        
        return pipeline
    
    def fit(self, X_text, y, X_amount=None, X_merchant=None):
        """
        Train the categorization model
        
        Args:
            X_text: Transaction descriptions (text)
            y: Category labels
            X_amount: Transaction amounts (optional, for future enhancements)
            X_merchant: Additional merchant features (optional)
        """
        self.pipeline = self._build_pipeline()
        
        # For now, focus on text features
        # Future: incorporate amount and merchant features
        self.pipeline.fit(X_text, y)
        
        return self
    
    def predict(self, X_text, X_amount=None, return_proba=False):
        """
        Predict categories for transactions
        
        Args:
            X_text: Transaction descriptions
            X_amount: Transaction amounts (optional)
            return_proba: Whether to return full probability matrix
            
        Returns:
            predictions, confidences (or probability matrix if return_proba=True)
        """
        if self.pipeline is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        predictions = self.pipeline.predict(X_text)
        probabilities = self.pipeline.predict_proba(X_text)
        
        if return_proba:
            return predictions, probabilities
        
        confidences = np.max(probabilities, axis=1)
        return predictions, confidences
    
    def predict_single(self, text, amount=None):
        """Predict a single transaction with detailed info"""
        pred, conf = self.predict([text])
        explanation = self.explain(text, pred[0])
        
        # Get top 3 predictions
        _, proba = self.predict([text], return_proba=True)
        top_3_idx = np.argsort(proba[0])[-3:][::-1]
        top_3 = [(self.categories[i], proba[0][i]) for i in top_3_idx]
        
        return {
            'category': pred[0],
            'confidence': float(conf[0]),
            'explanation': explanation,
            'top_3_predictions': top_3
        }
    
    def explain(self, text, pred):
        """
        Generate explanation for prediction using keyword matching
        
        Args:
            text: Transaction description
            pred: Predicted category
            
        Returns:
            Explanation string
        """
        words = set(str(text).lower().split())
        keywords = self.keyword_map.get(pred, [])
        
        # Find matching keywords
        matches = []
        for kw in keywords:
            if any(kw in word for word in words):
                matches.append(kw)
        
        if matches:
            return f"Matched keywords: {', '.join(matches[:3])}"
        else:
            return "Based on learned pattern (no direct keyword match)"
    
    def get_feature_importance(self):
        """Get feature importance from Random Forest in ensemble"""
        if self.pipeline is None:
            return None
        
        rf_classifier = self.pipeline.named_steps['classifier'].estimators_[0][1]
        
        if hasattr(rf_classifier, 'feature_importances_'):
            return rf_classifier.feature_importances_
        return None
    
    def get_categories(self):
        """Get list of all categories"""
        return self.categories
    
    def get_category_info(self, category):
        """Get detailed information about a category"""
        return {
            'name': category,
            'keywords': self.keyword_map.get(category, []),
            'description': self.category_descriptions.get(category, '')
        }
    
    def save(self, path='model.pkl'):
        """Save model to disk"""
        joblib.dump(self, path)
        
    @staticmethod
    def load(path='model.pkl'):
        """Load model from disk"""
        return joblib.load(path)