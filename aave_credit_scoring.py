import json
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

class AaveCreditScoring:
    def __init__(self, data_path):
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.wallet_features = None
        self.model = None
        self.scaler = None
        self.credit_scores = None
        
    def load_data(self):
        print("Loading transaction data...")
        try:
            with open(self.data_path, 'r') as f:
                self.raw_data = json.load(f)
            print(f"Successfully loaded {len(self.raw_data)} transactions")
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
        return True
    
    def engineer_features(self):
        print("Engineering features from transaction data...")
        
        df = pd.DataFrame(self.raw_data)
        
        action_data = pd.json_normalize(df['actionData'])
        df = pd.concat([df.drop('actionData', axis=1), action_data], axis=1)
        
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df['assetPriceUSD'] = pd.to_numeric(df['assetPriceUSD'], errors='coerce')
        
        df['usd_value'] = df['amount'] * df['assetPriceUSD'] / 1e18
        
        wallet_features = []
        
        for wallet, wallet_data in df.groupby('userWallet'):
            features = self._calculate_wallet_features(wallet_data)
            features['wallet'] = wallet
            wallet_features.append(features)
        
        self.wallet_features = pd.DataFrame(wallet_features)
        print(f"Generated features for {len(self.wallet_features)} wallets")
        
    def _calculate_wallet_features(self, wallet_data):
        features = {}
        
        features['total_transactions'] = len(wallet_data)
        features['unique_actions'] = wallet_data['action'].nunique()
        features['total_usd_volume'] = wallet_data['usd_value'].sum()
        features['avg_transaction_size'] = wallet_data['usd_value'].mean()
        features['median_transaction_size'] = wallet_data['usd_value'].median()
        features['std_transaction_size'] = wallet_data['usd_value'].std()
        
        action_counts = wallet_data['action'].value_counts()
        total_actions = len(wallet_data)
        
        features['deposit_ratio'] = action_counts.get('deposit', 0) / total_actions
        features['borrow_ratio'] = action_counts.get('borrow', 0) / total_actions
        features['repay_ratio'] = action_counts.get('repay', 0) / total_actions
        features['redeem_ratio'] = action_counts.get('redeemunderlying', 0) / total_actions
        features['liquidation_ratio'] = action_counts.get('liquidationcall', 0) / total_actions
        
        features['deposit_count'] = action_counts.get('deposit', 0)
        features['borrow_count'] = action_counts.get('borrow', 0)
        features['repay_count'] = action_counts.get('repay', 0)
        features['liquidation_count'] = action_counts.get('liquidationcall', 0)
        
        if features['borrow_count'] > 0:
            features['repay_to_borrow_ratio'] = features['repay_count'] / features['borrow_count']
        else:
            features['repay_to_borrow_ratio'] = 0
        
        wallet_data_sorted = wallet_data.sort_values('timestamp')
        features['activity_duration_days'] = (wallet_data_sorted['timestamp'].max() - 
                                            wallet_data_sorted['timestamp'].min()) / (24 * 3600)
        features['transactions_per_day'] = features['total_transactions'] / max(features['activity_duration_days'], 1)
        
        features['unique_assets'] = wallet_data['assetSymbol'].nunique()
        features['asset_concentration'] = 1 - (wallet_data['assetSymbol'].value_counts().iloc[0] / len(wallet_data))
        
        features['max_transaction_size'] = wallet_data['usd_value'].max()
        features['transaction_size_cv'] = features['std_transaction_size'] / max(features['avg_transaction_size'], 1)
        
        time_diffs = wallet_data_sorted['timestamp'].diff().dropna()
        if len(time_diffs) > 0:
            features['avg_time_between_transactions'] = time_diffs.mean()
            features['std_time_between_transactions'] = time_diffs.std()
        else:
            features['avg_time_between_transactions'] = 0
            features['std_time_between_transactions'] = 0
        
        features['network_consistency'] = (wallet_data['network'].value_counts().iloc[0] / len(wallet_data))
        
        return features
    
    def create_target_variable(self):
        print("Creating target variable based on behavioral patterns...")
        
        scores = []
        
        for _, row in self.wallet_features.iterrows():
            score = 500
            
            if row['repay_to_borrow_ratio'] > 0.8:
                score += 150
            elif row['repay_to_borrow_ratio'] > 0.5:
                score += 100
            elif row['repay_to_borrow_ratio'] > 0.2:
                score += 50
            
            if row['network_consistency'] > 0.9:
                score += 50
            
            if row['unique_assets'] > 3:
                score += 30
            
            if row['total_transactions'] > 50:
                score += 40
            elif row['total_transactions'] > 20:
                score += 20
            
            if row['liquidation_count'] > 0:
                score -= 200
            
            if row['transaction_size_cv'] < 0.1 and row['total_transactions'] > 100:
                score -= 100
            
            if row['transactions_per_day'] > 50:
                score -= 150
            
            if row['transaction_size_cv'] > 3:
                score -= 50
            
            score = max(0, min(1000, score))
            scores.append(score)
        
        self.wallet_features['target_score'] = scores
        print(f"Created target scores with mean: {np.mean(scores):.2f}, std: {np.std(scores):.2f}")
    
    def prepare_features(self):
        print("Preparing features for model training...")
        
        feature_columns = [col for col in self.wallet_features.columns 
                          if col not in ['wallet', 'target_score']]
        
        X = self.wallet_features[feature_columns]
        y = self.wallet_features['target_score']
        
        X = X.fillna(0)
        
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, feature_columns
    
    def train_model(self):
        print("Training credit scoring model...")
        
        X, y, feature_columns = self.prepare_features()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        }
        
        best_model = None
        best_score = -float('inf')
        
        for name, model in models.items():
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            print(f"{name} - CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            if cv_scores.mean() > best_score:
                best_score = cv_scores.mean()
                best_model = model
        
        self.model = best_model
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"Test R² Score: {test_r2:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
    
    def generate_credit_scores(self):
        print("Generating final credit scores...")
        
        X, _, feature_columns = self.prepare_features()
        
        predicted_scores = self.model.predict(X)
        
        predicted_scores = np.clip(predicted_scores, 0, 1000)
        
        self.credit_scores = pd.DataFrame({
            'wallet': self.wallet_features['wallet'],
            'credit_score': predicted_scores,
            'total_transactions': self.wallet_features['total_transactions'],
            'total_usd_volume': self.wallet_features['total_usd_volume'],
            'repay_to_borrow_ratio': self.wallet_features['repay_to_borrow_ratio'],
            'liquidation_count': self.wallet_features['liquidation_count']
        })
        
        self.credit_scores = self.credit_scores.sort_values('credit_score', ascending=False)
        
        print(f"Generated credit scores for {len(self.credit_scores)} wallets")
        print(f"Score distribution - Min: {predicted_scores.min():.2f}, Max: {predicted_scores.max():.2f}, Mean: {predicted_scores.mean():.2f}")
        
        return self.credit_scores
    
    def save_results(self, output_path='credit_scores.csv'):
        if self.credit_scores is not None:
            self.credit_scores.to_csv(output_path, index=False)
            print(f"Credit scores saved to {output_path}")
        else:
            print("No credit scores to save. Run generate_credit_scores() first.")
    
    def generate_analysis_plots(self):
        if self.credit_scores is None:
            print("No credit scores available. Run generate_credit_scores() first.")
            return
        
        plt.figure(figsize=(10, 6))
        
        score_ranges = ['0-100', '100-200', '200-300', '300-400', '400-500', 
                       '500-600', '600-700', '700-800', '800-900', '900-1000']
        range_counts = []
        for i in range(0, 1000, 100):
            count = len(self.credit_scores[
                (self.credit_scores['credit_score'] >= i) & 
                (self.credit_scores['credit_score'] < i + 100)
            ])
            range_counts.append(count)
        
        plt.bar(score_ranges, range_counts, color='skyblue')
        plt.title('Score Distribution by Ranges')
        plt.xlabel('Score Range')
        plt.ylabel('Number of Wallets')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('credit_score_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Analysis plot saved as 'credit_score_analysis.png'")
    
    def run_complete_pipeline(self):
        print("=== Aave V2 Credit Scoring Pipeline ===")
        
        if not self.load_data():
            return
        
        self.engineer_features()
        
        self.create_target_variable()
        
        self.train_model()
        
        credit_scores = self.generate_credit_scores()
        
        self.save_results()
        
        self.generate_analysis_plots()
        
        print("\n=== Pipeline Complete ===")
        print(f"Top 10 Highest Scoring Wallets:")
        print(credit_scores.head(10)[['wallet', 'credit_score', 'total_transactions', 'total_usd_volume']])
        
        print(f"\nBottom 10 Lowest Scoring Wallets:")
        print(credit_scores.tail(10)[['wallet', 'credit_score', 'total_transactions', 'liquidation_count']])
        
        return credit_scores

def main():
    data_path = "user-wallet-transactions.json"
    
    credit_scorer = AaveCreditScoring(data_path)
    results = credit_scorer.run_complete_pipeline()
    
    return results

if __name__ == "__main__":
    results = main()
