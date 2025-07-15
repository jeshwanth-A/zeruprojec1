# Aave V2 Credit Scoring System for zeru 

Machine learning model that assigns credit scores (0-1000) to DeFi wallets based on Aave V2 transaction behavior.

## Method
- **Data**: 100K transactions from 3,497 wallets
- **Features**: 20+ behavioral indicators (repayment ratios, liquidations, activity patterns)
- **Model**: Random Forest Regressor
- **Validation**: 5-fold cross-validation

## Architecture
```
Raw JSON → Feature Engineering → ML Training → Score Generation → Analysis
```

## Scoring Logic
- **Base Score**: 500 points
- **Positive**: Good repayment (+150), asset diversity (+30), high activity (+40)
- **Negative**: Liquidations (-200), bot behavior (-100), excessive frequency (-150)
- **Range**: [0, 1000]

## Usage
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python aave_credit_scoring.py
```

## Results
- **Accuracy**: 99.13% R² score
- **Score Range**: 350-770 points
- **Key Factor**: Repayment ratio (79.7% importance)

## Files
- `aave_credit_scoring.py`: Main implementation
- `credit_scores.csv`: Output results
- `analysis.md`: Detailed analysis
