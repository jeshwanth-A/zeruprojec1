# Aave V2 Credit Scoring Analysis

## Dataset Overview
- **Total Transactions**: 100,000 records
- **Unique Wallets**: 3,497 wallets
- **Score Range**: 350-770 points
- **Average Score**: 593.24

## Score Distribution by Ranges

| Score Range | Wallets | Percentage |
|-------------|---------|------------|
| 0-100       | 0       | 0.0%       |
| 100-200     | 0       | 0.0%       |
| 200-300     | 0       | 0.0%       |
| 300-400     | 29      | 0.8%       |
| 400-500     | 1,386   | 39.6%      |
| 500-600     | 1,041   | 29.8%      |
| 600-700     | 985     | 28.2%      |
| 700-800     | 56      | 1.6%       |
| 800-900     | 0       | 0.0%       |
| 900-1000    | 0       | 0.0%       |

## Lower Range Behavior (300-500 points)
- **Risk Indicators**: Higher liquidation counts
- **Repayment Patterns**: Poor repay-to-borrow ratios (<50%)
- **Activity**: Irregular transaction patterns
- **Asset Usage**: Limited to 1-2 assets
- **Liquidation Rate**: 15-25% experienced liquidations

## Higher Range Behavior (600-800 points)
- **Risk Management**: Minimal liquidations (<2%)
- **Repayment Patterns**: Excellent repay-to-borrow ratios (>80%)
- **Activity**: Consistent, regular transactions
- **Asset Diversity**: 4+ different assets
- **Liquidation Rate**: Near zero liquidations

## Key Insights
1. **Repayment Ratio**: 79.7% importance in scoring
2. **Liquidation Impact**: -200 points penalty per liquidation
3. **Quality Distribution**: 97.8% of wallets score 400-700 range
4. **Risk Identification**: 29 high-risk wallets identified

## Model Performance
- **Accuracy**: 99.13% R² score
- **Cross-Validation**: 99.26% average
- **RMSE**: 7.21 points
