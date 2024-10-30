import pandas as pd
import numpy as np

def strategy(original_df: pd.DataFrame):
    """
    Risk-Weighted Features and High-Risk Transaction Detection Strategy:
    This strategy extracts features associated with high-risk transactions.
    Features include:
    - High-risk flags for large transactions near credit limits and high-risk MCCs.
    - Ratios like amount-to-credit limit and debt-to-income.
    Useful for identifying transactions with characteristics linked to higher fraud probability.
    """

    columns_to_drop = ["Merchant City", "Merchant State", "Year", "Month", "Day", "CARD INDEX", "Card Number", 
                       "CVV", "Expires", "Address", "Apartment", "City", "State", "Zipcode"]

    df = original_df.drop(columns_to_drop, axis=1)

    # Date conversion and feature extraction
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df = df.sort_values(['User', 'Datetime'])

    # High-risk amount feature
    df['HighRiskAmount'] = df['Amount'] > (0.8 * df['Credit Limit'])
    df['HighRiskMCC'] = df['MCC'].apply(lambda x: 1 if x in [4829, 5411, 5813, 5999] else 0)

    # Ratio features
    df['AmountToCreditLimitRatio'] = df['Amount'] / df['Credit Limit']
    df['DebtToIncomeRatio'] = df['Total Debt'] / df['Yearly Income - Person']

    df = df.drop(["User", "Card", "Merchant Name", "Errors?", "Card Brand", "Acct Open Date", "Year PIN last Changed"], axis=1)
    return df
