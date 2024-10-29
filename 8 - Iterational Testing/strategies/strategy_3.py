import pandas as pd
import numpy as np

def strategy(original_df: pd.DataFrame):
    """
    User Demographic and Financial Profile Strategy:
    This strategy extracts features based on the user's demographic and financial characteristics.
    Features include:
    - Ratios like income-to-spending, debt-to-income, and card usage ratios.
    - User retirement status, age group classification.
    Useful for detecting unusual spending relative to the user's typical financial behavior.
    """

    columns_to_drop = ["Merchant City", "Merchant State", "Year", "Month", "Day", "CARD INDEX", "Card Number", 
                       "CVV", "Expires", "Address", "Apartment", "City", "State", "Zipcode"]

    df = original_df.drop(columns_to_drop, axis=1)

    # Calculate financial and demographic ratios
    df['IncomeToSpendingRatio'] = df['Yearly Income - Person'] / df['Amount']
    df['DebtToIncomeRatio'] = df['Total Debt'] / df['Yearly Income - Person']
    df['CardUsageRatio'] = df['Num Credit Cards'] / df['Cards Issued']
    df['YearsToRetirement'] = df['Retirement Age'] - df['Current Age']

    # User age groups
    df['Age Group'] = pd.cut(df['Current Age'], bins=[0, 25, 35, 45, 60, 100], labels=['18-25', '26-35', '36-45', '46-60', '60+'])
    df['Is Retired'] = df['Current Age'] >= df['Retirement Age']
    
    df = df.drop(["User", "Card", "Errors?", "Card Brand", "Acct Open Date", "Year PIN last Changed", 
                  "Birth Year", "Birth Month", "Zip"], axis=1)
    return df
