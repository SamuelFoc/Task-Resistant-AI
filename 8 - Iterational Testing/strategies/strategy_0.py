import pandas as pd
import numpy as np

def strategy(original_df: pd.DataFrame):
    """
    Transaction Frequency and Amount-Based Strategy:
    This strategy captures patterns in transaction frequency and amount-based statistics.
    Features include:
    - Time since the last transaction and rolling counts for recent weeks/months.
    - Standard deviation of transaction amounts over a week.
    - Ratios such as amount-to-credit limit and income-to-spending.
    Useful for detecting spikes in spending frequency or deviations in transaction amounts.
    """

    columns_to_drop = ["Merchant City", "Merchant State", "Year", "Month", "Day", "Person", "Zip", "CARD INDEX", 
                       "Card Number", "CVV", "Expires", "Address", "Apartment", "City", "State", "Zipcode"]

    df = original_df.drop(columns_to_drop, axis=1)
    
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df['Hour'] = df['Datetime'].dt.hour
    df['DayOfWeek'] = df['Datetime'].dt.dayofweek
    df = df.sort_values(['User', 'Datetime'])

    # Add transaction frequency features
    df['TimeSinceLastTransaction'] = df.groupby('User')['Datetime'].diff().dt.total_seconds() / 60  # in minutes
    df['TimeSinceLastTransaction'] = df['TimeSinceLastTransaction'].fillna(0)

    # Rolling statistics on amount
    df['WeeklyTransactionCount'] = df.groupby('User')['Amount'].transform(lambda x: x.rolling(window=7, min_periods=1).count())
    df['MonthlyTransactionCount'] = df.groupby('User')['Amount'].transform(lambda x: x.rolling(window=30, min_periods=1).count())
    df['StdDevTransactionAmount'] = df.groupby('User')['Amount'].transform(lambda x: x.rolling(window=7, min_periods=1).std()).fillna(0)

    # Ratios
    df['AmountToCreditLimitRatio'] = df['Amount'] / df['Credit Limit']
    df['IncomeToSpendingRatioPerson'] = df['Yearly Income - Person'] / df['Amount']
    df['DebtToIncomeRatio'] = df['Total Debt'] / df['Yearly Income - Person']

    df = df.drop(["User", "Card", "Merchant Name", "Errors?", "Card Brand", "Acct Open Date", "Year PIN last Changed"], axis=1)
    return df
