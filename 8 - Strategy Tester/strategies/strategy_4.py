import pandas as pd
import numpy as np

def strategy(original_df: pd.DataFrame):
    """
    Time-Series Patterns and Temporal Analysis Strategy:
    This strategy extracts features based on time-series and temporal patterns.
    Features include:
    - Time since the last transaction, weekday/weekend flags, and transaction time.
    - Rolling weekly transaction mean and standard deviation.
    Useful for detecting deviations in typical transaction times and frequencies.
    """

    columns_to_drop = ["Merchant City", "Merchant State", "CARD INDEX", "Card Number", "CVV", "Expires", "Address", 
                       "Apartment", "City", "State", "Zipcode", "Zip", "Card on Dark Web"]

    df = original_df.drop(columns_to_drop, axis=1)

    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df['Hour'] = df['Datetime'].dt.hour
    df['DayOfWeek'] = df['Datetime'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'] >= 5
    df = df.sort_values(['User', 'Datetime'])

    # User transaction frequency patterns
    df['TimeSinceLastTransaction'] = df.groupby('User')['Datetime'].diff().dt.total_seconds() / 60
    df['TimeSinceLastTransaction'] = df['TimeSinceLastTransaction'].fillna(df['TimeSinceLastTransaction'].mean())

    # Rolling transaction statistics
    df['WeeklyTransactionMean'] = df.groupby('User')['Amount'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    df['WeeklyTransactionStdDev'] = df.groupby('User')['Amount'].transform(lambda x: x.rolling(window=7, min_periods=1).std()).fillna(0)

    df = df.drop(["User", "Card", "Merchant Name", "Errors?", "Card Brand", "Acct Open Date", "Year PIN last Changed"], axis=1)
    return df
