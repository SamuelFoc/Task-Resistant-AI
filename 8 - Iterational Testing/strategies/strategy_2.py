import pandas as pd
import numpy as np

def strategy(original_df: pd.DataFrame):
    """
    Geolocation and Merchant-Based Strategy:
    This strategy extracts features based on geographic location and merchant behaviors.
    Features include:
    - Distance between consecutive transactions (indicative of geographic changes).
    - State change in transactions and high-risk MCC categorization.
    Useful for identifying unusual geographic shifts or transactions with high-risk merchant categories.
    """

    columns_to_drop = ["Year", "Month", "Day", "CARD INDEX", "Card Number", "CVV", "Expires", "Address", "Apartment", 
                       "City", "Zipcode", "Card on Dark Web"]

    df = original_df.drop(columns_to_drop, axis=1)

    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df['Hour'] = df['Datetime'].dt.hour
    df['DayOfWeek'] = df['Datetime'].dt.dayofweek
    df = df.sort_values(['User', 'Datetime'])

    # Add features based on geographical location
    df['TransactionDistance'] = df.groupby('User').apply(
        lambda x: np.sqrt((x['Latitude'] - x['Latitude'].shift()) ** 2 + (x['Longitude'] - x['Longitude'].shift()) ** 2)
    ).reset_index(level=0, drop=True).fillna(0)

    # Merchant type-based features
    df['MerchantStateChange'] = df['Merchant State'] != df.groupby('User')['Merchant State'].shift()
    df['HighRiskMCC'] = df['MCC'].apply(lambda x: 1 if x in [4814, 5411, 5813, 5999] else 0)

    df = df.drop(["User", "Card", "Merchant Name", "Merchant City", "Merchant State", "Errors?", "Card Brand", 
                  "Acct Open Date", "Year PIN last Changed", "Birth Year", "Birth Month", "Zip"], axis=1)
    return df
