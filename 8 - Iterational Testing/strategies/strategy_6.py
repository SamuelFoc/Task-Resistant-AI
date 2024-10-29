import pandas as pd
import numpy as np

def strategy(original_df: pd.DataFrame):
    """
    Super Strategy:
    This comprehensive feature extraction strategy combines the best features from various individual strategies.
    Key features include:
    - Transaction frequency and amount-based rolling statistics to capture spending patterns.
    - Geolocation-based distance and merchant information to identify unusual geographic changes.
    - Demographic and financial ratios for personalized spending behavior based on user profiles.
    - Temporal patterns such as day of week, hour, and time since last transaction for anomaly detection.
    - High-risk flags for transactions that are close to credit limits or associated with high-risk MCC codes.
    This strategy is designed to capture a wide range of features to identify potentially fraudulent transactions.
    """

    # Columns to drop that are less useful for predictive modeling
    columns_to_drop = ["Merchant City", "Year", "Month", "Day", "Person", "Zip", "CARD INDEX", 
                       "Card Number", "CVV", "Expires", "Address", "Apartment", "City", "State", "Zipcode", "Card on Dark Web"]

    df = original_df.drop(columns_to_drop, axis=1)
    
    # Convert date column to datetime format
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df = df.sort_values(['User', 'Datetime'])
    
    # Temporal features
    df['Hour'] = df['Datetime'].dt.hour
    df['DayOfWeek'] = df['Datetime'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'] >= 5
    df['TimeSinceLastTransaction'] = df.groupby('User')['Datetime'].diff().dt.total_seconds() / 60  # in minutes
    df['TimeSinceLastTransaction'] = df['TimeSinceLastTransaction'].fillna(df['TimeSinceLastTransaction'].mean())
    
    # Rolling statistics on amount
    df['WeeklyTransactionCount'] = df.groupby('User')['Amount'].transform(lambda x: x.rolling(window=7, min_periods=1).count())
    df['MonthlyTransactionCount'] = df.groupby('User')['Amount'].transform(lambda x: x.rolling(window=30, min_periods=1).count())
    df['StdDevTransactionAmount'] = df.groupby('User')['Amount'].transform(lambda x: x.rolling(window=7, min_periods=1).std()).fillna(0)
    df['WeeklyTransactionMean'] = df.groupby('User')['Amount'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())

    # Geolocation features
    df['TransactionDistance'] = df.groupby('User').apply(
        lambda x: np.sqrt((x['Latitude'] - x['Latitude'].shift()) ** 2 + (x['Longitude'] - x['Longitude'].shift()) ** 2)
    ).reset_index(level=0, drop=True).fillna(0)
    df['MerchantStateChange'] = df['Merchant State'] != df.groupby('User')['Merchant State'].shift()
    df['HighRiskMCC'] = df['MCC'].apply(lambda x: 1 if x in [4814, 5411, 5813, 5999] else 0)

    # High-risk features
    df['HighRiskAmount'] = df['Amount'] > (0.8 * df['Credit Limit'])

    # Financial and demographic ratios
    df['AmountToCreditLimitRatio'] = df['Amount'] / df['Credit Limit']
    df['IncomeToSpendingRatio'] = df['Yearly Income - Person'] / df['Amount']
    df['DebtToIncomeRatio'] = df['Total Debt'] / df['Yearly Income - Person']
    df['CardUsageRatio'] = df['Num Credit Cards'] / df['Cards Issued']
    df['YearsToRetirement'] = df['Retirement Age'] - df['Current Age']

    # User age-based segmentation
    df['Age Group'] = pd.cut(df['Current Age'], bins=[0, 25, 35, 45, 60, 100], labels=['18-25', '26-35', '36-45', '46-60', '60+'])
    df['Is Retired'] = df['Current Age'] >= df['Retirement Age']

    # Account-based feature
    df['Acct Open Date'] = pd.to_datetime(df['Acct Open Date'])
    df['Account Age (Days)'] = (df['Datetime'] - df['Acct Open Date']).dt.days
    
    # Additional binary flags
    df['Bad PIN Error'] = df['Errors?'] == "Bad PIN"

    # Drop columns that are no longer needed after feature extraction
    df = df.drop(["User", "Card", "Merchant Name", "Errors?", "Card Brand", "Acct Open Date", "Year PIN last Changed", 
                  "Birth Year", "Birth Month"], axis=1)
    
    return df
