import pandas as pd
import numpy as np

def strategy(original_df: pd.DataFrame):
    columns_to_drop = [ "Merchant City", "Merchant State", "Year", "Month", "Day", "Person", "Zip", "CARD INDEX", "Card Number", 
    "CVV", "Expires", "Address", "Apartment", "City", "State", "Zipcode", "Card on Dark Web"
    ]

    df = original_df.drop(columns_to_drop, axis=1)

    # Convert to datetime if not already done
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')

    # Extract hour, day of the week, and month
    df['Hour'] = df['Datetime'].dt.hour
    df['DayOfWeek'] = df['Datetime'].dt.dayofweek  # Monday=0, Sunday=6
    df['MonthOfYear'] = df['Datetime'].dt.month

    df = df.sort_values(['User', 'Datetime'])
    df['TimeSinceLastTransaction'] = df.groupby('User')['Datetime'].diff().dt.total_seconds() / 60  # in minutes
    df['TimeSinceLastTransaction'] = df['TimeSinceLastTransaction'].fillna(0)
    df['AvgTransactionAmountWeek'] = df.groupby('User')['Amount'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    df['AvgTransactionAmountMonth'] = df.groupby('User')['Amount'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())

    def handle_inf(col: str):
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        max_ratio = df[col].max()
        df[col] = df[col].replace(np.nan, max_ratio)

    df['AmountToCreditLimitRatio'] = df['Amount'] / df['Credit Limit']
    handle_inf("AmountToCreditLimitRatio")

    df['IncomeToSpendingRatioZip'] = df['Per Capita Income - Zipcode'] / df['Amount']
    handle_inf('IncomeToSpendingRatioZip')

    df['IncomeToSpendingRatioPerson'] = df['Yearly Income - Person'] / df['Amount']
    handle_inf('IncomeToSpendingRatioPerson')

    df['DebtToIncomeRatio'] = df['Total Debt'] / df['Yearly Income - Person']
    df['CardUsageRatio'] = df['Num Credit Cards'] / df['Cards Issued']
    df['YearsToRetirement'] = df['Retirement Age'] - df['Current Age']
    df['Acct Open Date'] = pd.to_datetime(df['Acct Open Date'])
    df['Account Age (Days)'] = (df['Datetime'] - df['Acct Open Date']).dt.days
    df['Age Group'] = pd.cut(df['Current Age'], bins=[0, 25, 35, 45, 60, 100], labels=['18-25', '26-35', '36-45', '46-60', '60+'])
    df['Is Retired'] = df['Current Age'] >= df['Retirement Age']
    df['Bad PIN Error'] = df['Errors?'] == "Bad PIN"

    df = df.drop(["User", "Card", "Merchant Name", "Errors?", "Card Brand", "Acct Open Date", "Year PIN last Changed", "Birth Year", "Birth Month"], axis=1)
    return df