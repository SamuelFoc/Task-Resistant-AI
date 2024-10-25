import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import itertools

def plot_category_counts_for(df: pl.DataFrame, categorical_features: list, group: str = None, layout_columns: int = 3, normalize: bool = False):
    """
    Plot category counts for each feature in `categorical_features` with specified layout, 
    and optionally split by a grouping variable and normalized to percentages.

    Parameters:
    - df (pl.DataFrame): DataFrame containing the data.
    - categorical_features (list): List of categorical feature names.
    - group (str): Optional. Column name to group data by for side-by-side comparison.
    - layout_columns (int): Number of plots per row. Default is 3.
    - normalize (bool): If True, normalize counts to percentages for each group. Default is False.
    """
    # Convert to Pandas DataFrame for compatibility with Seaborn
    df = df.to_pandas()

    # Check for features that exist in the DataFrame
    existing_features = [feature for feature in categorical_features if feature in df.columns]
    missing_features = [feature for feature in categorical_features if feature not in df.columns]

    # Warn about any missing features
    if missing_features:
        print(f"Warning: The following features were not found in the DataFrame and will be skipped: {missing_features}")

    # If no existing features are found, exit the function
    if not existing_features:
        print("No valid features found to plot.")
        return

    # Set threshold for many categories
    many_categories_threshold = 10

    # Separate features based on the number of categories
    regular_features = [feature for feature in existing_features if df[feature].nunique() <= many_categories_threshold]
    large_category_features = [feature for feature in existing_features if df[feature].nunique() > many_categories_threshold]

    # Calculate layout for regular features
    total_regular_features = len(regular_features)
    layout_rows = math.ceil(total_regular_features / layout_columns)

    # Set figure size for regular plots based on layout
    plt.figure(figsize=(layout_columns * 8, layout_rows * 4))
    
    # Plot each regular feature's value counts
    for idx, feature in enumerate(regular_features):
        # Create a subplot for each feature in the regular grid layout
        plt.subplot(layout_rows, layout_columns, idx + 1)
        
        if group and group in df.columns:
            # Group data by feature and group column to get counts
            group_counts = df.groupby([feature, group]).size().reset_index(name="count")
            
            # Normalize counts if requested
            if normalize:
                total_counts_per_group = group_counts.groupby(group)["count"].transform("sum")
                group_counts["count"] = group_counts["count"] / total_counts_per_group * 100  # Convert to percentage
                ylabel = "Percentage (%)"
            else:
                ylabel = "Counts"
            
            # Plot side-by-side grouped bar plots with hue
            sns.barplot(
                data=group_counts,
                x="count",
                y=feature,
                hue=group,
                palette="viridis"
            )
            plt.legend(title=group)
        else:
            # Plot without grouping
            feature_counts = df[feature].value_counts(normalize=normalize).reset_index()
            feature_counts.columns = [feature, "count"]

            if normalize:
                feature_counts["count"] *= 100  # Convert to percentage
                ylabel = "Percentage (%)"
            else:
                ylabel = "Counts"
            
            sns.barplot(
                x="count",
                y=feature,
                data=feature_counts,
                color="#7800C1"
            )
        
        # Set titles and labels
        plt.title(f"{feature} Counts")
        plt.xlabel(ylabel)
        plt.ylabel(feature)
    
    # Adjust layout for regular plots to avoid overlap
    plt.tight_layout()
    plt.show()

    # Plot large-category features with dynamic height
    base_width = 8     # Standard width for each subplot
    base_height = 0.5  # Height per category for large plots

    for feature in large_category_features:
        # Calculate the number of unique categories for dynamic height adjustment
        num_categories = df[feature].nunique()
        plot_height = max(4, num_categories * base_height)  # Minimum height of 4 for small plots

        plt.figure(figsize=(base_width, plot_height))
        
        if group and group in df.columns:
            # Group data by feature and group column to get counts
            group_counts = df.groupby([feature, group]).size().reset_index(name="count")
            
            # Normalize counts if requested
            if normalize:
                total_counts_per_group = group_counts.groupby(group)["count"].transform("sum")
                group_counts["count"] = group_counts["count"] / total_counts_per_group * 100  # Convert to percentage
                ylabel = "Percentage (%)"
            else:
                ylabel = "Counts"
            
            # Plot side-by-side grouped bar plots with hue
            sns.barplot(
                data=group_counts,
                x="count",
                y=feature,
                hue=group,
                palette="viridis"
            )
            plt.legend(title=group)
        else:
            # Plot without grouping
            feature_counts = df[feature].value_counts(normalize=normalize).reset_index()
            feature_counts.columns = [feature, "count"]

            if normalize:
                feature_counts["count"] *= 100  # Convert to percentage
                ylabel = "Percentage (%)"
            else:
                ylabel = "Counts"
            
            sns.barplot(
                x="count",
                y=feature,
                data=feature_counts,
                color="#7800C1"
            )
        
        # Set titles and labels for large plots
        plt.title(f"{feature} Counts")
        plt.xlabel(ylabel)
        plt.ylabel(feature)
        
        # Adjust layout to avoid overlap for individual plots
        plt.tight_layout()
        plt.show()


def plot_categorical_correlation_matrix(df: pd.DataFrame, categorical_features: list, against: str = None, normalize: bool = False):
    """
    Plot a "correlation matrix" for each pair of categorical features or each feature against a specified variable.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - categorical_features (list): List of categorical feature names.
    - against (str, optional): Optional column name to compare each feature against. Default is None.
    - normalize (bool): If True, normalize the co-occurrence counts to show percentages. Default is False.
    """
    # If 'against' is provided, check if it exists in the DataFrame
    if against and against not in df.columns:
        print(f"Error: '{against}' column not found in the DataFrame.")
        return

    # If 'against' is provided, compute co-occurrence matrices for each feature against the 'against' column
    if against:
        for feature in categorical_features:
            if feature not in df.columns:
                print(f"Warning: '{feature}' column not found in the DataFrame and will be skipped.")
                continue
            
            # Create a frequency matrix for the feature against the 'against' variable
            freq_matrix = pd.crosstab(df[feature], df[against])

            # Normalize the frequency matrix if requested
            if normalize:
                freq_matrix = freq_matrix.div(freq_matrix.sum(axis=1), axis=0) * 100  # Row-wise normalization to get percentages

            # Plot the frequency matrix as a heatmap
            plt.figure(figsize=(10, max(4, len(freq_matrix) * 0.5)))  # Adjust height based on categories
            sns.heatmap(
                freq_matrix,
                annot=True,
                fmt=".2f" if normalize else "d",
                cmap="viridis",
                cbar_kws={'label': 'Percentage (%)' if normalize else 'Count'}
            )
            
            # Set titles and labels
            plt.title(f'Co-occurrence Matrix: {feature} vs {against}')
            plt.xlabel(against)
            plt.ylabel(feature)
            plt.show()
    
    # If 'against' is not provided, compute pairwise co-occurrence matrices for each pair of categorical features
    else:
        for feature_x, feature_y in itertools.combinations(categorical_features, 2):
            # Create a frequency matrix for the pair of features
            freq_matrix = pd.crosstab(df[feature_x], df[feature_y])

            # Normalize the frequency matrix if requested
            if normalize:
                freq_matrix = freq_matrix.div(freq_matrix.sum(axis=1), axis=0) * 100  # Row-wise normalization to get percentages

            # Plot the frequency matrix as a heatmap
            plt.figure(figsize=(10, max(4, len(freq_matrix) * 0.5)))  # Adjust height based on categories
            sns.heatmap(
                freq_matrix,
                annot=True,
                fmt=".2f" if normalize else "d",
                cmap="viridis",
                cbar_kws={'label': 'Percentage (%)' if normalize else 'Count'}
            )
            
            # Set titles and labels
            plt.title(f'Co-occurrence Matrix: {feature_x} vs {feature_y}')
            plt.xlabel(feature_y)
            plt.ylabel(feature_x)
            plt.show()


def plot_correlations_with_target(df: pl.DataFrame, target_column: str):
    """
    Plot the correlation of each numerical column with the target column.

    Parameters:
    - df (pl.DataFrame): Polars DataFrame containing the data.
    - target_column (str): The name of the target column to correlate with.
    """
    # Check if the target column exists in the DataFrame
    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found in the DataFrame.")
        return
    
    # Select numerical columns, excluding the target column
    numerical_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64) and col != target_column]

    # Calculate correlation of each numerical column with the target column
    correlations = []
    for col in numerical_cols:
        corr_value = df.select(pl.corr(pl.col(col), pl.col(target_column))).to_numpy()[0][0]
        correlations.append((col, corr_value))

    # Convert the results to a Polars DataFrame and sort by absolute correlation values
    correlations_df = pl.DataFrame(correlations, schema=["Feature", "Correlation"], orient="row").sort("Correlation")

    # Convert to Pandas for easy plotting
    correlations_df_pd = correlations_df.to_pandas()

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(data=correlations_df_pd, x='Correlation', y='Feature', hue="Correlation", palette="coolwarm")
    plt.title(f'Correlation of Numerical Features with {target_column}')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Feature')
    plt.show()