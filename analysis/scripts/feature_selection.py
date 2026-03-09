import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

def plot_correlation_heatmap(df):
    plt.figure(figsize=(14, 12))
    
    # Calculate the matrix once
    corr_matrix = df.corr()
    
    # Render the visual heatmap
    sns.heatmap(
        corr_matrix, 
        annot=True,          
        cmap='coolwarm',     
        fmt=".2f",           
        linewidths=0.5,      
        vmin=-1, vmax=1      
    )
    
    plt.title('S&P 500 Regime Detection: Feature Correlation Heatmap')
    plt.tight_layout()
    plt.show()
    
    # Hand the raw data back for your extraction script
    return corr_matrix

def extract_correlation_pairs(corr_matrix, mod_low_bound, mod_up_bound, high_bound):
    # Unstack the matrix into a list of pairs and rename columns
    pairs = corr_matrix.unstack().reset_index()
    pairs.columns = ['Feature_1', 'Feature_2', 'Correlation']

    # Remove the diagonal and mirror duplicates
    unique_pairs = pairs[pairs['Feature_1'] < pairs['Feature_2']].copy()

    # Create the sorting column
    unique_pairs['Abs_Corr'] = unique_pairs['Correlation'].abs()
    sorted_pairs = unique_pairs.sort_values(by='Abs_Corr', ascending=False)

    # 1. Create the Highly Correlated DataFrame
    high_corr_df = sorted_pairs[
        sorted_pairs['Abs_Corr'] >= high_bound
    ].drop(columns=['Abs_Corr']).reset_index(drop=True)

    # 2. Create the Moderately Correlated DataFrame
    mod_corr_df = sorted_pairs[
        (sorted_pairs['Abs_Corr'] >= mod_low_bound) & (sorted_pairs['Abs_Corr'] < mod_up_bound)
    ].drop(columns=['Abs_Corr']).reset_index(drop=True)

    return high_corr_df, mod_corr_df


from statsmodels.tsa.stattools import adfuller

def run_adf_test(df):
    print("Augmented Dickey-Fuller Stationarity Test Results:\n")
    
    for column in df.columns:
        # Calculate the ADF test, dropping any residual NaNs
        result = adfuller(df[column].dropna())
        p_value = result[1]
        
        # Check against our 0.05 threshold
        if p_value < 0.05:
            print(f"[STATIONARY]     {column} (p-value: {p_value:.4f})")
        else:
            print(f"[NON-STATIONARY] {column} (p-value: {p_value:.4f})")