# Energy Consumption Forecasting - Feature Engineering and Data Preparation

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully")

# Load processed data from previous notebook
try:
    energy_data = pd.read_csv('../data/processed/energy_data_processed.csv')
    weather_data = pd.read_csv('../data/processed/weather_data_processed.csv')
    print(f"Energy data loaded: {energy_data.shape}")
    print(f"Weather data loaded: {weather_data.shape}")
except FileNotFoundError:
    print("Processed data not found. Run 01_data_exploration_and_analysis.ipynb first")

# Convert date columns
energy_data['day'] = pd.to_datetime(energy_data['day'])
weather_data['time'] = pd.to_datetime(weather_data['time'])

print(f"\nEnergy data date range: {energy_data['day'].min()} to {energy_data['day'].max()}")
print(f"Weather data date range: {weather_data['time'].min()} to {weather_data['time'].max()}")

# 1. TEMPORAL FEATURE ENGINEERING
print("\n=== TEMPORAL FEATURE ENGINEERING ===")

def create_temporal_features(df, date_col='day'):
    """Create comprehensive temporal features"""
    df = df.copy()
    
    # Basic time features
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day_of_month'] = df[date_col].dt.day
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['day_of_year'] = df[date_col].dt.dayofyear
    df['week_of_year'] = df[date_col].dt.isocalendar().week
    df['quarter'] = df[date_col].dt.quarter
    
    # Cyclical features (important for ML models)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Boolean features
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)
    
    # Season features
    df['season'] = df['month'].apply(lambda x: 
        1 if x in [12, 1, 2] else  # Winter
        2 if x in [3, 4, 5] else   # Spring
        3 if x in [6, 7, 8] else   # Summer
        4)                         # Autumn
    
    # Holiday approximations (simplified)
    df['is_december'] = (df['month'] == 12).astype(int)
    df['is_january'] = (df['month'] == 1).astype(int)
    
    print(f"Created {len([col for col in df.columns if col not in [date_col, 'LCLid', 'energy_sum']])} temporal features")
    return df

# Apply temporal features
energy_data = create_temporal_features(energy_data)
print("Temporal features created for energy data")

# 2. LAG FEATURES
print("\n=== LAG FEATURE ENGINEERING ===")

def create_lag_features(df, target_col='energy_sum', household_col='LCLid', lags=[1, 2, 3, 7, 14, 30]):
    """Create lag features for each household"""
    df = df.copy()
    df = df.sort_values([household_col, 'day'])
    
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df.groupby(household_col)[target_col].shift(lag)
    
    print(f"Created lag features for periods: {lags}")
    return df

# Create lag features
energy_data = create_lag_features(energy_data)

# 3. ROLLING WINDOW FEATURES
print("\n=== ROLLING WINDOW FEATURES ===")

def create_rolling_features(df, target_col='energy_sum', household_col='LCLid', windows=[7, 14, 30]):
    """Create rolling statistical features"""
    df = df.copy()
    df = df.sort_values([household_col, 'day'])
    
    for window in windows:
        # Rolling mean
        df[f'{target_col}_rolling_mean_{window}'] = df.groupby(household_col)[target_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        
        # Rolling std
        df[f'{target_col}_rolling_std_{window}'] = df.groupby(household_col)[target_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
        
        # Rolling min/max
        df[f'{target_col}_rolling_min_{window}'] = df.groupby(household_col)[target_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).min()
        )
        df[f'{target_col}_rolling_max_{window}'] = df.groupby(household_col)[target_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).max()
        )
    
    print(f"Created rolling features for windows: {windows}")
    return df

# Create rolling features
energy_data = create_rolling_features(energy_data)

# 4. HOUSEHOLD-SPECIFIC FEATURES
print("\n=== HOUSEHOLD-SPECIFIC FEATURES ===")

def create_household_features(df, target_col='energy_sum', household_col='LCLid'):
    """Create household-specific statistical features"""
    
    # Calculate household statistics
    household_stats = df.groupby(household_col)[target_col].agg([
        'mean', 'std', 'min', 'max', 'median'
    ]).reset_index()
    
    household_stats.columns = [household_col, 'household_mean', 'household_std', 
                              'household_min', 'household_max', 'household_median']
    
    # Merge back to main dataframe
    df = df.merge(household_stats, on=household_col, how='left')
    
    # Create relative features
    df['consumption_vs_household_mean'] = df[target_col] / df['household_mean']
    df['consumption_vs_household_median'] = df[target_col] / df['household_median']
    
    print("Created household-specific features")
    return df

# Create household features
energy_data = create_household_features(energy_data)

# 5. WEATHER DATA PREPARATION
print("\n=== WEATHER DATA PREPARATION ===")

# Convert weather time to date for merging
weather_data['date'] = weather_data['time'].dt.date
weather_daily = weather_data.groupby('date').agg({
    'temperature': ['mean', 'min', 'max'],
    'humidity': 'mean',
    'visibility': 'mean',
    'pressure': 'mean'
}).round(2)

# Flatten column names
weather_daily.columns = ['_'.join(col).strip() for col in weather_daily.columns.values]
weather_daily = weather_daily.reset_index()
weather_daily['date'] = pd.to_datetime(weather_daily['date'])

print(f"Weather data aggregated to daily: {weather_daily.shape}")
print("Weather features:", [col for col in weather_daily.columns if col != 'date'])

# 6. MERGE ENERGY AND WEATHER DATA
print("\n=== MERGING ENERGY AND WEATHER DATA ===")

# Convert energy day to date for merging
energy_data['date'] = energy_data['day'].dt.date
energy_data['date'] = pd.to_datetime(energy_data['date'])

# Merge datasets
merged_data = energy_data.merge(weather_daily, on='date', how='left')
print(f"Merged data shape: {merged_data.shape}")

# Check for missing weather data
weather_missing = merged_data[['temperature_mean', 'humidity_mean']].isnull().sum()
print(f"Missing weather data after merge: {weather_missing.sum()} records")

# Fill missing weather data with forward/backward fill
weather_cols = [col for col in merged_data.columns if any(w in col for w in ['temperature', 'humidity', 'visibility', 'pressure'])]
for col in weather_cols:
    merged_data[col] = merged_data[col].fillna(method='ffill').fillna(method='bfill')

print("Missing weather data filled")

# 7. FEATURE SELECTION AND CLEANING
print("\n=== FEATURE SELECTION AND CLEANING ===")

# Remove rows with missing lag features (early dates)
initial_rows = len(merged_data)
merged_data = merged_data.dropna(subset=['energy_sum_lag_7', 'energy_sum_lag_14'])
print(f"Removed {initial_rows - len(merged_data)} rows with missing lag features")

# Select final feature set
feature_columns = [
    # Target variable
    'energy_sum',
    
    # Temporal features
    'month', 'day_of_week', 'day_of_year', 'quarter',
    'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
    'day_of_year_sin', 'day_of_year_cos',
    'is_weekend', 'is_monday', 'is_friday', 'season',
    'is_december', 'is_january',
    
    # Lag features
    'energy_sum_lag_1', 'energy_sum_lag_2', 'energy_sum_lag_3',
    'energy_sum_lag_7', 'energy_sum_lag_14', 'energy_sum_lag_30',
    
    # Rolling features
    'energy_sum_rolling_mean_7', 'energy_sum_rolling_std_7',
    'energy_sum_rolling_mean_14', 'energy_sum_rolling_std_14',
    'energy_sum_rolling_mean_30', 'energy_sum_rolling_std_30',
    
    # Household features
    'household_mean', 'household_std', 'consumption_vs_household_mean',
    
    # Weather features
    'temperature_mean', 'temperature_min', 'temperature_max',
    'humidity_mean', 'pressure_mean',
    
    # Identifiers
    'LCLid', 'day', 'date'
]

# Keep only selected columns
available_columns = [col for col in feature_columns if col in merged_data.columns]
final_data = merged_data[available_columns].copy()

print(f"Final dataset shape: {final_data.shape}")
print(f"Selected {len(available_columns)-3} features (excluding target and identifiers)")

# 8. TRAIN/VALIDATION/TEST SPLIT
print("\n=== TRAIN/VALIDATION/TEST SPLIT ===")

# Sort by date for time series split
final_data = final_data.sort_values(['LCLid', 'day'])

# Time-based split (70% train, 15% validation, 15% test)
dates = sorted(final_data['day'].unique())
n_dates = len(dates)

train_end_idx = int(0.7 * n_dates)
val_end_idx = int(0.85 * n_dates)

train_dates = dates[:train_end_idx]
val_dates = dates[train_end_idx:val_end_idx]
test_dates = dates[val_end_idx:]

train_data = final_data[final_data['day'].isin(train_dates)]
val_data = final_data[final_data['day'].isin(val_dates)]
test_data = final_data[final_data['day'].isin(test_dates)]

print(f"Train data: {train_data.shape} ({train_dates[0]} to {train_dates[-1]})")
print(f"Validation data: {val_data.shape} ({val_dates[0]} to {val_dates[-1]})")
print(f"Test data: {test_data.shape} ({test_dates[0]} to {test_dates[-1]})")

# 9. FEATURE SCALING
print("\n=== FEATURE SCALING ===")

# Identify numerical features for scaling
numerical_features = [col for col in final_data.columns 
                     if final_data[col].dtype in ['int64', 'float64'] 
                     and col not in ['LCLid', 'energy_sum']]

print(f"Scaling {len(numerical_features)} numerical features")

# Initialize scalers
scaler = StandardScaler()

# Fit scaler on training data only
X_train_scaled = train_data[numerical_features].copy()
X_train_scaled[numerical_features] = scaler.fit_transform(X_train_scaled[numerical_features])

# Transform validation and test data
X_val_scaled = val_data[numerical_features].copy()
X_val_scaled[numerical_features] = scaler.transform(X_val_scaled[numerical_features])

X_test_scaled = test_data[numerical_features].copy()
X_test_scaled[numerical_features] = scaler.transform(X_test_scaled[numerical_features])

# Add back target and identifiers
for dataset_name, (original, scaled) in [
    ('train', (train_data, X_train_scaled)),
    ('val', (val_data, X_val_scaled)),
    ('test', (test_data, X_test_scaled))
]:
    scaled['energy_sum'] = original['energy_sum'].values
    scaled['LCLid'] = original['LCLid'].values
    scaled['day'] = original['day'].values

print("Feature scaling completed")

# 10. FEATURE IMPORTANCE ANALYSIS
print("\n=== FEATURE IMPORTANCE ANALYSIS ===")

# Calculate correlation with target
feature_correlations = train_data[numerical_features + ['energy_sum']].corr()['energy_sum'].abs().sort_values(ascending=False)

print("Top 10 features correlated with energy consumption:")
print(feature_correlations.head(10).round(3))

# Visualize feature correlations
plt.figure(figsize=(12, 8))
top_features = feature_correlations.head(15)
plt.barh(range(len(top_features)), top_features.values)
plt.yticks(range(len(top_features)), top_features.index)
plt.xlabel('Absolute Correlation with Energy Consumption')
plt.title('Top 15 Feature Correlations with Energy Consumption')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 11. DATA QUALITY CHECKS
print("\n=== DATA QUALITY CHECKS ===")

def check_data_quality(data, name):
    """Check data quality metrics"""
    print(f"\n{name} Data Quality:")
    print(f"Shape: {data.shape}")
    print(f"Missing values: {data.isnull().sum().sum()}")
    print(f"Infinite values: {np.isinf(data.select_dtypes(include=[np.number])).sum().sum()}")
    print(f"Unique households: {data['LCLid'].nunique()}")
    print(f"Date range: {data['day'].min()} to {data['day'].max()}")

check_data_quality(X_train_scaled, "Training")
check_data_quality(X_val_scaled, "Validation")
check_data_quality(X_test_scaled, "Testing")

# 12. SAVE PROCESSED DATA
print("\n=== SAVING PROCESSED DATA ===")

try:
    # Save train/val/test splits
    X_train_scaled.to_csv('../data/processed/train_data_features.csv', index=False)
    X_val_scaled.to_csv('../data/processed/val_data_features.csv', index=False)
    X_test_scaled.to_csv('../data/processed/test_data_features.csv', index=False)
    
    # Save feature names and scaler
    feature_info = {
        'numerical_features': numerical_features,
        'all_features': available_columns
    }
    
    pd.DataFrame({'feature_names': numerical_features}).to_csv('../data/processed/feature_names.csv', index=False)
    
    # Save scaler (using joblib would be better in practice)
    import pickle
    with open('../data/processed/feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Data successfully saved:")
    print("- train_data_features.csv")
    print("- val_data_features.csv") 
    print("- test_data_features.csv")
    print("- feature_names.csv")
    print("- feature_scaler.pkl")
    
except Exception as e:
    print(f"Error saving data: {e}")

# 13. SUMMARY STATISTICS
print("\n=== FEATURE ENGINEERING SUMMARY ===")
print(f"Original features: {len(energy_data.columns)}")
print(f"Final features: {len(numerical_features)}")
print(f"Total samples: {len(final_data):,}")
print(f"Training samples: {len(X_train_scaled):,}")
print(f"Validation samples: {len(X_val_scaled):,}")
print(f"Test samples: {len(X_test_scaled):,}")

print("\nFeature categories created:")
print("- Temporal features (cyclical encoding)")
print("- Lag features (1, 2, 3, 7, 14, 30 days)")
print("- Rolling window statistics (7, 14, 30 days)")
print("- Household-specific statistics")
print("- Weather features (temperature, humidity, pressure)")

print("\nData is ready for model training!")
print("Next steps:")
print("1. LSTM model development")
print("2. Traditional ML models (Random Forest, XGBoost)")
print("3. Model evaluation and comparison")
print("4. Hyperparameter tuning")