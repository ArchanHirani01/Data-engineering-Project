import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import display

# Additional libraries for data analysis and visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Libraries for data preprocessing
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Configure visualization settings
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("All libraries imported successfully!")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

print("Exploring Kaggle input directory structure:")
print("=" * 50)

data_files = []
total_size = 0

for dirname, _, filenames in os.walk('/kaggle/input'):
    print(f"\nDirectory: {dirname}")

    for filename in filenames:
        file_path = os.path.join(dirname, filename)
        print(f"   📄 {filename}")

        # Get file size
        try:
            file_size = os.path.getsize(file_path)
            total_size += file_size
            print(f"      Size: {file_size / (1024*1024):.2f} MB")
            data_files.append({
                'directory': dirname,
                'filename': filename,
                'full_path': file_path,
                'size_mb': file_size / (1024*1024)
            })
        except:
            print(f"      Size: Unable to determine")

print(f"\n📊 Summary:")
print(f"   Total files found: {len(data_files)}")
print(f"   Total data size: {total_size / (1024*1024):.2f} MB")

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

print(f"\nWorking directory: /kaggle/working/ (20GB available)")
print(f"Temp directory: /kaggle/temp/ (session only)")


# Load the NYC Taxi dataset
# Note: Adjust the path based on your specific dataset location in Kaggle
try:
    # Try common file paths for NYC taxi data
    possible_paths = [
        '/kaggle/input/nyc-yellow-taxi-trip-data/yellow_tripdata_2015-01.csv',
        '/kaggle/input/nyc-yellow-taxi-trip-data/yellow_tripdata_2016-01.csv',
        '/kaggle/input/nyc-yellow-taxi-trip-data/yellow_tripdata_2016-02.csv',
        '/kaggle/input/nyc-yellow-taxi-trip-data/yellow_tripdata_2016-03.csv'
    ]

    df = None
    used_path = None

    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading data from: {path}")
            df = pd.read_csv(path)
            used_path = path
            break


    print(f"\n📊 Dataset loaded successfully!")
    print(f"   Shape: {df.shape}")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    print("Creating minimal sample data for demonstration...")

    # Minimal sample for error case
    df = pd.DataFrame({
        'pickup_datetime': ['2023-01-01 12:00:00'] * 1000,
        'fare_amount': np.random.uniform(5, 50, 1000)
    })


# Initial data inspection
print("INITIAL DATA INSPECTION")
print("=" * 50)

# Display basic information
print("\n1.Dataset Overview:")
print(f"   Rows: {df.shape[0]:,}")
print(f"   Columns: {df.shape[1]}")
print(f"   Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Display first few rows
print("\n2.First 5 rows:")
print(df.head())

# Data types and null values
print("\n3.Data Types and Missing Values:")
info_df = pd.DataFrame({
    'Column': df.columns,
    'Data_Type': df.dtypes,
    'Non_Null_Count': df.count(),
    'Null_Count': df.isnull().sum(),
    'Null_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
})
display(info_df)

# Basic statistics
print("\n4.Statistical Summary:")
display(df.describe())

# Check for duplicate rows
duplicates = df.duplicated().sum()
print(f"\n5. Duplicate rows: {duplicates:,} ({duplicates/len(df)*100:.2f}%)")

# Initial data inspection
print("INITIAL DATA INSPECTION")
print("=" * 50)

# Display basic information
print("\n1.Dataset Overview:")
print(f"   Rows: {df.shape[0]:,}")
print(f"   Columns: {df.shape[1]}")
print(f"   Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Display first few rows
print("\n2.First 5 rows:")
display(df.head())

# Data types and null values
print("\n3.Data Types and Missing Values:")
info_df = pd.DataFrame({
    'Column': df.columns,
    'Data_Type': df.dtypes,
    'Non_Null_Count': df.count(),
    'Null_Count': df.isnull().sum(),
    'Null_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
})
display(info_df)

# Basic statistics
print("\n4.Statistical Summary:")
display(df.describe())

# Check for duplicate rows
duplicates = df.duplicated().sum()
print(f"\n5. Duplicate rows: {duplicates:,} ({duplicates/len(df)*100:.2f}%)")# Initial data inspection
print("INITIAL DATA INSPECTION")
print("=" * 50)

# Display basic information
print("\n1.Dataset Overview:")
print(f"   Rows: {df.shape[0]:,}")
print(f"   Columns: {df.shape[1]}")
print(f"   Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Display first few rows
print("\n2.First 5 rows:")
display(df.head())

# Data types and null values
print("\n3.Data Types and Missing Values:")
info_df = pd.DataFrame({
    'Column': df.columns,
    'Data_Type': df.dtypes,
    'Non_Null_Count': df.count(),
    'Null_Count': df.isnull().sum(),
    'Null_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
})
display(info_df)

# Basic statistics
print("\n4.Statistical Summary:")
display(df.describe())

# Check for duplicate rows
duplicates = df.duplicated().sum()
print(f"\n5. Duplicate rows: {duplicates:,} ({duplicates/len(df)*100:.2f}%)")

# Load the NYC Taxi dataset
# Note: Adjust the path based on your specific dataset location in Kaggle
try:
    # Try common file paths for NYC taxi data
    possible_paths = [
        '/kaggle/input/nyc-yellow-taxi-trip-data/yellow_tripdata_2015-01.csv',
        '/kaggle/input/nyc-yellow-taxi-trip-data/yellow_tripdata_2016-01.csv',
        '/kaggle/input/nyc-yellow-taxi-trip-data/yellow_tripdata_2016-02.csv',
        '/kaggle/input/nyc-yellow-taxi-trip-data/yellow_tripdata_2016-03.csv'
    ]

    df = None
    used_path = None

    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading data from: {path}")
            df = pd.read_csv(path)
            used_path = path
            break


    print(f"\n📊 Dataset loaded successfully!")
    print(f"   Shape: {df.shape}")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    print("Creating minimal sample data for demonstration...")

    # Minimal sample for error case
    df = pd.DataFrame({
        'pickup_datetime': ['2023-01-01 12:00:00'] * 1000,
        'fare_amount': np.random.uniform(5, 50, 1000)
    })


# Data cleaning and preprocessing
print("DATA CLEANING AND PREPROCESSING")
print("=" * 50)

# Store original shape for comparison
original_shape = df.shape
print(f"Original dataset shape: {original_shape}")

# 1. Handle datetime columns
datetime_columns = [col for col in df.columns if 'datetime' in col.lower() or 'pickup' in col.lower() or 'dropoff' in col.lower()]
print(f"\n1. Converting datetime columns: {datetime_columns}")

for col in datetime_columns:
    if col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col])
            print(f"   Converted {col} to datetime")
        except:
            print(f"  Failed to convert {col}")

# 2. Remove duplicates
duplicates_before = df.duplicated().sum()
df = df.drop_duplicates()
duplicates_removed = duplicates_before - df.duplicated().sum()
print(f"\n2.Removed {duplicates_removed:,} duplicate rows")

# 3. Handle missing values
print(f"\n3. Handling missing values:")
missing_before = df.isnull().sum().sum()

# Drop rows with missing critical columns (pickup/dropoff times, fare)
critical_columns = ['fare_amount'] + [col for col in df.columns if 'datetime' in col.lower()]
for col in critical_columns:
    if col in df.columns:
        before_count = len(df)
        df = df.dropna(subset=[col])
        dropped = before_count - len(df)
        if dropped > 0:
            print(f"   📉 Dropped {dropped:,} rows with missing {col}")

# Fill missing numerical values with median
numerical_columns = df.select_dtypes(include=[np.number]).columns
for col in numerical_columns:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"   🔧 Filled missing {col} with median: {median_val:.2f}")

missing_after = df.isnull().sum().sum()
print(f"   Missing values before: {missing_before:,} → after: {missing_after:,}")

# 4. Filter invalid data
print(f"\n4. Filtering invalid data:")

if 'fare_amount' in df.columns:
    # Remove negative fares and extremely high fares
    before_count = len(df)
    df = df[(df['fare_amount'] >= 0) & (df['fare_amount'] <= 500)]
    print(f"  Removed {before_count - len(df):,} rows with invalid fare amounts")

if 'passenger_count' in df.columns:
    # Remove invalid passenger counts
    before_count = len(df)
    df = df[(df['passenger_count'] >= 1) & (df['passenger_count'] <= 6)]
    print(f"   Removed {before_count - len(df):,} rows with invalid passenger counts")

if 'trip_distance' in df.columns:
    # Remove invalid trip distances
    before_count = len(df)
    df = df[(df['trip_distance'] >= 0) & (df['trip_distance'] <= 100)]
    print(f"   Removed {before_count - len(df):,} rows with invalid trip distances")

# 5. Create derived columns if datetime columns exist
datetime_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]']
if len(datetime_cols) >= 2:
    pickup_col = [col for col in datetime_cols if 'pickup' in col.lower()]
    dropoff_col = [col for col in datetime_cols if 'dropoff' in col.lower()]

    if pickup_col and dropoff_col:
        pickup_col = pickup_col[0]
        dropoff_col = dropoff_col[0]

        # Calculate trip duration
        df['trip_duration_minutes'] = (df[dropoff_col] - df[pickup_col]).dt.total_seconds() / 60

        # Remove trips with invalid duration (negative or too long)
        before_count = len(df)
        df = df[(df['trip_duration_minutes'] > 0) & (df['trip_duration_minutes'] <= 300)]
        print(f"   Removed {before_count - len(df):,} rows with invalid trip duration")

print(f"\nFinal dataset shape: {df.shape}")
print(f"Total rows removed: {original_shape[0] - df.shape[0]:,} ({(original_shape[0] - df.shape[0])/original_shape[0]*100:.1f}%)")
print(f"Cleaned dataset ready for analysis!")


# Exploratory Data Analysis
print(" EXPLORATORY DATA ANALYSIS")
print("=" * 50)

# 1. Basic statistics for numerical columns
print("\n1. Key Statistics Summary:")
numerical_cols = df.select_dtypes(include=[np.number]).columns

stats_summary = pd.DataFrame({
    'Column': numerical_cols,
    'Mean': [df[col].mean() for col in numerical_cols],
    'Median': [df[col].median() for col in numerical_cols],
    'Std': [df[col].std() for col in numerical_cols],
    'Min': [df[col].min() for col in numerical_cols],
    'Max': [df[col].max() for col in numerical_cols],
    'Unique_Values': [df[col].nunique() for col in numerical_cols]
}).round(2)

display(stats_summary)

# 2. Correlation analysis
print("\n2. Correlation Analysis:")
if len(numerical_cols) > 1:
    correlation_matrix = df[numerical_cols].corr()

    # Find strongest correlations
    correlation_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            col1 = correlation_matrix.columns[i]
            col2 = correlation_matrix.columns[j]
            corr_value = correlation_matrix.iloc[i, j]
            correlation_pairs.append({
                'Column_1': col1,
                'Column_2': col2,
                'Correlation': corr_value
            })

    corr_df = pd.DataFrame(correlation_pairs)
    corr_df = corr_df.reindex(corr_df['Correlation'].abs().sort_values(ascending=False).index)

    print("    Top 5 Strongest Correlations:")
    display(corr_df.head())

# 3. Data distribution insights
print("\n3. Data Distribution Insights:")

for col in numerical_cols[:5]:  # Analyze first 5 numerical columns
    q25, q50, q75 = df[col].quantile([0.25, 0.5, 0.75])
    iqr = q75 - q25
    outliers = df[(df[col] < (q25 - 1.5 * iqr)) | (df[col] > (q75 + 1.5 * iqr))][col].count()

    print(f"    {col}:")
    print(f"      Range: {df[col].min():.2f} - {df[col].max():.2f}")
    print(f"      IQR: {q25:.2f} - {q75:.2f}")
    print(f"      Outliers: {outliers:,} ({outliers/len(df)*100:.1f}%)")

# 4. Time-based analysis (if datetime columns exist)
datetime_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]']
if datetime_cols:
    print(f"\n4. Time-based Analysis:")
    pickup_col = [col for col in datetime_cols if 'pickup' in col.lower()]

    if pickup_col:
        pickup_col = pickup_col[0]
        df['hour'] = df[pickup_col].dt.hour
        df['day_of_week'] = df[pickup_col].dt.day_name()
        df['month'] = df[pickup_col].dt.month

        print(f"    Date range: {df[pickup_col].min()} to {df[pickup_col].max()}")
        print(f"    Peak hours: {df['hour'].mode().values}")
        print(f"    Busiest day: {df['day_of_week'].mode().values[0]}")

# 5. Business insights for batch processing
print(f"\n5.  Business Insights for Batch Processing:")

if 'fare_amount' in df.columns:
    avg_fare = df['fare_amount'].mean()
    print(f"    Average fare: ${avg_fare:.2f}")

    # Revenue calculations for quarterly processing
    daily_revenue = df['fare_amount'].sum() / df[pickup_col].dt.date.nunique() if pickup_col else 0
    quarterly_revenue = daily_revenue * 90  # 3 months

    print(f"    Estimated daily revenue: ${daily_revenue:,.2f}")
    print(f"    Estimated quarterly revenue: ${quarterly_revenue:,.2f}")

if 'trip_distance' in df.columns:
    avg_distance = df['trip_distance'].mean()
    print(f"    Average trip distance: {avg_distance:.2f} miles")

if 'passenger_count' in df.columns:
    avg_passengers = df['passenger_count'].mean()
    print(f"    Average passengers per trip: {avg_passengers:.1f}")

print(f"\n EDA completed! Ready for visualization and feature engineering.")

# Data Visualization
print(" DATA VISUALIZATION")
print("=" * 50)

# Set up the plotting style
plt.figure(figsize=(20, 15))

# 1. Distribution plots for key numerical variables
numerical_cols = df.select_dtypes(include=[np.number]).columns
n_cols = min(4, len(numerical_cols))

if n_cols > 0:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(' Distribution of Key Numerical Variables', fontsize=16, fontweight='bold')

    for i, col in enumerate(numerical_cols[:4]):
        row = i // 2
        col_idx = i % 2

        # Histogram with KDE
        axes[row, col_idx].hist(df[col], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[row, col_idx].set_title(f'{col} Distribution')
        axes[row, col_idx].set_xlabel(col)
        axes[row, col_idx].set_ylabel('Frequency')
        axes[row, col_idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# 2. Box plots for outlier analysis
if len(numerical_cols) > 0:
    fig, axes = plt.subplots(1, min(3, len(numerical_cols)), figsize=(15, 5))
    fig.suptitle(' Box Plots - Outlier Analysis', fontsize=16, fontweight='bold')

    if len(numerical_cols) == 1:
        axes = [axes]
    elif len(numerical_cols) == 2:
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]

    for i, col in enumerate(numerical_cols[:3]):
        if i < len(axes):
            axes[i].boxplot(df[col].dropna())
            axes[i].set_title(f'{col}')
            axes[i].set_ylabel(col)
            axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# 3. Correlation heatmap
if len(numerical_cols) > 1:
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[numerical_cols].corr()

    sns.heatmap(correlation_matrix,
                annot=True,
                cmap='coolwarm',
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={'shrink': 0.8})

    plt.title(' Correlation Matrix Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# 4. Time-based visualizations (if datetime columns exist)
datetime_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]']
pickup_col = [col for col in datetime_cols if 'pickup' in col.lower()]

if pickup_col and len(pickup_col) > 0:
    pickup_col = pickup_col[0]

    # Hourly trip patterns
    if 'hour' in df.columns:
        plt.figure(figsize=(12, 6))
        hourly_trips = df['hour'].value_counts().sort_index()

        plt.subplot(1, 2, 1)
        hourly_trips.plot(kind='bar', color='lightcoral')
        plt.title(' Trip Volume by Hour of Day')
        plt.xlabel('Hour')
        plt.ylabel('Number of Trips')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # Daily patterns
        if 'day_of_week' in df.columns:
            plt.subplot(1, 2, 2)
            daily_trips = df['day_of_week'].value_counts()
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_trips = daily_trips.reindex([day for day in days_order if day in daily_trips.index])

            daily_trips.plot(kind='bar', color='lightgreen')
            plt.title(' Trip Volume by Day of Week')
            plt.xlabel('Day')
            plt.ylabel('Number of Trips')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

# 5. Business metrics visualization
if 'fare_amount' in df.columns:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(' Business Metrics Dashboard', fontsize=16, fontweight='bold')

    # Fare distribution
    axes[0, 0].hist(df['fare_amount'], bins=50, color='gold', alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Fare Amount Distribution')
    axes[0, 0].set_xlabel('Fare ($)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)

    # Passenger count distribution
    if 'passenger_count' in df.columns:
        passenger_counts = df['passenger_count'].value_counts().sort_index()
        axes[0, 1].bar(passenger_counts.index, passenger_counts.values, color='lightblue')
        axes[0, 1].set_title('Passenger Count Distribution')
        axes[0, 1].set_xlabel('Number of Passengers')
        axes[0, 1].set_ylabel('Number of Trips')
        axes[0, 1].grid(True, alpha=0.3)

    # Trip distance vs fare
    if 'trip_distance' in df.columns:
        sample_size = min(5000, len(df))  # Sample for performance
        sample_df = df.sample(n=sample_size)

        axes[1, 0].scatter(sample_df['trip_distance'], sample_df['fare_amount'],
                          alpha=0.5, color='purple', s=10)
        axes[1, 0].set_title('Trip Distance vs Fare Amount')
        axes[1, 0].set_xlabel('Trip Distance (miles)')
        axes[1, 0].set_ylabel('Fare Amount ($)')
        axes[1, 0].grid(True, alpha=0.3)

    # Monthly revenue trend (if dates available)
    if pickup_col and 'month' in df.columns:
        monthly_revenue = df.groupby('month')['fare_amount'].sum()
        axes[1, 1].plot(monthly_revenue.index, monthly_revenue.values,
                       marker='o', linewidth=2, color='red')
        axes[1, 1].set_title('Monthly Revenue Trend')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Total Revenue ($)')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

print("\n Visualizations completed!")
print(" Key insights for batch processing pipeline:")
print("   • Hourly patterns will guide data partitioning strategies")
print("   • Fare distributions inform outlier detection rules")
print("   • Correlation patterns help feature engineering")
print("   • Revenue trends support quarterly aggregation logic")

# Feature Engineering
print("🔧 FEATURE ENGINEERING")
print("=" * 50)

# Store original column count
original_cols = len(df.columns)

# Fix coordinate columns that might have been incorrectly converted to datetime
coordinate_cols = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
for col in coordinate_cols:
    if col in df.columns and df[col].dtype == 'datetime64[ns]':
        print(f"⚠️ Fixing incorrectly converted coordinate column: {col}")
        # Try to convert back to numeric, fill invalid values with NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 1. Time-based features
datetime_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]']
pickup_col = [col for col in datetime_cols if 'pickup' in col.lower()]

if pickup_col:
    pickup_col = pickup_col[0]
    print(f"\n1️⃣ Creating time-based features from {pickup_col}:")

    # Extract temporal features
    df['year'] = df[pickup_col].dt.year
    df['month'] = df[pickup_col].dt.month
    df['day'] = df[pickup_col].dt.day
    df['hour'] = df[pickup_col].dt.hour
    df['minute'] = df[pickup_col].dt.minute
    df['day_of_week'] = df[pickup_col].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_name'] = df[pickup_col].dt.day_name()
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Time of day categories
    def get_time_period(hour):
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'

    df['time_period'] = df['hour'].apply(get_time_period)

    # Rush hour indicator
    df['is_rush_hour'] = ((df['hour'].isin([7, 8, 9, 17, 18, 19])) &
                         (df['day_of_week'] < 5)).astype(int)

    print("   ✅ Created: year, month, day, hour, minute, day_of_week, is_weekend")
    print("   ✅ Created: time_period, is_rush_hour")

# 2. Distance and location features
if 'trip_distance' in df.columns:
    print(f"\n2️⃣ Creating distance-based features:")

    # Distance categories
    def categorize_distance(distance):
        if distance <= 1:
            return 'Short'
        elif distance <= 5:
            return 'Medium'
        elif distance <= 15:
            return 'Long'
        else:
            return 'Very_Long'

    df['distance_category'] = df['trip_distance'].apply(categorize_distance)

    # Distance bins for analysis
    df['distance_bin'] = pd.cut(df['trip_distance'],
                               bins=[0, 1, 3, 5, 10, float('inf')],
                               labels=['0-1', '1-3', '3-5', '5-10', '10+'])

    print("   ✅ Created: distance_category, distance_bin")

# 3. Fare and payment features
if 'fare_amount' in df.columns:
    print(f"\n3️⃣ Creating fare-based features:")

    # Fare per mile (if trip distance available)
    if 'trip_distance' in df.columns:
        df['fare_per_mile'] = df['fare_amount'] / df['trip_distance'].replace(0, np.nan)
        df['fare_per_mile'] = df['fare_per_mile'].fillna(df['fare_per_mile'].median())

    # Fare categories
    def categorize_fare(fare):
        if fare <= 10:
            return 'Budget'
        elif fare <= 25:
            return 'Standard'
        elif fare <= 50:
            return 'Premium'
        else:
            return 'Luxury'

    df['fare_category'] = df['fare_amount'].apply(categorize_fare)

    # Tip features (if tip amount available)
    if 'tip_amount' in df.columns:
        df['tip_percentage'] = (df['tip_amount'] / df['fare_amount'] * 100).fillna(0)
        df['is_generous_tipper'] = (df['tip_percentage'] > 20).astype(int)

    print("   ✅ Created: fare_per_mile, fare_category")
    if 'tip_amount' in df.columns:
        print("   ✅ Created: tip_percentage, is_generous_tipper")

# 4. Trip duration features (if available)
if 'trip_duration_minutes' in df.columns:
    print(f"\n4️⃣ Creating duration-based features:")

    # Speed calculation
    if 'trip_distance' in df.columns:
        df['average_speed_mph'] = (df['trip_distance'] / (df['trip_duration_minutes'] / 60)).replace([np.inf, -np.inf], np.nan)
        df['average_speed_mph'] = df['average_speed_mph'].fillna(df['average_speed_mph'].median())

    # Duration categories
    def categorize_duration(duration):
        if duration <= 10:
            return 'Quick'
        elif duration <= 30:
            return 'Normal'
        elif duration <= 60:
            return 'Long'
        else:
            return 'Very_Long'

    df['duration_category'] = df['trip_duration_minutes'].apply(categorize_duration)

    print("   ✅ Created: average_speed_mph, duration_category")

# 5. Passenger and capacity features
if 'passenger_count' in df.columns:
    print(f"\n5️⃣ Creating passenger-based features:")

    # Group size categories
    def categorize_group_size(passengers):
        if passengers == 1:
            return 'Solo'
        elif passengers == 2:
            return 'Couple'
        elif passengers <= 4:
            return 'Small_Group'
        else:
            return 'Large_Group'

    df['group_size_category'] = df['passenger_count'].apply(categorize_group_size)

    # Revenue per passenger
    if 'fare_amount' in df.columns:
        df['fare_per_passenger'] = df['fare_amount'] / df['passenger_count']

    print("   ✅ Created: group_size_category, fare_per_passenger")

# 6. Location-based features (if coordinates available and properly formatted)
location_cols = [col for col in df.columns if 'longitude' in col.lower() or 'latitude' in col.lower()]
numeric_location_cols = [col for col in location_cols if pd.api.types.is_numeric_dtype(df[col])]

if len(numeric_location_cols) >= 4:  # pickup and dropoff coordinates
    print(f"\n6️⃣ Creating location-based features:")

    # Find coordinate columns
    pickup_lat_col = [col for col in numeric_location_cols if 'pickup' in col.lower() and 'lat' in col.lower()]
    pickup_lon_col = [col for col in numeric_location_cols if 'pickup' in col.lower() and 'lon' in col.lower()]
    dropoff_lat_col = [col for col in numeric_location_cols if 'dropoff' in col.lower() and 'lat' in col.lower()]
    dropoff_lon_col = [col for col in numeric_location_cols if 'dropoff' in col.lower() and 'lon' in col.lower()]

    if pickup_lat_col and pickup_lon_col and dropoff_lat_col and dropoff_lon_col:
        try:
            # Simplified distance calculation (not exact but good for features)
            lat_diff = df[dropoff_lat_col[0]] - df[pickup_lat_col[0]]
            lon_diff = df[dropoff_lon_col[0]] - df[pickup_lon_col[0]]
            df['straight_line_distance'] = np.sqrt(lat_diff**2 + lon_diff**2) * 111  # Approximate km

            print("   ✅ Created: straight_line_distance")
        except Exception as e:
            print(f"   ⚠️ Could not create location features: {e}")
            print(f"   📊 Coordinate column types: {[(col, df[col].dtype) for col in [pickup_lat_col[0], pickup_lon_col[0], dropoff_lat_col[0], dropoff_lon_col[0]]]}")
else:
    print(f"\n6️⃣ Skipping location-based features:")
    print(f"   ⚠️ Insufficient numeric coordinate columns found: {len(numeric_location_cols)}/4 needed")
    if location_cols:
        print(f"   📊 Available location columns: {[(col, df[col].dtype) for col in location_cols]}")

# 7. Quarterly aggregation features (for batch processing)
if pickup_col:
    print(f"\n7️⃣ Creating quarterly aggregation features:")

    # Quarter identification
    df['quarter'] = df[pickup_col].dt.quarter
    df['year_quarter'] = df['year'].astype(str) + '_Q' + df['quarter'].astype(str)

    # Monthly aggregation
    df['year_month'] = df[pickup_col].dt.to_period('M').astype(str)

    print("   ✅ Created: quarter, year_quarter, year_month")

# 8. One-hot encoding for categorical features
print(f"\n8️⃣ One-hot encoding categorical features:")

categorical_features = ['time_period', 'distance_category', 'fare_category', 'duration_category', 'group_size_category']
existing_categorical = [col for col in categorical_features if col in df.columns]

if existing_categorical:
    # Create dummy variables
    df_encoded = pd.get_dummies(df, columns=existing_categorical, prefix=existing_categorical, drop_first=True)

    # Update dataframe
    new_dummy_cols = [col for col in df_encoded.columns if col not in df.columns]
    for col in new_dummy_cols:
        df[col] = df_encoded[col]

    print(f"   ✅ Created {len(new_dummy_cols)} dummy variables from {len(existing_categorical)} categorical features")

# 9. Feature summary
new_cols = len(df.columns)
print(f"\n✅ FEATURE ENGINEERING COMPLETED!")
print(f"📊 Original columns: {original_cols}")
print(f"📊 New columns: {new_cols}")
print(f"📊 Features added: {new_cols - original_cols}")

# Display new feature summary
print(f"\n📋 New Feature Categories:")
time_features = [col for col in df.columns if col in ['year', 'month', 'day', 'hour', 'is_weekend', 'is_rush_hour', 'time_period']]
distance_features = [col for col in df.columns if 'distance' in col.lower() or 'speed' in col.lower()]
fare_features = [col for col in df.columns if 'fare' in col.lower() or 'tip' in col.lower()]
categorical_features = [col for col in df.columns if col.endswith(('_Budget', '_Standard', '_Premium', '_Solo', '_Couple', '_Morning', '_Afternoon', '_Evening', '_Night'))]

print(f"   ⏰ Time features ({len(time_features)}): {time_features[:5]}...")
print(f"   🛣️ Distance features ({len(distance_features)}): {distance_features}")
print(f"   💰 Fare features ({len(fare_features)}): {fare_features}")
print(f"   🏷️ Encoded features ({len(categorical_features)}): {categorical_features[:5]}...")

print(f"\n🎯 Dataset ready for batch processing and ML model training!")

# Export processed data
print("💾 EXPORTING PROCESSED DATA")
print("=" * 50)

# 1. Create export directory structure
export_base = '/kaggle/working/'
export_dirs = ['processed_data', 'quarterly_aggregations', 'ml_features']

for directory in export_dirs:
    dir_path = os.path.join(export_base, directory)
    os.makedirs(dir_path, exist_ok=True)
    print(f"📁 Created directory: {dir_path}")

# 2. Save main processed dataset
print(f"\n1️⃣ Saving main processed dataset:")

# Full dataset
main_output_path = os.path.join(export_base, 'processed_data', 'nyc_taxi_processed.csv')
df.to_csv(main_output_path, index=False)
print(f"   ✅ Saved full dataset: {main_output_path}")
print(f"      Rows: {len(df):,}, Columns: {len(df.columns)}")
print(f"      Size: {os.path.getsize(main_output_path) / (1024*1024):.2f} MB")

# Parquet format for better performance in Spark
parquet_output_path = os.path.join(export_base, 'processed_data', 'nyc_taxi_processed.parquet')
df.to_parquet(parquet_output_path, index=False)
print(f"   ✅ Saved parquet format: {parquet_output_path}")
print(f"      Size: {os.path.getsize(parquet_output_path) / (1024*1024):.2f} MB")

# 3. Create quarterly aggregations (for batch processing simulation)
if 'year_quarter' in df.columns:
    print(f"\n2️⃣ Creating quarterly aggregations:")

    # Quarterly summary statistics
    quarterly_agg = df.groupby('year_quarter').agg({
        'fare_amount': ['count', 'sum', 'mean', 'std'] if 'fare_amount' in df.columns else 'count',
        'trip_distance': ['mean', 'sum'] if 'trip_distance' in df.columns else 'count',
        'passenger_count': ['sum', 'mean'] if 'passenger_count' in df.columns else 'count',
        'tip_amount': ['sum', 'mean'] if 'tip_amount' in df.columns else 'count'
    })

    # Flatten column names
    quarterly_agg.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in quarterly_agg.columns]
    quarterly_agg = quarterly_agg.reset_index()

    # Save quarterly aggregations
    quarterly_output_path = os.path.join(export_base, 'quarterly_aggregations', 'quarterly_summary.csv')
    quarterly_agg.to_csv(quarterly_output_path, index=False)
    print(f"   ✅ Saved quarterly aggregations: {quarterly_output_path}")
    print(f"      Quarters: {len(quarterly_agg)}")

    # Display quarterly summary
    print(f"   📊 Quarterly Summary:")
    for _, row in quarterly_agg.head().iterrows():
        quarter = row['year_quarter']
        if 'fare_amount_count' in quarterly_agg.columns:
            trips = int(row['fare_amount_count'])
            revenue = row['fare_amount_sum'] if 'fare_amount_sum' in quarterly_agg.columns else 0
            print(f"      {quarter}: {trips:,} trips, ${revenue:,.2f} revenue")

# 4. Create ML-ready feature sets
print(f"\n3️⃣ Creating ML-ready feature sets:")

# Numerical features only
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
ml_numerical = df[numerical_features].copy()

ml_numerical_path = os.path.join(export_base, 'ml_features', 'numerical_features.csv')
ml_numerical.to_csv(ml_numerical_path, index=False)
print(f"   ✅ Saved numerical features: {ml_numerical_path}")
print(f"      Features: {len(numerical_features)}")

# Target variables for different ML tasks
if 'fare_amount' in df.columns:
    # Fare prediction dataset
    fare_prediction_features = ['trip_distance', 'passenger_count', 'hour', 'day_of_week', 'is_weekend', 'is_rush_hour']
    fare_prediction_features = [col for col in fare_prediction_features if col in df.columns]

    if fare_prediction_features:
        fare_ml_data = df[fare_prediction_features + ['fare_amount']].copy()
        fare_ml_path = os.path.join(export_base, 'ml_features', 'fare_prediction_data.csv')
        fare_ml_data.to_csv(fare_ml_path, index=False)
        print(f"   ✅ Saved fare prediction dataset: {fare_ml_path}")
        print(f"      Features for fare prediction: {len(fare_prediction_features)}")

# 5. Create data dictionary/schema
print(f"\n4️⃣ Creating data dictionary:")

# Generate schema information
schema_info = []
for col in df.columns:
    col_info = {
        'column_name': col,
        'data_type': str(df[col].dtype),
        'non_null_count': df[col].count(),
        'null_count': df[col].isnull().sum(),
        'null_percentage': round(df[col].isnull().sum() / len(df) * 100, 2),
        'unique_values': df[col].nunique(),
        'example_values': str(df[col].dropna().head(3).tolist()) if df[col].count() > 0 else 'No data'
    }
    schema_info.append(col_info)

schema_df = pd.DataFrame(schema_info)
schema_path = os.path.join(export_base, 'processed_data', 'data_schema.csv')
schema_df.to_csv(schema_path, index=False)
print(f"   ✅ Saved data schema: {schema_path}")

# 6. Create processing summary report
print(f"\n5️⃣ Creating processing summary report:")

processing_summary = {
    'processing_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'original_rows': original_shape[0] if 'original_shape' in locals() else len(df),
    'processed_rows': len(df),
    'rows_removed': (original_shape[0] - len(df)) if 'original_shape' in locals() else 0,
    'original_columns': original_cols if 'original_cols' in locals() else len(df.columns),
    'final_columns': len(df.columns),
    'features_created': len(df.columns) - (original_cols if 'original_cols' in locals() else len(df.columns)),
    'data_quality_score': round((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2),
    'dataset_size_mb': round(df.memory_usage(deep=True).sum() / (1024*1024), 2),
    'ready_for_ml': True
}

# Save processing summary
summary_df = pd.DataFrame([processing_summary])
summary_path = os.path.join(export_base, 'processed_data', 'processing_summary.csv')
summary_df.to_csv(summary_path, index=False)
print(f"   ✅ Saved processing summary: {summary_path}")

# Display summary
print(f"\n📊 PROCESSING SUMMARY:")
print(f"   🕐 Processed at: {processing_summary['processing_timestamp']}")
print(f"   📈 Rows: {processing_summary['original_rows']:,} → {processing_summary['processed_rows']:,}")
print(f"   📊 Columns: {processing_summary['original_columns']} → {processing_summary['final_columns']}")
print(f"   🔧 Features created: {processing_summary['features_created']}")
print(f"   ✨ Data quality score: {processing_summary['data_quality_score']}%")
print(f"   💾 Dataset size: {processing_summary['dataset_size_mb']} MB")

# 7. List all exported files
print(f"\n📁 EXPORTED FILES:")
for root, dirs, files in os.walk(export_base):
    level = root.replace(export_base, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}📁 {os.path.basename(root)}/")
    sub_indent = ' ' * 2 * (level + 1)
    for file in files:
        file_path = os.path.join(root, file)
        file_size = os.path.getsize(file_path) / (1024*1024)
        print(f"{sub_indent}📄 {file} ({file_size:.2f} MB)")

print(f"\n✅ DATA EXPORT COMPLETED!")
print(f"🎯 All files saved to /kaggle/working/ and ready for batch processing pipeline!")
print(f"🚀 Next steps: Deploy to Spark/Hadoop infrastructure for quarterly ML model training")

# Export processed data to local storage and AWS S3
print("EXPORTING PROCESSED DATA TO LOCAL AND AWS S3")
print("=" * 60)

# Import AWS SDK
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    s3_available = True
    print("AWS SDK (boto3) imported successfully")
except ImportError:
    print("Installing boto3 for AWS S3 integration...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3"])
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    s3_available = True
    print("boto3 installed and imported successfully")

# AWS Configuration
AWS_ACCESS_KEY_ID = ''
AWS_SECRET_ACCESS_KEY = 'c7Hqp5T2hHd0MaVVcx9k3CQltQoXR+v4pX8QhG34'
AWS_REGION = 'us-east-1'  # Default region
S3_BUCKET_NAME = 'nyc-taxi-batch-processing'  # You may need to change this to an existing bucket
S3_PREFIX = f'taxi-data/processed/{datetime.now().strftime("%Y/%m/%d")}'

# Initialize S3 client
try:
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    print(f"AWS S3 client initialized for region: {AWS_REGION}")
except Exception as e:
    print(f"Failed to initialize S3 client: {e}")
    s3_available = False

# Function to upload file to S3
def upload_to_s3(local_file_path, s3_key, description="file"):
    """Upload a file to S3 bucket"""
    try:
        file_size = os.path.getsize(local_file_path) / (1024*1024)
        print(f"   Uploading {description} to S3: s3://{S3_BUCKET_NAME}/{s3_key}")

        s3_client.upload_file(local_file_path, S3_BUCKET_NAME, s3_key)
        print(f"   Successfully uploaded {description} ({file_size:.2f} MB)")
        return True
    except FileNotFoundError:
        print(f"   Local file not found: {local_file_path}")
        return False
    except NoCredentialsError:
        print(f"    AWS credentials not found or invalid")
        return False
    except ClientError as e:
        print(f"    AWS S3 error: {e}")
        return False
    except Exception as e:
        print(f"    Unexpected error uploading {description}: {e}")
        return False

export_base = '/kaggle/working/'
export_dirs = ['processed_data', 'quarterly_aggregations', 'ml_features', 'emr_ready']

for directory in export_dirs:
    dir_path = os.path.join(export_base, directory)
    os.makedirs(dir_path, exist_ok=True)
    print(f"Created directory: {dir_path}")

# 2. Save main processed dataset
print(f"\n Saving main processed dataset:")

# Full dataset in CSV format
main_output_path = os.path.join(export_base, 'processed_data', 'nyc_taxi_processed.csv')
df.to_csv(main_output_path, index=False)
print(f"   Saved full dataset: {main_output_path}")
print(f"      Rows: {len(df):,}, Columns: {len(df.columns)}")
print(f"      Size: {os.path.getsize(main_output_path) / (1024*1024):.2f} MB")

# Parquet format for better performance in Spark/EMR
parquet_output_path = os.path.join(export_base, 'processed_data', 'nyc_taxi_processed.parquet')
df.to_parquet(parquet_output_path, index=False)
print(f"    Saved parquet format: {parquet_output_path}")
print(f"      Size: {os.path.getsize(parquet_output_path) / (1024*1024):.2f} MB")

# Upload to S3
if s3_available:
    print(f"\n    Uploading main dataset to AWS S3:")
    upload_to_s3(main_output_path, f"{S3_PREFIX}/processed_data/nyc_taxi_processed.csv", "main dataset (CSV)")
    upload_to_s3(parquet_output_path, f"{S3_PREFIX}/processed_data/nyc_taxi_processed.parquet", "main dataset (Parquet)")

#3. Create quarterly aggregations (for batch processing simulation)
if 'year_quarter' in df.columns:
    print(f"\n Creating quarterly aggregations:")

    # Quarterly summary statistics
    quarterly_agg = df.groupby('year_quarter').agg({
        'fare_amount': ['count', 'sum', 'mean', 'std'] if 'fare_amount' in df.columns else 'count',
        'trip_distance': ['mean', 'sum'] if 'trip_distance' in df.columns else 'count',
        'passenger_count': ['sum', 'mean'] if 'passenger_count' in df.columns else 'count',
        'tip_amount': ['sum', 'mean'] if 'tip_amount' in df.columns else 'count'
    })

    # Flatten column names
    quarterly_agg.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in quarterly_agg.columns]
    quarterly_agg = quarterly_agg.reset_index()

    # Save quarterly aggregations
    quarterly_output_path = os.path.join(export_base, 'quarterly_aggregations', 'quarterly_summary.csv')
    quarterly_agg.to_csv(quarterly_output_path, index=False)
    print(f"    Saved quarterly aggregations: {quarterly_output_path}")
    print(f"      Quarters: {len(quarterly_agg)}")

    # Upload quarterly data to S3
    if s3_available:
        upload_to_s3(quarterly_output_path, f"{S3_PREFIX}/quarterly_aggregations/quarterly_summary.csv", "quarterly aggregations")

    # Display quarterly summary
    print(f"    Quarterly Summary:")
    for _, row in quarterly_agg.head().iterrows():
        quarter = row['year_quarter']
        if 'fare_amount_count' in quarterly_agg.columns:
            trips = int(row['fare_amount_count'])
            revenue = row['fare_amount_sum'] if 'fare_amount_sum' in quarterly_agg.columns else 0
            print(f"      {quarter}: {trips:,} trips, ${revenue:,.2f} revenue")

# Create ML-ready feature sets for EMR
print(f"\n Creating ML-ready feature sets for AWS EMR:")

# Numerical features only
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
ml_numerical = df[numerical_features].copy()

ml_numerical_path = os.path.join(export_base, 'ml_features', 'numerical_features.csv')
ml_numerical.to_csv(ml_numerical_path, index=False)
print(f"   Saved numerical features: {ml_numerical_path}")
print(f"      Features: {len(numerical_features)}")

# Target variables for different ML tasks
if 'fare_amount' in df.columns:
    # Fare prediction dataset
    fare_prediction_features = ['trip_distance', 'passenger_count', 'hour', 'day_of_week', 'is_weekend', 'is_rush_hour']
    fare_prediction_features = [col for col in fare_prediction_features if col in df.columns]

    if fare_prediction_features:
        fare_ml_data = df[fare_prediction_features + ['fare_amount']].copy()
        fare_ml_path = os.path.join(export_base, 'ml_features', 'fare_prediction_data.csv')
        fare_ml_data.to_csv(fare_ml_path, index=False)
        print(f"    Saved fare prediction dataset: {fare_ml_path}")
        print(f"      Features for fare prediction: {len(fare_prediction_features)}")

        # Create EMR-optimized version (smaller chunks for distributed processing)
        chunk_size = 10000
        emr_dir = os.path.join(export_base, 'emr_ready', 'fare_prediction_chunks')
        os.makedirs(emr_dir, exist_ok=True)

        for i, chunk in enumerate(range(0, len(fare_ml_data), chunk_size)):
            chunk_data = fare_ml_data.iloc[chunk:chunk + chunk_size]
            chunk_path = os.path.join(emr_dir, f'fare_prediction_chunk_{i:03d}.csv')
            chunk_data.to_csv(chunk_path, index=False)

            # Upload chunk to S3
            if s3_available:
                upload_to_s3(chunk_path, f"{S3_PREFIX}/emr_ready/fare_prediction_chunks/fare_prediction_chunk_{i:03d}.csv", f"fare prediction chunk {i}")

        print(f"    Created {i+1} chunks for EMR distributed processing")

# Upload ML features to S3
if s3_available:
    print(f"\n    Uploading ML features to AWS S3:")
    upload_to_s3(ml_numerical_path, f"{S3_PREFIX}/ml_features/numerical_features.csv", "numerical features")
    if 'fare_amount' in df.columns:
        upload_to_s3(fare_ml_path, f"{S3_PREFIX}/ml_features/fare_prediction_data.csv", "fare prediction dataset")
