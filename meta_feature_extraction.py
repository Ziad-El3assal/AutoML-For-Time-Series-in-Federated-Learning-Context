import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def get_num_features(df):
    """
    Get the number of features in the dataset.
    If there is a 'Timestamp' column, it is not counted as a feature.
    """
    return len(df.columns) - 1 if 'Timestamp' in df.columns else len(df.columns)

def get_num_instances(df):
    """ Get the number of instances (rows) in the dataset. """
    return len(df)

def get_dataset_ratio(df):
    """ Get the ratio of the number of instances to the number of features. """
    return get_num_instances(df) / get_num_features(df)

def get_numerical_to_categorical_ratio(df):
    """ 
    Get the ratio of numerical features to categorical features.
    Returns infinity if there are no categorical features.
    """
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object', 'bool']).columns
    return len(numerical_features) / len(categorical_features) if len(categorical_features) > 0 else float("inf")

def get_num_missing_vals(df):
    """ Get the total number of missing values in the dataset. """
    return df.isnull().sum().sum()

def get_avg_missing_vals_per_feature(df):
    """ Get the average number of missing values per feature. """
    return get_num_missing_vals(df) / get_num_features(df)

def get_percentage_of_outliers(df):
    """
    Get the percentage of outliers in the dataset based on the IQR method.
    """
    numerical_data = df.select_dtypes(include=['int64', 'float64'])
    Q1 = numerical_data.quantile(0.25)
    Q3 = numerical_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = numerical_data[(numerical_data < lower_bound) | (numerical_data > upper_bound)].any(axis=1)
    percentage_of_outliers = outliers.sum() / get_num_instances(df)
    return percentage_of_outliers

def get_skewness_features(df):
    """
    Get the skewness of numerical features.
    Returns a series of skewness values and their mean, min, max, and standard deviation.
    """
    skewness = df.skew(axis=0, numeric_only=True).abs()
    return skewness, skewness.mean(), skewness.min(), skewness.max(), skewness.std()

def get_kurtosis_features(df):
    """
    Get the kurtosis of numerical features.
    Returns a series of kurtosis values and their mean, min, max, and standard deviation.
    """
    kurtosis = df.kurtosis(axis=0, numeric_only=True).abs()
    return kurtosis, kurtosis.mean(), kurtosis.min(), kurtosis.max(), kurtosis.std()

def get_sum_symbols(df):
    """ Get the sum of unique symbols in categorical features. """
    return df.select_dtypes(include=['object', 'bool']).nunique().sum()

def get_avg_symbols(df):
    """ Get the average number of unique symbols in categorical features. """
    return df.select_dtypes(include=['object', 'bool']).nunique().mean()

def get_std_symbols(df):
    """ Get the standard deviation of unique symbols in categorical features. """
    return df.select_dtypes(include=['object', 'bool']).nunique().std()



data_path=r"D:\AI ITI\Giza Graduation project\02_Datasets\02_Datasets\Regression\Multivariate\Real\1005.csv"
df=pd.read_csv(data_path)
num_features=get_num_features(df)
num_instances=get_num_instances(df)

print(f"""
      num_features={num_features},
      num_instances = {num_instances}
      dataset_ratio= {get_dataset_ratio(df)},
      ratio of numerical features to categorical features= {get_numerical_to_categorical_ratio(df)} ,
      num_missing_value= {get_num_missing_vals(df)}
      avg_missing_vals_per_feature= {get_avg_missing_vals_per_feature(df)}
      percentage_of_outliers= {get_percentage_of_outliers(df)}
      skewness_features= {get_skewness_features(df)}
      kurtosis_features= {get_kurtosis_features(df)}
      sum_symbols= {get_sum_symbols(df)} 
      avg_symbols= {get_avg_symbols(df)}
      std_symbols= {get_std_symbols(df)} """)         
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
