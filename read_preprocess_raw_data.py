import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_type = None
        self.data = None

    def load_check_data(self):
        self.data = pd.read_csv(self.data_path)
        
        # Check if Univariate or Multivariate
        if self.data.shape[1] <= 2:
            self.data_type = "Univariate"
        else: 
            self.data_type = "Multivariate"
        return self.data_type, self.data

    def general_processing(self):
        # 1: Check the Target name
        target_keywords = ['Close', 'close', 'value', 'Value']
        for col in self.data.columns:
            if any(keyword in col for keyword in target_keywords):
                self.data.rename(columns={col: 'Target'}, inplace=True)
                break

        # 2: Check the Timestamp name
        timestamp_keywords = ['timestamp', 'Timestamp']
        for col in self.data.columns:
            if any(keyword in col for keyword in timestamp_keywords):
                self.data.rename(columns={col: 'Timestamp'}, inplace=True)
                break

        # 3: Convert Timestamp to datetime
        if 'Timestamp' in self.data.columns:
            self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'])
            # 4: Sort by datetime
            self.data = self.data.sort_values(by='Timestamp')
        """
        The choice between the two methods: {
            1- Encoding First, Then Forward Fill
            2- Apply Linear Interpolation and Forward Fill Before Encoding }
        depends on the nature of your data and the assumptions you can make about the missing values. 
        For numerical data, performing interpolation before encoding ensures more accurate imputation. 
        For categorical data, handling missing values appropriately before encoding can prevent bias 
        and maintain data integrity
        """
        # 5: Handle Missing values
        if self.data_type == "Univariate":
            self.data = self.data.interpolate(method='linear')
        else:
            # Apply linear interpolation for numerical columns
            num_cols = self.data.select_dtypes(include=['number']).columns
            self.data[num_cols] = self.data[num_cols].interpolate(method='linear')
            
            # Forward fill for categorical columns
            cat_cols = self.data.select_dtypes(include=['object']).columns
            self.data[cat_cols] = self.data[cat_cols].ffill()
        
        # 6: Label Encoding for the categorical data
        categorical_features = self.data.select_dtypes(include='object').columns
        for col in categorical_features:
            label_encoder = LabelEncoder()
            self.data[col] = label_encoder.fit_transform(self.data[col])

        return self.data

# Example usage
# data_path = r"F:\ITI GP - Giza Systems\First Email\02_Task_Description\02_Datasets\Regression\Univariate\Real\571.csv"
data_path = r'F:\ITI GP - Giza Systems\First Email\02_Task_Description\02_Datasets\Regression\Multivariate\Real\1005.csv'

processor = DataProcessor(data_path)
data_type, data = processor.load_check_data()
preprocessed_data = processor.general_processing()

# To check
print(preprocessed_data.isnull().sum())
print(preprocessed_data.head())
print(preprocessed_data['Timestamp'].dtype)
print(preprocessed_data['Seasons'].unique())
