import pandas as pd
from math import log
from sklearn.preprocessing import LabelEncoder


class ReadPreprocessData:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = pd.read_csv(self.data_path)
        self.data.rename(columns={'timestamp': 'Timestamp','value':'Target'}, inplace=True)
        print(self.data.columns)
        self.columns_types = {'categorical': [], 'numerical': []}
        self.categorical_columns = None
        self.numerical_columns = None
        self.dataset_type = None

    def preprocess_data(self):
        self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'])
        self.data = self.data.sort_values(by='Timestamp')
        self.detect_columns_types()
        self.detect_dataset_type()
        self.categorical_columns = self.columns_types['categorical']
        self.numerical_columns = self.columns_types['numerical']
        self.encode_categorical()
        self.fill_missing()
        return self.data, self.columns_types, self.dataset_type

    def encode_categorical(self):
        le = LabelEncoder()

        # Encode each categorical column
        for col in self.categorical_columns:
            self.data[col] = le.fit_transform(self.data[col])

    def fill_missing(self):

        for col in self.categorical_columns:
            self.data[col] = self.data[col].ffill()

        # Linear interpolation for numerical columns
        for col in self.numerical_columns + [self.columns_types["target"]]:
            self.data[col] = self.data[col].interpolate(method='linear')
        self.data.dropna(inplace=True,axis=0)
    def detect_columns_types(self):
        num_samples = len(self.data)
        log_num_samples = log(num_samples)

        for column in self.data.columns:
            if column == 'Target':
                self.columns_types['target'] = column
            elif column == 'Timestamp':
                self.columns_types['timestamp'] = column
            else:
                unique_values = self.data[column].nunique()
                if unique_values < log_num_samples or isinstance(self.data[column], pd.CategoricalDtype):
                    self.columns_types['categorical'].append(column)
                else:
                    self.columns_types['numerical'].append(column)

    def detect_dataset_type(self):
        if self.columns_types['categorical'] or self.columns_types['numerical']:
            self.dataset_type = "multivariate"
        else:
            self.dataset_type = "univariate"