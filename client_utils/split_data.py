import pandas as pd
class SplitData:
    def __init__(self, data, train_freq=0.8, target_column="Target", selected_features=[]):
        self.data = data
        self.train_freq = train_freq
        self.target_column = target_column
        if len(selected_features) == 0:
            selected_features = [column for column in self.data.columns if column != self.target_column]
        self.selected_features = selected_features

    def x_y_split(self):
        X = self.data.drop(self.target_column, axis=1)
        y = self.data[self.target_column]  # Replace target_column_name with the actual name of your target column
        train_index = int(len(X) * self.train_freq / 1)
        X = X[self.selected_features]
        X = X.ffill()
        # for one step forecasting
        X = X.iloc[:len(X) - 1]
        y = y.iloc[1:]
        return X, y

    def train_test_split(self):
        self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'])
        self.data = self.data.sort_values(by='Timestamp')
        train_index = int(len(self.data) * self.train_freq / 1)
        train_data, test_data = self.data.iloc[:train_index], self.data.iloc[train_index:]
        return train_data, test_data
