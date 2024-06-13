

class TrainTestSplit:
    def __init__(self,data,train_freq=0.8,target_column = "target",selected_features = []):
        self.data = data
        self.train_freq = train_freq
        self.target_column = target_column
        if len(selected_features) == 0:
            selected_features = [column for column in self.data.columns if column!=self.target_column]
        self.selected_features = selected_features
    def train_test_split(self, test = False):
        X = self.data.drop(self.target_column, axis = 1)
        y = self.data[self.target_column]  # Replace target_column_name with the actual name of your target column
        train_index = int(len(X) * self.train_freq / 1)
        X = X[self.selected_features]
        X = X.ffill()
        # for one step forecasting
        X = X.iloc[:len(X)-1]
        y = y.iloc[1:]
        if test:
            X_train, y_train = X.iloc[:train_index], y.iloc[:train_index]
            X_test, y_test = X.iloc[train_index:], y.iloc[train_index:]
            return X_train,y_train,X_test,y_test
        else:
            return X,y
    def time_series_split(self):
        train_index = int(len(self.data) * self.train_freq / 1)
        train_data, test_data = self.data.iloc[:train_index], self.data.iloc[train_index:]
        return train_data, test_data
