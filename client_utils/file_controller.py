import json
import os
import pandas as pd

class FileController:
    """
    This class is designed to handle client features.
    """

    def get_file(self,file_name, type = "json"):
        """
        Return the last dictionary of client features from the JSON file.
        If the file doesn't exist, create a new file with an empty dictionary and return it.
        """
        if type == "json":
            file_path = os.path.join('./output', f"{file_name}.json")
            self._check_file_availability(file_path)
            with open(file_path, 'r') as file:
                # Read the file line by line in reverse order
                lines = file.readlines()
                for line in reversed(lines):
                    try:
                        # Try to parse JSON from the current line
                        last_features = json.loads(line.strip())
                        break  # Stop parsing when the last valid JSON object is found
                    except json.JSONDecodeError:
                        # Ignore lines that are not valid JSON
                        pass
            return last_features
        else:
            file_path = os.path.join('./output', f"{file_name}.csv")
            df = pd.read_csv(file_path)
            return df

    def save_file(self, data, file_name, type="json"):
        """
        Take dict of features as an input and append it to the JSON file.
        """
        if type == "json":
            file_path = os.path.join('./output', f"{file_name}.json")
            self._check_file_availability(file_path)
            with open(file_path, 'w') as file:
                file.write('\n')  # Ensure each dict is written on a new line
                json.dump(data, file)
        else:
            file_path = os.path.join('./output', f"{file_name}.csv")
            data.to_csv(file_path)

    def _check_file_availability(self,file_path):
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as file:
                json.dump({}, file)

    def save_file_append(self, data, file_name, type="json"):
        # output_dir = './output'
        # os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

        # file_path = os.path.join(output_dir, f"{file_name}.{type}")

        if type == "json":
            file_path = os.path.join('./output', f"{file_name}.json")
            self._check_file_availability(file_path)
            with open(file_path, 'a') as file:
                if os.path.getsize(file_path) > 0:
                    file.write('\n')  # Ensure each dict is written on a new line
                json.dump(data.to_dict(), file)  # Convert DataFrame to dictionary before dumping
                file.write('\n')  # Add a newline after each JSON object
        else:
            file_path = os.path.join('./output', f"{file_name}.csv")
            if os.path.exists(file_path):
                # Append to existing CSV file
                data.to_csv(file_path, mode='a', index=False, header=False)
            else:
                # Write new CSV file
                data.to_csv(file_path, index=False)

    def check_record_exists(self, dataset_name, num_clients, algorithm, prameters, file_name):
        file_path = os.path.join('./output',  f"{file_name}.csv")
        
        if os.path.exists(file_path):
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Construct the record id to check
            record_id = f"{dataset_name}-{num_clients}"
            
            # Check if there exists any record with matching dataset_name, num_clients, and algorithm
            exists = ((df['id'] == record_id) & (df['model'] == algorithm) & (df['hyperparameters'] == prameters)).any()
            
            return exists
        
        return False             


