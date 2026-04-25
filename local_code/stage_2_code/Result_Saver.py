from local_code.base_class.result import result
import pickle


class Result_Saver(result):
    data = None
    result_destination_folder_path = None
    result_destination_file_name = None

    def save(self):
        print('saving results...')
        path = self.result_destination_folder_path + self.result_destination_file_name
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f'Results saved to {path}')

    def load(self):
        path = self.result_destination_folder_path + self.result_destination_file_name
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
        return self.data
