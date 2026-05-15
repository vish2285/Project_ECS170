import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import torch

from local_code.stage_4_code.Dataset_Loader_Generation import Dataset_Loader_Generation
from local_code.stage_4_code.Method_RNN_Generation import Method_RNN_Generation
from local_code.stage_4_code.Result_Saver import Result_Saver
from local_code.stage_4_code.Setting_Generation import Setting_Generation

np.random.seed(42)
torch.manual_seed(42)

DATA_PATH = '../../data/stage_4_data/text_generation/'
RESULT_PATH = '../../result/stage_4_result/'

dataset_obj = Dataset_Loader_Generation('Jokes', '')
dataset_obj.dataset_source_folder_path = DATA_PATH
dataset_obj.dataset_source_file_name = 'data'
dataset_obj.seq_len = 100

method_obj = Method_RNN_Generation('GRU-Generation', '')
method_obj.rnn_type = 'GRU'
method_obj.max_epoch = 50
method_obj.learning_rate = 1e-3
method_obj.batch_size = 128
method_obj.generate_length = 400
method_obj.temperature = 0.8
method_obj.start_text = 'why did the'
method_obj.result_destination_folder_path = RESULT_PATH

result_obj = Result_Saver('saver', '')
result_obj.result_destination_folder_path = RESULT_PATH
result_obj.result_destination_file_name = 'GRU_generation_result'

setting_obj = Setting_Generation('generation', '')
setting_obj.prepare(dataset_obj, method_obj, result_obj)
setting_obj.print_setup_summary()

print('************ Start ************')
setting_obj.load_run_save_evaluate()
print('************ Finish ************')
