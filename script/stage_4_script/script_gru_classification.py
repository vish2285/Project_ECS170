import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import torch

from local_code.stage_4_code.Dataset_Loader_Classification import Dataset_Loader_Classification
from local_code.stage_4_code.Method_RNN_Classification import Method_RNN_Classification
from local_code.stage_4_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_4_code.Result_Saver import Result_Saver
from local_code.stage_4_code.Setting_Classification import Setting_Classification

np.random.seed(42)
torch.manual_seed(42)

DATA_PATH = '../../data/stage_4_data/text_classification/'
RESULT_PATH = '../../result/stage_4_result/'

dataset_obj = Dataset_Loader_Classification('IMDb', '')
dataset_obj.dataset_source_folder_path = DATA_PATH

method_obj = Method_RNN_Classification('GRU-Classification', '')
method_obj.rnn_type = 'GRU'
method_obj.max_epoch = 10
method_obj.learning_rate = 1e-3
method_obj.batch_size = 64
method_obj.result_destination_folder_path = RESULT_PATH

result_obj = Result_Saver('saver', '')
result_obj.result_destination_folder_path = RESULT_PATH
result_obj.result_destination_file_name = 'GRU_classification_prediction_result'

evaluate_obj = Evaluate_Metrics('metrics', '')

setting_obj = Setting_Classification('train test', '')
setting_obj.prepare(dataset_obj, method_obj, result_obj, evaluate_obj)
setting_obj.print_setup_summary()

print('************ Start ************')
metrics = setting_obj.load_run_save_evaluate()
print('************ Final Results ************')
for k, v in metrics.items():
    print(f'  {k}: {v:.4f}')
print('************ Finish ************')
