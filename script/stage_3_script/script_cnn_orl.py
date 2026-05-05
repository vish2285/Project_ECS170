import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import torch

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN import Method_CNN
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting_Train_Test import Setting_Train_Test

np.random.seed(42)
torch.manual_seed(42)

DATA_PATH = '../../data/stage_3_data/'
RESULT_PATH = '../../result/stage_3_result/'

train_data = Dataset_Loader('ORL-train', '')
train_data.dataset_source_folder_path = DATA_PATH
train_data.dataset_source_file_name = 'ORL'
train_data.dataset_type = 'ORL'
train_data.split = 'train'

test_data = Dataset_Loader('ORL-test', '')
test_data.dataset_source_folder_path = DATA_PATH
test_data.dataset_source_file_name = 'ORL'
test_data.dataset_type = 'ORL'
test_data.split = 'test'

method_obj = Method_CNN('CNN-ORL', '')
method_obj.dataset_type = 'ORL'
method_obj.max_epoch = 100
method_obj.learning_rate = 1e-3
method_obj.batch_size = 32
method_obj.result_destination_folder_path = RESULT_PATH

result_obj = Result_Saver('saver', '')
result_obj.result_destination_folder_path = RESULT_PATH
result_obj.result_destination_file_name = 'ORL_CNN_prediction_result'

evaluate_obj = Evaluate_Metrics('metrics', '')

setting_obj = Setting_Train_Test('train test', '')
setting_obj.prepare(train_data, test_data, method_obj, result_obj, evaluate_obj)
setting_obj.print_setup_summary()

print('************ Start ************')
metrics = setting_obj.load_run_save_evaluate()
print('************ Final Results ************')
for k, v in metrics.items():
    print(f'  {k}: {v:.4f}')
print('************ Finish ************')
