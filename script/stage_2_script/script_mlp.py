import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import torch

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_Train_Test import Setting_Train_Test

np.random.seed(42)
torch.manual_seed(42)

# ---- data ----
train_data = Dataset_Loader('MNIST-train', '')
train_data.dataset_source_folder_path = '../../data/stage_2_data/'
train_data.dataset_source_file_name = 'train.csv'

test_data = Dataset_Loader('MNIST-test', '')
test_data.dataset_source_folder_path = '../../data/stage_2_data/'
test_data.dataset_source_file_name = 'test.csv'

# ---- method ----
method_obj = Method_MLP('MLP', '')
method_obj.max_epoch = 100
method_obj.learning_rate = 1e-3
method_obj.batch_size = 256
method_obj.result_destination_folder_path = '../../result/stage_2_result/'

# ---- result ----
result_obj = Result_Saver('saver', '')
result_obj.result_destination_folder_path = '../../result/stage_2_result/'
result_obj.result_destination_file_name = 'MLP_prediction_result'

# ---- evaluate ----
evaluate_obj = Evaluate_Metrics('metrics', '')

# ---- setting ----
setting_obj = Setting_Train_Test('train test', '')
setting_obj.prepare(train_data, test_data, method_obj, result_obj, evaluate_obj)
setting_obj.print_setup_summary()

print('************ Start ************')
metrics = setting_obj.load_run_save_evaluate()
print('************ Final Results ************')
for k, v in metrics.items():
    print(f'  {k}: {v:.4f}')
print('************ Finish ************')
