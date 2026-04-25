from local_code.base_class.setting import setting


class Setting_Train_Test(setting):
    # holds two Dataset_Loader objects instead of one
    train_dataset = None
    test_dataset = None

    def __init__(self, sName=None, sDescription=None):
        super().__init__(sName, sDescription)

    def prepare(self, train_dataset, test_dataset, sMethod, sResult, sEvaluate):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.method = sMethod
        self.result = sResult
        self.evaluate = sEvaluate

    def print_setup_summary(self):
        print(
            'dataset (train):', self.train_dataset.dataset_name,
            '| dataset (test):', self.test_dataset.dataset_name,
            '| method:', self.method.method_name,
            '| setting:', self.setting_name,
        )

    def load_run_save_evaluate(self):
        print('loading train data...')
        train_data = self.train_dataset.load()
        print('loading test data...')
        test_data = self.test_dataset.load()

        self.method.data = {
            'train': {'X': train_data['X'], 'y': train_data['y']},
            'test':  {'X': test_data['X'],  'y': test_data['y']},
        }

        learned_result = self.method.run()

        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result
        return self.evaluate.evaluate()
