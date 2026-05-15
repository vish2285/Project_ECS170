from local_code.base_class.setting import setting


class Setting_Classification(setting):

    def __init__(self, sName=None, sDescription=None):
        super().__init__(sName, sDescription)

    def prepare(self, sDataset, sMethod, sResult, sEvaluate):
        self.dataset = sDataset
        self.method = sMethod
        self.result = sResult
        self.evaluate = sEvaluate

    def print_setup_summary(self):
        print(
            'dataset:', self.dataset.dataset_name,
            '| method:', self.method.method_name,
            '| setting:', self.setting_name,
        )

    def load_run_save_evaluate(self):
        data = self.dataset.load()

        self.method.vocab_size = data['vocab_size']
        self.method.data = {
            'train': {'X': data['X_train'], 'y': data['y_train']},
            'test':  {'X': data['X_test'],  'y': data['y_test']},
        }

        learned_result = self.method.run()

        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result
        return self.evaluate.evaluate()
