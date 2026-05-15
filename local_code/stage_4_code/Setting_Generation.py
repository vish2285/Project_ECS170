from local_code.base_class.setting import setting


class Setting_Generation(setting):

    def __init__(self, sName=None, sDescription=None):
        super().__init__(sName, sDescription)

    def prepare(self, sDataset, sMethod, sResult):
        self.dataset = sDataset
        self.method = sMethod
        self.result = sResult

    def print_setup_summary(self):
        print(
            'dataset:', self.dataset.dataset_name,
            '| method:', self.method.method_name,
            '| setting:', self.setting_name,
        )

    def load_run_save_evaluate(self):
        data = self.dataset.load()

        self.method.vocab_size = data['vocab_size']
        self.method.data = data

        result_data = self.method.run()

        self.result.data = result_data
        self.result.save()

        return result_data
