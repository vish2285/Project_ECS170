from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


class Evaluate_Metrics(evaluate):
    data = None

    def __init__(self, eName=None, eDescription=None):
        super().__init__(eName, eDescription)

    def evaluate(self):
        print('evaluating performance...')
        true_y = self.data['true_y']
        pred_y = self.data['pred_y']

        acc = accuracy_score(true_y, pred_y)

        f1_weighted = f1_score(true_y, pred_y, average='weighted')
        f1_macro = f1_score(true_y, pred_y, average='macro')
        f1_micro = f1_score(true_y, pred_y, average='micro')

        recall_weighted = recall_score(true_y, pred_y, average='weighted')
        recall_macro = recall_score(true_y, pred_y, average='macro')
        recall_micro = recall_score(true_y, pred_y, average='micro')

        precision_weighted = precision_score(true_y, pred_y, average='weighted')
        precision_macro = precision_score(true_y, pred_y, average='macro')
        precision_micro = precision_score(true_y, pred_y, average='micro')

        print(f'Accuracy:           {acc:.4f}')
        print(f'F1       weighted={f1_weighted:.4f}  macro={f1_macro:.4f}  micro={f1_micro:.4f}')
        print(f'Recall   weighted={recall_weighted:.4f}  macro={recall_macro:.4f}  micro={recall_micro:.4f}')
        print(f'Precision weighted={precision_weighted:.4f}  macro={precision_macro:.4f}  micro={precision_micro:.4f}')

        return {
            'accuracy': acc,
            'f1_weighted': f1_weighted, 'f1_macro': f1_macro, 'f1_micro': f1_micro,
            'recall_weighted': recall_weighted, 'recall_macro': recall_macro, 'recall_micro': recall_micro,
            'precision_weighted': precision_weighted, 'precision_macro': precision_macro, 'precision_micro': precision_micro,
        }
