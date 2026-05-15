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
        f1_w = f1_score(true_y, pred_y, average='weighted')
        f1_macro = f1_score(true_y, pred_y, average='macro')
        f1_micro = f1_score(true_y, pred_y, average='micro')
        rec_w = recall_score(true_y, pred_y, average='weighted')
        rec_macro = recall_score(true_y, pred_y, average='macro')
        rec_micro = recall_score(true_y, pred_y, average='micro')
        prec_w = precision_score(true_y, pred_y, average='weighted')
        prec_macro = precision_score(true_y, pred_y, average='macro')
        prec_micro = precision_score(true_y, pred_y, average='micro')

        print(f'Accuracy:            {acc:.4f}')
        print(f'F1        weighted={f1_w:.4f}  macro={f1_macro:.4f}  micro={f1_micro:.4f}')
        print(f'Recall    weighted={rec_w:.4f}  macro={rec_macro:.4f}  micro={rec_micro:.4f}')
        print(f'Precision weighted={prec_w:.4f}  macro={prec_macro:.4f}  micro={prec_micro:.4f}')

        return {
            'accuracy': acc,
            'f1_weighted': f1_w, 'f1_macro': f1_macro, 'f1_micro': f1_micro,
            'recall_weighted': rec_w, 'recall_macro': rec_macro, 'recall_micro': rec_micro,
            'precision_weighted': prec_w, 'precision_macro': prec_macro, 'precision_micro': prec_micro,
        }
