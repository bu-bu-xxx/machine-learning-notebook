import numpy as np
import pandas as pd

class Metrics:
    @staticmethod
    def confusion_matrix(y_true, y_pred)->pd.DataFrame:
        classes = np.sort(np.unique(y_true))
        classes = classes.reshape(-1, 1) # unique classes x 1
        y_true = np.array(y_true) 
        y_pred = np.array(y_pred)
        y_true = y_true.reshape(1, -1) # 1 x m
        y_pred = y_pred.reshape(1, -1) # 1 x m

        confusion_mat = (y_true == classes).astype(int) \
            .dot((y_pred == classes).astype(int).T) # unique classes x unique classes
        confusion_mat = pd.DataFrame(confusion_mat, columns=classes.flatten(), index=classes.flatten())
        return confusion_mat
    
    @staticmethod
    def accuracy(y_true, y_pred):
        confusion_mat = Metrics.confusion_matrix(y_true, y_pred).values
        return np.trace(confusion_mat) / np.sum(confusion_mat)
    
    @ staticmethod
    def scores(y_true, y_pred, type, score):
        confusion_mat = Metrics.confusion_matrix(y_true, y_pred).values
        # tp[class, :] = [tp, fp, fn, tn]
        tp = np.zeros((confusion_mat.shape[0], 4))
        for i in range(confusion_mat.shape[0]):
            tp[i, 0] = confusion_mat[i, i]
            tp[i, 1] = np.sum(confusion_mat[:, i]) - tp[i, 0]
            tp[i, 2] = np.sum(confusion_mat[i, :]) - tp[i, 0]
            tp[i, 3] = np.sum(confusion_mat) - tp[i, 0] - tp[i, 1] - tp[i, 2]
        if type == 'micro':
            micro_p = np.sum(tp[:, 0]) / (np.sum(tp[:, 0]) + np.sum(tp[:, 1]))
            micro_r = np.sum(tp[:, 0]) / (np.sum(tp[:, 0]) + np.sum(tp[:, 2]))
            if score == 'precision':
                return micro_p
            elif score =='recall':
                return micro_r
            elif score == 'f1':
                return 2 * micro_p * micro_r / (micro_p + micro_r)
        elif type == 'macro':
            macro_p = np.mean(tp[:, 0]/(tp[:, 0] + tp[:, 1]))
            macro_r = np.mean(tp[:, 0]/(tp[:, 0] + tp[:, 2]))
            if score == 'precision':
                return macro_p
            elif score =='recall':
                return macro_r
            elif score == 'f1':
                return 2 * macro_p * macro_r / (macro_p + macro_r)
        raise ValueError("Invalid type or score")
    
    @staticmethod
    def precision(y_true, y_pred, type='binary'):
        if type == 'binary':
            if np.unique(y_pred).shape[0] > 2:
                raise ValueError("y_pred should only have 2 unique values for binary classes")
            confusion_mat = Metrics.confusion_matrix(y_true, y_pred).values
            return confusion_mat[1, 1]/(confusion_mat[1, 1] + confusion_mat[0, 1])
        elif type == 'micro':
            return Metrics.scores(y_true, y_pred, type, 'precision')
        elif type =='macro':
            return Metrics.scores(y_true, y_pred, type, 'precision')
        else:
            raise ValueError("Invalid type")
    
    @staticmethod
    def recall(y_true, y_pred, type='binary'):
        if type == 'binary':
            if np.unique(y_pred).shape[0] > 2:
                raise ValueError("y_pred should only have 2 unique values for binary classes")
            confusion_mat = Metrics.confusion_matrix(y_true, y_pred).values
            return confusion_mat[1, 1]/(confusion_mat[1, 1] + confusion_mat[1, 0])
        elif type == 'micro':
            return Metrics.scores(y_true, y_pred, type,'recall')
        elif type =='macro':
            return Metrics.scores(y_true, y_pred, type,'recall')
        else:
            raise ValueError("Invalid type")
    
    @staticmethod
    def f1(y_true, y_pred, type='binary'):
        if type == 'binary':
            if np.unique(y_pred).shape[0] > 2:
                raise ValueError("y_pred should only have 2 unique values for binary classes")
            confusion_mat = Metrics.confusion_matrix(y_true, y_pred).values
            return 2 * confusion_mat[1, 1]/(2 * confusion_mat[1, 1] + confusion_mat[1, 0] + confusion_mat[0, 1])
        elif type == 'micro':
            return Metrics.scores(y_true, y_pred, type, 'f1')
        elif type =='macro':
            return Metrics.scores(y_true, y_pred, type, 'f1')
        else:
            raise ValueError("Invalid type")
            
