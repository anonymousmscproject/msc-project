import numpy as np 
import torch

# corresponds to CHAPTER 3.1

class fairness_scores:
    def __init__(self, data, labels, preds, group, n=-1.0):
        self.data = data # feautes of the data
        self.labels = labels # true lables
        self.preds = preds # model label predictions
        self.group = group # column in the data set containing grouup information
        self.n = n # negative outcome, usually 0 or -1.0. default -1.0

    def equal_opportunity(self):
        ''' Relaxed version of equalized odds. Asks that the rate of true positives (rather than true positive rates) is the same'''
        
        g_vals = list(set(self.data[:,self.group]))
        tp = [] 
        
        for i in g_vals:

            idx = np.where(self.data[:, self.group] == i)
            group_preds = self.preds[idx]
            group_labels = self.labels[idx]

            arr = []
            for i in range(len(list(group_labels))):
                if group_labels[i]==1.0 and group_preds[i]==1.0:
                    arr.append(1)
                else:
                    arr.append(0)

            arr2 = []
            for i in range(len(list(group_labels))):
                if group_labels[i]==1.0:
                    arr2.append(1)
                else:
                    arr2.append(0)
                
            tp_group = np.sum(arr) / np.sum(arr2)

            tp.append(tp_group)

        tp = np.array(tp)

        eo_diff = abs(tp[0] - tp[1])

        return eo_diff




    def equal_odds(self):
        ''' Requires that the true positive rates (tp/tp+fn) are the same'''

        g_vals = list(set(self.data[:,self.group]))
        tp = [] 
        fn = []

        for i in g_vals:
            idx = np.where(self.data[:, self.group] == i)
            group_preds = self.preds[idx]
            group_labels = self.labels[idx]
            
            arr = []
            for j in range(len(list(group_labels))):
                if group_labels[j]==1.0 and group_preds[j]==1.0:
                    arr.append(1)
                else:
                    arr.append(0)

            arr_t = []
            for i in range(len(list(group_labels))):
                if group_labels[i]==1.0:
                    arr_t.append(1)
                else:
                    arr_t.append(0)
                
            tp_group = np.sum(arr) / np.sum(arr_t)

            tp.append(tp_group)

            arr2 = []
            for j in range(len(list(group_labels))):
                if group_labels[j]==1.0 and group_preds[j]==self.n:
                    arr2.append(1)
                else:
                    arr2.append(0)

            arr2_t = []
            for i in range(len(list(group_labels))):
                if group_labels[i]==self.n:
                    arr2_t.append(1)
                else:
                    arr2_t.append(0)
                
            fn_group = np.sum(arr2) / np.sum(arr2_t)

            fn.append(fn_group)

        tpr_a = tp[0] / (tp[0] + fn[0])
        tpr_b = tp[1] / (tp[1] + fn[1])

        eo_diff = abs(tpr_a - tpr_b)

        return eo_diff


    def dem_parity(self):

        ''' requires positive outcomes be the same regardless of true/false status '''

        # this version returns tensor

        g_vals = list(set(self.data[:,self.group]))
        p = [] 
        
        for i in g_vals:
            idx = np.where(self.data[:, self.group] == i)
            group_preds = self.preds[idx]
            group_labels = self.labels[idx]
            
            arr = []
            for i in range(len(list(group_labels))):
                if group_preds[i]==1.0:
                    arr.append(1)
                else:
                    arr.append(0)

            p_group = np.sum(arr) / len(list(group_labels))

            p.append(p_group)

        p = np.array(p)
        p = torch.Tensor(p)

        dp = torch.abs(p[0] - p[1])

        return dp


    def dem_parity2(self):

        ''' requires positive outcomes be the same regardless of true/false status '''

        # this version returns array

        g_vals = list(set(self.data[:,self.group]))
        p = [] 
        
        for i in g_vals:
            idx = np.where(self.data[:, self.group] == i)
            group_preds = self.preds[idx]
            group_labels = self.labels[idx]
            
            arr = []
            for i in range(len(list(group_labels))):
                if group_preds[i]==1.0:
                    arr.append(1)
                else:
                    arr.append(0)

            p_group = np.sum(arr) / len(list(group_labels))

            p.append(p_group)

        p = np.array(p)
    
        dp = abs(p[0] - p[1])

        return dp



    def treatment_equality(self):

        ''' requires that the ratio of false positives and false negatives are equal. emphasis on disparate treatment '''
        g_vals = list(set(self.data[:,self.group]))
        fp = [] 
        fn = []

        for i in g_vals:
            idx = np.where(self.data[:, self.group] == i)
            group_preds = self.preds[idx]
            group_labels = self.labels[idx]
            
            arr = []
            for j in range(len(list(group_labels))):
                if group_labels[j]==self.n and group_preds[j]==1.0:
                    arr.append(1)
                else:
                    arr.append(0)
                
            fp_group = np.sum(arr) / len(group_labels)

            fp.append(fp_group)
    
            arr2 = []
            for j in range(len(list(group_labels))):
                if group_labels[j]==1.0 and group_preds[j]==self.n:
                    arr2.append(1)
                else:
                    arr2.append(0)
                
            fn_group = np.sum(arr2) / len(group_labels)

            fn.append(fn_group)
 
        ratio_a = fp[0] / fn[0]

        ratio_b = fp[1] / fn[1]


        diff = abs(ratio_a - ratio_b)

        if diff > 1.0: 
            diff = 1 / diff
  
        return diff


    def get_scores(self):

        # works better with meta-learning alg, returns tensor as opposed to array

        dp = self.dem_parity()

        model_scores = dp

        return model_scores

    def get_scores2(self):

        # returns array

        eo = self.equal_opportunity()
        eodds = self.equal_odds()
        dp = self.dem_parity2()
        te = self.treatment_equality()

        model_scores = [eo, eodds, dp, te]

        return model_scores