import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator 
import cvxopt
import cvxopt.solvers
from cvxopt import matrix
from collections import namedtuple
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.kernel_approximation import Nystroem

# CHAPTER 3.2

def FERM_plots(a, b, c, d, e, fig_name, test=True):
    ''' plotting scores and algorithms '''
    plt.figure()
    plt.rcParams['figure.dpi'] = 300
    
    plt.plot(a[1], a[0], 'c*', a[2], a[0], 'co', a[3], a[0], 'c^',
             b[1], b[0], 'b*', b[2], b[0], 'bo', b[3], b[0], 'b^', 
             c[1], c[0], 'm*', c[2], c[0], 'mo', c[3], c[0], 'm^', 
             d[1], d[0], 'r*', d[2], d[0], 'ro', d[3], d[0], 'r^',
             e[1], e[0], 'g*', e[2], e[0], 'go', e[3], e[0], 'g^')

    legend_elements = [Line2D([0], [0], marker='s', color='w', label='SVM standard', 
                              markerfacecolor='c', markersize=10),
                       Line2D([0], [0], marker='s', color='w', label='SVM with linear ferm', 
                              markerfacecolor='b', markersize=10), 
                       Line2D([0], [0], marker='s', color='w', label='RFC standard', 
                              markerfacecolor='m', markersize=10),
                       Line2D([0], [0], marker='s', color='w', label='RFC with linear ferm',
                              markerfacecolor='r', markersize=10), 
                       Line2D([0], [0], marker='s', color='w', label='Non-linear FERM', 
                              markerfacecolor='g', markersize=10),
                       Line2D([0], [0], marker='*', color='w', label='Equality of opportunity',
                              markerfacecolor='k', markersize=12), 
                       Line2D([0], [0], marker='o', color='w', label='Equalized odds',
                              markerfacecolor='k', markersize=8), 
                       Line2D([0], [0], marker='^', color='w', label='Demographic parity',
                              markerfacecolor='k', markersize=9), 
                       Line2D([0], [0], color='k', lw=1, ls='--',label='Change in score')
                       ]
    
    plt.plot([a[1],b[1]],[a[0],b[0]],'k--',
             [a[2],b[2]],[a[0],b[0]],'k--', 
             [a[3],b[3]],[a[0],b[0]],'k--', 
             [c[1],d[1]],[c[0],d[0]],'k--',
             [c[2],d[2]],[c[0],d[0]],'k--', 
             [c[3],d[3]],[c[0],d[0]],'k--')

    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=8)
    plt.xlim(-0.01,0.3)
    plt.grid()
    plt.ylim(0.5,1.05)
    plt.xlabel('Fairness score')
    plt.ylabel('Accuracy')
    if test:
        plt.title('Scores at test time')
    else:
        plt.title('Scores at train time')
    plt.axis('equal')

    columns = ('Accuracy', 'Equality of opportunity', 'Equalized odds', 'Demographic parity')
    rows = ('SVM', 'SVM with linear FERM', 'RFC', 'RFC with linear FERM', 'Non-linear FERM')
    data=[[round(a[0], 3), round(a[1], 3), round(a[2], 3), round(a[3], 3)],
          [round(b[0], 3), round(b[1], 3), round(b[2], 3), round(b[3], 3)], 
          [round(c[0], 3), round(c[1], 3), round(c[2], 3), round(c[3], 3)],
          [round(d[0], 3), round(d[1], 3), round(d[2], 3), round(d[3], 3)],
          [round(e[0], 3), round(e[1], 3), round(e[2], 3), round(e[3], 3)]]
    plt.table(cellText=data, rowLabels=rows,colLabels=columns, bbox=[0.2, -1.2, 0.8, 1.0])
    plt.savefig(fig_name, bbox_inches = "tight")

    plt.show()

########################################################


class Linear_FERM:
    def __init__(self, data, group, model=None):
        #corresponds to chapter 3.2.3
        ''' model: model used 
            data: dataset used
            group: marginalised group
        '''
        self.model = model
        self.data = data
        self.group = group
        self.group_vals = list(set(self.group))
        self.u = None
        self.i = None

    def fit(self, constraint='eo', binary_group=True):
        ''' constraint adjusted for different fairness metrics:
            'eo': equal opportunity
            'dp': demographic parity
            binary_group: True if categories in group are binary, False if have multiple categories
        '''

        if binary_group:
            if constraint=='eo':
                arr = []
                for i, x in enumerate(self.data[0]):
                    if self.data[1][i] == 1 and self.group[i] == np.max(self.group_vals):
                        arr.append(x)


                arr2 = []
                for i, x in enumerate(self.data[0]):
                    if self.data[1][i] == 1 and self.group[i] == np.min(self.group_vals):
                        arr2.append(x)
        

            elif constraint=='dp':
                arr = []
                for i, x in enumerate(self.data[0]):
                    if self.group[i] == np.max(self.group_vals):
                        arr.append(x)

                arr2 = []
                for i, x in enumerate(self.data[0]):
                    if self.group[i] == np.min(self.group_vals):
                        arr2.append(x)

            arr = np.array(arr)
            
            arr2 = np.array(arr2)
            
            u_a = np.mean(arr, 0) 
            u_b = np.mean(arr2, 0)

            self.u = -(u_a - u_b)
            self.i = np.argmax(self.u)

        else:
            means = [[]]
            for k in self.group_vals:
                arr = []
                for i, x in enumerate(self.data[0]):
                    if self.data[1][i] == 1 and self.group[i] == k:
                        arr.append(x)
                arr = np.array(arr)
                u_i = np.mean(arr)
                means.append(u_i)
            means = np.array(means)

            # need to satisfy k - 1 orthogonality constraints for non-binary groups
            # incomplete - for future work :)

        new = []
        for x in self.data[0]: # new data representation.
            upd = x - self.u * (x[self.i] / self.u[self.i])
            new.append(upd)
        new = np.array(new)
        new = np.delete(new, self.i, 1)

        self.data = namedtuple('_', 'data, label')(new, self.data[1])

        return self.data[0], self.data[1]



    def new_representation(self, data):
        data_new = []
        for x in data:
            upd = x - self.u * (x[self.i] / self.u[self.i])
            data_new.append(upd)
        data_new = np.array(data_new)
        data_new = np.delete(data_new, self.i, 1)
        return data_new

    def predict(self, data):
        prediction = self.model.predict(self.new_representation(data))
        return prediction


########################################################

class Non_Linear_FERM(BaseEstimator):
    # corresponds to chapter 3.2.2
    ''' Non-Linear FERM, similar to SVM with a fairness constraint'''

    def __init__(self, group, gamma, C, kernel='rbf'):

        self.kernel = kernel
        self.group = group
        self.gamma = gamma
        self.C = C
        self.w = None
    

    def fit(self, data, labels, constraint='eo', binary_group=True):

        self.group_vals = list(set(self.group))


        if self.kernel == 'rbf': # implements using rbf kernel 
            self.K = lambda x, y: rbf_kernel(x, y, self.gamma)
        else: raise NotImplementedError()

        if binary_group:
            if constraint=='eo':
                self.g1 = [idx for idx, ex in enumerate(data) if labels[idx] == 1
                                and self.group[idx] ==  np.max(self.group_vals)]

                self.g2 = [idx for idx, ex in enumerate(data) if labels[idx] == 1
                                and self.group[idx] == np.min(self.group_vals)]

            elif constraint=='dp':
                self.g1 = [idx for idx, ex in enumerate(data) if self.group[idx] ==  np.max(self.group_vals)]

                self.g2 = [idx for idx, ex in enumerate(data) if self.group[idx] == np.min(self.group_vals)]

            #feature_map_nystroem = Nystroem(gamma=.1, random_state=1, n_components=400)
            #data_transformed = feature_map_nystroem.fit_transform(data)

            #K = self.K(data_transformed, data_transformed)
            K = self.K(data, data)

            P = cvxopt.matrix(np.outer(labels, labels) * K)
            q = cvxopt.matrix(np.ones(data.shape[0]) * -1)
            A = cvxopt.matrix(labels.astype(np.double), (1, data.shape[0]), 'd')
            b = cvxopt.matrix(0.0)
            G = cvxopt.matrix(np.vstack((np.diag(np.ones(data.shape[0]) * -1), np.identity(data.shape[0]))))
            h = cvxopt.matrix(np.hstack((np.zeros(data.shape[0]), np.ones(data.shape[0]) * self.C)))

            fc = [(np.sum(K[self.g1, idx]) / len(self.g1)) - (np.sum(K[self.g2, idx]) / len(self.g2))
                   for idx in range(len(labels))]

        fair = matrix(labels * fc, (1, data.shape[0]), 'd')
        
        A = cvxopt.matrix(np.vstack([A, fair]))
        b = cvxopt.matrix([0.0, 0.0])
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        alpha = np.ravel(sol['x'])

        support_vectors = alpha > 1e-7
        i = np.arange(len(alpha))[support_vectors]
        self.alpha = alpha[support_vectors]
        self.support_vectors = data[support_vectors]
        self.support_vectors_labels = labels[support_vectors]

        self.b = 0
        for j in range(len(self.alpha)):
            self.b += self.support_vectors_labels[j]
            self.b -= np.sum(self.alpha * self.support_vectors_labels * K[i[j], support_vectors])
        self.b = self.b / len(self.alpha)

    def preds_acc(self, test_data, test_labels):

        if self.w is not None:
            return np.dot(test_data, self.w) + self.b
        else:
            XSV = self.K(test_data, self.support_vectors)
            pred = []
            for i in range(len(test_data)):
                p = np.sum(np.multiply(np.multiply(self.alpha, self.support_vectors_labels), XSV[i, :]))
                pred.append(p)
            pred = np.array(pred)

        preds = np.sign(pred + self.b)
        acc = accuracy_score(test_labels, preds)
        return preds, acc
