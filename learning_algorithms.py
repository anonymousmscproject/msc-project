import numpy as np
from FERM import Linear_FERM, Non_Linear_FERM
from TERM import Logistic_Regression
from fairness_metrics import fairness_scores

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier


''' Created for the purpose of having a library of algorithms (meta-distribution p_A)'''



LIBRARY_OF_LOSSES = {
    'MSE': 0,
    'CE': 1, 
    'HINGE': 2
}

LIBRARY_OF_FAIRNESS_TRANSFORMS = {
    None: 0,
    'DEO': 1,
    'DDP': 2,
}

LIBRARY_OF_NON_LINEAR_METHODS = {
    None: 0,
    'FERM': 1,
    'BOOSTING': 2,
}

#LIBRARY_OF_TILT_PARAM = {
#    None: 0,
#} 

class LearningAlgorithm:
  
    def __init__(self, fairness_transform=None):
        self.fairness_transform = fairness_transform
     
    def get_features(self):
        """
        :return: A feature vector that represents this algorithm
        """
        raise NotImplementedError()  

    def execute(self, data):
    
        raise NotImplementedError()

class LinearAlg(LearningAlgorithm):

    def __init__(self, loss_type='MSE', fairness_transform=None, tilt=None):
        super().__init__()
        self.loss_type = loss_type
        self.fairness_transform = fairness_transform
        self.tilt = tilt
        self.alg = 0

    def get_features(self):
        return np.array([
            0.,  # always linear
            LIBRARY_OF_LOSSES[self.loss_type],  # get the corresponding number
            LIBRARY_OF_FAIRNESS_TRANSFORMS[self.fairness_transform],  # get the corresponding number
            LIBRARY_OF_TILT_PARAM[self.tilt], 
        ])

    def execute(self, dataset, g=None):

        data_ = dataset[0]
        labels_ = dataset[1]

        if self.fairness_transform is None:
            data, labels = data_, labels_  

        elif self.fairness_transform == 'DEO':

            self.FT = Linear_FERM(dataset, data_[:, g])
            data, labels = self.FT.fit(constraint='eo')
            
        elif self.fairness_transform == 'DDP':
            
            self.FT = Linear_FERM(dataset, data_[:, g])
            data, labels = self.FT.fit(constraint='dp')

        else: raise NotImplementedError()


        if self.loss_type == 'MSE':
          
            self.alg = RidgeClassifier()
            self.alg.fit(data, labels)



        elif self.loss_type == 'CE' and self.tilt is None:

            self.alg = LogisticRegression(penalty='l1',solver='liblinear',max_iter=10000)
            self.alg.fit(data, labels)

        elif self.loss_type == 'HINGE':

            self.alg = make_pipeline(StandardScaler(), LinearSVC(penalty='l1', dual=False, max_iter=1000000))
            self.alg.fit(data, labels)

        if self.tilt is None:
            pass

        elif self.tilt and self.loss_type == 'CE':

            self.alg = Logistic_Regression(tilt=self.tilt, lr=0.1, iterations=10000)
            self.alg.fit(data, labels) 

        
        else: raise NotImplementedError()  


    def predict(self, data):

        if self.fairness_transform is None:
            preds = self.alg.predict(data)

        elif self.fairness_transform is not None:
            data_new = self.FT.new_representation(data)
            preds = self.alg.predict(data_new)

        return preds



class Non_LinearAlg(LearningAlgorithm):

    def __init__(self, algorithm='FERM', fairness_transform=None):
        super().__init__()
        self.algorithm = algorithm
        self.fairness_transform = fairness_transform
        self.alg = None

    def get_features(self):
        return np.array([
            0.,  # always non-linear
            LIBRARY_OF_NON_LINEAR_METHODS[self.algorithm],  # get the corresponding number
            LIBRARY_OF_FAIRNESS_TRANSFORMS[self.fairness_transform],  # same
            0.,  
        ])

    def execute(self, dataset, g = None, gamma = 0.1, C=0.1):

        data_ = dataset[0]
        labels_ = dataset[1]

        if self.fairness_transform is None:
            data, labels = data_, labels_ 

        elif self.fairness_transform == 'DEO':

            self.FT = Linear_FERM(dataset, data_[:, g])
            data, labels = self.FT.fit(constraint='eo')
            
        elif self.fairness_transform == 'DDP':
            self.FT = Linear_FERM(dataset, data_[:, g])
            data, labels = self.FT.fit(constraint='dp')

        else: raise NotImplementedError()


        if self.algorithm == 'FERM':
            self.alg = Non_Linear_FERM(data_[:, g], gamma, C)
            self.alg.fit(data, labels)

        elif self.algorithm == 'BOOSTING':

            clf = RandomForestClassifier(n_estimators=50) 
            param_grid = { 'max_features': ['auto', 'sqrt', 'log2']}
            self.alg = GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1)
            self.alg.fit(data, labels)
    

        else: raise NotImplementedError() 

    def predict(self, data, labels):
    
        if self.fairness_transform is None and self.algorithm != 'FERM':
            preds = self.alg.predict(data)
            acc = None

        elif self.fairness_transform is not None:
            data_new = self.FT.new_representation(data)
            preds = self.alg.predict(data_new)
            acc = None

        elif self.fairness_transform is None and self.algorithm == 'FERM':
            preds, acc = self.alg.preds_acc(data, labels)

        else: raise NotImplementedError() 

        return preds, acc 

