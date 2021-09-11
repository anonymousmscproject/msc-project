import numpy as np
import scipy.special

# TERM Logistic Regression. Due to use of exponentials, transforms of the data may be required to avoid overflow.
# CHAPTER 3.3

class Logistic_Regression():
    def __init__(self, tilt=0.1, lr=0.01, iterations=10000):
        self.t = tilt # tilt paramter
        self.lr = lr # learning rate
        self.num_iter = iterations # maximum number of iterations
        
    
    def sigmoid(self, z):

        return 1 / (1 + np.exp(-z))
    
    def fit(self, data, label):

        self.w = np.zeros(data.shape[1])
        obj = []
        
        for i in range(self.num_iter):
            
            z = np.dot(data, self.w)
        
            loss = np.log(1 + np.exp(-z)) + np.multiply(1 - label, z)
            
            # new TERM objective and gradient
            grad = np.dot(data.T, np.multiply(np.exp(loss * self.t), self.sigmoid(z) - label)) / label.size
            Z = np.mean(np.exp(self.t * loss))
    
            self.w -= self.lr / Z * grad
  
            tmp = (1/self.t) * np.log(np.mean(np.exp(self.t * loss)))
            obj.append(tmp)
    
    def predict(self, data):
        pred = self.sigmoid(np.dot(data, self.w)).round()
        return pred 

