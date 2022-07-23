from typing import List

class Linear_Regression:
    def __init__(self, learning_rate: int, epoch: int):
        self.root = None
        self.m1 = 1
        self.m2 = 2
        self.b = 5
        self.learning_rate = learning_rate
        self.epoch = epoch

    def fit(self, X : List[int], Y : List[int], Z : List[int]): 
        m1 = self.m1
        m2 = self.m2
        b = self.b
        train_data = []
        results = []
        for j in range(X.__len__()):
            adding_data = [ X[j], Y[j], Z[j] ]
            train_data.append(adding_data)
        for i in range(self.epoch):
            m1, m2, b= self.gradient_descent(m1, m2, b, train_data, self.learning_rate)
            results.append([m1,m2,b])
            

        return results
    
    def gradient_descent(self, m1N, m2N, bN, points, L):
        m1_gradient = 0
        m2_gradient = 0
        b_gradient = 0
        L = self.learning_rate

        n = len(points)
        loss = 0
        for i in range(n):
            x = points[i][0]
            y = points[i][1]
            z = points[i][2]

            m1_gradient += (2/n)*x*(m1N*x + m2N*y + bN - z)
            m2_gradient += (2/n)*y*(m1N*x + m2N*y + bN - z)
            b_gradient += (2/n)*(m1N*x + m2N*y + bN - z)
            
        m1 = m1N - m1_gradient*L
        m2 = m2N - m2_gradient*L
        b = bN - b_gradient*L

        self.m1 = m1
        self.m2 = m2
        self.b = b

        return m1,m2,b
    
    def predict(self, x_test, y_test):
        z_pred = []
        
        for i in range (len(x_test)):
            z_pred.append(self.m1*x_test[i] + self.m2*y_test[i] + self.b)
        return z_pred

