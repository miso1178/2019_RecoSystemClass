# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 09:54:07 2019

@author: kd556
"""

import numpy as np
import pandas as pd
import random

user = pd.read_csv('C:/RecoSys/Data/user.csv')
book = pd.read_csv('C:/RecoSys/Data/book.csv')
rating = pd.read_csv('C:/RecoSys/Data/BX-Book-Ratings.csv')

# data merge
data = pd.merge(pd.merge(user,rating, how='outer'), book, how='outer', on='ISBN')

# Rating = 0인 애들 빼기
data = data[data['Book-Rating'] != 0]

rating_matrix = data.pivot(index = 'User-ID', columns ='ISBN', values = 'Book-Rating')
rating_matrix.shape

class QMF():
    # Initializing the object
    def __init__(self, rating_matrix, K, alpha, beta, iterations, verbose=True):
        
        self.R_ = rating_matrix
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.verbose = verbose
        
        # Calculate and sorting by user mean
        self.user_mean = self.R_.mean(axis=1).sort_values(ascending=False)
        # Set divide size(# of user in each Matrix)
        self.div_size = int(np.ceil(len(self.user_mean)/4))
        
        # Get indexes of each Matrix
        self.idx1 = self.user_mean.index[:self.div_size]
        self.idx2 = self.user_mean.index[self.div_size:self.div_size*2]
        self.idx3 = self.user_mean.index[self.div_size*2:self.div_size*3]
        self.idx4 = self.user_mean.index[self.div_size*3:]
        
        # Make quartered Matrixes
        self.R1 = np.array(self.R_.loc[self.idx1].fillna(0))
        self.R2 = np.array(self.R_.loc[self.idx2].fillna(0))
        self.R3 = np.array(self.R_.loc[self.idx3].fillna(0))
        self.R4 = np.array(self.R_.loc[self.idx4].fillna(0))
        
        # Make full Matrix for validation
        self.R0 = np.concatenate([self.R1, self.R2, self.R3, self.R4])
        
    def train(self):
        # Initializing user-feature and movie-feature matrix 
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Initializing the bias terms
        self.b_u = np.zeros(self.num_users)
        self.b_d = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        # List of training samples
        self.samples = [
        (i, j, self.R[i, j])
        for i in range(self.num_users)
        for j in range(self.num_items)
        if self.R[i, j] > 0
        ]

        # Stochastic gradient descent for given number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            self.full_matrix = self.full_prediction()
            measure = self.rmse()
            training_process.append((i, measure))
            if self.verbose:
                if (i+1) % 10 == 0:
                    print("Iteration: %d ; RMSE = %.4f" % (i+1, measure))
        return training_process

    # Computing mean squared error
    def rmse(self):
        xs, ys = self.R.nonzero()
        self.predictions = []
        self.errors = []
        error = 0
        for x, y in zip(xs, ys):
            self.predictions.append(self.full_matrix[x, y])
            self.errors.append(self.R[x, y] - self.full_matrix[x, y])
        self.predictions = np.array(self.predictions)
        self.errors = np.array(self.errors)
        return np.sqrt(np.mean(self.errors**2))

    # Stochastic gradient descent to get optimized P and Q matrix
    def sgd(self):
        for i, j, r in self.samples:
            prediction = self.get_prediction(i, j)
            e = (r - prediction)

            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_d[j] += self.alpha * (e - self.beta * self.b_d[j])

            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

    # Ratings for user i and moive j
    def get_prediction(self, i, j):
        prediction = self.b + self.b_u[i] + self.b_d[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    # Full user-movie rating matrix
    def full_prediction(self):
        return self.b + self.b_u[:,np.newaxis] + self.b_d[np.newaxis,:] + self.P.dot(self.Q.T)

    
class NEW_QMF(QMF):

    # Get test set from whole matrix R0(4777,22222), consists of four concatenated matrixes.
    def set_total_test(self, test_size=0.3): 
        xs, ys = self.R0.nonzero()
        total_test_set = []
        for x, y in zip(xs, ys):        
            if random.random() < test_size:
                total_test_set.append([x,y,self.R0[x,y]])
                self.R0[x,y] = 0
        self.total_test_set = total_test_set
        return total_test_set        

    
    def total_rmse(self):
        t_error = 0
        for t_one_set in self.total_test_set:
            t_predicted = self.total_matrix[t_one_set[0], t_one_set[1]]
            t_error += pow(t_one_set[2] - t_predicted, 2)
        return np.sqrt(t_error/len(self.total_test_set))    
    
    
    def set_test(self, test_size=0.3):         # Setting test set
        xs, ys = self.R.nonzero() # 인덱스를 가져옴.
        test_set = []
        for x, y in zip(xs, ys):                # Random selection
            if random.random() < test_size:
                test_set.append([x,y,self.R[x,y]]) # random하게 25%를 골라서 test_set에 붙임(?)
                self.R[x,y] = 0
        self.test_set = test_set
        return test_set                         # Return test set

    # Get RMSE from total matrix, that consists of four matrixes that which are trained by sgd()
    def test_rmse(self):
        error = 0
        for one_set in self.test_set:
            predicted = self.full_matrix[one_set[0], one_set[1]]
            error += pow(one_set[2] - predicted, 2)
        return np.sqrt(error/len(self.test_set))
    
    def test(self):
  

        # Make empty matrix for concatenation of 4 quartered Matrixes
        self.total_matrix = np.zeros((1,22222))
        
        total_training_process = []
        
        total_test_set = self.set_total_test()
        
        # Train each matrixes
        for self.idx, self.R in enumerate([self.R1, self.R2, self.R3, self.R4]):
            
            # Make test set
            test_set = self.set_test(0.3)             
            
            print('<%d Quarter matrix>' % (self.idx+1))
            
            self.num_users, self.num_items = np.shape(self.R)
            
            # Initializing user-feature and movie-feature matrix 
            self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
            self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

            # Initializing the bias terms
            self.b_u = np.zeros(self.num_users)
            # Set initial Bias of item using Books table
            self.b_d = np.array([data[data['major_author']==1]['Book-Rating'].mean() - data[data['major_author']==0]['Book-Rating'].mean()]*self.num_items)
            self.b = np.mean(self.R[np.where(self.R != 0)])

            # List of training samples
            self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
            ]
           
            # Stochastic gradient descent for given number of iterations
            training_process = []
            for iter_ in range(self.iterations):
                np.random.shuffle(self.samples)
                self.sgd()
                self.full_matrix = self.full_prediction()
                
                rmse1 = self.rmse()
                rmse2 = self.test_rmse()
                training_process.append((iter_, rmse1, rmse2))
                if self.verbose:
                    if (iter_+1) % 10 == 0:
                        print("Iteration: %d ; Train RMSE = %.4f ; Test RMSE = %4f" % (iter_+1, rmse1, rmse2))
            
            total_training_process.append(training_process)
            # Concatenate four trained matrixes
            self.total_matrix = np.concatenate((self.total_matrix, self.full_matrix), axis=0)
        # Drop first row, that is filled with np.zeros(1,22222)
        self.total_matrix = self.total_matrix[1:,:]
        rmse3 = self.total_rmse()
        print("Final RMSE = %4f" % (rmse3))
        return total_training_process

rating_matrix = data.pivot(index = 'User-ID', columns ='ISBN', values = 'Book-Rating')
R_temp = rating_matrix.copy()             # Save original data
mf = NEW_QMF(R_temp, K=30, alpha=0.001, beta=0.02, iterations=50, verbose=True)
total_test_set = mf.set_total_test(test_size=0.3)
result = mf.test()