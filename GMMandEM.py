import numpy as np
import pandas as pd
import math
import copy
import random
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt

class GMM(object):
    """docstring for GMM"""
    def __init__(self, epochs = 200, K = 4, precision = 1e-4):

        self.epochs = epochs;
        self.K = K;
        self.precision = precision;
        
        
    def fit(self, data_array):

        self.data_array = data_array;
        self.initWeights();

        for T in range(self.epochs):

            pi_arrayY, mean_listY, cov_listY = self.pi_array, self.mean_list, self.cov_list;

            self.pi_array, self.mean_list, self.cov_list = self.updataWeights();

            print(self.calculateTheLogLikehood(self.data_array, self.pi_array, self.mean_list, self.cov_list));


            if self.checkWehtherConverge(self.pi_array, self.mean_list, self.cov_list, pi_arrayY, mean_listY, cov_listY, 1e-6):
                print("epochs " + str(T));
                break;

    def predict(self, point):

        maxval = 0;
        cluster = 0;
        temp = 0;
        for i in range(self.K):
            temp = self.calculateGaussianPDF(point, self.mean_list[i], self.cov_list[i]) * self.pi_array[i];
            if temp  > maxval:
                maxval = temp;
                cluster = i;
        return cluster;


    def getParms(self):
        return self.pi_array, self.mean_list, self.cov_list;


    def initWeights(self):

        self.pi_array = np.array([0.25 for _ in range(self.K)]);

        module = KMeans(self.K);
        module.fit(self.data_array);
        
        MeanOfKCluster = module.cluster_centers_
        self.mean_list = module.cluster_centers_;

        clusterDataList = [[] for _ in range(self.K)];

        sampleNumber = self.data_array.shape[0];

        for i in range(sampleNumber):

            Mindistance = np.linalg.norm(self.data_array[i] - MeanOfKCluster[0]);
            whichCluster = 0;

            for j in range(self.K):

                if np.linalg.norm(self.data_array[i] - MeanOfKCluster[j]) < Mindistance:

                    Mindistance = np.linalg.norm(self.data_array[i] - MeanOfKCluster[j]);
                    whichCluster = j;

            clusterDataList[whichCluster].append(self.data_array[i]);

        self.cov_list = [];

        for i in range(self.K):
            self.cov_list.append(
                
                np.matrix(
                    np.cov(np.array(clusterDataList[i])[:, 0],  np.array(clusterDataList[i])[:, 1])
                )
            );

    def calculateGaussianPDF(self, point, mean, cov):

        temp = np.matrix(point - mean);
        temp2 = float(temp * cov.I * temp.T);
        temp4 = np.e ** (-temp2 * 0.5);
        temp5 = temp4 / (math.sqrt(abs(np.linalg.det(cov)) ) * 2 * np.pi);
        return temp5;

    def calculateYnk(self, xn, k, pi_array, mean_list, cov_list):

        temp = self.calculateGaussianPDF(xn, mean_list[k], cov_list[k]) * pi_array[k];
        total = 0;

        for i in range(len(pi_array)):
            total += self.calculateGaussianPDF(xn, mean_list[i], cov_list[i]) * pi_array[i];

        return temp / total;

    def calculateTheLogLikehood(self, data_array, pi_array, mean_list, cov_list):
     
        QLikeHood = 0;

        for n in range(data_array.shape[0]):
            for k in range(len(pi_array)):
                ynk = self.calculateYnk(data_array[n], k, pi_array, mean_list, cov_list);
                
                QLikeHood += ynk * (math.log(self.calculateGaussianPDF(data_array[n], mean_list[k], cov_list[k])) + math.log(pi_array[k]));

        return QLikeHood;

    def checkWehtherConverge(self, pi_array, mean_list, cov_list, pi_arrayY, mean_listY, cov_listY, precision):
        judge = (abs(pi_array - pi_arrayY) <= precision).all();
        judge = judge and (abs(mean_list - mean_listY) <= precision).all()
        judge = judge and (abs(np.array(cov_list) - np.array(cov_listY)) <= precision).all();
        return judge;

    def updataWeights(self):

        temp_pi_array = np.array([0 for _ in range(self.K)], dtype = np.float64);
        temp_mean_list = np.array([[0, 0] for _ in range(self.K)], dtype = np.float64);
        temp_cov_list = [np.matrix([[0,0],[0,0]], dtype = np.float64) for _ in range(self.K)];

        EffectiveNumber = [0 for _ in range(self.K)];

        for k in range(self.K):
            for n in range(self.data_array.shape[0]):
                ynk = self.calculateYnk(self.data_array[n], k, self.pi_array, self.mean_list, self.cov_list);

                EffectiveNumber[k] += ynk;
                temp_mean_list[k] += ynk * self.data_array[n];
                temp_cov_list[k] += ynk * np.matrix(self.data_array[n] - self.mean_list[k]).T * np.matrix(self.data_array[n] - self.mean_list[k]);

            temp_mean_list[k] /= EffectiveNumber[k];
            temp_cov_list[k] /= EffectiveNumber[k];
            temp_pi_array[k] = EffectiveNumber[k] / self.data_array.shape[0];

        return temp_pi_array, temp_mean_list, temp_cov_list;


def main():


    csv_data = pd.read_csv('TrainingData_GMM.csv', header = None);
    data_array = np.array(csv_data);


    gmm = GMM(K = 4);
    gmm.fit(data_array);
    print(gmm.getParms());
    #print(calculateTheLogLikehood(data_array, pi_array, mean_list, cov_list));

    clusterList = [[] for _ in range(4)];

    for i in range(data_array.shape[0]):
        cluster = gmm.predict(data_array[i]);

        clusterList[cluster].append(data_array[i]);

    plt.scatter(np.array(clusterList[0])[:, 0], np.array(clusterList[0])[:, 1], color = 'green')
    plt.scatter(np.array(clusterList[1])[:, 0], np.array(clusterList[1])[:, 1], color = 'red')
    plt.scatter(np.array(clusterList[2])[:, 0], np.array(clusterList[2])[:, 1], color = 'yellow')
    plt.scatter(np.array(clusterList[3])[:, 0], np.array(clusterList[3])[:, 1], color = 'blue')

    plt.show();
    print('\n');




#main();
