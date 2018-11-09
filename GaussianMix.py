import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
import math

def getData(sample_number):

    NormalData = np.random.multivariate_normal([0, 0],  [[1, 0], [0, 3]], sample_number)

    x1, y1 = np.random.multivariate_normal([1,1], [[1, 0], [0, 4]], sample_number).T;
    
    x2, y2 = np.random.multivariate_normal([2, -1], [[3, 0], [0, 2.5]], sample_number).T;

    M = [];

    for i in range(sample_number):

        M.append([0.3 * x1[i] + 0.7 * x2[i] ,  0.3 * y1[i] + 0.7 * y2[i]]);

    return NormalData, np.array(M);

def calThePDFOfGaussin(miu, cov, x, y):
    temp = np.matrix(np.array([x, y]) - miu);
    temp2 = float(temp * (cov.I).T * temp.T);
    temp4 = np.e ** (-temp2 * 0.5);
    temp5 = temp4 / (math.sqrt(np.linalg.det(cov)));
    return temp5;

def main():

    bound = 4
    sample_number = 1000

    normalData, mistureData = getData(sample_number);

    clfNormal = mixture.GaussianMixture(n_components=1, covariance_type='full')
    clfMisture = mixture.GaussianMixture(n_components=2, covariance_type='full')

    clfNormal.fit(normalData);
    clfMisture.fit(mistureData);

    normalMean = clfNormal.means_;
    #normalMean = np.array([0, 0]);
    normalCov = np.matrix(clfNormal.covariances_);
    #normalCov = np.matrix( [[1, 0], [0, 3]]);

    mistureMeanOne = clfMisture.means_[0];
    #mistureMeanOne = np.array([1,1]);
    mistureMeanTwo = clfMisture.means_[1];
    #mistureMeanTwo = np.array([2, -1]);

    mistureCovOne = np.matrix(clfMisture.covariances_[0]);
    #mistureCovOne = np.matrix([[1, 0], [0, 4]]);
    mistureCovTwo = np.matrix(clfMisture.covariances_[1]);
    #mistureCovTwo = np.matrix([[3, 0], [0, 2.5]]);

    mistureWeightOne = clfMisture.weights_[0];
    #mistureWeightOne = 0.3;
    mistureWeightTwo = clfMisture.weights_[1];
    #mistureWeightTwo = 0.7;

    #可视化
    plt.plot(normalData.T[0], normalData.T[1], 'rx')
    plt.plot(mistureData.T[0], mistureData.T[1], 'bo')
    plt.axis('equal')
    plt.axis([-bound,bound,-bound,bound]) #限制显示范围
    plt.show()

    num = 400;
    step = 2 * bound / num;
    tempX = -bound;
    tempY = -bound;

    listAX = [];
    listAY = []
    listBX = [];
    listBY = []

    DELEA = 1 / (2 * math.pi);

    for i in range(num):

        tempY = 0;

        for j in range(num):


            A = calThePDFOfGaussin(normalMean, normalCov, tempX, tempY);
            B = calThePDFOfGaussin(mistureMeanOne, mistureCovOne, tempX, tempY);
            C = calThePDFOfGaussin(mistureMeanTwo, mistureCovTwo, tempX, tempY);
            D = mistureWeightOne * B + mistureWeightTwo * C;

            if A > D:
                listAX.append(tempX);
                listAY.append(tempY);

            else:
                listBX.append(tempX);
                listBY.append(tempY);

            tempY += step;


        tempX += step;

    plt.scatter(listAX, listAY, c = 'g');
    plt.scatter(listBX, listBY, c = 'r');
    plt.show();

    totalA = 0;

    for val in normalData:

        A = calThePDFOfGaussin(normalMean, normalCov, val[0], val[1]);
        B = calThePDFOfGaussin(mistureMeanOne, mistureCovOne, val[0], val[1]);
        C = calThePDFOfGaussin(mistureMeanTwo, mistureCovTwo, val[0], val[1]);
        D = mistureWeightOne * B + mistureWeightTwo * C; 

        if A < D:
            totalA += 1;


    totalB = 0;
    for val in mistureData:

        A = calThePDFOfGaussin(normalMean, normalCov, val[0], val[1]);
        B = calThePDFOfGaussin(mistureMeanOne, mistureCovOne, val[0], val[1]);
        C = calThePDFOfGaussin(mistureMeanTwo, mistureCovTwo, val[0], val[1]);
        D = mistureWeightOne * B + mistureWeightTwo * C; 

        if A > D:
            totalB += 1;

    print('Corret Rate of samples from Normal distribution');        
    print(1 - totalA / ( sample_number));
   
    print('Corret Rate of samples from Mixture distribution');        
    print(1 - totalB / ( sample_number));

main();
