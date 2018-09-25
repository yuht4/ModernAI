import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import copy
from sklearn.linear_model import Ridge

def PolyRegression(degree = 9,  sampleNumber = 100, lam = 0):

    xs = np.linspace(0, 1, sampleNumber);
    ys = np.sin(2 * np.pi * xs)  + np.random.normal(0, 0.1, sampleNumber);

    xs = xs.reshape(xs.shape[0],1)
    ys = ys.reshape(ys.shape[0],1)

    plt.xlabel('x')
    plt.ylabel('y')

    plt.title('N = ' + str(sampleNumber) + " lnλ = " + str(np.log(lam)))
    plt.scatter(xs, ys, color = 'r', label = 'Original Data')

    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(xs)

    regressor = LinearRegression();
    regressor.fit(X_train_poly, ys);

    rige_poly = Ridge(lam)
    rige_poly.fit(X_train_poly, ys)

    print(rige_poly.coef_);

    plt.plot(xs, rige_poly.predict(X_train_poly), color ='g', label = 'Regresson Data')

    plt.show();

#PolyRegression(9, 15, 0);
#PolyRegression(9, 100, 0);
#PolyRegression(9, 10, np.exp(-18));
#PolyRegression(9, 10, 1);
#PolyRegression(9, 10, 0);
