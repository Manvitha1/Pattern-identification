# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

#import the data set
dataset = pd.read_csv("data_number_pattern.csv")
x=dataset.iloc[:,0:1].values
y=dataset.iloc[:,1].values

"""
#create svr regression
from sklearn.svm import SVR
regressor= SVR(kernel='rbf')
regressor.fit(x,y)

#SVR didnt give a good prediction
"""

#create polynomial regression to try the prediction
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=5) # Here the data set is added with 
        #polynomial features before applying the linear regression model
x_poly=poly_reg.fit_transform(x)
lin_reg=LinearRegression()
lin_reg.fit(x_poly,y)


#visualize using polynomial model
import matplotlib.pyplot as plt
X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x,y,color='red')
plt.plot(X_grid, lin_reg.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Pattern identification (Polynomial Regression)')
plt.xlabel('Azimuth')
plt.ylabel('Elevation')
plt.show()
# gave a good result

#predict using random forest model
from sklearn.ensemble import RandomForestRegressor
random_reg = RandomForestRegressor(n_estimators = 1000, random_state = 0)
random_reg.fit(x, y)

#Plot graph using random forest regressor
X_grid = np.arange(min(x), max(x), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, random_reg.predict(X_grid), color = 'blue')
plt.title('Pattern identification (Random Forest Regression)')
plt.xlabel('Azimuth')
plt.ylabel('Elevation')
plt.show()

# Verify for specific X values
y_random=random_reg.predict([[322.645]])
y_poly=lin_reg.predict(poly_reg.fit_transform([[322.645]]))

