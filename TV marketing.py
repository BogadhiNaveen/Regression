import pandas as pd
advertising = pd.read_csv("tvmarketing.csv")
# Display the first 5 rows
advertising.head()
# Display the last 5 rows
advertising.tail()
# Let's check the columns
advertising.info()
# Check the shape of the DataFrame (rows, columns)
advertising.shape
# Let's look at some statistical information about the dataframe.
advertising.describe()
import seaborn as sns
%matplotlib inline
# Visualise the relationship between the features and the response using scatterplots
sns.pairplot(advertising, x_vars=['TV'], y_vars='Sales',size=7, aspect=0.7, kind='scatter')
# Putting feature variable to X
X = advertising['TV']
y = advertising['Sales']
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 , random_state=100)
import numpy as np
X_train = X_train[:, np.newaxis]
X_test = X_test[:, np.newaxis]
# import LinearRegression from sklearn
from sklearn.linear_model import LinearRegression
# Representing LinearRegression as lr(Creating LinearRegression Object)
lr = LinearRegression()
# Fit the model using lr.fit()
lr.fit(X_train, y_train)
# Print the intercept and coefficients
print(lr.intercept_)
print(lr.coef_)
y_pred = lr.predict(X_test)
# Actual vs Predicted
import matplotlib.pyplot as plt
c = [i for i in range(1,61,1)] # generating index
fig = plt.figure()
plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-")
plt.plot(c,y_pred, color="red", linewidth=2.5, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20) # Plot heading
plt.xlabel('Index', fontsize=18) # X-label
plt.ylabel('Sales', fontsize=16) # Y-label
# Error terms
c = [i for i in range(1,61,1)]
fig = plt.figure()
plt.plot(c,y_test-y_pred, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20) # Plot heading
plt.xlabel('Index', fontsize=18) # X-label
plt.ylabel('ytest-ypred', fontsize=16) # Y-label
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)
import matplotlib.pyplot as plt
plt.scatter(y_test,y_pred)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
