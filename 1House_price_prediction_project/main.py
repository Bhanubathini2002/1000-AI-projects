import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error





#this is training data 
X = np.array([[650], [800], [950], [1100], [1250], [1400], [1550], [1700], [1850], [2000]])
y = np.array([325, 400, 475, 550, 625, 700, 775, 850, 925, 1000])


# create a linear regression mode 
model = LinearRegression()


#train model with data 

model.fit(X,y)



#make predictions 

y_pred = model.predict(X)

# printmodel parameters 

print("Model coefficient (slope):", model.coef_[0])
print("Model Intercept:", model.intercept_)
print("Mean squared error:", mean_squared_error(y,y_pred))


#predict price for a new house 
new_area = [[1900]]
predicted_price = model.predict(new_area)[0]
print(f"\nPredicted price for 1600 sq.ft: ${predicted_price * 1000:.2f}")


# Plotting
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.scatter(new_area, predicted_price, color='green', label='Prediction (1600 sq.ft)')
plt.xlabel("Area (sq.ft)")
plt.ylabel("Price ($1000s)")
plt.title("House Price Prediction using Linear Regression")
plt.legend()
plt.grid(True)
plt.show()