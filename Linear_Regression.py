#Sri Rama Jayam
#Coding Linear Regression

#Imports
import numpy as np
import matplotlib.pyplot as plt

#Visualize the training data
x = np.array([0.01,0.02,0.03,0.04,0.05,0.06]) #Features - In Our Case Location of the thermocouple
y = np.array([15.46,14.59,12.66,12.55,11.57,11.42]) #Output - In our case the temperature as measured experimentally

'''
plt.figure(figsize=(8,6))
plt.scatter(x,y)
plt.xlabel('Location of Thermocouple (in m)')
plt.ylabel('Experimental Temperature (in ^C)')
plt.title('Visualization of the training data')
plt.show()
'''

#Onto the Linear Regression Model
#Defining the Cost Function
#Our Model is like y = wx + b
#Goal is to determine 'w' and 'b'
#Initialize Random Values of w and b

w = -90.00
b = 17.00

#Visualize the plot for these values
y_init = w*x+b

#Plotting
'''
plt.figure(figsize=(8,6))
plt.scatter(x,y)
plt.plot(x,y_init)
plt.xlabel('Location of Thermocouple (in m)')
plt.ylabel('Experimental Temperature (in ^C)')
plt.title('Visualization of the training data')
plt.show()
'''

#Defining the cost function
#Taking the mean squared error
def cost_derivative(x,y,w,b):
    #x - Numpy array
    #y - Numpy array
    #w - weight (scalar)
    #b - bias (scalar)
    y_init = w*x + b
    m = x.shape[0]
    J = 0
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        J+=(y[i]-y_init[i])**2/(2*m)
        dj_dw+=(y[i]-y_init[i])*x[i]/m
        dj_db+=(y[i]-y_init[i])/m
    return J,dj_dw,dj_db


iterations = 10000
alpha = 0.001 #Learning Rate
cost_history = []
for iteration in range(iterations):
    J,dj_dw,dj_db = cost_derivative(x,y,w,b)
    cost_history.append(J)
    w+=dj_dw*alpha
    b+=dj_db*alpha

#Plotting the Variation of cost function with number of iterations
'''
x_iteration = np.arange(1,1001)
plt.figure(figsize = (8,6))
plt.plot(x_iteration, cost_history)
plt.xlabel('Number of iterations')
plt.ylabel('Cost')
plt.title('Plot showing variation of the cost function with the number of iterations')
plt.show()
'''

#Plotting the values of w and b obtained along with the training data points to observe the fit.
'''
y_final = w*x + b
plt.figure(figsize=(8,6))
plt.scatter(x,y)
plt.plot(x,y_final)
plt.xlabel('Location of Thermocouple (in m)')
plt.ylabel('Experimental Temperature (in ^C)')
plt.title('Visualization of the training data')
plt.show()
'''

print("The value of the weight (w) thus obtained is: ",w)
print("The value of the bias(b) thus obtained is: ",b)

