import numpy as np
import matplotlib.pyplot as plt

#INPUT DATA
x = np.array([10,15,20,25,35,40,60,100,105,115,140])            #FEATURE
y = np.array([102,148,245,240,300,384,510,928,1140,1167,1401])  #TARGET VARIABLE (TO BE PREDICTED)

#VISUAL REPRESENTATION
plt.figure(figsize=(8,4))
plt.scatter(x, y)
plt.xlabel('X (Feature)')
plt.ylabel('Y (Target)')
plt.show()

#WITH INTERCEPT
θ1 = 150 #Initialize θ1 as any random value
θ0 = 25  #Initialize θ2 as any random value
lr = 0.0001 #Set the Learning Rate
all_θ1 = []  #Create an empty list to add all the updated θ1s during gradient descent (for error graph plotting)
all_θ0 = []  #Create an empty list to add all the updated θ0s during gradient descent (for error graph plotting)
all_error = [] #Create an empty list to store the errors (RMSE) at every iteration. 
prev_θ0 = θ0   #Assign prev_θ0 as θ0 and update it as we pass by every iteration.
prev_θ1 = θ1   #Assign prev_θ1 as θ1 and update it as we pass by every iteration.
i = 1
#The difference between the [prev_θ0, prev_θ1 and θ0, θ1] is used as tthe breaking-condition for the loop. (convergence)
while True:
    print('Iteration: ', i)
    i+=1
    y_pred = θ0+(θ1*x)    #Prediction using linear regression. y = θ0 + θ1x (same as y=c+mx)
    error = (1/(2*len(x)))*(sum((y_pred - y)**2)) #Cost Function that we have to minimize using gradiant descent.
    slope_0 = sum((y_pred - y))/len(x)            #Calculate the gradient (slope) of Cost Function wrt θ0.
    slope_1 = sum(x*(y_pred - y))/len(x)          #Calculate the gradient (slope) of Cost Function wrt θ0.
    θ0 = θ0 - (lr*slope_0)                        #Update θ0 according to its slope and the learning rate.   
    θ1 = θ1 - (lr*slope_1)                        #Update θ1 according to its slope and the learning rate.   
    print('θ0 = ', θ0, ' Slope(θ0) = ', slope_0, ' θ1 = ', θ1, ' Slope(θ1) = ', slope_1)
    if (np.abs(prev_θ0-θ0) <= 0.00001) & (np.abs(prev_θ1-θ1) <= 0.00001): #If prev_θ0,θ0, prev_θ1,θ1 are very close/almost same, it means we have reached the minima.
        break
    else:
        prev_θ0 = θ0       #Else, assign prev_θ0,prev_θ1 as θ0, θ1 and run the loop again. (next iteration: Predict with updated θ1 and caluclate error, slope, update etc.) 
        prev_θ1 = θ1      
        all_θ0.append(θ0) #Appending θ0, θ1  and error to the list to plot the error graph. 
        all_θ1.append(θ1) 
        all_error.append(error)
		

#VISUAL REPRESENTATION OF BEST FIT LINE
plt.figure(figsize=(8,4))
plt.scatter(x, y)
plt.plot(x, y_pred)
plt.xlabel('X (Feature)')
plt.ylabel('Y (Target)')
plt.ylim(0,1600)
plt.show()