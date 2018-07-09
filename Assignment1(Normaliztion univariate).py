# Normalization Method for univarite Linear Regression


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt

# Opening the file
file = open("Assignment-1 data\\ex1data1.txt", "r")

IndependentVariable = []
DependentVariable = []
list_of_ones = []


for i in range(97):
    list_of_ones.append(1)

two_d_array_of_list_of_1 = np.array(list_of_ones)[np.newaxis]


# Getting the independent and the dependent Variables in the separate Arrays
for line in file: 
    x,y = line.split(",")
    IndependentVariable.append(x)
    DependentVariable.append(y)




# Visualising the Graph
plt.scatter(IndependentVariable, DependentVariable, color = 'red')
plt.title('Prediting the Profit (Regression Model)')
plt.xlabel('Populations')
plt.ylabel('Profit')
plt.show()
    


# First Converting the X-array into 2D array and then taking its Transpose     
transpose_of_two_d_array_of_x = np.array(IndependentVariable)[np.newaxis]



# Apending a column of ones at the beginning of 2D array
transpose_of_two_d_array_of_x = np.insert(transpose_of_two_d_array_of_x, 0, np.array(list_of_ones), 0)  
two_d_array_of_x = transpose_of_two_d_array_of_x.T



# Converting the y-array into the 2D so that multiplication can be performed
two_d_array_of_y = np.array(DependentVariable)[np.newaxis]
two_d_array_of_y = two_d_array_of_y.T




# Now Performing the multiplication of the Transposed array and the TwoDyarray
a = np.array(transpose_of_two_d_array_of_x, dtype = float)
b = np.array( two_d_array_of_x , dtype = float)
x_transpose_multiply_x = a.dot(b)


# Now calculating the term (Xtransposey) inverse
inverse = np.linalg.inv(x_transpose_multiply_x)




# Now calculating the term Xtranspose multiply y
c = np.array(transpose_of_two_d_array_of_x, dtype = float)
d = np.array( two_d_array_of_y , dtype = float)
x_transpose_multiply_y = c.dot(d)



# Now here multipling multiplyin the two 2D arrays x_transpose_multiply_x and x_transpose_multiply_y
# This gives us the optimal values of Thetas

e = np.array(inverse, dtype = float)
f = np.array(x_transpose_multiply_y , dtype = float)
x_transpose_x_inverse_multiply_x_transpose_y = e.dot(f)



array_containing_predicted_values = []


hypothesis = 0

for i in IndependentVariable:
    hypothesis = 0
    hypothesis =  x_transpose_x_inverse_multiply_x_transpose_y[0] + x_transpose_x_inverse_multiply_x_transpose_y[1]* float(i)
    array_containing_predicted_values.append(hypothesis)
    
    


# Visualising the Graph
plt.scatter(IndependentVariable, DependentVariable, color = 'red')
plt.plot(IndependentVariable,array_containing_predicted_values)
plt.title('Prediting the Profit (Regression Model)')
plt.xlabel('Populations')
plt.ylabel('Profit')
plt.show()




#plt.scatter(IndependentVariable, DependentVariable)
#plt.plot(IndependentVariable, yfit)
#plt.show()







 
 




















#print(TransposedMatrix)
#print(TransposedMatrix.T)


   