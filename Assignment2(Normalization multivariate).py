# Normalization for Multivariate Linear Regression



# Importing the libraries
import numpy as np
import pandas as pd


# Opening the file
file = pd.read_csv('Assignment-1 data\\ex1data22.csv',header = None)


# Separating the Independent and the Dependent Variables
IndependentVariables = file.iloc[:, 0:2].values
IndependentVariables = pd.DataFrame(IndependentVariables)
DependentVariables = file.iloc[:, 2].values



# Adding a column of 1's at the beginning of a dataframe
location_of_one_column = 0
new_col = [1] * len(DependentVariables) # can be a list, a Series, an array or a scalar   
IndependentVariables.insert(location_of_one_column, 'Ones Column', new_col)


# Transpose of the Independent Variables
transpose_of_independent_variables = IndependentVariables.T



# Now Performing the multiplication of the Transposed array and the TwoDyarray
a = np.array(transpose_of_independent_variables, dtype = float)
b = np.array( IndependentVariables , dtype = float)
x_transpose_multiply_x = a.dot(b)


# Now calculating the term (Xtransposey) inverse
inverse = np.linalg.inv(x_transpose_multiply_x)




# Now calculating the term Xtranspose multiply y
c = np.array(transpose_of_independent_variables, dtype = float)
d = np.array( DependentVariables , dtype = float)
x_transpose_multiply_y = c.dot(d)



# Now here multipling multiplyin the two 2D arrays x_transpose_multiply_x and x_transpose_multiply_y
# This gives us the optimal values of Thetas

e = np.array(inverse, dtype = float)
f = np.array(x_transpose_multiply_y , dtype = float)
x_transpose_x_inverse_multiply_x_transpose_y = e.dot(f)



array_containing_predicted_values = []


print(IndependentVariables[1][3])

hypothesis = 0

for i in range(0,len(IndependentVariables)):
    hypothesis = 0
    hypothesis =  x_transpose_x_inverse_multiply_x_transpose_y[0] + x_transpose_x_inverse_multiply_x_transpose_y[1]* float(IndependentVariables[0][i]) + x_transpose_x_inverse_multiply_x_transpose_y [2] * float(IndependentVariables[1][i])
    array_containing_predicted_values.append(hypothesis)





# Predicting the test Data Result


# Hypothesis Equation 
h1 = x_transpose_x_inverse_multiply_x_transpose_y[0]
+ 1650 * x_transpose_x_inverse_multiply_x_transpose_y[1] 
+ 3 * x_transpose_x_inverse_multiply_x_transpose_y[2]

print("The price of the house with size = 1650 foot and noofbedrooms = 3 is",h1)




 
 



















