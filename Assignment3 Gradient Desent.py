# Gradient Descent Implementation to calculate the optimal Values of Thetas
# Gradient Descent for Univariate and Multivariate Regression
# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy


# Opening the file
file = pd.read_csv('Assignment-1 data\\ex1data22.csv',header = None)


# Separating the Independent and the Dependent Variables
IndependentVariable = file.iloc[:, 0:2].values
IndependentVariable = pd.DataFrame(IndependentVariable)
DependentVariable = file.iloc[:, 2].values


list_containing_cost_at_each_thetas  = []
No_of_thetas = len(file.columns)
no_of_training_examples = len(IndependentVariable)
predicted_values = []
mean_of_columns = []
standard_deviation_of_columns = []
array_containing_corresponding_thetas = []
Normalized_Independent_Variables = deepcopy(IndependentVariable)

       

# Checking if no of features are greater than 2 then there is a need for 
# Feature Normalization 
# In this case we perform the mean Normalization of each feature
if No_of_thetas > 2:
    for loop_control_variable in range(0,No_of_thetas-1):
        mean = IndependentVariable[[loop_control_variable]].mean()
        std = IndependentVariable[[loop_control_variable]].std() 
        mean_of_columns.append(mean)
        standard_deviation_of_columns.append(std)
        for loop_control_variable_1 in range(0,len(IndependentVariable)):
            Normalized_Independent_Variables[loop_control_variable][loop_control_variable_1] = ((Normalized_Independent_Variables[loop_control_variable][loop_control_variable_1] - mean_of_columns[loop_control_variable])/standard_deviation_of_columns[loop_control_variable])   
            
            
    





Values_of_thetas = [0] * No_of_thetas
h_of_x = 0
x_note = 1
array_containing_difference_of_actual_and_predicted_values = []
actual_values = deepcopy(DependentVariable)
array_storing_h_of_x_minus_y_into_x_values = []
updated_values_of_thetas = []       
alpha_value = 0.01
array_containing_iteration_no = []


   

for biggest_loop_control_variable in range(0,50):
    predicted_values[:] = []
    # Hypothesis Equation 
    
    array_containing_iteration_no.append(biggest_loop_control_variable)
    for i in range(0,len(IndependentVariable)):
        h_of_x = 0
        for j in range(0,len(Values_of_thetas)):  
            if j == 0:
                h_of_x = h_of_x + Values_of_thetas[j] * x_note
            else: 
                h_of_x = h_of_x + Values_of_thetas[j] * Normalized_Independent_Variables[j-1][i]
        predicted_values.append(h_of_x)
        
    
    
    
    # Now calculating (h(x) - y)^2 to be used in calculating Theta and updating theta
    array_containing_difference_of_actual_and_predicted_values[:] = []
    temp = 0
    for k in range(0,len(actual_values)):
        temp = (predicted_values[k] - float(actual_values[k]))**2
        array_containing_difference_of_actual_and_predicted_values.append(temp) 
        
    
            
    # Now calculating Summation(h(x) - y)^2    
    sum_term = 0
    for l in range(0,len(array_containing_difference_of_actual_and_predicted_values)):
        sum_term = sum_term + array_containing_difference_of_actual_and_predicted_values[l]
        
        
    sum_term = (1 / (2* no_of_training_examples)) * sum_term
    list_containing_cost_at_each_thetas.append(sum_term)
    array_containing_corresponding_thetas.append(Values_of_thetas[:])
 
    
    #latest_cost = list_containing_cost_at_each_thetas[-1] 
    
    # Now here i ll apply the gradient descent Method
    # to calculate the new values of thetas
    array_storing_h_of_x_minus_y_into_x_values[:] = []
    updated_values_of_thetas[:] = []
    
    for m in range(0,No_of_thetas):
        for n in range(0,len(IndependentVariable)):
            if m == 0:
                temp = (predicted_values[n] - float(actual_values[n])) * x_note
                array_storing_h_of_x_minus_y_into_x_values.append(temp) 
            else:
                temp = ((predicted_values[n] - float(actual_values[n])) * Normalized_Independent_Variables[m-1][n])
                array_storing_h_of_x_minus_y_into_x_values.append(temp) 
        sum_of_h_of_x_minus_y_into_x_array = 0 
        for o in range(0,len(array_storing_h_of_x_minus_y_into_x_values)):
            sum_of_h_of_x_minus_y_into_x_array = sum_of_h_of_x_minus_y_into_x_array + array_storing_h_of_x_minus_y_into_x_values[o] 
        
        sum_of_h_of_x_minus_y_into_x_array = sum_of_h_of_x_minus_y_into_x_array / no_of_training_examples    
        sum_of_h_of_x_minus_y_into_x_array = sum_of_h_of_x_minus_y_into_x_array * alpha_value
        updated_values_of_thetas.append(Values_of_thetas[m] - sum_of_h_of_x_minus_y_into_x_array)
    
    #Values_of_thetas = updated_values_of_thetas        
    for loop in range(0,len(Values_of_thetas)):
        Values_of_thetas[loop] = deepcopy(updated_values_of_thetas[loop])



# Visualising the Graph
plt.scatter(array_containing_iteration_no, list_containing_cost_at_each_thetas, color = 'red')
plt.title('Cost Deviation')
plt.xlabel('No of iterations')
plt.ylabel('Costs')
plt.show()




min_cost = min(list_containing_cost_at_each_thetas)
thetas_corresponding_to_min_cost = array_containing_corresponding_thetas[list_containing_cost_at_each_thetas.index(min(list_containing_cost_at_each_thetas))]

array_containing_predicted_values = []
hypothesis = 0
    
if No_of_thetas == 2:    
    for i in range(0,len(Normalized_Independent_Variables)):
        hypothesis = 0
        hypothesis =  thetas_corresponding_to_min_cost[0] + thetas_corresponding_to_min_cost[1]* float(Normalized_Independent_Variables[0][i])
        array_containing_predicted_values.append(hypothesis)


# This is only for Univariate Linear Regression
#plt.scatter(IndependentVariable, DependentVariable, color = 'red')
#plt.plot(IndependentVariable,array_containing_predicted_values)
#plt.title('Prediting the Profit (Regression Model)')
#plt.xlabel('Populations')
#plt.ylabel('Profit')
#plt.show()




# This test case is only for Multivariate Linear Regression
# First of all Normalize the Features and then put into the hypothesis Equation
test_size = 1650
test_bedrooms = 3
test_values = []
test_values.append(test_size)
test_values.append(test_bedrooms)
h_of_x_predicted = 0  


# Normalization
for i in range(0,len(mean_of_columns)):
    test_values[i] = (test_values[i]-mean_of_columns[i])/standard_deviation_of_columns[i]
    

print(test_values[0])


for j in range(0,No_of_thetas):  
          if j == 0:
                h_of_x_predicted = h_of_x_predicted + thetas_corresponding_to_min_cost[j] * x_note
          else: 
                h_of_x_predicted = h_of_x_predicted + thetas_corresponding_to_min_cost[j] * float(test_values[j-1])


print(h_of_x_predicted)












 
 



















