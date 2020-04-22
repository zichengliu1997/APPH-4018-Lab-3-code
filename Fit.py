import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# https://ourworldindata.org/grapher/total-deaths-covid-19
# Read the CSV into a pandas data frame and extract the data for the US specifically. Then get the natural logs of the data
c19_data = pd.read_csv('total-deaths-covid-19.csv')
Y_us = c19_data.loc[c19_data["Entity"] == "United States"]["Total confirmed deaths due to COVID-19 (deaths)"]
# From Mar 7 to Apr 15
Y_us = Y_us[len(Y_us)-40:len(Y_us)-25].to_numpy()
ln_Y_us = np.log(Y_us)
#%%
# Create the design matrix
T = np.empty((0,3))
for i in range(len(ln_Y_us)):
    T_row = np.array([[1, i, i**2]])
    T = np.concatenate((T, T_row), axis = 0)
#%%
# Get the SVD of the design matrix and create the singular value matrix W
U, w, V = np.linalg.svd(T, full_matrices=False)
W = np.empty((0,3))
for i in range(len(w)):
    W_row = np.zeros(((1,3)))
    W_row[0][i] = w[i]
    W = np.append(W, W_row, axis = 0)
#%%
# Transpose U and V and invert W. Then get the coefficient vector
U_t = np.transpose(U)
W_i = np.linalg.inv(W)
V_t = np.transpose(V)
V_tW_i = np.dot(V_t, W_i)
V_tW_iU_t = np.dot(V_tW_i, U_t)
a_vector = np.dot(V_tW_iU_t, ln_Y_us)
#%%
# Show the fitted curve and the actual curve
fitted_ln_Y_us = np.array([])
a0 = a_vector[0]
a1 = a_vector[1]
a2 = a_vector[2]
t = np.array([])
for i in range(len(ln_Y_us)):
    fitted_ln_Y_us = np.append(fitted_ln_Y_us, a0 + a1*i + a2*i**2)
    t = np.append(t, i)
plt.figure()
plt.plot(t,fitted_ln_Y_us, label = "Fitted")
plt.plot(t,ln_Y_us, label = "Actual")
plt.title("ln(D(t)) vs. t (days)")
plt.grid()
plt.xlabel("t (days)")
plt.ylabel("D(t)")
plt.legend()
plt.show()
#%%
# Get the covariance matrix and the standard errors
sum = 0
for i in range(len(fitted_ln_Y_us)):
    sum += abs(ln_Y_us[i] - fitted_ln_Y_us[i])
var = sum/len(fitted_ln_Y_us)
cov_mat = np.dot(np.dot(np.dot(V_t, W), W), V) * var
#%%
# Print the results
print("a0 = "+ str(a0) + " +/- " + str(np.sqrt(cov_mat[0][0])) + ", a1 = " + str(a1)+ " +/- " + str(np.sqrt(cov_mat[1][1])) + ", a2 = " + str(a2) + " +/- " + str(np.sqrt(cov_mat[2][2])))