import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# https://ourworldindata.org/grapher/total-deaths-covid-19
c19_data = pd.read_csv('total-deaths-covid-19.csv')
Y_us = c19_data.loc[c19_data["Entity"] == "United States"]["Total confirmed deaths due to COVID-19 (deaths)"]
# From Mar 7 to Apr 15
Y_us = Y_us[len(Y_us)-40:len(Y_us)-25].to_numpy()
ln_Y_us = np.log(Y_us)
#%%
A = np.empty((0,3))
for i in range(len(ln_Y_us)):
    A_row = np.array([[1, i, i**2]])
    A = np.concatenate((A, A_row), axis = 0)
#%%
U, d, V = np.linalg.svd(A, full_matrices=False)
D = np.empty((0,3))
for i in range(len(d)):
    D_row = np.zeros(((1,3)))
    D_row[0][i] = d[i]
    D = np.append(D, D_row, axis = 0)
#%%
U_t = np.transpose(U)
D_i = np.linalg.inv(D)
V_t = np.transpose(V)
V_tD_i = np.dot(V_t, D_i)
V_tD_iU_t = np.dot(V_tD_i, U_t)
a_vector = np.dot(V_tD_iU_t, ln_Y_us)
#%%
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
sum = 0
for i in range(len(fitted_ln_Y_us)):
    sum += abs(ln_Y_us[i] - fitted_ln_Y_us[i])
var = sum/len(fitted_ln_Y_us)
#%%
cov_mat = np.dot(np.dot(np.dot(V_t, D), D), V) * var