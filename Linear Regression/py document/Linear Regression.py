import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Define The Load Data Functions
def read_txt_file(file_path):
    data = np.loadtxt(file_path)  
    x_data, y_data = data[:, 0], data[:, 1]
    return x_data, y_data

# Define The Drawing Function
def plot_fit_sklearn(x, y, model):
    fit_line = model.predict(x.reshape(-1, 1))
    plt.plot(x, fit_line, 'g', label='Linear Fit')
    plt.plot(x, y, 'r+', label='Data')
    plt.legend()
    plt.xlabel('x values')
    plt.ylabel('y values')
    plt.title('Best fit to our data (sklearn)')
    plt.show()

 # Define The Computed Residual Function
def evaluate_residuals_sklearn(x, y, model):
    fit_line = model.predict(x.reshape(-1, 1))
    residuals = y - fit_line
    sum_residuals = np.sum(residuals)
    mean_residuals = np.mean(residuals)
    std_residuals = np.std(residuals)
    
    print(f"Sum of Residuals: {sum_residuals:.8f}")
    print(f"Mean of Residuals: {mean_residuals:.8f}")
    print(f"Standard Deviation of Residuals: {std_residuals:.4f}")

# Reading Data
x_data, y_data = read_txt_file('./Linear Regression/dataset/data 1.txt')

# Linear Fitting Using scikit-learn
x_data_2d = x_data.reshape(-1, 1)
model = LinearRegression().fit(x_data_2d, y_data)

# Calculated Residual
evaluate_residuals_sklearn(x_data, y_data, model)

# Plot Data And Fit Curves
plot_fit_sklearn(x_data, y_data, model)

