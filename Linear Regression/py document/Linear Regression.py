import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 从文本文件中读取数据，并返回 NumPy 数组
def read_txt_file(file_path):
    data = np.loadtxt(file_path)  
    x_data, y_data = data[:, 0], data[:, 1]
    return x_data, y_data

# 定义画图函数
def plot_fit_sklearn(x, y, model):
    fit_line = model.predict(x.reshape(-1, 1))
    plt.plot(x, fit_line, 'g', label='Linear Fit')
    plt.plot(x, y, 'r+', label='Data')
    plt.legend()
    plt.xlabel('x values')
    plt.ylabel('y values')
    plt.title('Best fit to our data (sklearn)')
    plt.show()

 # 定义计算残差函数
def evaluate_residuals_sklearn(x, y, model):
    fit_line = model.predict(x.reshape(-1, 1))
    residuals = y - fit_line
    sum_residuals = np.sum(residuals)
    mean_residuals = np.mean(residuals)
    std_residuals = np.std(residuals)
    
    print(f"Sum of Residuals: {sum_residuals:.8f}")
    print(f"Mean of Residuals: {mean_residuals:.8f}")
    print(f"Standard Deviation of Residuals: {std_residuals:.4f}")

# 读取数据
x_data, y_data = read_txt_file('./Linear Regression/dataset/data 1.txt')

# 使用 scikit-learn 进行线性拟合
x_data_2d = x_data.reshape(-1, 1)
model = LinearRegression().fit(x_data_2d, y_data)

# 计算残差
evaluate_residuals_sklearn(x_data, y_data, model)

# 绘制数据和拟合曲线
plot_fit_sklearn(x_data, y_data, model)

