import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from colorama import Fore, Style, init
from tabulate import tabulate


DATA_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "students_set.csv")
)
data = pd.read_csv(DATA_PATH)
data = data.replace({'Yes': 1, 'No': 0})

print("Статистика по данным:")
print(tabulate(data.describe(), headers='keys', tablefmt='pretty'))

data.hist(bins=50, figsize=(20, 15), color='purple')
plt.show()

data = data.dropna()

# нормализация данных
data_normalized = (data - data.mean()) / data.std()

# синтетический признак (взаимодействие признаков)
data_normalized["hours_scores_interaction"] = (
    data_normalized["Hours Studied"] * data_normalized["Previous Scores"]
)

# обучающий и тестовый набор
train_set, test_set = train_test_split(data_normalized, test_size=0.2, random_state=42)

# линейная регрессия
def linear_regression(X, y):
    X_transpose = X.T
    return np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y) #обратная

# оценка производительности (коэффициент детерминации R^2)
def r_squared(X, y, theta): 
    y_pred = X.dot(theta) #предсказание целевого признака 
    ss_res = np.sum((y - y_pred) ** 2) #остатки
    ss_tot = np.sum((y - np.mean(y)) ** 2) #общая сумма квадратов
    return 1 - (ss_res / ss_tot) #объясняет ли модель изменчивость данных

# модель 1: несколько признаков
X_train_1 = train_set[["Hours Studied", "Previous Scores"]].values
y_train_1 = train_set["Performance Index"].values
X_train_1 = np.c_[np.ones(X_train_1.shape[0]), X_train_1] #добавляем столбец для смещения
theta_1 = linear_regression(X_train_1, y_train_1)
r2_1 = r_squared(X_train_1, y_train_1, theta_1)

# модель 2: больше признаков
X_train_2 = train_set[["Hours Studied", "Previous Scores", "Sleep Hours"]].values
y_train_2 = train_set["Performance Index"].values
X_train_2 = np.c_[np.ones(X_train_2.shape[0]), X_train_2]
theta_2 = linear_regression(X_train_2, y_train_2)
r2_2 = r_squared(X_train_2, y_train_2, theta_2)

# модель 3: больше признаков + синтетический признак
X_train_3 = train_set[
    ["Sleep Hours", "Sample Question Papers Practiced", "hours_scores_interaction"]
].values
y_train_3 = train_set["Performance Index"].values
X_train_3 = np.c_[np.ones(X_train_3.shape[0]), X_train_3]
theta_3 = linear_regression(X_train_3, y_train_3)
r2_3 = r_squared(X_train_3, y_train_3, theta_3)

# сравнение моделей
models_data = [
    ["Модель 1", r2_1],
    ["Модель 2", r2_2],
    ["Модель 3 (с синтетическим признаком)", r2_3]
]
print("\nСравнение моделей (коэффициент детерминации R²):")
print(tabulate(models_data, headers=["Модель", "R²"], tablefmt='pretty'))

# оценка на тестовом наборе для первой модели
X_test_1 = test_set[["Hours Studied", "Previous Scores"]].values
y_test_1 = test_set["Performance Index"].values
X_test_1 = np.c_[np.ones(X_test_1.shape[0]), X_test_1]
r2_test_1 = r_squared(X_test_1, y_test_1, theta_1)

# оценка на тестовом наборе для второй модели
X_test_2 = test_set[["Hours Studied", "Previous Scores", "Sleep Hours"]].values
y_test_2 = test_set["Performance Index"].values
X_test_2 = np.c_[np.ones(X_test_2.shape[0]), X_test_2]
r2_test_2 = r_squared(X_test_2, y_test_2, theta_2)

# оценка на тестовом наборе для третьей модели
X_test_3 = test_set[
    ["Sleep Hours", "Sample Question Papers Practiced", "hours_scores_interaction"]
].values
y_test_3 = test_set["Performance Index"].values
X_test_3 = np.c_[np.ones(X_test_3.shape[0]), X_test_3]
r2_test_3 = r_squared(X_test_3, y_test_3, theta_3)

# сравнение моделей на тестовых данных
test_models_data = [
    ["Модель 1", r2_test_1],
    ["Модель 2", r2_test_2],
    ["Модель 3 (с синтетическим признаком)", r2_test_3]
]

print("\nСравнение моделей на тестовых данных (коэффициент детерминации R²):")
print(tabulate(test_models_data, headers=["Модель", "R² на тестовых данных"], tablefmt='pretty'))


