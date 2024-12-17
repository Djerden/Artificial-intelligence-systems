import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tabulate import tabulate

# Функция для вывода статистики по данным
def print_data_statistics(data):
    print("Статистика по данным:")
    print(tabulate(data.describe(), headers='keys', tablefmt='pretty'))

# Визуализация данных
def plot_data_histograms(data):
    data.hist(bins=50, figsize=(20, 15), color='skyblue')
    plt.show()

# Нормализация данных
def normalize_data(data):
    return (data - data.mean()) / data.std()

# Добавление синтетического признака
def add_synthetic_feature(data):
    data["hours_scores_interaction"] = (
        data["Hours Studied"] * data["Previous Scores"]
    )
    return data

# Функция для линейной регрессии
def linear_regression(X, y):
    X_transpose = X.T
    return np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)

# Оценка модели с помощью R²
def r_squared(X, y, theta):
    y_pred = X.dot(theta)  # предсказание целевого признака
    ss_res = np.sum((y - y_pred) ** 2)  # остатки
    ss_tot = np.sum((y - np.mean(y)) ** 2)  # общая сумма квадратов
    return 1 - (ss_res / ss_tot)  # объясняет ли модель изменчивость данных

# Функция для подготовки данных (добавление столбца для смещения)
def prepare_data(X):
    return np.c_[np.ones(X.shape[0]), X]  # добавляем столбец для смещения

# Функция для тренировки и оценки модели
def train_and_evaluate_model(train_set, test_set, features, target):
    # Обучение модели
    X_train = train_set[features].values
    y_train = train_set[target].values
    X_train = prepare_data(X_train)
    
    # Вектор коэффициентов, который описывает как признаки влиябт на целевой признак
    theta = linear_regression(X_train, y_train)
    
    r2_train = r_squared(X_train, y_train, theta)
    
    # Оценка на тестовом наборе
    X_test = test_set[features].values
    y_test = test_set[target].values
    X_test = prepare_data(X_test)
    
    r2_test = r_squared(X_test, y_test, theta)
    
    return r2_train, r2_test, theta

# Функция для вычисления и вывода матрицы корреляции
def print_correlation_matrix(data):
    correlation_matrix = data.corr()  # Вычисляем корреляцию между признаками
    print("\nМатрица корреляции:")
    print(tabulate(correlation_matrix, headers='keys', tablefmt='fancy_grid'))

# Основной код
if __name__ == "__main__":

    # Загружаем датасет про обучение студентов
    data = pd.read_csv("students_set.csv")

    # Заменяем категориальные значения
    data = data.replace({'Yes': 1, 'No': 0})


    # Вывод статистики по данным
    print_data_statistics(data)
    
    # Гистограммы данных
    plot_data_histograms(data)
    
    # Предобработка данных
    data = data.dropna() # удаление пропущенных значений (NaN)
    data_normalized = normalize_data(data)
    data_normalized = add_synthetic_feature(data_normalized)
    
    # Вычисление и вывод матрицы корреляции до нормализации
    print_correlation_matrix(data)

    # Вычисление и вывод матрицы корреляции после нормализации
    print_correlation_matrix(data_normalized)


    # Разделение на обучающий и тестовый наборы
    train_set, test_set = train_test_split(data_normalized, test_size=0.2, random_state=42)
    
    # Описание признаков 
    # Целевой признак
    target = "Performance Index"
    
    # Модель 1: несколько признаков
    features_1 = ["Hours Studied", "Previous Scores"]
    
    # Модель 2: больше признаков
    features_2 = ["Hours Studied", "Previous Scores", "Sleep Hours"]
    
    # Модель 3: больше признаков + синтетический признак
    features_3 = ["Sleep Hours", "Sample Question Papers Practiced", "hours_scores_interaction"]
    
    # Тренировка и оценка моделей
    r2_train_1, r2_test_1, _ = train_and_evaluate_model(train_set, test_set, features_1, target)
    r2_train_2, r2_test_2, _ = train_and_evaluate_model(train_set, test_set, features_2, target)
    r2_train_3, r2_test_3, _ = train_and_evaluate_model(train_set, test_set, features_3, target)

    # Сравнение моделей на обучающих данных
    models_data_train = [
        ["Модель 1", r2_train_1],
        ["Модель 2", r2_train_2],
        ["Модель 3 (с синтетическим признаком)", r2_train_3]
    ]
    print("\nСравнение моделей на обучающих данных (коэффициент детерминации R²):")
    print(tabulate(models_data_train, headers=["Модель", "R²"], tablefmt='pretty'))

    # Сравнение моделей на тестовых данных
    models_data_test = [
        ["Модель 1", r2_test_1],
        ["Модель 2", r2_test_2],
        ["Модель 3 (с синтетическим признаком)", r2_test_3]
    ]
    print("\nСравнение моделей на тестовых данных (коэффициент детерминации R²):")
    print(tabulate(models_data_test, headers=["Модель", "R² на тестовых данных"], tablefmt='pretty'))
