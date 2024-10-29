import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Загрузка и предобработка данных
data = pd.read_csv('diabet_set.csv')

# Заменяем нули на NaN в тех колонках, где они не могут быть валидными значениями
cols_with_zero_values = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zero_values:
    data[col] = data[col].replace(0, np.nan)

# Заполняем пропуски медианой
imputer = SimpleImputer(strategy="median")
data[cols_with_zero_values] = imputer.fit_transform(data[cols_with_zero_values])

# Нормализация признаков
scaler = StandardScaler()
features = data.drop(columns=["Outcome"])
features_scaled = scaler.fit_transform(features)

# Преобразуем данные обратно в DataFrame и добавляем столбец "Outcome"
data_scaled = pd.DataFrame(features_scaled, columns=features.columns)
data_scaled['Outcome'] = data['Outcome']

# Разделение на обучающую и тестовую выборки
train_data, test_data = train_test_split(data_scaled, test_size=0.2, random_state=42)

# Функция для расчёта расстояния Евклида
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Оптимизированная функция k-NN
def k_nearest_neighbors_optimized(train_data, test_point, k):
    train_features = train_data.drop(columns=['Outcome']).values
    test_features = test_point[:-1].values

    distances = np.sqrt(np.sum((train_features - test_features) ** 2, axis=1))
    nearest_indices = np.argsort(distances)[:k]

    nearest_labels = train_data.iloc[nearest_indices]['Outcome']
    prediction = Counter(nearest_labels).most_common(1)[0][0]
    return prediction

# Оценка модели для разных значений k
def evaluate_knn_model_optimized(train_data, test_data, k_values):
    results = {}
    for k in k_values:
        predictions = [k_nearest_neighbors_optimized(train_data, test_point, k) for _, test_point in test_data.iterrows()]
        actuals = test_data['Outcome'].tolist()
        
        accuracy = np.mean([pred == actual for pred, actual in zip(predictions, actuals)])
        results[k] = accuracy
    return results

# Расчёт и отображение матриц ошибок для каждого k
def get_confusion_matrices(train_data, test_data, k_values):
    matrices = {}
    for k in k_values:
        predictions = [k_nearest_neighbors_optimized(train_data, test_point, k) for _, test_point in test_data.iterrows()]
        actuals = test_data['Outcome'].tolist()
        matrices[k] = confusion_matrix(actuals, predictions)
    return matrices

# Определяем значения k
k_values = [3, 5, 10]

# Запускаем оценку и отображаем результаты
knn_accuracies = evaluate_knn_model_optimized(train_data, test_data, k_values)
knn_confusion_matrices = get_confusion_matrices(train_data, test_data, k_values)

# Вывод результатов
print("Результаты оценки k-NN для разных значений k:\n")
for k in k_values:
    print(f"Для k = {k}:")
    print(f"  - Точность: {knn_accuracies[k] * 100:.2f}%")
    tn, fp, fn, tp = knn_confusion_matrices[k].ravel()
    print("  - Матрица ошибок:")
    print(f"      TP (Истинные положительные): {tp}")
    print(f"      FP (Ложные положительные): {fp}")
    print(f"      TN (Истинные отрицательные): {tn}")
    print(f"      FN (Ложные отрицательные): {fn}\n")

# Построение 3D-графика для визуализации признаков
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Выбираем три признака для 3D-визуализации
x, y, z = data_scaled['Glucose'], data_scaled['BMI'], data_scaled['Age']
ax.scatter(x, y, z, c=data_scaled['Outcome'], cmap='coolwarm', marker='o', alpha=0.6)

# Назначение осей
ax.set_xlabel('Glucose')
ax.set_ylabel('BMI')
ax.set_zlabel('Age')
ax.set_title('3D Визуализация признаков: Glucose, BMI и Age')

plt.show()
