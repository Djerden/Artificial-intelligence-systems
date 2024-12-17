import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os
from tabulate import tabulate

data = pd.read_csv("diabet_set.csv")

# Обработка отсутствующих значений
# data = data.replace(0, np.nan)
data['Glucose'] = data['Glucose'].replace(0, np.nan)
data['BloodPressure'] = data['BloodPressure'].replace(0, np.nan)
data['SkinThickness'] = data['SkinThickness'].replace(0, np.nan)
data['Insulin'] = data['Insulin'].replace(0, np.nan)
data['BMI'] = data['BMI'].replace(0, np.nan)

#Заполняем пустые значения средними значениями по столбцу
data.fillna(data.mean(), inplace=True)

data.hist(bins=50, figsize=(20, 15), color='skyblue')
plt.show()

data = data.dropna() # удаление строк с NaN

# Масштабирование данных
scaler = StandardScaler()
scaled_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Pedigree', 'Age']
data[scaled_columns] = scaler.fit_transform(data[scaled_columns])

data_normalized = (data - data.mean()) / data.std()

# Создание 3D-области для визуализации
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Построение 3D-графика случайными признаками
ax.scatter(data['Pregnancies'], data['Glucose'], data['BloodPressure'], marker='^')
ax.scatter(data['SkinThickness'], data['Insulin'], data['BMI'], marker='o')
ax.scatter(data['Pedigree'], data['Age'], data['Outcome'], marker='d')

ax.set_xlabel('Glucose')
ax.set_ylabel('BMI')
ax.set_zlabel('Age')

plt.show()

# Евклидово расстояние между двумя точками
def distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# X_data - обучающая, x_class - целевой класс, y - новый пациент > в-т массив классов большинства соседей для конкретного случая
def neighbors(X_data, X_classification, y, K=3):
    # проходимся по каждому случаю, для которого находим соседей и класс
    predictions = np.zeros((len(y), 1))
    for j in range(len(y)):
        Y = y[j]
        dist = []
        # Считаем расстояния между искомой точкой и остальными
        for i in range(len(X_data)):
            dist.append((distance(X_data[i], Y), X_classification[i]))
            
        # Сортируем по расстояниям
        dist = sorted(dist, key=lambda x: x[0])
            
        # Извлекаем классы K ближайших соседей
        nearest_classes = [d[1] for d in dist[:K]]
            
        # Преобразуем nearest_classes в одномерный массив
        nearest_classes = np.array(nearest_classes).flatten()
        # Определяем класс по большинству
        predicted_class = round(sum(nearest_classes) / len(nearest_classes))
        
        predictions[j, 0] = predicted_class            
    return predictions


X = data.drop('Outcome', axis=1)
y = data['Outcome']


# Разделение на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Pedigree', 'Age']

# Модель 1 (Признаки выбираются случайно)
count_of_columns = random.randint(2, len(columns) - 1)
model_columns = np.random.choice(columns, size=count_of_columns, replace=False)
X1 = X_train[model_columns]
print(model_columns)

# Модель 2 (Заранее заданные признаки)
X2 = X_train[['Glucose', 'Insulin', 'BMI', 'Pedigree', 'Age']]

models = {'Model 1': X1, 'Model 2': X2}

# Вывод результатов для каждой модели и разных значений K
for model_name, X_model in models.items():
    print(f"\nРезультаты для {model_name}:\n")
    for k in [3, 5, 10]:
        # Предсказания 
        predictions = neighbors(X_model.values, y_train.values, X_test[X_model.columns].values, k)
        # Матрица ошибок на основе предсказаний и реальных значений
        cm = confusion_matrix(y_test.values, predictions)
        
        correct = cm[0][0] + cm[1][1]
        incorrect = cm[0][1] + cm[1][0]
        accuracy = 100 * correct / (correct + incorrect)
        
        print(f"\nМатрица ошибок при k={k}:\n")
        print(tabulate(cm, headers=["Предсказано 0", "Предсказано 1"], showindex=["Факт 0", "Факт 1"], tablefmt="pretty"))
        
        print(f"Правильных классификаций: {correct}")
        print(f"НЕправильных классификаций: {incorrect}")
        print(f"Точность: {accuracy:.2f}%")
