import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Шаг 1: Загрузка данных и вычисление статистики
data = pd.read_csv('students_set.csv')

# Визуализация статистики по данным (исключая 'count')
def display_statistics(data):
    stats = data.describe()
    stats = stats.drop(['count'])  # Убираем строку 'count', так как она не несет смысла для графика

    # Построение графика на одной фигуре
    stats.loc[['mean', 'std', 'min', '25%', '50%', '75%', 'max']].plot(kind='bar', figsize=(10, 6))
    plt.title("Остальные статистические показатели")
    plt.ylabel("Значение")
    plt.xticks(rotation=45)
    plt.show()


display_statistics(data)

# Шаг 2: Предварительная обработка данных
def preprocess_data(data):
    # Преобразование категориальных текстовых значений в числовой формат
    for column in data.select_dtypes(include=['object']).columns:
        data[column] = data[column].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Обработка отсутствующих значений
    data = data.fillna(data.mean())
    
    # Кодирование категориальных признаков (если остались)
    data = pd.get_dummies(data, drop_first=True)
    
    # Нормализация данных
    normalized_data = (data - data.mean()) / data.std()
    return normalized_data

processed_data = preprocess_data(data)

# Замените 'Performance Index' на фактическое название целевой переменной
X = processed_data.drop('Performance Index', axis=1)
y = processed_data['Performance Index']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Шаг 3: Линейная регрессия методом наименьших квадратов
class LinearRegression:
    def __init__(self):
        self.coefficients = None
    
    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # добавление столбца единиц для свободного члена
        self.coefficients = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.coefficients)

# Построение трех моделей с разными наборами признаков
def build_and_evaluate_models(X_train, y_train, X_test, y_test):
    feature_sets = [
        ['Hours Studied', 'Previous Scores'],  # первый набор признаков
        ['Sleep Hours', 'Sample Question Papers Practiced'],  # второй набор признаков
        ['Hours Studied', 'Sleep Hours', 'Previous Scores']  # третий набор признаков
    ]
    
    for i, features in enumerate(feature_sets):
        model = LinearRegression()
        model.fit(X_train[features], y_train)
        predictions = model.predict(X_test[features])
        
        # Оценка производительности
        r_squared = 1 - sum((y_test - predictions) ** 2) / sum((y_test - y_test.mean()) ** 2)
        print(f"Модель {i+1} (Признаки: {features}): Коэффициент детерминации R^2 = {r_squared:.4f}")

build_and_evaluate_models(X_train, y_train, X_test, y_test)

# Бонусное задание: добавление синтетического признака
X_train['synthetic_feature'] = X_train['Hours Studied'] * X_train['Sleep Hours']
X_test['synthetic_feature'] = X_test['Hours Studied'] * X_test['Sleep Hours']

# Построение модели с синтетическим признаком
model_with_synthetic = LinearRegression()
model_with_synthetic.fit(X_train[['Hours Studied', 'Sleep Hours', 'synthetic_feature']], y_train)
predictions_with_synthetic = model_with_synthetic.predict(X_test[['Hours Studied', 'Sleep Hours', 'synthetic_feature']])

# Оценка производительности модели с синтетическим признаком
r_squared_synthetic = 1 - sum((y_test - predictions_with_synthetic) ** 2) / sum((y_test - y_test.mean()) ** 2)
print(f"Модель с синтетическим признаком: Коэффициент детерминации R^2 = {r_squared_synthetic:.4f}")
