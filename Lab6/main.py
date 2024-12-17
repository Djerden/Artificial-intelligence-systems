import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler


file_path = 'diabetes.csv'  
df = pd.read_csv(file_path)

# Заменяем 0 на медиану в соответствующих колонках без использования inplace
df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].median())
df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].median())
df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].median())
df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].median())

# Визуализация: гистограммы для каждого признака
df.hist(bins=50, figsize=(20, 15), color='skyblue')
plt.tight_layout()
plt.show()

# 3. Разделение данных на обучающий и тестовый наборы
X = df.drop(columns='Outcome')
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Нормализация данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 5. Реализация логистической регрессии
# Сигмоидная функция
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Функция для вычисления логистической потери (log loss)
def compute_log_loss(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    epsilon = 1e-5  # Для предотвращения логарифма нуля
    loss = (-y.dot(np.log(h + epsilon)) - ((1 - y).dot(np.log(1 - h + epsilon)))) / m
    return loss

# Градиентный спуск
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = []
    
    for i in range(iterations):
        gradient = X.T.dot(sigmoid(X.dot(theta)) - y) / m
        theta -= learning_rate * gradient
        cost_history.append(compute_log_loss(X, y, theta))
    
    return theta, cost_history

# Добавление столбца единиц для смещения (intercept)
X_train_ = np.c_[np.ones(X_train_scaled.shape[0]), X_train_scaled]
X_test_ = np.c_[np.ones(X_test_scaled.shape[0]), X_test_scaled]

# Инициализация параметров (случайные значения)
theta_initial = np.zeros(X_train_.shape[1])

# 6. Исследование гиперпараметров
learning_rates = [0.1, 0.01, 0.001]
iterations_list = [100, 500, 1000]

results = []  

for lr in learning_rates:
    for it in iterations_list:
        # Обучение модели с текущими гиперпараметрами
        theta_opt, _ = gradient_descent(X_train_, y_train, theta_initial, lr, it)
        
        # Получаем предсказания для тестового набора
        predictions = sigmoid(X_test_.dot(theta_opt)) >= 0.5
        
        # Оценка метрик
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        
        # Сохраняем результаты
        results.append({
            "Learning Rate": lr,
            "Iterations": it,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

# Создаем DataFrame для удобного вывода результатов
results_df = pd.DataFrame(results)
print(results_df)

# 7. Оценка модели с оптимальными гиперпараметрами
# Выбираем лучший набор гиперпараметров
best_result = results_df.loc[results_df['F1 Score'].idxmax()]

# Получаем предсказания с лучшими гиперпараметрами
best_lr = best_result['Learning Rate']
best_it = int(best_result['Iterations'])  # Приводим к целому числу

theta_opt, _ = gradient_descent(X_train_, y_train, theta_initial, best_lr, best_it)
predictions = sigmoid(X_test_.dot(theta_opt)) >= 0.5

# Оценка финальной модели
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print(f"\nBest Hyperparameters: Learning Rate = {best_lr}, Iterations = {best_it}")
print(f"Final Model Evaluation:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
