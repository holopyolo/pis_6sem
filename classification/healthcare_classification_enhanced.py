import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
from catboost import CatBoostClassifier
warnings.filterwarnings('ignore')

# 1. ЗАГРУЗКА И ПЕРВИЧНАЯ ОБРАБОТКА
print("=== ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ ===")
df = pd.read_csv('healthcare_dataset.csv')
print(f"Исходный размер: {df.shape}")

# Выбираем признаки и целевую переменную
target = 'Test Results'
features = ['Age', 'Gender', 'Blood Type', 'Medical Condition', 
           'Insurance Provider', 'Billing Amount', 'Admission Type', 'Medication']

df_work = df[features + [target]].copy()

# 2. ПРЕДОБРАБОТКА ДАННЫХ
print("\n=== ПРЕДОБРАБОТКА ===")

# Проверяем пропуски
print(f"Пропуски: {df_work.isnull().sum().sum()}")

# Обработка выбросов в Billing Amount
Q1 = df_work['Billing Amount'].quantile(0.25)
Q3 = df_work['Billing Amount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_before = len(df_work[(df_work['Billing Amount'] < lower_bound) | 
                             (df_work['Billing Amount'] > upper_bound)])
print(f"Выбросы в Billing Amount: {outliers_before}")

# Ограничиваем выбросы (не удаляем, а ограничиваем)
df_work.loc[df_work['Billing Amount'] < lower_bound, 'Billing Amount'] = lower_bound
df_work.loc[df_work['Billing Amount'] > upper_bound, 'Billing Amount'] = upper_bound

# Feature Engineering - создаем новые признаки
df_work['Age_Group'] = pd.cut(df_work['Age'], bins=[0, 30, 50, 70, 100], 
                             labels=['Young', 'Middle', 'Senior', 'Elderly'])
df_work['High_Cost'] = (df_work['Billing Amount'] > df_work['Billing Amount'].median()).astype(int)

# Добавляем новые признаки в список
features_enhanced = features + ['Age_Group', 'High_Cost']

# Кодируем категориальные переменные
le_dict = {}
categorical_cols = ['Gender', 'Blood Type', 'Medical Condition', 
                   'Insurance Provider', 'Admission Type', 'Medication', 'Age_Group']

for col in categorical_cols:
    le_dict[col] = LabelEncoder()
    df_work[col] = le_dict[col].fit_transform(df_work[col].astype(str))

# Кодируем целевую переменную
le_target = LabelEncoder()
y = le_target.fit_transform(df_work[target])
X = df_work[features_enhanced]

print(f"Целевые классы: {le_target.classes_}")
print(f"Распределение классов: {np.bincount(y)}")
print(f"Итоговый размер данных: {X.shape}")

# Стандартизация числовых признаков
scaler = StandardScaler()
numeric_cols = ['Age', 'Billing Amount']
X_scaled = X.copy()
X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# 3. РАЗДЕЛЕНИЕ НА ОБУЧЕНИЕ И ТЕСТ
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, 
                                                    random_state=42, stratify=y)
print(f"Обучение: {X_train.shape}, Тест: {X_test.shape}")

# 4. ОБУЧЕНИЕ МОДЕЛЕЙ
print("\n=== ОБУЧЕНИЕ МОДЕЛЕЙ ===")

# Модель 1: Random Forest (меньше деревьев для скорости)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Модель 2: Logistic Regression
lr = LogisticRegression(random_state=42, max_iter=5000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Модель 3: CatBoostClassifier
cat_features_idx = [X.columns.get_loc(col) for col in categorical_cols if col in X.columns]
cb = CatBoostClassifier(verbose=0, random_state=42)
cb.fit(X_train, y_train, cat_features=cat_features_idx)
cb_pred = cb.predict(X_test)

# 5. ПРОСТОЙ АНСАМБЛЬ
# print("Создание ансамбля...")
# ensemble_pred = []
# for i in range(len(rf_pred)):
#     # Простое голосование
#     votes = [rf_pred[i], lr_pred[i], cb_pred[i]]
#     ensemble_pred.append(max(set(votes), key=votes.count))

# ensemble_pred = np.array(ensemble_pred)

# 6. ОЦЕНКА КАЧЕСТВА
print("\n=== РЕЗУЛЬТАТЫ ===")

models = {
    'Random Forest': rf_pred,
    'Logistic Regression': lr_pred,
    'CatBoost': cb_pred,
    # 'Simple Ensemble': ensemble_pred
}

best_acc = 0
best_model = ""

for name, pred in models.items():
    acc = accuracy_score(y_test, pred)
    if acc > best_acc:
        best_acc = acc
        best_model = name
    
    print(f"\n{name}: Accuracy = {acc:.3f}")

print(f"\nЛучшая модель: {best_model} (Accuracy: {best_acc:.3f})")

# Детальный отчет для лучшей модели
best_pred = models[best_model]
print(f"\nDetailed Report ({best_model}):")
print(classification_report(y_test, best_pred, target_names=le_target.classes_))

# Важность признаков
feature_importance = rf.feature_importances_
importance_df = pd.DataFrame({
    'feature': features_enhanced,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(f"\nВажность признаков:\n{importance_df}")
