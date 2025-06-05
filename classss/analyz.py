import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import catboost as cb
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression



sample_df = pd.read_csv('train.csv', nrows=1000).sample(50, random_state=42)
print("Случайная выборка из 50 строк:")
print(sample_df.head())

df_info = pd.read_csv('train.csv', nrows=5)  
print(f"Количество колонок: {len(df_info.columns)}")
print(f"Имена колонок: {df_info.columns.tolist()}")
print("\nТипы данных:")
print(df_info.dtypes)


file_lines = sum(1 for line in open('train.csv')) - 1  
print(f"\nОбщее количество строк в файле: {file_lines}")


print(sample_df.describe())


print(sample_df.isnull().sum())


print(sample_df['satisfaction'].value_counts())
print(sample_df['satisfaction'].value_counts(normalize=True) * 100)


plt.figure(figsize=(8, 6))
sns.countplot(x='satisfaction', data=sample_df)
plt.title('Распределение уровня удовлетворенности (выборка)')
plt.savefig('satisfaction_distribution.png')
plt.close()


categorical_features = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=feature, hue='satisfaction', data=sample_df)
    plt.title(f'Распределение {feature} по уровню удовлетворенности (выборка)')
    plt.xticks(rotation=45)
    plt.savefig(f'{feature}_distribution.png')
    plt.close()


numeric_features = sample_df.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(15, 12))
correlation_matrix = sample_df[numeric_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Корреляционная матрица числовых признаков (выборка)')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()


df = pd.read_csv('train.csv')



print(df['satisfaction'].value_counts())



missing_values = df.isnull().sum()
print("\nКоличество пропущенных значений в полном датасете:")
print(missing_values[missing_values > 0])  


if 'Arrival Delay in Minutes' in df.columns and df['Arrival Delay in Minutes'].isnull().sum() > 0:
    print("Заполнение пропущенных значений в колонке 'Arrival Delay in Minutes'...")
    
    df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].median(), inplace=True)


print("Разделение данных на признаки и целевую переменную...")
X = df.drop(columns=['satisfaction', 'id'])
y = df['satisfaction']


print("Преобразование категориальных переменных...")
categorical_features = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
for feature in categorical_features:
    print(f"Уникальные значения для {feature}: {df[feature].unique()}")


numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"Числовые признаки: {numeric_features}")


print("Разделение данных на обучающую и тестовую выборки...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")


print("Создание препроцессора для данных...")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])




X_train_catboost = X_train.copy()
X_test_catboost = X_test.copy()

for feature in categorical_features:
    X_train_catboost[feature] = X_train_catboost[feature].astype('category')
    X_test_catboost[feature] = X_test_catboost[feature].astype('category')


cat_features_indices = [X_train_catboost.columns.get_loc(col) for col in categorical_features]
print(f"Индексы категориальных признаков для CatBoost: {cat_features_indices}")


print("\nОбучение модели CatBoost...")
catboost_model = cb.CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    verbose=100,  
    cat_features=cat_features_indices,
    random_state=42,
    custom_loss=['F1'],
    eval_metric='F1'
)

catboost_model.fit(X_train_catboost, y_train, eval_set=(X_test_catboost, y_test), plot=True)
y_pred_catboost = catboost_model.predict(X_test_catboost)

print("\nРезультаты модели CatBoost:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_catboost):.4f}")
print("\nОтчет о классификации:")
print(classification_report(y_test, y_pred_catboost))


plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_catboost)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказанные значения')
plt.ylabel('Истинные значения')
plt.title('Матрица ошибок для CatBoost')
plt.savefig('confusion_matrix_catboost.png')
plt.close()


plt.figure(figsize=(12, 8))
feature_importance = catboost_model.feature_importances_
feature_names = X_train_catboost.columns
sorted_idx = np.argsort(feature_importance)
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
plt.title('Важность признаков в модели CatBoost')
plt.savefig('feature_importance_catboost.png')
plt.close()


print("\nОбучение модели Логистической регрессии...")
lr_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("\nРезультаты модели Логистической регрессии:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print("\nОтчет о классификации:")
print(classification_report(y_test, y_pred_lr))





print("\nСравнение моделей:")
models = {
    'Логистическая регрессия': y_pred_lr,
    'CatBoost': y_pred_catboost,
}

results = []
for model_name, predictions in models.items():
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    results.append({
        'Модель': model_name,
        'Accuracy': accuracy,
        'Precision (satisfied)': report['satisfied']['precision'],
        'Recall (satisfied)': report['satisfied']['recall'],
        'F1-score (satisfied)': report['satisfied']['f1-score']
    })

results_df = pd.DataFrame(results)
print(results_df)

plt.figure(figsize=(12, 8))
sns.barplot(x='Модель', y='Accuracy', data=results_df)
plt.title('Сравнение точности моделей')
plt.ylim(0.8, 1.0)  
plt.savefig('model_comparison.png')
plt.close()






print("\n" + "="*50)
print("ВАЖНОСТЬ ПРИЗНАКОВ ПО РЕЗУЛЬТАТАМ CATBOOST:")
print("="*50)


feature_importance = catboost_model.feature_importances_
feature_names = X_train_catboost.columns.tolist()


importance_df = pd.DataFrame({
    'Признак': feature_names,
    'Важность': feature_importance
})


importance_df = importance_df.sort_values('Важность', ascending=False).reset_index(drop=True)


total_importance = importance_df['Важность'].sum()
importance_df['Процент'] = importance_df['Важность'] / total_importance * 100


print("\nТоп-10 наиболее важных признаков:")
print(importance_df.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))