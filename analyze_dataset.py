import pandas as pd

# Загружаем датасет
df = pd.read_csv('healthcare_dataset.csv')

print('Размер датасета:', df.shape)
print('\nПервые 5 строк:')
print(df.head())
print('\nИнформация о столбцах:')
print(df.info())
print('\nУникальные значения в каждом столбце:')
print(df.nunique())
print('\nПропуски:')
print(df.isnull().sum())
print('\nСтатистика числовых столбцов:')
print(df.describe()) 