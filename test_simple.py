print("Python работает!")
import pandas as pd
print("Pandas импортирован успешно")

# Проверяем наличие файла
import os
if os.path.exists("groceries - groceries.csv"):
    print("Файл groceries найден!")
    df = pd.read_csv("groceries - groceries.csv", nrows=5)
    print("Первые 5 строк:")
    print(df.head())
else:
    print("Файл groceries не найден")
    print("Содержимое директории:")
    for file in os.listdir("."):
        print(f"  {file}") 