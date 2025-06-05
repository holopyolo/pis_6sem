#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
print("Python version:", sys.version)
print("Current working directory:", __file__)

try:
    import pandas as pd
    print("Pandas импортирован успешно, версия:", pd.__version__)
    
    # Пробуем загрузить файл
    df = pd.read_csv("groc.csv", nrows=3)
    print("Файл загружен успешно!")
    print("Размер:", df.shape)
    print("Первые строки:")
    print(df.head())
    
except Exception as e:
    print("ОШИБКА:", str(e))
    import traceback
    traceback.print_exc()

print("Тест завершен!") 