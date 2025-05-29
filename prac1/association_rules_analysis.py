import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

def load_data(file_path, sample_size=1000):
    """Загрузка и подготовка данных"""
    try:
        print("Загрузка данных...")
        df = pd.read_csv(file_path, nrows=sample_size, encoding='utf-8')
        print(f"Размер данных: {df.shape}")
        print(f"Столбцы: {df.columns.tolist()}")
        print(f"Первые 5 строк:\n{df.head()}")
        
        # Преобразование в транзакции
        transactions = []
        for _, row in df.iterrows():
            transaction = []
            for col in df.columns[1:]:  # Пропускаем первый столбец с количеством
                if pd.notna(row[col]) and row[col] != '':
                    transaction.append(row[col])
            transactions.append(transaction)
        
        print(f"Количество транзакций: {len(transactions)}")
        print(f"Пример транзакции: {transactions[0]}")
        return transactions
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return []

def prepare_data(transactions):
    """Подготовка данных для анализа ассоциативных правил"""
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    print(f"Уникальных товаров: {len(te.columns_)}")
    return df_encoded

def find_frequent_itemsets(df_encoded, min_support=0.01):
    """Поиск частых наборов элементов"""
    print(f"\nПоиск частых наборов с минимальной поддержкой: {min_support}")
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    print(f"Найдено частых наборов: {len(frequent_itemsets)}")
    return frequent_itemsets

def generate_rules(frequent_itemsets, min_confidence=0.6):
    """Генерация ассоциативных правил"""
    print(f"\nГенерация правил с минимальной уверенностью: {min_confidence}")
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    # Фильтруем правила с количеством элементов >= 2
    rules = rules[rules['antecedents'].apply(len) + rules['consequents'].apply(len) >= 2]
    
    print(f"Найдено правил: {len(rules)}")
    return rules

def analyze_metrics(rules):
    """Анализ метрик ассоциативных правил"""
    print("\n=== АНАЛИЗ МЕТРИК ===")
    
    print("\nОписание метрик:")
    print("• Support (Поддержка) - частота встречаемости набора элементов")
    print("• Confidence (Доверие) - условная вероятность P(B|A)")  
    print("• Lift (Лифт) - отношение наблюдаемой и ожидаемой поддержки")
    print("• Leverage (Рычаг) - разность наблюдаемой и ожидаемой поддержки")
    print("• Conviction (Убежденность) - мера зависимости правила")
    
    if len(rules) == 0:
        print("Нет правил для анализа")
        return
    
    # Топ-5 правил по разным метрикам
    print("\n=== ТОП-5 ПРАВИЛ ПО УВЕРЕННОСТИ ===")
    top_confidence = rules.nlargest(5, 'confidence')
    for idx, rule in top_confidence.iterrows():
        print(f"{set(rule['antecedents'])} → {set(rule['consequents'])}")
        print(f"  Support: {rule['support']:.3f}, Confidence: {rule['confidence']:.3f}")
        print(f"  Lift: {rule['lift']:.3f}, Leverage: {rule['leverage']:.3f}")
        print(f"  Conviction: {rule['conviction']:.3f}\n")

def main():
    try:
        # Загрузка данных
        file_path = "groc.csv"  # Исправленный путь
        transactions = load_data(file_path, sample_size=2000)
        
        if not transactions:
            print("Не удалось загрузить данные")
            return
        
        # Подготовка данных
        df_encoded = prepare_data(transactions)
        
        # Поиск частых наборов (алгоритм Apriori)
        frequent_itemsets = find_frequent_itemsets(df_encoded, min_support=0.005)
        
        # Генерация ассоциативных правил
        rules = generate_rules(frequent_itemsets, min_confidence=0.3)
        
        # Анализ метрик
        analyze_metrics(rules)
        
        print("\n=== СТАТИСТИКА ===")
        print(f"Обработано транзакций: {len(transactions)}")
        print(f"Найдено частых наборов: {len(frequent_itemsets)}")
        print(f"Сгенерировано правил: {len(rules)}")
        
    except Exception as e:
        print(f"Общая ошибка: {e}")

if __name__ == "__main__":
    main() 