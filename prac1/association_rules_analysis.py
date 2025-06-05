import pandas as pd
import numpy as np
from itertools import combinations

def custom_apriori(transactions, min_support):    
    all_items = set()
    for transaction in transactions:
        all_items.update(transaction)
    all_items = list(all_items)
    print(f"Всего уникальных товаров: {len(all_items)}")
    
    
    item_counts = {}
    for item in all_items:
        count = sum(1 for transaction in transactions if item in transaction)
        support = count / len(transactions)
        if support >= min_support:
            item_counts[frozenset([item])] = support
    
    print(f"Частые 1-элементные наборы: {len(item_counts)}")
    
    
    frequent_itemsets = {1: item_counts}
    k = 2
    
    
    while frequent_itemsets[k-1]:
        print(f"Поиск {k}-элементных наборов...")
        
        
        candidates = generate_candidates(list(frequent_itemsets[k-1].keys()), k)
        
        
        candidate_counts = {}
        for candidate in candidates:
            count = sum(1 for transaction in transactions 
                       if candidate.issubset(set(transaction)))
            support = count / len(transactions)
            if support >= min_support:
                candidate_counts[candidate] = support
        
        if candidate_counts:
            frequent_itemsets[k] = candidate_counts
            print(f"Найдено {k}-элементных наборов: {len(candidate_counts)}")
        else:
            print(f"Нет частых {k}-элементных наборов")
            break
        
        k += 1
    
    
    all_frequent = {}
    for level_itemsets in frequent_itemsets.values():
        all_frequent.update(level_itemsets)
    
    print(f"Всего найдено частых наборов: {len(all_frequent)}")
    return all_frequent

def generate_candidates(frequent_itemsets, k):
    
    candidates = []
    n = len(frequent_itemsets)
    
    for i in range(n):
        for j in range(i+1, n):
            
            union_set = frequent_itemsets[i].union(frequent_itemsets[j])
            if len(union_set) == k:
                candidates.append(union_set)
    
    return candidates

def custom_generate_rules(frequent_itemsets, min_confidence):
    
    print(f"\n=== ГЕНЕРАЦИЯ ПРАВИЛ (min_confidence={min_confidence}) ===")
    
    rules = []
    
    
    for itemset, support in frequent_itemsets.items():
        if len(itemset) < 2:
            continue
            
        
        items = list(itemset)
        for i in range(1, len(items)):
            for antecedent in combinations(items, i):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                
                
                antecedent_support = frequent_itemsets.get(antecedent, 0)
                if antecedent_support > 0:
                    confidence = support / antecedent_support
                    
                    if confidence >= min_confidence:
                        
                        consequent_support = frequent_itemsets.get(consequent, 0)
                        lift = confidence / consequent_support if consequent_support > 0 else 0
                        leverage = support - (antecedent_support * consequent_support)
                        conviction = (1 - consequent_support) / (1 - confidence) if confidence < 1 else float('inf')
                        
                        rules.append({
                            'antecedent': antecedent,
                            'consequent': consequent,
                            'support': support,
                            'confidence': confidence,
                            'lift': lift,
                            'leverage': leverage,
                            'conviction': conviction
                        })
    
    print(f"Сгенерировано правил: {len(rules)}")
    return rules

def analyze_rules(rules):
    
    print("\n=== АНАЛИЗ МЕТРИК ===")
    
    if not rules:
        print("Нет правил для анализа")
        return
    
    
    rules_sorted = sorted(rules, key=lambda x: x['confidence'], reverse=True)
    
    print("\n=== ТОП-5 ПРАВИЛ ПО УВЕРЕННОСТИ ===")
    for i, rule in enumerate(rules_sorted[:5], 1):
        print(f"{i}. {set(rule['antecedent'])} → {set(rule['consequent'])}")
        print(f"   Support: {rule['support']:.3f}")
        print(f"   Confidence: {rule['confidence']:.3f}")
        print(f"   Lift: {rule['lift']:.3f}")
        print(f"   Leverage: {rule['leverage']:.3f}")
        print(f"   Conviction: {rule['conviction']:.3f}\n")
        
    
    print("=== ТОП-3 ПРАВИЛА ПО ЛИФТУ ===")
    rules_by_lift = sorted(rules, key=lambda x: x['lift'], reverse=True)
    for i, rule in enumerate(rules_by_lift[:3], 1):
        print(f"{i}. {set(rule['antecedent'])} → {set(rule['consequent'])}")
        print(f"   Lift: {rule['lift']:.3f}, Confidence: {rule['confidence']:.3f}\n")

def load_data(file_path, sample_size=1000):
    
    try:
        print("Загрузка данных...")
        df = pd.read_csv(file_path, nrows=sample_size)
        print(f"Размер данных: {df.shape}")
        print(f"Столбцы: {df.columns.tolist()}")
        print(f"Первые 5 строк:\n{df.head()}")
        
        
        transactions = []
        for _, row in df.iterrows():
            transaction = []
            for col in df.columns[1:]:  
                if pd.notna(row[col]) and row[col] != '':
                    transaction.append(row[col])
            transactions.append(transaction)
        
        print(f"Количество транзакций: {len(transactions)}")
        print(f"Пример транзакции: {transactions[0]}")
        return transactions
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        import traceback
        traceback.print_exc()
        return []

def main():
    try:
        
        file_path = "groc.csv"
        transactions = load_data(file_path, sample_size=1000)
        
        if not transactions:
            print("Не удалось загрузить данные")
            return
        
        
        print("\n" + "="*50)
        print("АНАЛИЗ АССОЦИАТИВНЫХ ПРАВИЛ АЛГОРИТМОМ APRIORI")
        print("="*50)
        
        frequent_itemsets = custom_apriori(transactions, min_support=0.01)
        rules = custom_generate_rules(frequent_itemsets, min_confidence=0.5)
        analyze_rules(rules)
        
        
        
    except Exception as e:
        print(f"Общая ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 