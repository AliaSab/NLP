"""Скрипт для проверки созданного корпуса."""

import json
import os

dirs = ['binary_classification', 'multiclass_classification', 'multilabel_classification']
splits = ['train', 'validation', 'test']

print("Проверка созданного корпуса:\n")

for d in dirs:
    print(f"{d}:")
    for s in splits:
        f = f'corpus/{d}/{s}.jsonl'
        if os.path.exists(f):
            with open(f, 'r', encoding='utf-8') as file:
                count = sum(1 for line in file if line.strip())
                print(f"  {s}: {count} строк")
            
            # Показываем пример первой строки
            with open(f, 'r', encoding='utf-8') as file:
                first_line = file.readline()
                if first_line.strip():
                    example = json.loads(first_line)
                    print(f"    Пример: {list(example.keys())}")
    print()


