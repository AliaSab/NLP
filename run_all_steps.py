#!/usr/bin/env python3
"""
Главный скрипт для выполнения всех этапов анализа токенизации
Система анализа текстов и токенизации для русскоязычных новостей
"""
import os
import sys
import subprocess
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_step(step_number: int, script_name: str) -> bool:
    """Запуск отдельного этапа"""
    print(f"\n{'='*60}")
    print(f"ЗАПУСК ЭТАПА {step_number}")
    print(f"{'='*60}")
    
    if not os.path.exists(script_name):
        print(f"❌ Скрипт {script_name} не найден!")
        return False
    
    try:
        result = subprocess.run([sys.executable, script_name], check=True, capture_output=True, text=True)
        print(f"✅ Этап {step_number} выполнен успешно")
        if result.stdout:
            print("Вывод:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка в этапе {step_number}: {e}")
        if e.stdout:
            print("Вывод:")
            print(e.stdout)
        if e.stderr:
            print("Ошибки:")
            print(e.stderr)
        return False
    except KeyboardInterrupt:
        print(f"\n⏹️ Этап {step_number} прерван пользователем")
        return False

def check_dependencies():
    """Проверка зависимостей"""
    print("🔍 Проверка зависимостей...")
    
    required_packages = [
        'requests', 'beautifulsoup4', 'pandas', 'numpy', 
        'plotly', 'matplotlib', 'seaborn', 'streamlit'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Отсутствуют пакеты: {', '.join(missing_packages)}")
        print("💡 Установите их командой: pip install -r requirements.txt")
        return False
    
    print("✅ Все основные зависимости установлены")
    return True

def main():
    """Основная функция"""
    print("🚀 СИСТЕМА АНАЛИЗА ТОКЕНИЗАЦИИ ТЕКСТОВ")
    print("=" * 60)
    print("Этапы:")
    print("1. Формирование экспериментального корпуса текстов")
    print("2. Предварительная обработка и очистка текста")
    print("3. Проектирование универсального модуля предобработки")
    print("4. Сравнительный анализ методов токенизации и нормализации")
    print("5. Обучение подсловных моделей токенизации")
    print("6. Разработка веб-интерфейса для интерактивного анализа")
    print()
    
    # Проверка зависимостей
    if not check_dependencies():
        return
    
    # Запуск этапов
    steps = [
        (1, "step1.py", "Парсинг статей Коммерсанта (ID 8060000-8069999)"),
        (2, "step2.py", "Очистка и предобработка текстов"),
        (3, "step3.py", "Универсальная предобработка с токенами"),
        (4, "step4.py", "Анализ методов токенизации"),
        (5, "step5.py", "Обучение подсловных моделей"),
        (6, "step6.py", "Веб-интерфейс для анализа")
    ]
    
    completed_steps = []
    
    for step_number, script_name, description in steps:
        print(f"\n📋 Этап {step_number}: {description}")
        
        if run_step(step_number, script_name):
            completed_steps.append(step_number)
        else:
            print(f"\n❌ Этап {step_number} завершился с ошибкой")
            break
    
    # Итоговый отчет
    print(f"\n{'='*60}")
    print(f"ИТОГОВЫЙ ОТЧЕТ")
    print(f"{'='*60}")
    
    if len(completed_steps) == len(steps):
        print("🎉 Все этапы выполнены успешно!")
        print("\n📁 Созданные файлы:")
        files = [
            "kommersant_articles.jsonl",
            "kommersant_articles_cleaned.jsonl",
            "kommersant_articles_processed.jsonl", 
            "preprocessing_config.json",
            "tokenization_analysis_results.json",
            "tokenization_analysis_results.csv",
            "subword_models_results.json",
            "subword_models_comparison.csv",
            "corpus.txt"
        ]
        
        for file in files:
            if os.path.exists(file):
                size = os.path.getsize(file)
                print(f"   ✅ {file} ({size:,} байт)")
            else:
                print(f"   ❌ {file} (не найден)")
        
        print(f"\n🌐 Для веб-интерфейса запустите:")
        print(f"   streamlit run step6.py")
        
        print(f"\n📊 Требования к корпусу:")
        print(f"   • Общий объём: не менее 50 000 слов")
        print(f"   • Формат хранения: JSONL")
        print(f"   • Структура: заголовок, текст, дата, URL, категория")
        
    else:
        print(f"⚠️ Выполнено этапов: {len(completed_steps)}/{len(steps)}")
        print(f"✅ Завершенные этапы: {completed_steps}")
        
        if completed_steps:
            print(f"\n💡 Для продолжения запустите:")
            next_step = completed_steps[-1] + 1
            if next_step <= len(steps):
                print(f"   python step{next_step}.py")
    
    print(f"\n📝 Дополнительные команды:")
    print(f"   • Парсинг только: python step1.py")
    print(f"   • Очистка только: python step2.py")
    print(f"   • Предобработка только: python step3.py")
    print(f"   • Анализ токенизации только: python step4.py")
    print(f"   • Обучение моделей только: python step5.py")
    print(f"   • Веб-интерфейс только: streamlit run step6.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n⏹️ Выполнение прервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Критическая ошибка: {e}")
        sys.exit(1)

