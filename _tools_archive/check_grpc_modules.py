import os
import importlib.util

def list_finampy_modules():
    print("🔍 Ищем библиотеку FinamPy.grpc...")
    
    # Находим физический путь к установленному модулю
    spec = importlib.util.find_spec("FinamPy.grpc")
    
    if spec is None or not spec.submodule_search_locations:
        print("❌ Модуль FinamPy.grpc не найден. Проверьте установку.")
        return
        
    grpc_path = spec.submodule_search_locations[0]
    print(f"✅ Папка найдена: {grpc_path}\n")
    
    print("📂 Доступные сгенерированные файлы gRPC:")
    print("-" * 40)
    
    # Получаем все .py файлы в папке
    modules = [f for f in os.listdir(grpc_path) if f.endswith('.py') and not f.startswith('__')]
    
    for mod in sorted(modules):
        # Отрезаем расширение .py для красоты
        print(f" 📦 {mod[:-3]}")
        
    print("-" * 40)
    print("\n💡 Чтобы импортировать один из них, используйте:")
    print(f"from FinamPy.grpc.{modules[0][:-3]} import НазваниеКласса")

if __name__ == "__main__":
    list_finampy_modules()