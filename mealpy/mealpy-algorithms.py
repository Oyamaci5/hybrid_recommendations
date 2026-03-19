import mealpy
import inspect
import pkgutil

# Tüm algoritmaları listele
algorithms = []
for importer, modname, ispkg in pkgutil.walk_packages(
    path=mealpy.__path__,
    prefix=mealpy.__name__ + '.',
    onerror=lambda x: None
):
    try:
        module = __import__(modname, fromlist="dummy")
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if hasattr(obj, 'solve'):
                algorithms.append({
                    'module': modname,
                    'class': name
                })
    except:
        pass

# DataFrame olarak gör
import pandas as pd
df = pd.DataFrame(algorithms).drop_duplicates()
print(f"Toplam algoritma sayısı: {len(df)}")
print(df.to_string())