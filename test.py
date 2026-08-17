from pathlib import Path

candidates = Path(__file__).resolve().parent
b = ", ".join(str(path) for path in candidates)
print(b)