import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print(f"BASE_DIR: {BASE_DIR}")


database_path = os.path.abspath(os.path.join(BASE_DIR, 'notebook', 'ShiftData.db'))
data_artifact = os.path.abspath(os.path.join(BASE_DIR, 'data_artifact', 'data'))

target_column = "Shift_efficiency_score"
print(database_path)