import os
import shutil

src_dir = "src"
cpp_dir = os.path.join(src_dir, "cpp")
py_dir = os.path.join(src_dir, "python", "accretive_mas")

# Create directories
os.makedirs(cpp_dir, exist_ok=True)
os.makedirs(py_dir, exist_ok=True)

# Move cpp files
for file in os.listdir(src_dir):
    if file.endswith(".cpp"):
        shutil.move(os.path.join(src_dir, file), os.path.join(cpp_dir, file))

# Move python file
if os.path.exists("accretive_mas_llm.py"):
    shutil.move("accretive_mas_llm.py", os.path.join(py_dir, "llm.py"))

# Create __init__.py
with open(os.path.join(py_dir, "__init__.py"), "w") as f:
    f.write("from ._core import *\nfrom .llm import LatentSemanticAgent, PhysicsOrchestrator\n")

print("Restructuring complete.")
