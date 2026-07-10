import json
import sys
from pathlib import Path


path = Path(sys.argv[1])
nb = json.loads(path.read_text(encoding="utf-8"))
code_text = "\n".join(
    "".join(cell.get("source", []))
    for cell in nb["cells"]
    if cell.get("cell_type") == "code"
)

for term in ["AU_NAMES", "EmojiHero", "DANN", "keras-tcn", "DataType"]:
    print(term, term in code_text)

for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        compile("".join(cell.get("source", [])), f"cell_{i}", "exec")

print("cells", len(nb["cells"]))
print("syntax ok")
