import json
import sys
from pathlib import Path


path = Path(sys.argv[1])
start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
end = int(sys.argv[3]) if len(sys.argv) > 3 else start + 10

nb = json.loads(path.read_text(encoding="utf-8"))
for i in range(start, min(end, len(nb["cells"]))):
    cell = nb["cells"][i]
    print(f"\n--- CELL {i} {cell['cell_type']} ---")
    print("".join(cell.get("source", [])))
