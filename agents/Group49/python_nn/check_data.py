import json
from pathlib import Path

print("Checking files for corruption...")
for path in Path("data").glob("*.jsonl"):
    print(f"Scanning {path.name}...", end="\r")
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                print(f"\n[!] CORRUPTION FOUND in {path.name} at line {line_num}")
                print(f"    Error: {e}")
                print(f"    Snippet: {line[:100]}...")
print("\nScan complete.")