import ast
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

NOTEBOOK = BASE_DIR / "LONG_TERM_SERIES_FORECASTING_ETT.ipynb"
OUTPUT = BASE_DIR / "backend" / "models.py"

# All model classes required by the application
required_classes = [
    "MovingAvg",
    "FourierBlock",
    "FEDformer",
    "PatchEmbedding",
    "PositionalEncoding",
    "PatchTST",
    "ProbSparseAttention",
    "InformerEncoderLayer",
    "Informer",
    "TransformLayer",
    "ExponentialSmoothingLayer",
    "ETSformerEncoderLayer",
    "ETSformer",
    "DSWEmbedding",
    "TwoStageAttention",
    "Crossformer",
]

if not NOTEBOOK.exists():
    raise FileNotFoundError(f"Notebook not found: {NOTEBOOK}")

with NOTEBOOK.open("r", encoding="utf-8") as f:
    notebook = json.load(f)

extracted = {}

# Search every code cell for the required classes
for cell in notebook["cells"]:

    if cell["cell_type"] != "code":
        continue

    source = "".join(cell["source"])

    try:
        tree = ast.parse(source)
    except SyntaxError:
        continue

    lines = source.splitlines()

    for node in tree.body:

        if isinstance(node, ast.ClassDef):
            if node.name in required_classes:

                if node.name in extracted:
                    raise RuntimeError(
                        f"Duplicate class found: {node.name}"
                    )

                class_code = "\n".join(
                    lines[node.lineno - 1:node.end_lineno]
                )

                extracted[node.name] = class_code

# Check that every required class was found
missing = [
    name for name in required_classes
    if name not in extracted
]

if missing:
    raise RuntimeError(
        f"Missing model classes: {missing}"
    )

# Write classes in dependency order
output_parts = [
    '"""Model classes extracted from the forecasting notebook."""',
    "",
    "import math",
    "import numpy as np",
    "import torch",
    "import torch.nn as nn",
    "",
]

for name in required_classes:
    output_parts.append(extracted[name])
    output_parts.append("")


OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(
    "\n".join(output_parts),
    encoding="utf-8"
)

print("Successfully generated backend/models.py")
print(f"Extracted {len(extracted)} classes:")

for name in required_classes:
    print("-", name)
