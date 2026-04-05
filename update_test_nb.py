import json
import os

filepath = "models/Decoder/test.ipynb"
with open(filepath, "r") as f:
    nb = json.load(f)

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "print(\"=\" * 80)\\n",
        "print(\"VISUALIZING ONE DOCUMENT AND ITS CORRESPONDING ENTITY\")\\n",
        "print(\"=\" * 80)\\n",
        "print()\\n",
        "\\n",
        "# Pick the first entity that has a corresponding text description\\n",
        "sample_ent_uri = next(e for e in entity_list if e in entity_text_dict)\\n",
        "sample_doc_text = list(entities[sample_ent_uri])[0]\\n",
        "ent_text = entity_text_dict[sample_ent_uri]\\n",
        "\\n",
        "print(\"--- 1. DOCUMENT TEXT ---\")\\n",
        "print(\"\\\"\\\"\\\"\")\\n",
        "print(sample_doc_text)\\n",
        "print(\"\\\"\\\"\\\"\")\\n",
        "print()\\n",
        "print(\"--- 2. ENTITY DESCRIPTION ---\")\\n",
        "print(\"\\\"\\\"\\\"\")\\n",
        "print(ent_text)\\n",
        "print(\"\\\"\\\"\\\"\")\\n"
    ]
}

nb["cells"].append(new_cell)

with open(filepath, "w") as f:
    json.dump(nb, f, indent=4)
