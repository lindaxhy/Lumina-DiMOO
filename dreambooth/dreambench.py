"""Canonical DreamBench subjects and prompt templates."""

from __future__ import annotations

import csv
from pathlib import Path

OBJECT_PROMPTS = [
    "a {token} {class_name} in the jungle",
    "a {token} {class_name} in the snow",
    "a {token} {class_name} on the beach",
    "a {token} {class_name} on a cobblestone street",
    "a {token} {class_name} on top of pink fabric",
    "a {token} {class_name} on top of a wooden floor",
    "a {token} {class_name} with a city in the background",
    "a {token} {class_name} with a mountain in the background",
    "a {token} {class_name} with a blue house in the background",
    "a {token} {class_name} on top of a purple rug in a forest",
    "a {token} {class_name} with a wheat field in the background",
    "a {token} {class_name} with a tree and autumn leaves in the background",
    "a {token} {class_name} with the Eiffel Tower in the background",
    "a {token} {class_name} floating on top of water",
    "a {token} {class_name} floating in an ocean of milk",
    "a {token} {class_name} on top of green grass with sunflowers around it",
    "a {token} {class_name} on top of a mirror",
    "a {token} {class_name} on top of the sidewalk in a crowded street",
    "a {token} {class_name} on top of a dirt road",
    "a {token} {class_name} on top of a white rug",
    "a red {token} {class_name}",
    "a purple {token} {class_name}",
    "a shiny {token} {class_name}",
    "a wet {token} {class_name}",
    "a cube shaped {token} {class_name}",
]

LIVE_PROMPTS = OBJECT_PROMPTS[:10] + [
    "a {token} {class_name} wearing a red hat",
    "a {token} {class_name} wearing a santa hat",
    "a {token} {class_name} wearing a rainbow scarf",
    "a {token} {class_name} wearing a black top hat and a monocle",
    "a {token} {class_name} in a chef outfit",
    "a {token} {class_name} in a firefighter outfit",
    "a {token} {class_name} in a police outfit",
    "a {token} {class_name} wearing pink glasses",
    "a {token} {class_name} wearing a yellow shirt",
    "a {token} {class_name} in a purple wizard outfit",
] + OBJECT_PROMPTS[-5:]

LIVE_CLASSES = {"dog", "cat"}


def load_subjects(classes_file: Path) -> dict[str, str]:
    lines = classes_file.read_text().splitlines()
    start = lines.index("subject_name,class")
    rows = []
    for line in lines[start + 1 :]:
        if not line.strip():
            break
        rows.append(line)
    return {row["subject_name"]: row["class"] for row in csv.DictReader([lines[start], *rows])}


def prompts_for(class_name: str, token: str = "sks") -> list[str]:
    templates = LIVE_PROMPTS if class_name in LIVE_CLASSES else OBJECT_PROMPTS
    return [template.format(token=token, class_name=class_name) for template in templates]
