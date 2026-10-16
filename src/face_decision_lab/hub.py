"""The training dataset on the Hugging Face Hub: one row per face crop, manifest columns plus the image."""

import json
from collections import Counter
from pathlib import Path

from .manifest import COLUMNS, FaceSample, load_manifest

FIELDS = tuple(column for column in COLUMNS if column != "image_path")

CARD = """---
license: other
tags:
- face-anti-spoofing
- deepfake-detection
pretty_name: face-decision-lab training crops
---

# face-decision-lab training crops

Face crops in the format the [face-decision-lab](https://github.com/paulohenriquevn/face-decision-lab)
training notebook reads. Every crop is a 256x256 JPEG (quality 95) around one detected face with
25% margin, so resolution and compression carry no hint of the label.

**Private on purpose.** The crops come from third-party datasets and show real people's faces,
which are biometric data. Do not make this repository public or share the files.

| Source | License | What it contributes |
|---|---|---|
| [OpenRL/DeepFakeFace](https://huggingface.co/datasets/OpenRL/DeepFakeFace) | OpenRAIL | real IMDB-WIKI photos, InsightFace swaps, SD inpainting, SD 1.5 images |
| [AxonData/face-anti-spoofing-dataset](https://huggingface.co/datasets/AxonData/face-anti-spoofing-dataset) | CC BY-NC 4.0 | selfies, replay and mask attacks (small sample) |

Columns: `image`, {columns}.

## Rows

{counts}

## Face detection outcomes per generator

Images where RetinaFace found no face, or two faces of similar size, were dropped. Compare the
saved share across generators before trusting a result: a detector that drops one class more than
another biases the data.

```json
{report}
```
"""


def dataset_rows(samples: list[FaceSample], root: str | Path) -> dict[str, list]:
    """Column-oriented rows for datasets.Dataset.from_dict, with the image as a file path."""
    root = Path(root)
    columns = {"image": [str(root / sample.image_path) for sample in samples]}
    for field in FIELDS:
        columns[field] = [getattr(sample, field) for sample in samples]
    return columns


def dataset_card(samples: list[FaceSample], report: dict) -> str:
    counts = Counter((sample.source_dataset, sample.generator, sample.attack_type) for sample in samples)
    table = ["| source_dataset | generator | attack_type | rows |", "|---|---|---|---|"]
    table += [f"| {s} | {g} | {a} | {n} |" for (s, g, a), n in sorted(counts.items())]
    return CARD.format(
        columns=", ".join(f"`{field}`" for field in FIELDS),
        counts="\n".join(table),
        report=json.dumps(report, indent=2, sort_keys=True),
    )


def push_build(output_dir: str | Path, repo_id: str) -> str:
    """Uploads a finished build as a private dataset and returns the commit URL."""
    from datasets import Dataset, Image
    from huggingface_hub import HfApi

    output_dir = Path(output_dir)
    samples = load_manifest(output_dir / "manifest.csv")
    report = json.loads((output_dir / "report.json").read_text())
    dataset = Dataset.from_dict(dataset_rows(samples, output_dir)).cast_column("image", Image())
    dataset.push_to_hub(repo_id, private=True, max_shard_size="500MB")
    commit = HfApi().upload_file(
        path_or_fileobj=dataset_card(samples, report).encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Describe sources, licenses and detection outcomes",
    )
    return commit.commit_url


def load_hub_samples(repo_id: str, revision: str | None = None) -> tuple[list[FaceSample], list[dict]]:
    """Samples and their undecoded images ({"bytes", "path"} dicts, which Unsloth accepts)."""
    from datasets import Image, load_dataset

    dataset = load_dataset(repo_id, split="train", revision=revision).cast_column(
        "image", Image(decode=False)
    )
    samples = [
        FaceSample(image_path=f"hub:{index}", **{field: row[field] for field in FIELDS})
        for index, row in enumerate(dataset.select_columns(list(FIELDS)))
    ]
    return samples, dataset["image"]
