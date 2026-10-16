"""The manifest: one CSV row per face crop, shared by every dataset adapter and the training notebook."""

import csv
from dataclasses import dataclass
from pathlib import Path

from .errors import ManifestError
from .labels import ATTACK_TYPES, BONA_FIDE

COLUMNS = ("image_path", "attack_type", "subject_id", "video_id", "source_dataset")


@dataclass(frozen=True)
class FaceSample:
    image_path: str
    attack_type: str
    # The person whose identity the capture claims. Splits group by it so a person never lands on
    # both sides; for a face swap it is the target (the person being impersonated).
    subject_id: str
    video_id: str
    source_dataset: str

    @property
    def is_bona_fide(self) -> bool:
        return self.attack_type == BONA_FIDE


def parse_row(row: dict, line: int) -> FaceSample:
    missing = [column for column in COLUMNS if not (row.get(column) or "").strip()]
    if missing:
        raise ManifestError(f"line {line}: empty or missing columns {missing}")
    sample = FaceSample(**{column: row[column].strip() for column in COLUMNS})
    if sample.attack_type not in ATTACK_TYPES:
        raise ManifestError(
            f"line {line}: attack_type {sample.attack_type!r} is not one of {sorted(ATTACK_TYPES)}"
        )
    return sample


def load_manifest(path: str | Path) -> list[FaceSample]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        absent = set(COLUMNS) - set(reader.fieldnames or ())
        if absent:
            raise ManifestError(f"{path}: header lacks columns {sorted(absent)}")
        # Line 1 is the header.
        samples = [parse_row(row, line) for line, row in enumerate(reader, start=2)]
    if not samples:
        raise ManifestError(f"{path}: no samples")
    return samples


def write_manifest(samples: list[FaceSample], path: str | Path) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for sample in samples:
            writer.writerow({column: getattr(sample, column) for column in COLUMNS})
