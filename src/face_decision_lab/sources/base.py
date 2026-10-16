"""What an adapter knows about one source file before any face is cropped from it."""

from dataclasses import dataclass

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
VIDEO_EXTENSIONS = (".mp4", ".mov")


class SourceLayoutError(ValueError):
    """A source file does not match the layout its adapter expects."""


@dataclass(frozen=True)
class SourceItem:
    # Path of the file inside the source (zip member or repository path).
    path: str
    attack_type: str
    subject_id: str
    video_id: str
    source_dataset: str
    generator: str

    @property
    def is_video(self) -> bool:
        return self.path.lower().endswith(VIDEO_EXTENSIONS)
