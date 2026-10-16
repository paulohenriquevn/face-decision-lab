# Data

The training notebook reads one manifest: a CSV with one row per face crop.

| Column | Meaning |
|---|---|
| `image_path` | Crop path, relative to the data directory |
| `attack_type` | One of `bona_fide`, `print`, `replay`, `mask`, `face_swap`, `face_reenactment`, `fully_synthetic` |
| `subject_id` | The person the capture claims to be. For a face swap, the target being impersonated |
| `video_id` | The video the frame came from. Each video belongs to one subject |
| `source_dataset` | Short dataset name, such as `ffpp` or `oulu_npu`. Cross-dataset splits use it |

`subject_id` and `video_id` exist to stop leakage. Frames of one video look almost the same, so a
model tested on frames from a video it trained on scores well without having learned anything
useful. `face_decision_lab.splits` keeps each subject on one side.

## Cropping faces

`face_decision_lab.extract.extract_face_crops` samples evenly spaced frames from a video and saves
one aligned face crop per frame with DeepFace (`pip install ".[extract]"`). It skips frames with no
face or with several faces and counts them in the returned report. Check those counts: a dataset
where most frames are skipped needs a different detector backend.

## Candidate datasets

All of them are research-only and need an access request to their authors. Read each license
before downloading. None of them may train a commercial model.

| Dataset | Attacks it covers | Label mapping |
|---|---|---|
| FaceForensics++ | face swap, reenactment | `original` → `bona_fide`; Deepfakes, FaceSwap, FaceShifter → `face_swap`; Face2Face, NeuralTextures → `face_reenactment` |
| Celeb-DF v2 | face swap | `Celeb-real`, `YouTube-real` → `bona_fide`; `Celeb-synthesis` → `face_swap` |
| DF40 | 40 forgery methods | per method; check the dataset's method list |
| OULU-NPU | print, replay | access type 1 → `bona_fide`; 2 and 3 → `print`; 4 and 5 → `replay` |
| CelebA-Spoof | print, replay, mask | photo, poster, A4 → `print`; PC, pad, phone → `replay`; masks → `mask` |

The mappings above come from the datasets' published descriptions. Confirm them against the files
you download before writing the adapter.

No public dataset covers injection through the capture channel (virtual cameras, emulators, hooked
drivers) at a useful size. A model trained on these datasets sees the injected content, not the
channel, so it cannot flag a genuine video of the victim replayed through a virtual camera.
