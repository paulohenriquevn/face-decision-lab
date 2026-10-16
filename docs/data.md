# Data

Training reads a private Hugging Face dataset with one row per face crop. `notebooks/build_dataset.ipynb`
builds it on a Colab GPU from the sources below and uploads it.

| Column | Meaning |
|---|---|
| `image` | 256x256 JPEG crop around one face |
| `attack_type` | One of `bona_fide`, `print`, `replay`, `mask`, `face_swap`, `face_reenactment`, `fully_synthetic` |
| `subject_id` | The person the capture claims to be. For a face swap, the target being impersonated |
| `video_id` | The video or photo the crop came from. Each one belongs to a single subject |
| `source_dataset` | Short dataset name, such as `deepfakeface` or `axondata` |
| `generator` | What produced the capture: a manipulation method (`insight`, `text2img`) or an attack instrument (`latex_mask`) |

The smoke run and local tools use the same columns in a CSV manifest, with `image_path` (relative to
the data directory) in place of `image`.

`subject_id` and `video_id` exist to stop leakage. Frames of one video look almost the same, so a
model tested on frames from a video it trained on scores well without having learned anything
useful. `face_decision_lab.splits` keeps each subject on one side.

## Face crops

Every image and every sampled video frame (16 per video, evenly spaced) goes through RetinaFace and
the same crop: a square around the face with 25% margin, padded with black past the image border,
resized to 256x256 and saved as JPEG quality 95.

The fixed size and encoding matter. In DeepFakeFace every fake is 512x512 and the real photos come
in their original sizes, so uncropped images would let a model separate the classes by resolution.

An image is dropped when RetinaFace finds no face, or when the largest face is less than twice the
area of the second (two people, no way to tell which one the label is about). The build counts drops
per generator in `report.json` and on the dataset card. Compare the saved share across generators:
a detector that misses one class more often biases the data.

## Sources in the dataset

| Source | License | Mapping |
|---|---|---|
| [OpenRL/DeepFakeFace](https://huggingface.co/datasets/OpenRL/DeepFakeFace) | OpenRAIL | `wiki` → `bona_fide`; `insight`, `inpainting` → `face_swap`; `text2img` → `fully_synthetic` |
| [AxonData/face-anti-spoofing-dataset](https://huggingface.co/datasets/AxonData/face-anti-spoofing-dataset) | CC BY-NC 4.0 | selfies → `bona_fide`; screen and phone replays → `replay`; paper, cutout, latex, silicone and textile masks → `mask` |

DeepFakeFace subjects come from the IMDB-WIKI person id in each path, so a person's real photo and
the three fakes made from it share a subject. Its real photos are press and red-carpet photos, not
live selfies, and all its manipulations come from one paper, so a test split from it measures
generalization to new people, not to new generators or datasets.

AxonData is a free sample of a commercial dataset: a few dozen videos per instrument. Its card does
not say which participant appears in which file, so most files are their own subject. Treat its
metrics as a format check. Its two CC BY 4.0 sub-datasets (`3D_paper_mask_attack_dataset_for_Liveness`
and `Anti_Spoofing_Cut_print_attack`) hold the same files as folders of this one, so they are not
added again.

Unofficial copies of FaceForensics++, Celeb-DF and CelebA-Spoof on the Hub are left out: they skip
the access agreements of the original authors.

## Datasets to add through their authors

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
