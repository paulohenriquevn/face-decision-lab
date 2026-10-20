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

## Unseen test set: Celeb-DF v2

`notebooks/build_celebdf_test.ipynb` crops the 518 official Celeb-DF v2 test videos
(`List_of_testing_videos.txt`: 178 real, 340 synthesized) with the same pipeline, 16 frames per
video, into a separate private dataset. Nothing trains on it. It is the test side of the
`cross_celebdf` protocol in `face_decision_lab.protocol`.

| Folder | Label | Subject |
|---|---|---|
| `Celeb-real/id<N>_<clip>.mp4` | `bona_fide`, generator `celeb_real` | `celebdf_id<N>` |
| `YouTube-real/<clip>.mp4` | `bona_fide`, generator `youtube_real` | one per video |
| `Celeb-synthesis/id<T>_id<S>_<clip>.mp4` | `face_swap`, generator `celebdf_synthesis` | `celebdf_id<T>`, the person in the target clip |

`ALIGN = True` builds a second private dataset (`face-decision-celebdf-test-aligned`) the way
GenD's authors crop: 32 frames per video, each face warped onto five reference landmarks with a
1.3x margin (`faces.aligned_crop`, a port of their MIT-licensed `align_face`).

The test list labels real videos 1 and fakes 0; the adapter checks each label against the folder
the video sits in. Access comes from the authors' form, for non-commercial research only, and the
archive stays in the researcher's own Google Drive.

## Unseen test set: DiFF

`notebooks/build_diff_test.ipynb` crops the official test split of DiFF (Diffusion Facial Forgery,
ACM MM 2024) into a separate private dataset, the test side of the `cross_diff` protocol. Nothing
trains on it. The synthesized images are public on the authors' Google Drive, one zip per method;
the notebook downloads them one at a time. In October 2026 the 13 test zips held 57,060 images of
143 identities.

| Subset | Methods (zip folder) | Label |
|---|---|---|
| T2I, text to image | `Midjourney`, `SDXL`, `FreeDoM_T`, `HPS` | `fully_synthetic` |
| I2I, image to image | `SDXL_Refine`, `LoRA`, `DreamBooth`, `FreeDoM_I` | `fully_synthetic` |
| FS, face swapping | `DiffFace`, `DCFace` | `face_swap` |
| FE, face editing | `Imagic`, `CoDiff`, `cycle_diff` | `face_reenactment` |

Every member sits under `<method>/id_<number>/`; the generator is `diff_<subset>_<method>` (for
example `diff_t2i_sdxl`) and the subject is `diff_id_<number>`. The lab's taxonomy has no face
editing class, so FE maps to `face_reenactment`, the class for a real face changed in expression or
attributes. The LoRA zip carries macOS resource forks, which the adapter skips.

The pristine images, the bona fide side, come only through the authors' access form, for academic
research and without redistribution. Their layout is not in the adapter yet; it fails on the first
pristine member until it is.

## Unseen test set: DiffusionFace

`notebooks/build_diffusionface_test.ipynb` crops a fixed sample of DiffusionFace part 1 (Chen et
al., arXiv 2403.18471) into a separate private dataset, the test side of the
`cross_diffusionface` protocol. Nothing trains on it. Part 1 is open on Zenodo (record 10865300,
CC BY 4.0, research use only per the authors): one tar of about 3 GB per folder, each holding
`<folder>/<number>.png` at 256x256.

| Folder | Label |
|---|---|
| `Real` (CelebA-HQ photos) | `bona_fide` |
| `ADM`, `DDIM`, `DDPM`, `PNDM`, `LDM` | `fully_synthetic` |
| `SDv15_DS{0.3,0.5,0.7}`, `SDv21_DS{0.3,0.5,0.7}` (image to image) | `fully_synthetic` |
| `DiffSwap`, `Inpaint` (images in `Inpaint/images/`) | `face_swap` |

An image's number is the number of the real photo it was made from, so the subject is
`diffusionface_<number>` and a fake shares it with its source photo. The unconditional
generators reuse the numbering without depicting the same person. `MM_CelebA.tar` holds no images,
only the captions behind the text prompts, so the build skips it; so are Inpaint's masks and JSON
files and DiffSwap's `target_source.csv`. The authors publish no test
split; `diffusionface.sampled` keeps about one number in 20, the same numbers in every folder,
and the notebook downloads every tar in parallel (Zenodo served about 0.6 MB/s per connection) before cropping them.

## FaceForensics++

`notebooks/build_faceforensics.ipynb` downloads the c23 videos of the 1,000 originals and the four
manipulation methods of the paper with the authors' script, crops 16 frames per video with the same
pipeline, and uploads them to a separate private dataset. The official train/val/test splits
(360/70/70 pairs of sequences) ship with the package in `sources/faceforensics_splits.json`
(MIT, from the authors' repository).

| Path | Label | Subject |
|---|---|---|
| `original_sequences/youtube/c23/videos/<target>.mp4` | `bona_fide`, generator `ffpp_original` | `ffpp_<target>` |
| `manipulated_sequences/{Deepfakes,FaceSwap,FaceShifter}/c23/videos/<target>_<source>.mp4` | `face_swap` | `ffpp_<target>` |
| `manipulated_sequences/{Face2Face,NeuralTextures}/c23/videos/<target>_<source>.mp4` | `face_reenactment` | `ffpp_<target>` |

A manipulated video keeps the target video's body and background, so its subject is the target, and
batches keep an original with every manipulation of it (`pairs.pair_key`). The protocols use the
four original methods only; the adapter also reads FaceShifter. The download URL arrives by email
after the access form and lives in a Colab secret, never in this repository.

Unofficial copies of FaceForensics++, Celeb-DF and CelebA-Spoof on the Hub are left out: they skip
the access agreements of the original authors.

## DF40

`notebooks/build_df40.ipynb` crops DF40 (Yan et al., arXiv 2406.13495; CC BY-NC 4.0, non-commercial research only, access form per researcher) into a private dataset with
source_dataset `df40` and generator `df40_<method>` (`sources/df40.py`). The authors ship frames
they already cropped; the build detects the face again and crops it at the lab's margin.

| Part | Archive layout | Label | Subject | Kept |
|---|---|---|---|---|
| Known face swap and reenactment (21 methods) | `<method>/frames/<target>_<source>/<n>.png` | `face_swap` or `face_reenactment` | `ffpp_<target>` | FF++ train and val sequences, one frame in 16 |
| Known entire-face synthesis (10 methods) | `<method>/<sequence>/<name>.png` | `fully_synthetic` | `ffpp_<sequence>` | same |
| FF++ real frames | `FaceForensics++/original_sequences/youtube/c23/frames/<sequence>/<n>.png` | `bona_fide`, generator `df40_ffpp_real` | `ffpp_<sequence>` | all |
| Unknown methods (9) | `<method>/{fake,real}/...` | the method's attack type; its reals `bona_fide`, generator `df40_<method>_real` | one per image or clip | all, e4e only for FF++ test sequences |

What the authors' files hold differs from what their email says:

- The folder the email calls "testing data" holds the FF++-domain training frames (for example
  the pair 402_453, which the authors' JSON lists as training); the folder it calls "training
  data" holds the FF++-domain test frames, the Celeb-DF domain and the unknown methods.
- The link to the Celeb-DF real frames points to a script, so the Celeb-DF domain is left out. It
  would also put Celeb-DF identities into training, and Celeb-DF is the lab's video benchmark.
- MidJourney ships 247 of its 882 fakes as GIF, which OpenCV cannot decode; they are skipped.
- One training archive (ddim in the "training data" folder) is not a valid zip; the build only
  uses that folder for the unknown methods.

Face editing (e4e, StarGAN, StarGAN v2, StyleCLIP, CollabDiff) maps to `face_reenactment`, as
DiFF's FE subset does. The unknown methods' reals differ from their fakes in size and format
(MidJourney: 1024 px PNG fakes, 256 px JPEG reals); the lab's crop resizes both to 256 px JPEG, but
the detail lost or kept in that resize still differs, so their AUCs carry that difference, as the
published ones do.

`df40_ff` trains on the known methods (training sequences; validation sequences calibrate) and
tests on the unknown methods; `auc_within_groups` with `df40.unknown_group` scores each against
its own reals. `lab_df40` adds the same DF40 sides to `lab_ffpp_real`.

## Super-resolved real crops

`notebooks/build_superresolved.ipynb` takes every real crop on the training side of
`lab_ffpp_real` (DeepFakeFace photos and FaceForensics++ original frames), shrinks it 4x and
enlarges it back with Real-ESRGAN x4plus (BSD-3-Clause) or Swin2SR real-world x4 (Apache-2.0),
half the crops each, and uploads the copies to a separate private dataset. Each copy is labeled
`bona_fide` with source_dataset `superresolved` and generator `sr_<network>`, and keeps the
subject and pair group of its original (`sources/superresolved.py`). The copies inherit the terms
of the crops they come from.

Only `lab_ffpp_real_sr` uses them, and only for originals on its training side; every other
protocol leaves them out. They test the diagnosis that the lab's detectors flag DiffusionFace's
CelebA-HQ photos because a super-resolution network made them (`docs/results.md`).

## Datasets to add through their authors

All of them are research-only and need an access request to their authors. Read each license
before downloading. None of them may train a commercial model.

| Dataset | Attacks it covers | Label mapping |
|---|---|---|
| OULU-NPU | print, replay | access type 1 → `bona_fide`; 2 and 3 → `print`; 4 and 5 → `replay` |
| CelebA-Spoof | print, replay, mask | photo, poster, A4 → `print`; PC, pad, phone → `replay`; masks → `mask` |

The mappings above come from the datasets' published descriptions. Confirm them against the files
you download before writing the adapter.

No public dataset covers injection through the capture channel (virtual cameras, emulators, hooked
drivers) at a useful size. A model trained on these datasets sees the injected content, not the
channel, so it cannot flag a genuine video of the victim replayed through a virtual camera.
