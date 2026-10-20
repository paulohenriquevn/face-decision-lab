# Changelog

## [Unreleased]

### Added
- Face crop manifest with an attack taxonomy for presentation attacks and deepfakes
- Subject-level and cross-dataset splits that refuse to leak a subject or video across sides
- ISO/IEC 30107-3 metrics: APCER per attack type, BPCER, ACER, BPCER at an APCER limit, AUC and EER
- Dataset build that crops one face per image or video frame to a fixed 256x256 JPEG and reports dropped images per generator
- Source adapters for DeepFakeFace and the AxonData anti-spoofing sample
- Celeb-DF v2 adapter and a `cross_celebdf` protocol that trains on the lab data and tests on the official Celeb-DF v2 test videos
- DiFF adapter for the official test split (T2I, I2I, FS and FE subsets) and a `cross_diff` protocol that tests on it
- DiFF pristine test images (`diff.parse_pristine_member`), built into their own dataset next to the attack side
- DiffusionFace adapter, a `cross_diffusionface` protocol, and `tar_members` to build a sample of a tar archive while streaming it
- FaceForensics++ adapter with the official splits, and the `ffpp`, `ffpp_to_celebdf` and `lab_ffpp_to_celebdf` protocols
- Video-level AUC (mean frame score per video) and the mean score of each generator in every evaluation report
- AUC of each generator against all bona fide samples in every evaluation report, and `auc_by_group` for groups of generators such as DiFF's subsets
- `lab_ffpp_real` protocol: the lab's training side plus the FF++ originals as bona fide video
- `lab_ffpp_real_sr` protocol and the `superresolved` source: copies of the training-side real crops enlarged by a super-resolution network, labeled bona fide
- DF40 adapter for the FF++-domain frames of its 31 known methods and its 9 unknown methods, the `df40_ff` and `lab_df40` protocols, and `auc_within_groups` to score each method against its own reals
- `http_range_reader` and `remote_zip_members` to crop a sample of a remote zip archive from HTTP byte ranges, without downloading it
- `bootstrap_interval` for confidence intervals that resample whole subjects, and `score_files` to save every crop's detector scores
- Face alignment on 5 landmarks with GenD's reference points (`aligned_crop`), selectable per build, and landmarks kept on detected boxes
- `tighten_crop` to reframe a built crop to the face margin a pretrained detector expects
- `lower_resolution` to show a crop as it would look had the face been smaller in the source
- Gated mixture of two detectors by predicted domain (`gated_mixture`, `in_lab_domain`)
- Learned fusion of detector scores (logistic regression over logits, balanced across domains and classes)
- Scoring of the official GenD CLIP ViT-L/14 checkpoint under the lab protocols, alone and fused with the lab detector by the larger of the two scores
- Upload of the build as a private Hugging Face dataset, and training from it
- Evaluation broken down by generator as well as by attack type
- Shared evaluation protocol (split, test subset, APCER limit) so every model is scored on the same subjects
- CLIP ViT-L/14 linear probe baseline
- Unseen-generator protocols that train without one manipulation method and test only on it
- Option to train LoRA on the vision tower of the decision model
- LayerNorm-tuned CLIP detector trained on paired real and fake batches with the GenD loss (cross entropy, alignment and uniformity) and augmentation, checkpointed every epoch
- Decision model training pushes checkpoints to a private Hub repository and resumes from the last one
- Upload of the trained adapter and its evaluation report to a private Hugging Face model repository
- Smoke mode that runs the whole training pipeline on generated noise images, with no face data
- Colab notebook that trains, calibrates and evaluates a Qwen3.5-4B decision model on face crops
