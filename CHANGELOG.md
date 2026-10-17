# Changelog

## [Unreleased]

### Added
- Face crop manifest with an attack taxonomy for presentation attacks and deepfakes
- Subject-level and cross-dataset splits that refuse to leak a subject or video across sides
- ISO/IEC 30107-3 metrics: APCER per attack type, BPCER, ACER, BPCER at an APCER limit, AUC and EER
- Dataset build that crops one face per image or video frame to a fixed 256x256 JPEG and reports dropped images per generator
- Source adapters for DeepFakeFace and the AxonData anti-spoofing sample
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
