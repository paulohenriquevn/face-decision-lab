# Changelog

## [Unreleased]

### Added
- Face crop manifest with an attack taxonomy for presentation attacks and deepfakes
- Subject-level and cross-dataset splits that refuse to leak a subject or video across sides
- ISO/IEC 30107-3 metrics: APCER per attack type, BPCER, ACER, BPCER at an APCER limit, AUC and EER
- Face crop extraction from videos with DeepFace
- Smoke mode that runs the whole training pipeline on generated noise images, with no face data
- Colab notebook that trains, calibrates and evaluates a Qwen3.5-4B decision model on face crops
