# DReAM-SAM
Repository for source code submission (Anonymous Version)

## Abstract
The Segment Anything Model (SAM) has demonstrated unprecedented zero-shot segmentation capabilities across general visual domains. However, its effectiveness in Medical Image Segmentation (MIS) remains significantly constrained, particularly when dealing with small, ill-defined shapes, and low-contrast with ambiguous boundaries, due to insufficient modeling of local context and a lack of awareness of surrounding anatomical relationships. In this paper, we propose Dynamic Region-Adaptive Modulated SAM (DReAM-SAM), a novel framework designed to enhance SAM’s spatial adaptability and fidelity for fine-grained MIS. Our framework introduces three key technical pillars: (1) Dynamically Adaptive Modulation: We integrate an agile CNN branch to complement SAM’s encoder, which enforces fine structural details and boundary-sensitive localization via dynamically adaptive feature modulation across regions. (2) Bidirectional Feature Unification: We develop a reciprocal neighborhood attention to facilitate mutual reinforcement between two feature streams. (3) Self-Prompting Mechanism: To eliminate the manual or explicit prompt intervention inherent in SAM’s original design, we devise a self-prompting strategy that autonomously derives spatially semantic prompts from model-inferred target cues. Extensive experiments across diverse medical modalities and datasets demonstrate that DReAM-SAM outperforms state-of-the-art SAM variants and task-specific MIS approaches.


## Project Structure
```none
DReAM-SAM
├── segment_anything/ # SAM backbone and the proposed modules
├── utils/ # Helper functions for data preprocessing/postprocessing and objective functions
├── train.py # Training script
├── inference.py # Inference & evaluation script
├── config.yaml # Configuration file (directory, hyperparameters, and optimization settings)
```
