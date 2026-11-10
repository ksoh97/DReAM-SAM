# DReAM-SAM

This repository provides a PyTorch implementation of the following paper:
> **DReAM-SAM: Dynamic Region-Adaptive Modulated Segment Anything Model for Fully Automatic Medical Image Segmentation**<br>
> [Kwanseok Oh](https://scholar.google.co.kr/citations?user=EMYHaHUAAAAJ&hl=ko)<sup>1, 2</sup>, Dahye Lee<sup>2</sup>, and [Heung-Il Suk](https://scholar.google.co.kr/citations?user=dl_oZLwAAAAJ&hl=ko)<sup>1</sup><br/>
> (<sup>1</sup>Department of Artificial Intelligence, Korea University) <br/>
> (<sup>2</sup>DEEPNOID Inc.) <br/>
> Published in IEEE TMI: https://doi.org/10.1109/TMI.2026.3752818 <br/>
> 
> **Abstract:** *The Segment Anything Model (SAM) has shown strong zero-shot segmentation capabilities across general visual domains. However, its effectiveness in Medical Image Segmentation (MIS) remains constrained by the medical-domain gap, reliance on explicit user prompts, and limited adaptability to small, heterogeneous, or ill-defined targets. In particular, feature adaptation in existing SAM approaches often remains shared across anatomical targets, making it difficult to tailor local representations to target identity, morphology, and surrounding anatomical context. In this paper, we propose **D**ynamic **Re**gion-**A**daptive **M**odulated **SAM** (**DReAM-SAM**), a fully automatic framework that introduces target- and region-adaptive representation learning for fine-grained MIS. Our framework introduces three key technical pillars: 
(1) Target-Conditioned Dynamic Modulation: We propose an agile CNN (ACNN) that complements SAM's encoder by adapting local feature extraction to target identity and morphology while promoting boundary-sensitive details. (2) Query-Adaptive Feature Interaction: We develop a bidirectional neighborhood attention mechanism to integrate complementary ACNN and SAM features over query-adaptive multi-scale contexts. (3) Target-Aware Self-Prompting: We further devise a self-prompting mechanism that exploits target-existence cues and target-aware representations to automatically generate surrogate prompts. Extensive experiments across six MIS datasets and diverse medical imaging modalities reveal that DReAM-SAM outperforms state-of-the-art SAM variants and task-specific MIS models.*

## Overall Framework
<p align="center"><img width="3896" height="2505" alt="Image" src="https://github.com/user-attachments/assets/e559054c-661a-4c3a-90de-56e4831f3e0e" />

The main contributions are summarized as follows:
- We propose DReAM-SAM, a fully automatic SAM adaptation framework that explicitly introduces target-/region-adaptive representation learning to address the lack of target-adaptive local representation in challenging MIS.
- We introduce an ACNN branch equipped with the DyRA-Block, which performs target-conditioned dynamic feature extraction via a shared--specific kernel decomposition and spatially adaptive sampling, complemented by kernel decorrelation and boundary-aware region decoupling.
- We further develop D$^2$NA for query-adaptive multi-scale interaction between the complementary ACNN and SAM representations; and a self-prompting module (TEC + TAPG) that derives surrogate prompts from inferred target cues without manual intervention.
- Extensive experiments across six MIS datasets and diverse imaging modalities demonstrate that DReAM-SAM outperforms state-of-the-art task-specific and SAM-based segmentation approaches, with particularly pronounced gains on small, heterogeneous, and ambiguous targets.


## Project Structure
```none
DReAM-SAM
├── segment_anything/ # SAM backbone and the proposed modules
├── utils/ # Helper functions for data preprocessing/postprocessing and objective functions
├── tests/dummy_run.py # Dummy-tensor smoke test (DyRA-Block kernels, L_kd, full forward/backward)
├── train.py # Training script
├── inference.py # Inference & evaluation script
├── config.yaml # Configuration file (directory, hyperparameters, and optimization settings)
```



## Citation
If you find this work useful for your research, please cite the following paper:

```
@article{oh2026dreamsam,
  title={DReAM-SAM: Dynamic Region-Adaptive Modulated Segment Anything Model for Fully Automatic Medical Image Segmentation},
  author={Oh, Kwanseok and Lee, Dahye and Suk, Heung-Il},
  journal={IEEE Transactions on Medical Imaging},
  year={2026},
  publisher={IEEE}
}
```

## Acknowledgements
This work was supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. RS-2019-II190079, Department of Artificial Intelligence (Korea University));
the Institute of Information & Communications Technology Planning & Evaluation (IITP) under the Artificial Intelligence Star Fellowship Support Program to Nurture the Best Talents (IITP-2025-RS-2025-02304828) grant funded by the Korea government (MSIT);
and the Technology Innovation Program (RS-2025-02221011, Development of Medical-Specialized Multimodal Hyperscale Generative AI Technology for Global Integration) funded by the Ministry of Trade, Industry & Energy (MOTIE, Korea).
