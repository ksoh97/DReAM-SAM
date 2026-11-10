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

## Experimental Results
We evaluate DReAM-SAM on six MIS benchmarks covering CT/CTA, MRI/MRA, ultrasound, and X-ray.

| Dataset | Modality | Target | Train / Test |
|:--|:--|:--|:--:|
| LIASD | CTA | Intracranial aneurysm | 1,186 / 152 |
| ADAM | ToF-MRA | Intracranial aneurysm | 79 / 34 |
| BTCV | CT | 13 abdominal organs | 21 / 9 |
| BraTS2021 | MRI (T1, T1CE, T2, FLAIR) | Brain tumor (ET / TC / WT) | 875 / 376 |
| TDSC-ABUS | 3D ultrasound | Breast tumor | 70 / 30 |
| QaTa-COV19 | Chest X-ray | COVID-19 pneumonia | 7,145 / 2,113 |

Performance is reported with DSC, boundary IoU (BIoU), and HD95. Unless otherwise noted, results correspond to DReAM-SAM with the 2D SAM (ViT-B) backbone.

### 1. Comparison with State-of-the-Art Methods

#### 1.1 Quantitative Results (Table II)
DReAM-SAM is compared with 6 CNN/ViT-based models and 14 SAM-based methods, all reproduced under the same environment.
- DReAM-SAM achieves the **best DSC on all six datasets**.
- The margin over the second-best method is largest on datasets dominated by small, ill-defined targets (**+8.18%** on ADAM, **+5.88%** on TDSC, **+3.99%** on LIASD), and narrower on QaTa (+3.46%), where pneumonia regions are much larger.
- The framework is **backbone-agnostic**: built on MedSAM (2D) or SAM-Med3D (3D), it still outperforms every competing method in DSC.

<p align="center"><img width="2892" height="1501" alt="Table II: Quantitative comparison on six datasets" src="https://github.com/user-attachments/assets/c89a60c7-d688-4b6c-9b77-f682c8a57b27" /></p>

#### 1.2 Class-wise Results on BTCV and BraTS (Table III)
- **BTCV:** DReAM-SAM ranks first on large organs (liver 97.9%, kidneys 96.1%) and shows larger gains on small structures such as the gallbladder (**+5.7%**) and adrenal glands (**+3.8%**).
- **BraTS:** DReAM-SAM improves ET / TC / WT by **+5.3% / +4.6% / +4.2%**. The largest gain is on ET, the smallest and most heterogeneous sub-region, which suggests the benefit grows with target difficulty.

<p align="center"><img width="65%" alt="Table III: Class-wise DSC on BTCV and BraTS" src="https://github.com/user-attachments/assets/f83ab7e4-cc93-454d-98bd-ce1dbddc5b1c" /></p>

#### 1.3 Qualitative Results (Fig. 5)
On small or low-contrast targets, DReAM-SAM produces tighter and more coherent boundaries. Competing methods often miss the lesion entirely or over-segment into surrounding tissue. Green boxes show zoomed-in regions.

<p align="center"><img width="2902" height="1671" alt="Fig. 5: Qualitative comparison across six datasets" src="https://github.com/user-attachments/assets/9fa06e14-c7cb-4977-81c6-adda6b788f5c" /></p>

### 2. Data Efficiency on Small vs. Large Targets (Fig. 6)
BTCV organs are split into small-volume (mean ≈ 13,263 mm³) and large-volume (mean ≈ 198,267 mm³) groups, about a 15× size gap, and models are trained with 10–100% of the labels. DReAM-SAM is compared with adaptation methods (SAMed, 3DSAM-adt) and with semi-supervised methods (H-SAM, CPC-SAM, SynFoC), which also use the remaining unlabeled data.
- Semi-supervised methods stay stable on large organs but drop sharply on small ones.
- DReAM-SAM uses **no unlabeled data**, yet it gives the **best small-volume DSC at 50% labels and above**.

<p align="center"><img width="65%" alt="Fig. 6: DSC under varying labeled data fractions on BTCV" src="https://github.com/user-attachments/assets/098458d3-36b3-4b74-b4c3-9e542c5db0e5" /></p>

### 3. Computational Efficiency (Table IV)
Parameters, FLOPs, GPU memory, and inference time are measured on LIASD (1 target) / BTCV (13 targets) with 1024 × 1024 inputs on a single NVIDIA Quadro RTX 8000.
- Compared with AutoSAMUS, another CNN-augmented SAM, DReAM-SAM improves DSC by **+27.0 / +10.5 %p** and also runs faster (233.2 → 164.6 ms / 373.7 → 353.8 ms).
- Compared with Med-SA, DReAM-SAM improves DSC by **+4.0 / +5.0 %p** with **~4.2–4.6× fewer total parameters**.
- These results suggest the gains do not come simply from added model capacity or computation.

<p align="center"><img width="65%" alt="Table IV: Computational cost and DSC" src="https://github.com/user-attachments/assets/4ffa0d0f-7d39-43bb-ac2f-f5d2ecceea80" /></p>

### 4. Analysis

#### 4.1 Learned Target Prototypes (Fig. 7)
- **(a) Prototype similarity:** On BTCV, bilateral organ pairs show among the highest off-diagonal similarities (kidneys 0.56, adrenal glands 0.58). This indicates that the learned prototype space reflects anatomical relationships between targets.
- **(b) Prototype intervention:** We replace the prototypes at inference without retraining. Using the mean prototype for every target lowers DSC from 89.50% to 81.92% (BTCV) and from 94.03% to 89.47% (BraTS). Shuffling prototypes across targets lowers it further, to 77.47% and 86.30%. Correct target–prototype correspondence is therefore essential for generating effective target-conditioned kernels.

<p align="center"><img width="65%" alt="Fig. 7: Analysis of learned target prototypes" src="https://github.com/user-attachments/assets/f25bf1ce-eb3a-42cb-b697-5f66552b0e14" /></p>

#### 4.2 Feature Visualization (Fig. 8)
The figure shows averaged feature maps from the ACNN ($F_c \rightarrow F'_c$) and SAM ($F_s \rightarrow F'_s$) branches before and after accentuation, together with the generated dense prompt $P_d$ and the final prediction $Y'$.
- The self-generated dense prompt closely matches the GT in both location and shape.
- The original features mainly capture local textures (ACNN) and global semantics (SAM). After accentuation, both show finer detail and a clear focus on the target.

<p align="center"><img width="65%" alt="Fig. 8: Feature visualization of ACNN and SAM branches" src="https://github.com/user-attachments/assets/039f89ff-0df2-4aa9-b0fe-172a85a6c65d" /></p>

#### 4.3 Failure Case Analysis (Fig. 9)
Representative failure cases on LIASD, annotated with the target-existence probability $y'$ predicted by TEC:
- **(a) TEC false negative** ($y'$ = 0.29): the target gets too little guidance, and the lesion is missed.
- **(b) TEC false positive** ($y'$ = 0.71): a localized activation in the dense prompt produces a spurious foreground.
- **(c) TAPG partial response:** in a multi-lesion slice, the prompt responds strongly to the dominant lesion but only weakly to the smaller one.
- **(d) TAPG misplacement:** a spatially misaligned dense prompt produces a shifted and enlarged mask.

TEC decides *whether* target evidence is activated, and TAPG decides *where* it is concentrated. Uncertainty-aware existence estimation and multi-peak prompt generation are promising directions for handling small and multifocal lesions.

<p align="center"><img width="65%" alt="Fig. 9: Representative failure cases on LIASD" src="https://github.com/user-attachments/assets/f7a45202-ce08-4c74-a90c-d7df54d03715" /></p>

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
