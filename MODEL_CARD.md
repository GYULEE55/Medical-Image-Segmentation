# Model Card — Medical Image Segmentation AI

## Model Details

| Property | Value |
|----------|-------|
| Model Architecture | YOLOv8n-seg (Ultralytics) |
| Task | Instance Segmentation |
| Parameters | ~3.4M |
| Training Hardware | Google Colab T4 GPU |
| Framework | PyTorch + Ultralytics |
| Input | RGB images (640×640) |
| Output | Bounding boxes + segmentation masks + class labels |

**Two model variants:**
- `best.pt` — Polyp segmentation (Kvasir-SEG, 1-class)
- `best_dentex.pt` — Dental X-ray detection (DENTEX, 4-class)

---

## Intended Use

**Primary Use Case**: Proof-of-concept medical AI assistant for radiologist/endoscopist workflow support.

**Intended Users**: Medical AI researchers, developers building clinical decision support prototypes.

**Out-of-Scope Uses**:
- Clinical diagnosis or treatment decisions
- Replacement of licensed medical professionals
- Deployment in regulated medical device contexts without proper validation

> **IMPORTANT**: This model is a research prototype. It is NOT approved for clinical use and should NOT be used for actual patient diagnosis.

---

## Training Data

### Kvasir-SEG (Polyp Model)
- **Source**: Kvasir-SEG dataset (publicly available)
- **Size**: 1,000 colonoscopy images with pixel-level polyp annotations
- **Split**: 800 train / 200 validation
- **Classes**: 1 (polyp)
- **Annotation**: Binary segmentation masks → converted to YOLO polygon format

### DENTEX (Dental Model)
- **Source**: DENTEX Challenge dataset
- **Size**: ~700 panoramic dental X-ray images
- **Split**: ~560 train / ~140 validation
- **Classes**: 4 (Caries, Deep Caries, Periapical Lesion, Impacted Tooth)
- **Annotation**: COCO JSON format → converted to YOLO polygon format

---

## Evaluation Results

### Kvasir-SEG (50 epochs, Colab T4)

| Metric | Box | Mask |
|--------|-----|------|
| Precision | 0.920 | 0.930 |
| Recall | 0.887 | 0.897 |
| mAP@50 | 0.939 | **0.942** |
| mAP@50-95 | 0.777 | 0.786 |

**Interpretation**: Strong performance on single-class polyp segmentation. mAP@50=0.942 indicates reliable detection with good mask quality.

### DENTEX (100 epochs, Colab T4)

| Metric | Box | Mask |
|--------|-----|------|
| Precision | 0.485 | 0.485 |
| Recall | 0.334 | 0.334 |
| mAP@50 | 0.377 | **0.344** |
| mAP@50-95 | 0.242 | 0.225 |

**Interpretation**: Lower performance due to dataset challenges (limited data, 4-class complexity, subtle X-ray features). See [docs/DENTEX_ANALYSIS.md](docs/DENTEX_ANALYSIS.md) for detailed analysis.

---

## Limitations

1. **Dataset Size**: DENTEX model trained on only ~700 images — insufficient for robust 4-class detection
2. **Single GPU Training**: No distributed training; limited hyperparameter search
3. **Domain Specificity**: Polyp model optimized for colonoscopy images; may not generalize to other endoscopy types
4. **No External Validation**: Models not validated on independent test sets from different institutions
5. **Image Quality Dependency**: Performance degrades with low-resolution or artifact-heavy images
6. **Class Imbalance**: DENTEX dataset has uneven class distribution affecting per-class recall

---

## Ethical Considerations

- **Not for Clinical Use**: These models are research prototypes and must not be used for actual patient diagnosis
- **Bias Risk**: Training data may not represent diverse patient populations, equipment types, or imaging protocols
- **False Negatives**: Missing a lesion (false negative) can have serious clinical consequences — recall optimization is critical for medical AI
- **Explainability**: Segmentation masks provide visual evidence, but model decisions are not fully interpretable
- **Regulatory Status**: Not FDA/CE approved; would require extensive clinical validation before any regulated use

---

## Caveats and Recommendations

**Before any production deployment:**
1. Validate on diverse, institution-specific datasets
2. Conduct prospective clinical trials
3. Obtain appropriate regulatory approvals (FDA 510(k), CE marking)
4. Implement human-in-the-loop review for all predictions
5. Establish monitoring for distribution shift

**For research use:**
- Use as baseline for transfer learning experiments
- Benchmark against domain-specific architectures (e.g., SAM, MedSAM)
- Consider data augmentation strategies for DENTEX improvement
