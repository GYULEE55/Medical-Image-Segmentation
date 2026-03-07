# DENTEX Performance Analysis

## Dataset Overview

The DENTEX (Dental Enumeration and Diagnosis) dataset presents unique challenges for instance segmentation:

| Property | Value |
|----------|-------|
| Total Images | ~700 panoramic dental X-rays |
| Classes | 4 (Caries, Deep Caries, Periapical Lesion, Impacted Tooth) |
| Annotation Format | COCO JSON → converted to YOLO polygon |
| Image Type | Grayscale panoramic X-ray |
| Train/Val Split | ~80/20 |

### Class Distribution (Approximate)
- **Impacted Tooth**: Most common (teeth that haven't erupted properly)
- **Caries**: Common (tooth decay)
- **Deep Caries**: Less common (advanced decay near pulp)
- **Periapical Lesion**: Least common (infection at tooth root)

This class imbalance directly impacts per-class recall, particularly for rare classes.

---

## Training Configuration

Three training variants were tested:

### Variant 1: Baseline (`train_colab_dentex.py`)
- Epochs: 100, Batch: 16, imgsz: 640
- Optimizer: SGD, lr0: 0.01
- patience: 15, cos_lr: True

### Variant 2: Recall Boost (`train_colab_dentex_recall_boost.py`)
- Increased augmentation intensity
- Adjusted class weights to improve recall on rare classes

### Variant 3: V2 Augmented (`train_colab_dentex_v2_augmented.py`)
- Enhanced data augmentation pipeline
- Mosaic + MixUp augmentation

---

## Performance Analysis

### Final Results (Best Checkpoint)

| Metric | Box | Mask |
|--------|-----|------|
| Precision | 0.485 | 0.485 |
| Recall | 0.334 | 0.334 |
| mAP@50 | 0.377 | **0.344** |
| mAP@50-95 | 0.242 | 0.225 |

### Comparison: Kvasir-SEG vs DENTEX

| Factor | Kvasir-SEG | DENTEX | Impact |
|--------|-----------|--------|--------|
| Dataset size | 1,000 images | ~700 images | Smaller → less generalization |
| Classes | 1 | 4 | More classes → harder optimization |
| Class balance | Balanced | Imbalanced | Rare classes underperform |
| Image type | Color endoscopy | Grayscale X-ray | Different feature space |
| Lesion visibility | High contrast | Low contrast | Harder to detect boundaries |
| mAP@50 (Mask) | **0.942** | **0.344** | 2.7× performance gap |

---

## Root Cause Analysis

### Why mAP@50 = 0.344?

**1. Insufficient Training Data**
- 700 images for 4-class detection is significantly below recommended minimums
- YOLOv8 typically needs 1,500+ images per class for robust performance
- Effective per-class data: ~175 images average (far below threshold)

**2. Multi-Class Complexity**
- Single-class Kvasir-SEG: model focuses all capacity on one pattern
- 4-class DENTEX: model must simultaneously learn 4 distinct pathology patterns
- Feature space competition between similar-looking pathologies (Caries vs Deep Caries)

**3. Class Imbalance**
- Periapical Lesion and Deep Caries are rare → model biased toward common classes
- Standard cross-entropy loss doesn't compensate for imbalance
- Recall for rare classes is particularly low

**4. Dental X-ray Characteristics**
- Panoramic X-rays have lower resolution than endoscopy images
- Overlapping dental structures create ambiguous boundaries
- Pathologies can be subtle (early caries vs healthy enamel)
- Grayscale images lack color information that aids polyp detection

**5. Annotation Complexity**
- Dental pathologies require expert radiologist annotation
- Inter-annotator variability is higher for subtle lesions
- COCO polygon annotations may have inconsistent granularity

---

## Learning Curve Analysis

Training showed steady improvement but plateaued early:
- Epochs 1-30: Rapid initial learning
- Epochs 30-70: Gradual improvement, high variance
- Epochs 70-100: Plateau with minor fluctuations
- Best checkpoint: ~epoch 83 (early stopping triggered)

This pattern suggests the model reached its capacity limit given the available data.

---

## Improvement Roadmap

### Short-term (Data-focused)
1. **Data Augmentation**: Aggressive augmentation (rotation, brightness, contrast, elastic deformation)
2. **Synthetic Data**: Generate synthetic X-rays using GAN or diffusion models
3. **Transfer Learning**: Pre-train on larger dental datasets (e.g., TUFTS Dental Database)
4. **Focal Loss**: Replace standard loss with focal loss to address class imbalance

### Medium-term (Architecture)
1. **Larger Model**: Upgrade from YOLOv8n to YOLOv8s or YOLOv8m
2. **Ensemble**: Combine multiple model checkpoints
3. **Two-stage Detection**: Use detection + classification pipeline for rare classes
4. **Domain Adaptation**: Fine-tune from medical imaging pre-trained weights

### Long-term (Data Collection)
1. **More Data**: Target 2,000+ images per class
2. **Multi-institution Data**: Diverse equipment and patient populations
3. **Expert Annotation**: Standardized annotation protocol with radiologist consensus

---

## Interview Positioning

When asked "Why is DENTEX performance low?":

> "The DENTEX dataset represents a genuinely challenging real-world scenario: 4-class detection on only 700 images with significant class imbalance. This is a common situation in medical AI — high-quality annotated data is expensive and scarce. The 2.7× performance gap between Kvasir-SEG (single-class, 1000 images) and DENTEX (4-class, 700 images) empirically demonstrates the impact of dataset size and class complexity. Rather than hiding this result, I documented it thoroughly and identified concrete improvement strategies, which reflects production-grade ML engineering thinking."
