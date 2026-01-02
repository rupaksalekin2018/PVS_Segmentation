# 3D Perivascular Space Segmentation with UNet++

This project implements a deep learning pipeline for the **3D semantic segmentation of Perivascular Spaces (PVS)** in T2-weighted fMRI/MRI data. It utilizes the **MONAI** framework for medical image processing and **Segmentation Models PyTorch 3D** for the architecture.

The pipeline is designed to handle NIfTI (`.nii`, `.nii.gz`) formats, perform robust data augmentation, and train a **UNet++** model with an **EfficientNet-B7** encoder.

---

## 🌟 Key Features

* **Architecture:** 3D UNet++ with a pre-trained **EfficientNet-B7** encoder.
* **Hybrid Loss Function:** Combines **Binary Cross Entropy (BCE)** and **Dice Loss** for improved boundary delineation.
* **MONAI Preprocessing:** Includes spatial resampling ($1.5mm^3$), orientation normalization (RAS), and intensity scaling.
* **Robust Augmentation:** Random affine transformations, elastic cropping, contrast adjustment, and Gaussian noise.
* **Comprehensive Metrics:** Automatically calculates **Dice**, **IoU (Jaccard)**, **Precision**, and **Recall**.
* **Visualizations:** Generates high-resolution overlays (**300 DPI**) comparing Ground Truth vs. Predictions.

---

## 🛠️ Installation

Ensure you have **Python** installed. Install the required dependencies:

```bash
pip install monai-weekly[gdown,nibabel,tqdm,ignite]
pip install segmentation_models_pytorch_3d
pip install torch matplotlib scikit-learn numpy
````

-----

## 📂 Dataset Structure

The script expects the data to be organized in a standard BIDS-like format. The default data directory is configured to: `/home/ubuntu/FMRI_Perivascular_Space_Dataset/ds005595-1.0.0`.

The file loader looks for the following patterns:

  * **Images (T2w):** `sub-*/anat/*_T2w.nii*`
  * **Masks (PVS):** `derivatives/mask/sub-*/sub-*_desc-mask_PVS.nii*`

> **Note:** The script automatically filters out subjects with empty masks to ensure valid training data.

-----

## ⚙️ Configuration

Key hyperparameters are defined in the `Config` class at the top of the script. You can modify these to suit your hardware capabilities:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `IMG_PATCH_SIZE` | `(96, 96, 96)` | Input size for the 3D model. |
| `BATCH_SIZE` | `2` | Number of samples per batch. |
| `EPOCHS` | `1000` | Maximum training epochs. |
| `LR` | `1e-3` | Initial learning rate. |
| `PIX_DIM` | `(1.5, 1.5, 1.5)` | Target voxel resolution (mm). |
| `EARLY_STOPPING` | `100` | Patience for stopping if validation Dice stagnates. |

-----

## 🚀 Usage

To start the training pipeline, run the script:

```bash
python unet_3d_final_1000_epoch.py
```

### Training Pipeline

1.  **Preprocessing:** Images are loaded, reoriented to RAS, resampled, and normalized.
2.  **Training:** The model trains using **AdamW** optimizer and a **ReduceLROnPlateau** scheduler.
3.  **Validation:** Occurs every epoch using **Sliding Window Inference** to handle full volumes.
4.  **Checkpointing:** The model state is saved only when the Validation Dice score improves.

-----

## 📊 Outputs

Results are saved in the `outputs_3d` directory:

  * `checkpoints/best_model.pth`: The saved model weights with the highest validation Dice score.
  * `logs/training_history.json`: Loss and Dice curves for training and validation.
  * `logs/comprehensive_test_metrics.json`: Aggregate Mean/Std for Dice, IoU, Precision, and Recall on the test set.
  * `test_set_visualization_grid_300dpi.jpg`: A visual grid showing the original slice, Ground Truth overlay (Red), and Prediction overlay (Green).

-----

## 📐 Evaluation Metrics

The `calculate_metrics` function computes the following pixel-level metrics for every test sample:

  * **Dice Score (F1):** Harmonic mean of precision and recall.
  * **IoU (Jaccard Index):** Intersection over Union.
  * **Precision:** Accuracy of positive predictions.
  * **Recall (Sensitivity):** Ability to find all positive samples.

-----

## 📜 License

This project relies on `monai` and `segmentation_models_pytorch_3d`. Please refer to their respective licenses for usage terms.

```
```

## 3D Multi-Plane Segmentation Analysis Results

This section presents the performance metrics and visual results for the 3D multi-plane segmentation model. The analysis includes aggregate statistics across all test subjects and detailed visualizations for individual subjects.

### Aggregate Performance Metrics

The following table summarizes the overall performance of the segmentation model across the test dataset. The metrics include the mean and standard deviation for Dice Score, IoU (Intersection over Union), Precision, and Recall.

| Metric | Mean | Standard Deviation |
| :--- | :--- | :--- |
| **Dice Score** | 0.6073 | 0.0395 |
| **IoU Score** | 0.4372 | 0.0408 |
| **Precision** | 0.6225 | 0.0392 |
| **Recall** | 0.5951 | 0.0554 |

---

### Individual Subject Analysis

Visualizations for each test subject display the segmentation results across three anatomical planes: Axial, Coronal, and Sagittal. Each set includes:
1.  **Original Image:** The raw MRI scan slice.
2.  **Ground Truth:** The manually annotated segmentation (Red).
3.  **Prediction:** The model's predicted segmentation (Green).

#### Subject 000

![Subject 000 Analysis](0.png)

#### Subject 001

![Subject 001 Analysis](1.png)

#### Subject 002

![Subject 002 Analysis](2.png)

#### Subject 003

![Subject 003 Analysis](3.png)

#### Subject 004

![Subject 004 Analysis](4.png)

