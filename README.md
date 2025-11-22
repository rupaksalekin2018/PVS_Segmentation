3D Perivascular Space Segmentation with UNet++ & MONAIThis repository contains a deep learning pipeline for the 3D semantic segmentation of Perivascular Spaces (PVS) from T2-weighted MRI/fMRI images. It leverages the MONAI framework for robust medical image preprocessing and Segmentation Models PyTorch 3D for a state-of-the-art UNet++ architecture.📋 Table of ContentsKey FeaturesInstallationDataset StructureConfigurationModel ArchitectureUsageOutputs & Metrics🌟 Key FeaturesAdvanced Architecture: Implements a 3D UNet++ with an EfficientNet-B7 encoder, pre-trained on ImageNet.Medical Preprocessing: Uses MONAI for voxel spacing ($1.5mm^3$), orientation (RAS), and intensity scaling (-1000 to 3000).Robust Augmentation: Includes random 3D elastic cropping, affine transformations, contrast adjustment, and Gaussian noise.Hybrid Loss: Trains using a weighted combination of Dice Loss and Binary Cross Entropy (BCE).Comprehensive Evaluation: Calculates Dice, IoU (Jaccard), Precision, and Recall on a hold-out test set.Visual Validation: Automatically generates high-resolution (300 DPI) visualization grids comparing input, ground truth, and prediction.🛠 InstallationPrerequisites: Python 3.8+ and CUDA-enabled GPU (recommended).Install Dependencies:# Core medical imaging libraries
pip install -q "monai-weekly[gdown, nibabel, tqdm, ignite]"

# 3D Segmentation models
pip install segmentation_models_pytorch_3d

# Standard data science stack
pip install matplotlib scikit-learn numpy torch
📂 Dataset StructureThe script is designed to parse a BIDS-compliant directory structure. By default, it looks for data in /home/ubuntu/FMRI_Perivascular_Space_Dataset/ds005595-1.0.0.Expected File Hierarchy:dataset_root/
├── sub-01/
│   └── anat/
│       └── sub-01_T2w.nii.gz          <-- Image
├── derivatives/
│   └── mask/
│       └── sub-01/
│           └── sub-01_desc-mask_PVS.nii.gz  <-- Label
└── ...
Note: The script automatically filters out subjects that contain empty masks to ensure training stability.⚙ ConfigurationHyperparameters are managed via the Config class within the script. You can modify the following key parameters:ParameterDefault ValueDescriptionIMG_PATCH_SIZE(96, 96, 96)The 3D volume size fed into the network.BATCH_SIZE2Batch size (adjust based on GPU VRAM).EPOCHS1000Maximum number of training epochs.LR1e-3Learning rate (AdamW optimizer).PIX_DIM(1.5, 1.5, 1.5)Target resampling resolution in mm.A_MIN, A_MAX-1000.0, 3000.0Intensity clipping range for T2w images.EARLY_STOPPING100Patience for validation loss stagnation.🧠 Model ArchitectureThe pipeline uses segmentation_models_pytorch_3d to instantiate the model:model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b7",
    encoder_weights="imagenet",
    in_channels=1, 
    classes=1,
)
Training Strategy:Optimizer: AdamW with weight decay 1e-5.Scheduler: ReduceLROnPlateau (Factor: 0.5, Patience: 5).Inference: Uses MONAI sliding_window_inference with 50% overlap to handle full 3D volumes during validation/testing.🚀 UsageRun the script directly to start the training, validation, and testing pipeline:python unet_3d_final_1000_epoch(3).py
Process Flow:Data Split: Data is split into Train, Validation, and Test sets (approx 90/5/5 split).Training: The model trains until convergence or early stopping.Checkpointing: The model is saved to outputs_3d/checkpoints/best_model.pth only when Validation Dice improves.Testing: The best model is loaded to evaluate the test set and generate visualizations.📊 Outputs & MetricsResults are saved to the outputs_3d directory.1. Quantitative MetricsSaved in logs/comprehensive_test_metrics.json. The pipeline tracks:Dice Score (F1 Score)IoU (Intersection over Union)PrecisionRecall2. Training LogsSaved in logs/training_history.json, containing loss and metric curves for every epoch.3. VisualizationsA high-resolution JPEG (test_set_visualization_grid_300dpi.jpg) is generated, showing:Panel 1: Normalized Original T2w Slice.Panel 2: Ground Truth Mask Overlay (Red).Panel 3: Model Prediction Overlay (Green).
