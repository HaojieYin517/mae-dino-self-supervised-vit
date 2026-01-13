"""
Create Kaggle Submission with KNN Classifier
=============================================

This script provides a simple baseline using:
- Pretrained DINO model for feature extraction
- KNN classifier for predictions

NOTE: This is a BASELINE example. For the competition, you MUST:
- Train your own SSL model from scratch (no pretrained weights!)
- This script is just to understand the submission format

Usage:
    python create_submission_knn.py \
        --data_dir ./kaggle_data \
        --output submission.csv \
        --k 5
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, Dinov2Model
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys
sys.path.append("../../dino")
from vision_transformer import vit_small_patch8_512, vit_base_patch8
import torch.serialization
import numpy as np
import torchvision.transforms as transforms
from sklearn.preprocessing import normalize
from cuml.linear_model import LogisticRegression  # GPU version
import cupy as cp
from sklearn.preprocessing import StandardScaler, normalize



# ============================================================================
#                          MODEL SECTION (Modular)
# ============================================================================

class FeatureExtractor:
    """
    Feature extractor using your DINO ViT-S checkpoint.
    """
    def __init__(
        self,
        checkpoint_path="/workspace/dino_output_small_299/checkpoint.pth",
        # checkpoint_path="/workspace/dino_output_small_patch8/checkpoint0199.pth",
        device="cuda",
    ):
        print(f"Loading DINO checkpoint: {checkpoint_path}")
        self.device = device

        # ---- allow numpy scalar in torch.load (PyTorch 2.6 safety) ----
        import numpy as np
        import torch.serialization as serialization

        serialization.add_safe_globals([np.core.multiarray.scalar])

        # ---- load CLEAN checkpoint (encoder-only dict) ----
        backbone_state = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False
        )

        if not isinstance(backbone_state, dict):
            raise RuntimeError("Checkpoint is not a state_dict")

        # ---- build backbone model ----
        # self.model = vit_small(patch_size=16) # note for patch16 model
        # self.model = vit_small_patch8_512(patch_size=8)
        self.model = vit_base_patch8(patch_size = 8)

        msg = self.model.load_state_dict(backbone_state, strict=True)
        print("Backbone loaded with message:", msg)

        self.model.to(self.device)
        self.model.eval()

        self.sanity_test()

        # ---- transform: must match your training resolution (96x96) ----
        self.transform = transforms.Compose([
            transforms.Resize((96, 96), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.512, 0.489, 0.461),
                std=(0.230, 0.225, 0.232),
            ),
        ])
    
    def sanity_test(self):
        # --------------------------------------------------------------
        # 🔍 SANITY CHECK: Verify parameters + dummy forward pass
        # --------------------------------------------------------------
        def count_params(model):
            return sum(p.numel() for p in model.parameters())

        total_params = count_params(self.model)
        print(f"\n[Sanity] Total Backbone Params: {total_params:,}")

        # Encoder-only params (patch_embed + blocks + pos_embed + cls_token + norm)
        encoder_param_names = ["patch_embed", "blocks", "pos_embed", "cls_token", "norm"]
        encoder_params = 0
        for name, param in self.model.named_parameters():
            if any(name.startswith(prefix) for prefix in encoder_param_names):
                encoder_params += param.numel()

        print(f"[Sanity] Encoder-Only Params: {encoder_params:,}")
        print(f"[Sanity] PASS <100M requirement? {encoder_params < 100_000_000}")

        # --------------------------------------------------------------
        # 🔍 FORWARD TEST WITH RANDOM 96×96 IMAGE
        # --------------------------------------------------------------
        try:
            dummy = torch.randn(1, 3, 96, 96).to(self.device)
            with torch.no_grad():
                out = self.model(dummy)

            if out.ndim == 3:
                print(f"[Sanity] Forward OK — output token shape = {tuple(out.shape)}")
                print(f"[Sanity] CLS token shape = {tuple(out[:,0].shape)}")
            else:
                print(f"[Sanity] Forward OK — output shape = {tuple(out.shape)}")

        except Exception as e:
            print("\n❌ SANITY FORWARD FAILED:")
            print(e)

    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward through ViT backbone and return CLS token features.

        Input:
            x: (B, 3, H, W)

        Output:
            cls: (B, D)
        """
        with torch.no_grad():
            feats = self.model(x)  # DINO ViT usually returns (B, N_tokens, D)

        if feats.ndim == 3:
            # (B, N_tokens, D) → take CLS token
            cls = feats[:, 0]  # (B, D)
        elif feats.ndim == 2:
            # already (B, D)
            cls = feats
        else:
            raise RuntimeError(f"Unexpected feature shape from backbone: {feats.shape}")

        return cls

    def extract_features(self, image):
        """
        Extract features from a single PIL Image.

        Returns:
            1D numpy array of shape (feature_dim,)
        """
        x = self.transform(image).unsqueeze(0).to(self.device)  # (1, 3, H, W)
        cls = self._forward_backbone(x)  # (1, D)
        return cls[0].cpu().numpy()      # (D,)

    def extract_batch_features(self, images):
        """
        Extract features from a batch of PIL Images.

        Returns:
            2D numpy array of shape (batch_size, feature_dim)
        """
        x = torch.stack([self.transform(img) for img in images]).to(self.device)  # (B, 3, H, W)
        cls = self._forward_backbone(x)  # (B, D)
        return cls.cpu().numpy()         # (B, D)


# ============================================================================
#                          DATA SECTION
# ============================================================================

class ImageDataset(Dataset):
    """Simple dataset for loading images"""
    
    def __init__(self, image_dir, image_list, labels=None, resolution=224):
        """
        Args:
            image_dir: Directory containing images
            image_list: List of image filenames
            labels: List of labels (optional, for train/val)
            resolution: Image resolution (96 for competition, 224 for DINO baseline)
        """
        self.image_dir = Path(image_dir)
        self.image_list = image_list
        self.labels = labels
        self.resolution = resolution
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = self.image_dir / img_name
        
        # Load and resize image
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.resolution, self.resolution), Image.BILINEAR)
        
        if self.labels is not None:
            return image, self.labels[idx], img_name
        return image, img_name


def collate_fn(batch):
    """Custom collate function to handle PIL images"""
    if len(batch[0]) == 3:  # train/val (image, label, filename)
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        filenames = [item[2] for item in batch]
        return images, labels, filenames
    else:  # test (image, filename)
        images = [item[0] for item in batch]
        filenames = [item[1] for item in batch]
        return images, filenames


# ============================================================================
#                          FEATURE EXTRACTION
# ============================================================================

def extract_features_from_dataloader(feature_extractor, dataloader, split_name='train'):
    """
    Extract features from a dataloader.
    
    Args:
        feature_extractor: FeatureExtractor instance
        dataloader: DataLoader
        split_name: Name of split (for progress bar)
    
    Returns:
        features: numpy array (N, feature_dim)
        labels: list of labels (or None for test)
        filenames: list of filenames
    """
    all_features = []
    all_labels = []
    all_filenames = []
    
    print(f"\nExtracting features from {split_name} set...")
    
    for batch in tqdm(dataloader, desc=f"{split_name} features"):
        if len(batch) == 3:  # train/val
            images, labels, filenames = batch
            all_labels.extend(labels)
        else:  # test
            images, filenames = batch
        
        # Extract features for batch
        features = feature_extractor.extract_batch_features(images)
        all_features.append(features)
        all_filenames.extend(filenames)
    
    features = np.concatenate(all_features, axis=0)
    print(features.shape)
    labels = all_labels if all_labels else None
    
    print(f"  Extracted {features.shape[0]} features of dimension {features.shape[1]}")
    
    return features, labels, all_filenames


# ============================================================================
#                          KNN CLASSIFIER
# ============================================================================

def train_knn_classifier(train_features, train_labels,
                         val_features, val_labels,
                         k_list=[1,3,5,10,20,50,100],
                         metrics=["cosine","euclidean"],
                         weights=["uniform","distance"]):

    print("\n========== 🔍 KNN PARAMETER SEARCH ==========")

    train_features = normalize(train_features, axis=1)
    val_features   = normalize(val_features, axis=1)

    best_acc = -1
    best_model = None

    for metric in metrics:
        for w in weights:
            for k in k_list:
                knn = KNeighborsClassifier(
                    n_neighbors=k,
                    metric=metric,
                    weights=w,
                    n_jobs=-1
                )
                knn.fit(train_features, train_labels)
                acc = knn.score(val_features, val_labels)

                print(f"K={k:3d} | metric={metric:8s} | weight={w:8s} → Val Acc={acc:.4f}")

                if acc > best_acc:
                    best_acc = acc
                    best_model = knn

    print("\n🎯 Best KNN Model:")
    print(f"   n_neighbors={best_model.n_neighbors}, metric={best_model.metric}")
    print(f"   Validation Accuracy = {best_acc:.4f}")

    return best_model, best_acc

def train_linear_probe_classifier(train_features, train_labels,
                                  val_features, val_labels, C_values):

    print("\n========== 🚀 GPU LINEAR PROBE (cuML) ==========")

    # Convert CPU numpy → GPU CuPy
    X_train = cp.asarray(train_features)
    y_train = cp.asarray(train_labels)
    X_val   = cp.asarray(val_features)
    y_val   = cp.asarray(val_labels)

    best_acc = -1
    best_model = None
    best_C = None

    for C in C_values:
        clf = LogisticRegression(
            C=C,
            max_iter=2000,          # much faster convergence on GPU
            penalty="l2",
            tol=1e-4
        )
        clf.fit(X_train, y_train)

        preds = clf.predict(X_val)
        acc = (preds == y_val).mean()

        print(f"GPU Linear Probe → C={C}  |  Acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model = clf
            best_C = C

    print("\n🎯 Best GPU Linear Probe:")
    print(f"   C={best_C}  |  Val Accuracy={best_acc:.4f}")

    return best_model, best_acc

# ============================================================================
#                          SUBMISSION CREATION
# ============================================================================

def create_submission(test_features, test_filenames, classifier, output_path):
    """
    Create submission.csv for Kaggle.
    
    Args:
        test_features: Test features (N_test, feature_dim)
        test_filenames: List of test image filenames
        classifier: Trained KNN classifier
        output_path: Path to save submission.csv
    """
    print("\nGenerating predictions on test set...")
    predictions = classifier.predict(test_features)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'id': test_filenames,
        'class_id': predictions
    })
    
    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"Submission file created: {output_path}")
    print(f"{'='*60}")
    print(f"Total predictions: {len(submission_df)}")
    print(f"\nFirst 10 predictions:")
    print(submission_df.head(10))
    print(f"\nClass distribution in predictions:")
    print(submission_df['class_id'].value_counts().head(10))
    
    # Validate submission format
    print(f"\nValidating submission format...")
    assert list(submission_df.columns) == ['id', 'class_id'], "Invalid columns!"
    assert submission_df['class_id'].min() >= 0, "Invalid class_id < 0"
    assert submission_df['class_id'].max() <= 199, "Invalid class_id > 199"
    assert submission_df.isnull().sum().sum() == 0, "Missing values found!"
    print("✓ Submission format is valid!")


# ============================================================================
#                          MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Create Kaggle Submission with KNN')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory containing train/val/test folders')
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='Output path for submission file')
    parser.add_argument('--model_name', type=str, 
                        default='facebook/webssl-dino300m-full2b-224',
                        help='HuggingFace model name (baseline only!)')
    parser.add_argument('--resolution', type=int, default=96,
                        help='Image resolution (96 for competition, 224 for DINO)')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of neighbors for KNN')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument("--checkpoint", type=str, default="/workspace/dino_output_small_299/checkpoint.pth")
    parser.add_argument("--classifier", type=str, default="linear",
                    choices=["knn", "linear", "both"],
                    help="Choose classifier: KNN, Linear Probe, or both")
    parser.add_argument("--normalization", type=str, default="none",
                choices=["none","l2","standard"],
                help="Feature scaling method:\n"
                     "none = raw embeddings\n"
                     "l2 = cosine norm like current KNN\n"
                     "standard = StandardScaler(train→fit, val/test→transform)")

    
    args = parser.parse_args()
    
    # Check device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    data_dir = Path(args.data_dir)
    
    # Load CSV files
    print("\nLoading dataset metadata...")
    train_df = pd.read_csv(data_dir / 'train_labels.csv')
    val_df = pd.read_csv(data_dir / 'val_labels.csv')
    test_df = pd.read_csv(data_dir / 'test_images.csv')
    
    print(f"  Train: {len(train_df)} images")
    print(f"  Val: {len(val_df)} images")
    print(f"  Test: {len(test_df)} images")
    print(f"  Classes: {train_df['class_id'].nunique()}")
    
    # Create datasets
    print(f"\nCreating datasets (resolution={args.resolution}px)...")
    train_dataset = ImageDataset(
        data_dir / 'train',
        train_df['filename'].tolist(),
        train_df['class_id'].tolist(),
        resolution=args.resolution
    )
    
    val_dataset = ImageDataset(
        data_dir / 'val',
        val_df['filename'].tolist(),
        val_df['class_id'].tolist(),
        resolution=args.resolution
    )
    
    test_dataset = ImageDataset(
        data_dir / 'test',
        test_df['filename'].tolist(),
        labels=None,
        resolution=args.resolution
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    # Initialize feature extractor
    feature_extractor = feature_extractor = FeatureExtractor(
        checkpoint_path=args.checkpoint,
        device=device
    )
    
    # Extract features
    train_features, train_labels, _ = extract_features_from_dataloader(
        feature_extractor, train_loader, 'train'
    )
    val_features, val_labels, _ = extract_features_from_dataloader(
        feature_extractor, val_loader, 'val'
    )
    test_features, _, test_filenames = extract_features_from_dataloader(
        feature_extractor, test_loader, 'test'
    )

    if args.normalization == "l2":   # 🔹 your original behavior
        train_features = normalize(train_features, axis=1)
        val_features   = normalize(val_features, axis=1)
        test_features  = normalize(test_features, axis=1)
        print("\n📌 Using L2 normalization (cosine-friendly)")

    elif args.normalization == "standard":  # 🔥 NEW (train-fit only)
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)   # fit only on train
        val_features   = scaler.transform(val_features)
        test_features  = scaler.transform(test_features)
        print("\n📌 Using StandardScaler (NO test leakage)")

    else:
        print("\n📌 No normalization applied")
    

    C_values1 = [0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 50.0, 100.0]
    C_values2 = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3,
          3e-3, 1e-2, 3e-2, 1e-1,
          0.3, 1, 3, 10, 30, 100, 300]
    if args.normalization == "standard":
        C_values_use = C_values2
    else:
        C_values_use = C_values1

    if args.classifier == "knn":
        classifier, knn_acc = train_knn_classifier(
            train_features, train_labels, val_features, val_labels
        )
        print(f"\nSelected model = KNN (Acc={knn_acc:.4f})")

    elif args.classifier == "linear":
        classifier, linear_acc = train_linear_probe_classifier(
            train_features, train_labels, val_features, val_labels, C_values_use
        )
        print(f"\nSelected model = Linear Probe (Acc={linear_acc:.4f})")

    elif args.classifier == "both":
        print("\n🔄 Running BOTH KNN + Linear Probe...")

        knn_model, knn_acc = train_knn_classifier(train_features, train_labels, val_features, val_labels)
        lin_model, lin_acc = train_linear_probe_classifier(train_features, train_labels, val_features, val_labels, C_values_use)

        if lin_acc >= knn_acc:
            classifier = lin_model
            print(f"\n Selected BEST = Linear Probe (Acc={lin_acc:.4f})")
        else:
            classifier = knn_model
            print(f"\n Selected BEST = KNN (Acc={knn_acc:.4f})")
            
    else:
        raise ValueError(f"Unknown classifier option: {args.classifier}")
    
    # Create submission
    create_submission(test_features, test_filenames, classifier, args.output)
    
    print("\n" + "="*60)
    print("DONE! Now upload your submission.csv to Kaggle.")
    print("="*60)
    print("\nREMINDER: This baseline uses pretrained weights!")
    print("For the competition, you MUST train your own SSL model from scratch.")


if __name__ == "__main__":
    main()

