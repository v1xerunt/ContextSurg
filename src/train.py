"""
Minimal training script for ContextSurg model.

This script provides a straightforward example of training the ContextSurg model
with essential functionality:
- Train model with early stopping
- Evaluate on test set (AUROC, AUPRC)
- Compute and save feature importance (gradient-based)
"""

import os
import argparse
import pickle
import logging
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

from model import ContextSurg
from prepare_dataset import prepare_dataset


# Default Hyperparameters
DEFAULT_CONFIG = {
    # Model architecture
    'num_experts': 5,
    'embedding_dim': 64,
    'hospital_embedding_dim': 128,
    'expert_dim': 128,
    'shared_hidden_dims': [128, 64],
    'router_hidden_dims': [64],
    'expert_hidden_dims': [256, 128],

    # Training parameters
    'batch_size': 256,
    'num_epochs': 100,
    'learning_rate': 0.001,
    'early_stopping_patience': 10,

    # Loss weights
    'lambda_fair': 0.1,
    'lambda_adv': 0.0,
    'lambda_pred': 1.0,
}


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    hospital_train: torch.Tensor,
    location_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    hospital_val: torch.Tensor,
    location_val: torch.Tensor,
    config: Dict,
    device: str = 'cuda'
):
    """Train the ContextSurg model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    best_val_loss = float('inf')
    patience_counter = 0
    patience = config.get('early_stopping_patience', 10)

    for epoch in range(config['num_epochs']):
        # Training
        model.train()
        train_loss = 0
        num_batches = 0

        for i in range(0, len(X_train), config['batch_size']):
            batch_X = X_train[i:i+config['batch_size']].to(device)
            batch_y = y_train[i:i+config['batch_size']].to(device)
            batch_hosp = hospital_train[i:i+config['batch_size']].to(device)
            batch_loc = location_train[i:i+config['batch_size']].to(device)

            optimizer.zero_grad()
            outputs = model(batch_X, batch_hosp)
            loss_dict = model.compute_loss(outputs, batch_y, batch_loc)
            loss = loss_dict['total_loss']

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss /= num_batches

        # Validation
        if X_val.nelement() > 0:
            model.eval()
            val_loss = 0
            num_val_batches = 0

            with torch.no_grad():
                for i in range(0, len(X_val), config['batch_size']):
                    batch_X = X_val[i:i+config['batch_size']].to(device)
                    batch_y = y_val[i:i+config['batch_size']].to(device)
                    batch_hosp = hospital_val[i:i+config['batch_size']].to(device)
                    batch_loc = location_val[i:i+config['batch_size']].to(device)

                    outputs = model(batch_X, batch_hosp)
                    loss_dict = model.compute_loss(outputs, batch_y, batch_loc)
                    val_loss += loss_dict['total_loss'].item()
                    num_val_batches += 1

            val_loss /= num_val_batches

            logging.info(f"Epoch {epoch+1}/{config['num_epochs']} - "
                        f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"Early stopping at epoch {epoch+1}")
                    break
        else:
            logging.info(f"Epoch {epoch+1}/{config['num_epochs']} - Train Loss: {train_loss:.4f}")

    return model


def evaluate_model(
    model: nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    hospital_test: torch.Tensor,
    config: Dict,
    device: str = 'cuda'
) -> Dict:
    """Evaluate the model on test set."""
    model = model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for i in range(0, len(X_test), config['batch_size']):
            batch_X = X_test[i:i+config['batch_size']].to(device)
            batch_y = y_test[i:i+config['batch_size']].to(device)
            batch_hosp = hospital_test[i:i+config['batch_size']].to(device)

            outputs = model(batch_X, batch_hosp)
            predictions = outputs['final_prediction'].cpu().numpy()

            all_predictions.extend(predictions)
            all_labels.extend(batch_y.cpu().numpy())

    all_predictions = np.array(all_predictions).flatten()
    all_labels = np.array(all_labels).flatten()

    # Calculate metrics
    auroc = roc_auc_score(all_labels, all_predictions)
    auprc = average_precision_score(all_labels, all_predictions)

    logging.info(f"\nTest Results:")
    logging.info(f"AUROC: {auroc:.4f}")
    logging.info(f"AUPRC: {auprc:.4f}")

    return {
        'auroc': auroc,
        'auprc': auprc,
        'predictions': all_predictions,
        'labels': all_labels
    }


def compute_feature_importance(
    model: nn.Module,
    X_test: torch.Tensor,
    hospital_test: torch.Tensor,
    config: Dict,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Compute average feature importance using gradient-based method.

    Returns importance score for each patient feature (averaged over all test patients).
    """
    model = model.to(device)
    model.eval()

    num_features = X_test.shape[1]
    importance_sum = np.zeros(num_features)
    num_samples = 0

    for i in range(0, len(X_test), config['batch_size']):
        batch_X = X_test[i:i+config['batch_size']].to(device)
        batch_hosp = hospital_test[i:i+config['batch_size']].to(device)

        # Enable gradient computation for input
        batch_X.requires_grad = True

        # Forward pass
        outputs = model(batch_X, batch_hosp)
        predictions = outputs['final_prediction']

        # Compute gradients with respect to input features
        predictions.sum().backward()

        # Get absolute gradients as importance scores
        gradients = batch_X.grad.abs().cpu().numpy()
        importance_sum += gradients.sum(axis=0)
        num_samples += len(batch_X)

        # Clear gradients
        batch_X.grad = None

    # Average importance across all samples
    avg_importance = importance_sum / num_samples

    return avg_importance


def main():
    parser = argparse.ArgumentParser(description='Train ContextSurg model')

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data/processed',
                       help='Directory containing processed data')
    parser.add_argument('--target', type=str, default='comps_chole_2l',
                       choices=['comps_chole_2l', 'comps_major_2l', 'comps_any_2l'],
                       help='Target variable to predict')

    # Model arguments
    parser.add_argument('--num_experts', type=int, default=None,
                       help='Number of expert networks (default: 5)')
    parser.add_argument('--embedding_dim', type=int, default=None,
                       help='Patient embedding dimension (default: 64)')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for training (default: 256)')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--lambda_fair', type=float, default=None,
                       help='Weight for fairness loss (default: 0.1)')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Directory to save outputs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Build configuration from defaults and command-line overrides
    config = DEFAULT_CONFIG.copy()
    if args.num_experts is not None:
        config['num_experts'] = args.num_experts
    if args.embedding_dim is not None:
        config['embedding_dim'] = args.embedding_dim
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.num_epochs is not None:
        config['num_epochs'] = args.num_epochs
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
    if args.lambda_fair is not None:
        config['lambda_fair'] = args.lambda_fair

    # Prepare dataset
    logging.info("Preparing dataset...")
    dataset = prepare_dataset(
        data_dir=args.data_dir,
        target=args.target
    )

    # Extract data
    X_train = dataset['X_train']
    y_train = dataset['y_train']
    hospital_train = dataset['hospital_train']
    location_train = dataset['location_train']
    X_val = dataset['X_val']
    y_val = dataset['y_val']
    hospital_val = dataset['hospital_val']
    location_val = dataset['location_val']
    X_test = dataset['X_test']
    y_test = dataset['y_test']
    hospital_test = dataset['hospital_test']

    patient_dim = X_train.shape[1]
    hospital_dim = hospital_train.shape[1]
    num_locations = len(dataset['location_map'])

    logging.info(f"Patient features: {patient_dim}, Hospital features: {hospital_dim}")
    logging.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Create model
    logging.info("Creating ContextSurg model...")
    logging.info(f"Model config: num_experts={config['num_experts']}, "
                f"embedding_dim={config['embedding_dim']}, "
                f"expert_dim={config['expert_dim']}")

    model = ContextSurg(
        patient_feature_dim=patient_dim,
        hospital_feature_dim=hospital_dim,
        num_experts=config['num_experts'],
        num_locations=num_locations,
        embedding_dim=config['embedding_dim'],
        hospital_embedding_dim=config['hospital_embedding_dim'],
        expert_dim=config['expert_dim'],
        shared_hidden_dims=config['shared_hidden_dims'],
        router_hidden_dims=config['router_hidden_dims'],
        expert_hidden_dims=config['expert_hidden_dims'],
        lambda_fair=config['lambda_fair'],
        lambda_adv=config['lambda_adv'],
        lambda_pred=config['lambda_pred'],
        all_train_hospital_features=dataset['all_train_hospital_features'].to(device)
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model parameters: {num_params:,}")

    # Train model
    logging.info("Training model...")
    model = train_model(
        model, X_train, y_train, hospital_train, location_train,
        X_val, y_val, hospital_val, location_val,
        config, device
    )

    # Evaluate model
    logging.info("Evaluating model...")
    results = evaluate_model(
        model, X_test, y_test, hospital_test, config, device
    )

    # Compute feature importance
    logging.info("Computing feature importance...")
    feature_importance = compute_feature_importance(
        model, X_test, hospital_test, config, device
    )

    # Save feature importance to CSV
    # Try to get feature names from dataset if available
    feature_names = [f'feature_{i}' for i in range(len(feature_importance))]
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    })
    importance_df = importance_df.sort_values('importance', ascending=False)

    importance_path = os.path.join(args.output_dir, f'feature_importance_{args.target}.csv')
    importance_df.to_csv(importance_path, index=False)
    logging.info(f"Feature importance saved to {importance_path}")
    logging.info(f"Top 5 features: {list(importance_df['feature'].head(5))}")

    # Save model and results
    model_path = os.path.join(args.output_dir, f'model_{args.target}.pth')
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved to {model_path}")

    results_path = os.path.join(args.output_dir, f'results_{args.target}.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    logging.info(f"Results saved to {results_path}")

    logging.info("Training complete!")


if __name__ == '__main__':
    main()
