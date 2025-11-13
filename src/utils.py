import logging
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.optim as optim
import os

# =============================================================================
# Plotting (unchanged)
# =============================================================================
def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Plot the training history (losses and validation metrics).

    Args:
        history: Dictionary containing lists of metrics per epoch.
        save_path: Path to save the plot image. If None, shows the plot.
    """
    num_epochs = len(history.get('train_loss', []))
    if num_epochs == 0:
        logging.info("No training history found to plot.")
        return

    epochs = range(1, num_epochs + 1)
    #plt.style.use('seaborn-v0_8-grid') # Use a pleasant style
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    fig.suptitle('Training History', fontsize=16)

    # Plot Total Losses
    if 'train_loss' in history:
        axs[0, 0].plot(epochs, history['train_loss'], marker='.', linestyle='-', label='Train Loss')
    if 'val_loss' in history:
        axs[0, 0].plot(epochs, history['val_loss'], marker='.', linestyle='-', label='Validation Loss')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title('Total Loss')
    if 'train_loss' in history or 'val_loss' in history:
        axs[0, 0].legend()
    axs[0, 0].grid(True)


    # Plot Component Losses (if available)
    comp_loss_plotted = False
    if 'pred_loss' in history:
        axs[0, 1].plot(epochs, history['pred_loss'], marker='.', linestyle='--', label='Prediction Loss')
        comp_loss_plotted = True
    if 'adv_loss' in history:
        axs[0, 1].plot(epochs, history['adv_loss'], marker='.', linestyle='--', label='Adversarial Loss')
        comp_loss_plotted = True
    if 'diversity_loss' in history:
        axs[0, 1].plot(epochs, history['diversity_loss'], marker='.', linestyle='--', label='Diversity Loss')
        comp_loss_plotted = True
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].set_title('Loss Components (Train Avg)')
    if comp_loss_plotted:
        axs[0, 1].legend()
    axs[0, 1].grid(True)


    # Plot Validation Metrics (AUROC/AUPRC)
    if 'auroc' in history:
        axs[1, 0].plot(epochs, history['auroc'], marker='.', linestyle='-', label='Validation AUROC')
    if 'auprc' in history:
        axs[1, 1].plot(epochs, history['auprc'], marker='.', linestyle='-', label='Validation AUPRC')

    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Score')
    axs[1, 0].set_title('Validation AUROC')
    if 'auroc' in history: axs[1, 0].legend()
    axs[1, 0].grid(True)


    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Score')
    axs[1, 1].set_title('Validation AUPRC')
    if 'auprc' in history: axs[1, 1].legend()
    axs[1, 1].grid(True)


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Training history plot saved to {save_path}")
        except Exception as e:
            logging.error(f"Error saving plot: {e}")
    else:
        plt.show()
    plt.close(fig) # Close the figure after saving/showing
    
# =============================================================================
# Results Printing (MODIFIED for ECE)
# =============================================================================
def print_eval_results(results_dict: Dict, title: str = "Evaluation Results"):
    """
    Helper function to print evaluation results dictionary nicely.
    MODIFIED: Includes ECE scores.
    """
    logging.info(f"\n--- {title} ---")
    # Define print order preference
    key_order = [
        'auc', 'ap', 'accuracy_at_threshold',
        'ece_raw', 'ece_calibrated', # ++ Added ECE ++
        'fair_auc', 'fair_ap',
        'adv_accuracy', 'diversity', 'adv_confusion', 'expert_counts',
        'fairness_metrics_prediction', 'fairness_metrics_fair_risk',
        'threshold' # Added threshold for clarity
    ]
    printed_keys = set()

    # Print keys in preferred order
    for key in key_order:
        if key in results_dict:
            value = results_dict[key]
            if key == 'adv_confusion':
                logging.info(f"\n{key}:")
                # Ensure confusion matrix is printed reasonably
                try:
                    if isinstance(value, np.ndarray) and value.size > 0:
                         logging.info(np.array_str(value, precision=0, suppress_small=True))
                    elif value.size == 0 :
                         logging.info(" (Not Available / Empty)")
                    else:
                         logging.info(value)
                except Exception:
                    logging.info(value) # Fallback
            # elif key.startswith('fairness_metrics_'):
            #     fairness_type = key.replace('fairness_metrics_', '').upper()
            #     logging.info(f"\n--- Fairness Metrics ({fairness_type}) ---")
            #     if not isinstance(value, dict) or not value:
            #         logging.info("   (No fairness data calculated or provided)")
            #         continue
            #     for group_name, metrics in value.items():
            #         logging.info(f"\n   Grouping: '{group_name}'")
            #         has_details = 'group_samples' in metrics or 'group_auc' in metrics
            #         # Safely get threshold for title
            #         threshold_val = metrics.get('threshold', 'N/A')
            #         logging.info(f"     (Threshold: {threshold_val}, Details: {'Included' if has_details else 'Omitted'})")

            #         # Print summary keys first
            #         summary_keys = ['privileged_group_spec', 'privileged_group_used',
            #                         'auc_disparity', 'ap_disparity', 'max_abs_eod', 'max_abs_dpd']
            #         for sub_key in summary_keys:
            #             if sub_key in metrics:
            #                 sub_value = metrics[sub_key]
            #                 logging.info(f"     {sub_key}: {sub_value:.4f}" if isinstance(sub_value, (float, np.floating)) and pd.notna(sub_value) else f"     {sub_key}: {sub_value}")

            #         # Print detailed keys if they exist (controlled by show_detailed_fairness)
            #         detailed_keys = ['group_samples', 'group_auc', 'group_ap', 'group_tpr', 'group_ppr',
            #                          'eod_difference', 'dpd_difference']
            #         for sub_key in detailed_keys:
            #             if sub_key in metrics:
            #                 sub_value = metrics[sub_key]
            #                 if isinstance(sub_value, dict):
            #                     logging.info(f"     {sub_key}:")
            #                     sorted_items = sorted(sub_value.items())
            #                     if not sorted_items: logging.info("       (No data or N/A)")
            #                     for grp_label, metric in sorted_items:
            #                         logging.info(f"       {grp_label}: {metric:.4f}" if isinstance(metric, (float, np.floating)) and pd.notna(metric) else f"       {grp_label}: {metric}")
            #                 else: # Should not happen for these keys, but as fallback
            #                     logging.info(f"     {sub_key}: {sub_value:.4f}" if isinstance(sub_value, (float, np.floating)) and pd.notna(sub_value) else f"     {sub_key}: {sub_value}")
            elif isinstance(value, (float, np.floating)):
                logging.info(f"{key}: {value:.4f}")
            else:
                logging.info(f"{key}: {value}") # Print other types (like dicts, strings) as is
            printed_keys.add(key)

    # Print any remaining keys not in the preferred order
    for key, value in results_dict.items():
        if key not in printed_keys:
             logging.info(f"{key} (other): {value}") # Indicate it wasn't in the main list
    logging.info("-" * (len(title) + 6))
    
# =============================================================================
# Fairness Metrics Calculation (unchanged)
# =============================================================================
def calculate_fairness_metrics_grouped(
    scores: np.ndarray,
    y_true: np.ndarray,
    grouping_data: Dict[str, np.ndarray],
    threshold: float,
    num_samples: int,
    privileged_groups: Optional[Dict[str, Any]] = None,
    show_detailed_fairness: bool = False
) -> Dict[str, Dict]:
    """
    Calculates grouped fairness metrics (disparities, EOD, DPD) based on groupings.

    Metrics calculated:
    - AUC/AP Disparity: Max - Min performance across groups.
    - Equal Opportunity Difference (EOD): Difference in True Positive Rates (TPR)
      between unprivileged and privileged groups. `max_abs_eod` is the max absolute difference.
    - Demographic Parity Difference (DPD): Difference in Positive Prediction Rates (PPR)
      between unprivileged and privileged groups. `max_abs_dpd` is the max absolute difference.

    Args:
        scores: Model prediction scores (probabilities). Should be calibrated for EOD/DPD.
        y_true: True binary labels (0 or 1).
        grouping_data: Dict mapping group names (e.g., 'iso2', 'wb') to NumPy arrays
                           of group labels for each sample.
        threshold: Classification threshold to binarize scores for EOD/DPD.
        num_samples: Total number of samples (for validation).
        privileged_groups: Dict mapping group names to the privileged group value
                           (e.g., {'iso2': 'US', 'wb': 2}). Required for EOD/DPD.
        show_detailed_fairness: If True, include per-group metrics in the result.

    Returns:
        Dict where keys are grouping names. Each value is a dict containing
        fairness metrics for that grouping.
    """
    fairness_results = {}
    if not isinstance(grouping_data, dict) or not grouping_data:
        logging.warning("calculate_fairness_metrics_grouped: No valid grouping_data provided.")
        return fairness_results

    if privileged_groups is None:
        privileged_groups = {}
        logging.warning("No privileged_groups specified. EOD/DPD metrics will be skipped.")

    predicted_labels = (scores >= threshold).astype(int) # Binarize for EOD/DPD

    # --- Loop through each specified grouping ---
    for group_name, group_array in grouping_data.items():
        logging.info(f"   Calculating fairness for grouping: '{group_name}'...")

        # Basic validation
        if not isinstance(group_array, np.ndarray):
            logging.warning(f"   Data for grouping '{group_name}' is not a NumPy array. Skipping.")
            continue
        if len(group_array) != num_samples:
            logging.warning(f"   Length mismatch: grouping '{group_name}' ({len(group_array)}) vs expected ({num_samples}). Skipping.")
            continue

        # Initialize results for this grouping
        fm_results = {
            'threshold': threshold,
            'privileged_group_spec': privileged_groups.get(group_name),
            'privileged_group_used': None, # Will be set if EOD/DPD are calculated
            'auc_disparity': np.nan, 'ap_disparity': np.nan,
            'max_abs_eod': np.nan, 'max_abs_dpd': np.nan,
        }
        if show_detailed_fairness:
            fm_results.update({
                'group_samples': {}, 'group_auc': {}, 'group_ap': {},
                'group_tpr': {}, 'group_ppr': {},
                'eod_difference': {}, 'dpd_difference': {}
            })
        fairness_results[group_name] = fm_results

        try:
            unique_groups = sorted(list(np.unique(group_array)))
        except Exception as e:
            logging.warning(f"   Could not get unique groups for '{group_name}': {e}. Skipping.")
            continue

        if not unique_groups:
             logging.warning(f"   No unique groups found for '{group_name}'. Skipping.")
             continue

        # --- Calculate metrics per group (needed internally & for detailed output) ---
        group_metrics_cache = {} # Store {group_value: {'tpr': float, 'ppr': float, 'auc': float, 'ap': float}}
        temp_group_aucs, temp_group_aps = {}, {} # Store {group_label: value}

        for group in unique_groups:
            mask = (group_array == group)
            group_label = f"{group_name}_{str(group)}" # For dictionary keys

            if not np.any(mask):
                logging.warning(f"   Group '{group}' in '{group_name}' has no samples. Assigning NaN metrics.")
                group_metrics_cache[group] = {'tpr': np.nan, 'ppr': np.nan, 'auc': np.nan, 'ap': np.nan, 'n': 0}
                temp_group_aucs[group_label], temp_group_aps[group_label] = np.nan, np.nan
                if show_detailed_fairness:
                    fm_results['group_samples'][group_label] = 0
                    # Initialize detailed metrics to NaN
                    for metric_dict in ['group_auc', 'group_ap', 'group_tpr', 'group_ppr']:
                        fm_results[metric_dict][group_label] = np.nan
                continue

            y_group = y_true[mask]
            pred_group_scores = scores[mask]
            pred_group_labels = predicted_labels[mask]
            n_group = len(y_group)
            group_metrics_cache[group] = {'n': n_group}

            if show_detailed_fairness:
                fm_results['group_samples'][group_label] = n_group

            # Calculate AUC/AP
            auc_group, ap_group = np.nan, np.nan
            if len(np.unique(y_group)) > 1: # Need both classes for AUC/AP
                try:
                    auc_group = roc_auc_score(y_group, pred_group_scores)
                    ap_group = average_precision_score(y_group, pred_group_scores)
                except ValueError as e:
                    logging.warning(f"   Could not calculate AUC/AP for group '{group}' in '{group_name}': {e}")
            else:
                 logging.warning(f"   Group '{group}' in '{group_name}' has only one class; AUC/AP are undefined.")
            temp_group_aucs[group_label], temp_group_aps[group_label] = auc_group, ap_group
            group_metrics_cache[group].update({'auc': auc_group, 'ap': ap_group})
            if show_detailed_fairness:
                fm_results['group_auc'][group_label] = auc_group
                fm_results['group_ap'][group_label] = ap_group

            # Calculate TPR/PPR (threshold-dependent)
            actual_positives = np.sum(y_group == 1)
            predicted_positives = np.sum(pred_group_labels == 1)
            true_positives = np.sum((y_group == 1) & (pred_group_labels == 1))

            tpr = true_positives / actual_positives if actual_positives > 0 else np.nan
            ppr = predicted_positives / n_group if n_group > 0 else np.nan
            group_metrics_cache[group].update({'tpr': tpr, 'ppr': ppr})
            if show_detailed_fairness:
                fm_results['group_tpr'][group_label] = tpr
                fm_results['group_ppr'][group_label] = ppr

        # --- Calculate Summary Disparities (AUC/AP) ---
        valid_aucs = [m['auc'] for m in group_metrics_cache.values() if pd.notna(m['auc'])]
        valid_aps = [m['ap'] for m in group_metrics_cache.values() if pd.notna(m['ap'])]
        fm_results['auc_disparity'] = max(valid_aucs) - min(valid_aucs) if len(valid_aucs) >= 2 else (0.0 if len(valid_aucs) == 1 else np.nan)
        fm_results['ap_disparity'] = max(valid_aps) - min(valid_aps) if len(valid_aps) >= 2 else (0.0 if len(valid_aps) == 1 else np.nan)
        logging.info(f"     Summary Disparities: AUC={fm_results['auc_disparity']:.4f}, AP={fm_results['ap_disparity']:.4f}")


        # --- Calculate EOD/DPD Differences and Summaries ---
        priv_group_value = fm_results['privileged_group_spec']
        temp_eod_diffs, temp_dpd_diffs = {}, {} # {group_label: value}

        if priv_group_value is None:
            logging.warning(f"     No privileged group specified for '{group_name}'. Skipping EOD/DPD calculation.")
        elif priv_group_value not in group_metrics_cache:
            logging.warning(f"     Specified privileged group '{priv_group_value}' not found in calculated metrics for '{group_name}'. Skipping EOD/DPD.")
        else:
            fm_results['privileged_group_used'] = priv_group_value
            priv_metrics = group_metrics_cache[priv_group_value]
            priv_tpr, priv_ppr = priv_metrics['tpr'], priv_metrics['ppr']

            if pd.isna(priv_tpr): logging.warning(f"     Privileged group '{priv_group_value}' has NaN TPR. EOD metrics will be NaN.")
            if pd.isna(priv_ppr): logging.warning(f"     Privileged group '{priv_group_value}' has NaN PPR. DPD metrics will be NaN.")

            for group in unique_groups:
                if group == priv_group_value: continue # Compare others *to* the privileged group

                group_label = f"{group_name}_{str(group)}"
                unpriv_metrics = group_metrics_cache[group]
                unpriv_tpr, unpriv_ppr = unpriv_metrics['tpr'], unpriv_metrics['ppr']

                # Calculate difference: unprivileged - privileged
                eod = unpriv_tpr - priv_tpr if pd.notna(unpriv_tpr) and pd.notna(priv_tpr) else np.nan
                dpd = unpriv_ppr - priv_ppr if pd.notna(unpriv_ppr) and pd.notna(priv_ppr) else np.nan

                temp_eod_diffs[group_label], temp_dpd_diffs[group_label] = eod, dpd

            # Store detailed differences if requested
            if show_detailed_fairness:
                fm_results['eod_difference'] = temp_eod_diffs
                fm_results['dpd_difference'] = temp_dpd_diffs

            # Calculate summary max absolute differences
            valid_eods = [abs(eod) for eod in temp_eod_diffs.values() if pd.notna(eod)]
            valid_dpds = [abs(dpd) for dpd in temp_dpd_diffs.values() if pd.notna(dpd)]
            # Max difference is 0 if only the privileged group exists or has valid metrics
            fm_results['max_abs_eod'] = max(valid_eods) if valid_eods else (0.0 if pd.notna(priv_tpr) else np.nan)
            fm_results['max_abs_dpd'] = max(valid_dpds) if valid_dpds else (0.0 if pd.notna(priv_ppr) else np.nan)

            logging.info(f"     Summary EOD/DPD: Max Abs EOD={fm_results['max_abs_eod']:.4f}, Max Abs DPD={fm_results['max_abs_dpd']:.4f}")

    return fairness_results

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error (ECE).

    Args:
        y_true: True binary labels (0 or 1).
        y_prob: Predicted probabilities for the positive class.
        n_bins: Number of bins to divide the probability space into.

    Returns:
        ECE score.
    """
    if len(y_true) == 0 or len(y_prob) == 0:
        return np.nan

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # Ensure y_prob is within [0, 1]
    y_prob = np.clip(y_prob, 0.0, 1.0)

    bin_limits = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_limits[:-1]
    bin_uppers = bin_limits[1:]

    ece = 0.0
    total_samples = len(y_true)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples whose predicted probability falls into the current bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin) # Proportion of samples in this bin

        if prop_in_bin > 0:
            # Accuracy within the bin (mean true label)
            accuracy_in_bin = np.mean(y_true[in_bin])
            # Average predicted probability within the bin
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            # Add weighted absolute difference to ECE
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return ece

# =============================================================================
# Model Training (unchanged)
# =============================================================================
def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    hospital_train: torch.Tensor,
    location_train: torch.Tensor, # Indices
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    hospital_val: torch.Tensor,
    location_val: torch.Tensor, # Indices
    batch_size: int = 64,
    num_epochs: int = 10, # Default was 5, example call used 10
    learning_rate: float = 5e-4, # Default was 0.001, example call used 5e-4
    lambda_fair: float = 0.1, # Weight for diversity loss (if applicable)
    lambda_adv: float = 0.1, # Weight for adversarial loss (if applicable)
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    output_dir: str = './major/output/',
    wb: str = '',
    seed: int = 42 # Pass seed for worker_init_fn consistency
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train the model with specified hyperparameters and track history.

    Args:
        model: The PyTorch model instance (ContextSurg or BaselineMLP).
        X_train, y_train, hospital_train, location_train: Training tensors.
        X_val, y_val, hospital_val, location_val: Validation tensors.
        batch_size: Number of samples per batch.
        num_epochs: Number of training epochs.
        learning_rate: Optimizer learning rate.
        lambda_fair: Weight for the diversity/fairness loss component.
        lambda_adv: Weight for the adversarial loss component.
        device: Computation device ('cuda' or 'cpu').
        output_dir: Directory to save the best model checkpoint.
        seed: Random seed for DataLoader workers.

    Returns:
        Tuple containing:
        - The best model based on validation AUROC.
        - Dictionary containing training history (losses, metrics per epoch).
    """
    logging.info(f"Starting model training on {device}...")

    # Handle empty validation set case
    has_validation_set = X_val is not None and len(X_val) > 0

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- Data Loaders ---
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train, hospital_train, location_train)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=True if device == 'cuda' else False
    )

    if has_validation_set:
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val, hospital_val, location_val)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True,
            pin_memory=True if device == 'cuda' else False
        )
    else:
        val_loader = None
        logging.warning("No validation data provided to train_model. Model saving based on training loss (last epoch).")

    # --- Training History & Best Model Tracking ---
    history = {
        'train_loss': [], 'val_loss': [],
        'pred_loss': [], 'adv_loss': [], 'diversity_loss': [], # Model-specific losses
        'auroc': [], 'auprc': [] # Validation metrics
    }
    best_val_metric = -1.0 # Use AUROC if available, else negative train loss
    best_model_path = os.path.join(output_dir, f'output/best_model{wb}.pt')
    os.makedirs(output_dir, exist_ok=True) # Ensure directory exists
    os.makedirs(os.path.join(output_dir, 'output'), exist_ok=True) # Ensure output subdirectory exists

    # --- Training Loop ---
    for epoch in range(num_epochs):
        model.train()
        epoch_train_losses = {'total': 0.0, 'pred': 0.0, 'adv': 0.0, 'diversity': 0.0}
        num_train_batches = len(train_loader)

        for batch_idx, (batch_X, batch_y, batch_hosp, batch_loc) in enumerate(train_loader):
            batch_X, batch_y, batch_hosp, batch_loc = \
                batch_X.to(device), batch_y.to(device), batch_hosp.to(device), batch_loc.to(device)


            outputs = model(batch_X, batch_hosp) # Forward pass

            # Compute loss (model should have this method)
            # Assumes model.compute_loss returns a dict with 'total_loss', 'pred_loss', etc.
            loss_dict = model.compute_loss(
                outputs, batch_y, batch_loc,
                lambda_fair=lambda_fair, lambda_adv=lambda_adv
            )
            total_loss = loss_dict['total_loss']

            optimizer.zero_grad()
            total_loss.backward() # Backward pass
            optimizer.step()      # Update weights

            # Accumulate losses for reporting
            epoch_train_losses['total'] += total_loss.item()
            epoch_train_losses['pred'] += loss_dict.get('pred_loss', torch.tensor(0.0)).item()
            epoch_train_losses['adv'] += loss_dict.get('adv_loss', torch.tensor(0.0)).item()
            epoch_train_losses['diversity'] += loss_dict.get('diversity_loss', torch.tensor(0.0)).item()

        # --- Validation Phase ---
        avg_train_loss = epoch_train_losses['total'] / num_train_batches
        avg_pred_loss = epoch_train_losses['pred'] / num_train_batches
        avg_adv_loss = epoch_train_losses['adv'] / num_train_batches
        avg_div_loss = epoch_train_losses['diversity'] / num_train_batches
        history['train_loss'].append(avg_train_loss)
        history['pred_loss'].append(avg_pred_loss)
        history['adv_loss'].append(avg_adv_loss)
        history['diversity_loss'].append(avg_div_loss)

        val_auroc, val_auprc, avg_val_loss = np.nan, np.nan, np.nan
        current_epoch_metric = -avg_train_loss # Default if no validation

        if has_validation_set and val_loader is not None:
            model.eval()
            epoch_val_loss = 0.0
            val_predictions_list = []
            val_targets_list = []
            num_val_batches = len(val_loader)

            with torch.no_grad():
                for batch_X, batch_y, batch_hosp, batch_loc in val_loader:
                    batch_X, batch_y, batch_hosp, batch_loc = \
                        batch_X.to(device), batch_y.to(device), batch_hosp.to(device), batch_loc.to(device)

                    outputs = model(batch_X, batch_hosp)
                    # Ensure compute_loss exists and handles eval mode if necessary
                    if hasattr(model, 'compute_loss') and callable(model.compute_loss):
                        loss_dict = model.compute_loss(outputs, batch_y, batch_loc, lambda_fair, lambda_adv)
                        epoch_val_loss += loss_dict['total_loss'].item()
                    else:
                        criterion = nn.BCEWithLogitsLoss() # Or BCELoss if model outputs probabilities
                        if 'final_prediction' in outputs: # Assuming this is the primary output score/logit
                            pred_output = outputs['final_prediction']
                            loss = criterion(pred_output, batch_y)
                            epoch_val_loss += loss.item()
                        else:
                            logging.warning("Cannot compute validation loss: 'final_prediction' not in model outputs and no compute_loss method found.")


                    if 'final_prediction' in outputs:
                        val_predictions_list.append(outputs['final_prediction'].cpu().numpy())
                        val_targets_list.append(batch_y.cpu().numpy())
                    else:
                         logging.warning("Cannot compute validation metrics: 'final_prediction' not in model outputs.")


            avg_val_loss = epoch_val_loss / num_val_batches if num_val_batches > 0 else np.nan

            if val_predictions_list: # Check if we collected predictions
                val_predictions = np.concatenate(val_predictions_list)
                val_targets = np.concatenate(val_targets_list)

                # Handle cases with only one class in validation batch/epoch
                if len(np.unique(val_targets)) > 1:
                    try:
                        val_auroc = roc_auc_score(val_targets, val_predictions)
                        val_auprc = average_precision_score(val_targets, val_predictions)
                    except ValueError as e:
                        logging.warning(f"Could not compute validation AUC/AUPRC in epoch {epoch+1}: {e}")
                else:
                    logging.warning(f"Only one class present in validation set epoch {epoch+1}, AUC/AUPRC undefined.")

                current_epoch_metric = val_auprc # Use AUROC for saving best model
            else:
                 current_epoch_metric = -avg_val_loss # Fallback if predictions weren't generated

        # --- Update History ---
        history['val_loss'].append(avg_val_loss)
        history['auroc'].append(val_auroc)
        history['auprc'].append(val_auprc)

        # --- Print Progress ---
        log_msg = (f"Epoch {epoch+1}/{num_epochs} | "
                   f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                   f"Val AUROC: {val_auroc:.4f} | Val AUPRC: {val_auprc:.4f}")
        if not has_validation_set:
             log_msg = (f"Epoch {epoch+1}/{num_epochs} | "
                        f"Train Loss: {avg_train_loss:.4f} | (No Validation Set)")

        logging.info(log_msg)
        # Log components only if they are non-zero (or always if preferred)
        # logging.info(f"  Loss Components: Pred={avg_pred_loss:.4f}, Adv={avg_adv_loss:.4f}, Div={avg_div_loss:.4f}")


        # --- Save Best Model ---
        # Prioritize validation AUROC if available and valid
        metric_to_compare = best_val_metric
        if not np.isnan(current_epoch_metric):
             # Check if current metric is better than best metric so far
             # Note: If using loss, lower is better (-loss means higher is better)
             is_better = (current_epoch_metric > metric_to_compare) if not np.isnan(val_auroc) else \
                         (current_epoch_metric > metric_to_compare) # Assumes current_epoch_metric = -avg_train_loss here

             if is_better:
                 best_val_metric = current_epoch_metric
                 torch.save(model.state_dict(), best_model_path)
                 logging.info(f"   => New best model saved (Metric: {best_val_metric:.4f})")
        elif epoch == num_epochs - 1: # Save last model if no metric improved or available
             torch.save(model.state_dict(), best_model_path)
             logging.info(f"   => Saving model from last epoch (no improvement or no valid metric).")


    # --- Load Best Model ---
    if os.path.exists(best_model_path):
        logging.info(f"Training finished. Loading best model from {best_model_path} (Best Metric: {best_val_metric:.4f})")
        model.load_state_dict(torch.load(best_model_path))
    else:
        logging.warning("Best model path not found after training. Returning the last epoch model.")

    return model, history