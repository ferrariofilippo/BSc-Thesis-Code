"""Evaluate the saved NN, RF, DT, GP and RSM on the same held-out dataset.

Run ``python final_evaluation.py`` to write one <model>_predictions.csv per
approach and a shared metrics.json containing training, validation and final
evaluation metrics. Use --models to evaluate a subset, or
--checkpoint-dir / --<model>-checkpoint to select different saved checkpoints.
No models are trained and no preprocessing statistics are fitted here.

New checkpoints contain their own log-space preprocessing statistics. Legacy
NN raw-scale checkpoints can still use the optional standardization sidecar.
Invalid rows are preserved with diagnostics and excluded from the applicable
metrics; their counts are always reported. Missing inputs are never imputed.

Smooth L1 uses z = (log10(tol) - training_mean) / training_std, as in NN
training. Compare it with standardized_log10_mse, not log10_mse: the latter
uses unstandardized log10 residuals. At beta=0.5 and with identical scored
rows, Smooth L1 <= standardized_log10_mse = log10_mse / training_std**2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.predict.neuralnetwork.model import ACTIVATIONS, NeuralNetworkModel


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "final_validation_preprocessed.csv"
DEFAULT_TRAINING_DATASET = PROJECT_ROOT / "data" / "preprocessed.csv"
APPROACHES = ("nn", "rf", "dt", "gp", "rsm")
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "conf" / "tolerance-prediction"
DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINT_DIR / "nn.pt"
DEFAULT_STANDARDIZATION = PROJECT_ROOT / "data" / "standardization.npz"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "final_evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the five saved tolerance-prediction approaches (no training)."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--training-dataset", type=Path, default=DEFAULT_TRAINING_DATASET)
    parser.add_argument("--split-seed", type=int, default=180)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--rows-per-problem", type=int, default=5)
    parser.add_argument(
        "--smooth-l1-beta", type=float, default=0.5,
        help="Fallback beta for checkpoints that do not save their training loss metadata.",
    )
    parser.add_argument("--models", nargs="+", choices=APPROACHES, default=list(APPROACHES))
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument(
        "--nn-checkpoint", "--checkpoint", dest="nn_checkpoint", type=Path,
        help="NN checkpoint override (--checkpoint is retained as an alias).",
    )
    for approach in APPROACHES[1:]:
        parser.add_argument(f"--{approach}-checkpoint", type=Path)
    parser.add_argument(
        "--gp-point-estimate", choices=("mean", "median"), default="median",
        help=(
            "GP tolerance estimate: mean matches GaussianProcess.predict; median "
            "exponentiates its log-space mean, as in training validation."
        ),
    )
    parser.add_argument(
        "--standardization", type=Path, default=DEFAULT_STANDARDIZATION
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-column", default="tol")
    parser.add_argument(
        "--id-column",
        default="id",
        help=(
            "Column to use as the prediction ID. If it is absent, the input "
            "row's zero-based index is used."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--strict", action="store_true",
        help="Fail without exporting reports if any row cannot be fully evaluated.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="NN device (default: CUDA when available); sklearn models always use CPU.",
    )
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CheckpointNeuralNetwork(torch.nn.Module):
    """NN reconstructed from serialized layer shapes, including older layouts."""

    def __init__(self, dimensions, hidden_activation, output_activation):
        super().__init__()
        modules = []
        for index, (input_dim, output_dim) in enumerate(zip(dimensions, dimensions[1:])):
            modules.append(torch.nn.Linear(input_dim, output_dim))
            if index < len(dimensions) - 2:
                modules.extend((ACTIVATIONS[hidden_activation](), torch.nn.Dropout(p=0.0)))
            else:
                modules.append(ACTIVATIONS[output_activation]())
        self.model = torch.nn.Sequential(*modules)

    def forward(self, features):
        return self.model(features)


def build_model(checkpoint: dict, device: torch.device) -> torch.nn.Module:
    """Rebuild the NN from saved tensor shapes, not today's mutable model class."""
    try:
        cfg = checkpoint["model_cfg"]
        model_state = checkpoint["model_state"]
    except KeyError as exc:
        raise KeyError(f"Checkpoint is missing required key: {exc.args[0]}") from exc

    hidden_activation = cfg["hidden_activation"]
    output_activation = cfg["output_activation"]
    if hidden_activation not in ACTIVATIONS or output_activation not in ACTIVATIONS:
        raise ValueError("Checkpoint specifies an unknown NN activation.")
    weights = sorted(
        (
            (int(name.split(".")[1]), value)
            for name, value in model_state.items()
            if name.startswith("model.") and name.endswith(".weight") and value.ndim == 2
        ),
        key=lambda item: item[0],
    )
    if not weights:
        raise ValueError("Checkpoint contains no fully connected NN layers.")
    expected_indices = list(range(0, 3 * len(weights), 3))
    if [index for index, _ in weights] != expected_indices:
        raise ValueError("Checkpoint NN layer layout is unsupported.")
    dimensions = [int(weights[0][1].shape[1])]
    for _, weight in weights:
        if int(weight.shape[1]) != dimensions[-1]:
            raise ValueError("Checkpoint NN layer dimensions are inconsistent.")
        dimensions.append(int(weight.shape[0]))
    if dimensions[0] != int(cfg["input_dim"]) or dimensions[-1] != int(cfg["output_dim"]):
        raise ValueError("Checkpoint NN tensors disagree with model_cfg input/output dimensions.")
    model = CheckpointNeuralNetwork(
        dimensions, hidden_activation, output_activation
    ).to(device)
    model.load_state_dict(model_state)
    invalid_parameters = [
        name for name, value in model.state_dict().items()
        if not torch.isfinite(value).all()
    ]
    if invalid_parameters:
        raise ValueError(f"Checkpoint contains non-finite parameters: {invalid_parameters}")
    model.eval()
    return model


@torch.inference_mode()
def predict_in_batches(
    model,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    predictions: list[np.ndarray] = []
    for start in range(0, len(features), batch_size):
        batch = features[start : start + batch_size]
        if isinstance(model, torch.nn.Module):
            tensor = torch.as_tensor(batch, dtype=torch.float32, device=device)
            output = model(tensor).cpu().numpy()
        else:
            output = model.predict(batch)
        predictions.append(single_target_output(output, len(batch)))

    return np.concatenate(predictions) if predictions else np.empty(0, dtype=np.float64)


def single_target_output(output, count):
    output = np.asarray(output, dtype=np.float64)
    if output.shape not in ((count,), (count, 1)):
        raise ValueError(f"Expected one prediction per row, got output shape {output.shape}.")
    return output.reshape(count)


class GPMeanPredictor:
    """Express the raw lognormal mean as a standardized log10 point estimate.

    If z is the GP's standardized Gaussian target, then
    log10(E[tol]) = target_mean + target_std * E[z]
                   + 0.5 * ln(10) * (target_std * std[z])**2.
    This matches src/predict/gp.py without exponentiating prematurely.
    """

    def __init__(self, model, target_std):
        self.model = model
        self.target_std = float(target_std)

    def predict(self, features):
        mean, std = self.model.predict(features, return_std=True)
        mean = single_target_output(mean, len(features))
        std = single_target_output(std, len(features))
        with np.errstate(over="ignore", invalid="ignore"):
            result = mean + 0.5 * np.log(10.0) * self.target_std * np.square(std)
        return np.where(np.isfinite(std) & (std >= 0), result, np.nan)


def checkpoint_path(args, approach):
    override = getattr(args, f"{approach}_checkpoint")
    if override is not None:
        return override
    path = args.checkpoint_dir / ("nn.pt" if approach == "nn" else f"{approach}.joblib")
    if approach == "nn" and not path.exists():
        legacy = args.checkpoint_dir / "best_model.pt"
        if legacy.exists():
            return legacy
    return path


def load_approach(approach, path, device, standardization, feature_columns,
                  gp_point_estimate, fallback_beta=0.5):
    """Load once per approach, using only that checkpoint's training statistics."""
    if approach == "nn":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = build_model(checkpoint, device)
        input_dim = model.model[0].in_features
        preprocessing = checkpoint.get("preprocessing")
    else:
        checkpoint = joblib.load(path)
        if not isinstance(checkpoint, dict) or not {"model", "preprocessing"}.issubset(checkpoint):
            raise ValueError("Regression checkpoint must contain model and log-space preprocessing.")
        model = checkpoint["model"]
        input_dim = model.n_features_in_
        preprocessing = checkpoint["preprocessing"]
        if preprocessing is None:
            raise ValueError("Regression checkpoint is missing log-space preprocessing.")

    if len(feature_columns) != input_dim:
        raise ValueError(f"Dataset has {len(feature_columns)} features, but the model expects {input_dim}.")
    if preprocessing is None:
        with np.load(standardization) as stats:
            means = stats["means"].astype(np.float64)
            stds = stats["stds"].astype(np.float64)
    else:
        means = np.asarray(preprocessing["means"], dtype=np.float64)
        stds = np.asarray(preprocessing["stds"], dtype=np.float64)
    validate_preprocessing(means, stds, feature_columns, preprocessing)

    prediction_kind = "inverse_log10_point_estimate" if preprocessing is not None else "raw_point_estimate"
    if approach == "gp":
        prediction_kind = f"lognormal_{gp_point_estimate}"
        if gp_point_estimate == "mean":
            model = GPMeanPredictor(model, stds[0])
    metadata = {
        "checkpoint": str(path.resolve()),
        "standardization": None if preprocessing is not None else str(standardization.resolve()),
        "device": str(device),
        "prediction_kind": prediction_kind,
        "training_loss": checkpoint.get("loss", "smooth_l1"),
        "smooth_l1_beta": float(checkpoint.get("huber_beta", fallback_beta)),
        "smooth_l1_beta_source": "checkpoint" if "huber_beta" in checkpoint else "fallback",
        "target_log10_mean": float(means[0]) if preprocessing is not None else None,
        "target_log10_std": float(stds[0]) if preprocessing is not None else None,
        "saved_validation_loss": (
            float(checkpoint["val_loss"]) if checkpoint.get("val_loss") is not None else None
        ),
    }
    return model, means, stds, preprocessing, metadata


def validate_preprocessing(means, stds, feature_columns, preprocessing):
    expected_shape = (len(feature_columns) + 1,)
    if means.shape != expected_shape or stds.shape != expected_shape:
        raise ValueError(
            "Scaling statistics must have one value for the target and each feature: "
            f"expected {expected_shape}, got means={means.shape}, stds={stds.shape}."
        )
    if not np.isfinite(means).all() or not np.isfinite(stds).all() or np.any(stds <= 0):
        raise ValueError("Scaling statistics must be finite, with strictly positive stds.")
    if preprocessing is not None:
        if (
            preprocessing.get("version") != 1
            or preprocessing.get("transform") != "log10_target_and_error"
        ):
            raise ValueError("Unsupported checkpoint preprocessing version or transform.")
        error_index = preprocessing["error_feature_index"]
        if not isinstance(error_index, (int, np.integer)) or not 0 <= error_index < len(feature_columns):
            raise ValueError("Checkpoint error-feature index is invalid.")
        expected_name = preprocessing.get("error_feature_name", "err")
        if feature_columns[error_index] != expected_name:
            raise ValueError(
                f"Expected feature {expected_name!r} at index {error_index}, "
                f"found {feature_columns[error_index]!r}."
            )


def _finite_mean(values: np.ndarray) -> float | None:
    """Average finite nonnegative errors without overflowing their sum."""
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    largest = float(values.max())
    return largest * float(np.mean(values / largest)) if largest else 0.0


def load_training_splits(path, target_column, id_column, feature_columns,
                         seed, validation_fraction, rows_per_problem):
    """Recreate the grouped train/validation split used by IMLEngine."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between zero and one.")
    if rows_per_problem <= 0:
        raise ValueError("rows_per_problem must be greater than zero.")
    data = pd.read_csv(path)
    if data.empty or target_column not in data.columns:
        raise ValueError(f"Training dataset is empty or lacks {target_column!r}: {path}")
    actual_features = [
        column for column in data.columns if column not in {target_column, id_column}
    ]
    if actual_features != list(feature_columns):
        raise ValueError("Training and final-evaluation feature columns/order do not match.")
    numeric = data[[target_column, *actual_features]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float32)
    if not np.isfinite(numeric).all():
        raise ValueError("Training dataset contains missing, non-numeric or non-finite values.")
    if len(numeric) % rows_per_problem:
        raise ValueError(
            f"Training rows ({len(numeric)}) must be a multiple of rows_per_problem "
            f"({rows_per_problem})."
        )
    problem_count = len(numeric) // rows_per_problem
    validation_problem_count = max(1, int(problem_count * validation_fraction))
    training_problem_count = problem_count - validation_problem_count
    if training_problem_count <= 0:
        raise ValueError("validation_fraction leaves no training problems.")
    permutation = np.random.RandomState(seed).permutation(problem_count)

    def rows_for(problem_indices):
        indices = (
            problem_indices[:, None] * rows_per_problem
            + np.arange(rows_per_problem)[None, :]
        ).ravel()
        selected = numeric[indices].copy()
        return {
            "targets": selected[:, 0].astype(np.float64),
            "features": selected[:, 1:].astype(np.float64),
        }

    return {
        "training": rows_for(permutation[:training_problem_count]),
        "validation": rows_for(permutation[training_problem_count:]),
    }


def validate_reconstructed_split(training_split, preprocessing):
    """Prove that CLI split settings reproduce a checkpoint's fitted scaler."""
    if preprocessing is None:
        return None
    combined = np.column_stack((
        training_split["targets"], training_split["features"]
    )).astype(np.float32)
    if np.any(combined[:, :2] <= 0):
        raise ValueError("Training tolerance and error must be positive for log10 scaling.")
    combined[:, 0] = np.log10(combined[:, 0])
    error_index = int(preprocessing["error_feature_index"]) + 1
    combined[:, error_index] = np.log10(combined[:, error_index])
    computed_means = combined.mean(axis=0, dtype=np.float64).astype(np.float32)
    computed_stds = combined.std(axis=0, dtype=np.float64).astype(np.float32)
    computed_stds[computed_stds == 0] = 1.0
    saved_means = np.asarray(preprocessing["means"], dtype=np.float32)
    saved_stds = np.asarray(preprocessing["stds"], dtype=np.float32)
    mean_delta = float(np.max(np.abs(computed_means - saved_means)))
    std_delta = float(np.max(np.abs(computed_stds - saved_stds)))
    if not (
        np.allclose(computed_means, saved_means, rtol=1e-6, atol=1e-7)
        and np.allclose(computed_stds, saved_stds, rtol=1e-6, atol=1e-7)
    ):
        raise ValueError(
            "Reconstructed training split does not match checkpoint preprocessing "
            f"(max mean delta={mean_delta:.3g}, std delta={std_delta:.3g}). "
            "Use the seed, validation fraction and rows-per-problem used for training."
        )
    return {"matches_checkpoint": True, "max_mean_delta": mean_delta, "max_std_delta": std_delta}


def short_metrics(metrics):
    keys = (
        "mse", "log10_mse", "standardized_log10_mse",
        "smooth_l1_standardized_log10", "smooth_l1_beta",
        "num_examples", "num_predictions_valid", "num_invalid_predictions",
        "num_mse_examples", "num_log10_mse_examples",
        "num_standardized_log10_mse_examples", "num_smooth_l1_examples",
        "num_rows_with_issues", "status", "error",
    )
    return {key: metrics[key] for key in keys if key in metrics}


def evaluate_rows(model, features, targets, feature_columns, means, stds,
                  preprocessing, device, batch_size, smooth_l1_beta=0.5):
    """Predict valid inputs only; keep every source row and disclose metric coverage."""
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    stds = np.asarray(stds, dtype=np.float64)
    validate_preprocessing(means, stds, feature_columns, preprocessing)
    if features.ndim != 2 or features.shape != (len(targets), len(feature_columns)):
        raise ValueError("Feature/target shapes do not match the dataset schema.")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    count = len(targets)
    status = np.full(count, "ok", dtype=object)
    reasons = [[] for _ in range(count)]

    def flag(mask, code, reason):
        for row in np.flatnonzero(mask):
            if status[row] == "ok":
                status[row] = code
            reasons[row].append(reason(row) if callable(reason) else reason)

    input_valid = np.isfinite(features).all(axis=1)
    flag(~input_valid, "invalid_input", lambda row: "Missing, non-numeric or non-finite input: " + ", ".join(
        feature_columns[col] for col in np.flatnonzero(~np.isfinite(features[row]))
    ))
    transformed = features.copy()
    if preprocessing is not None:
        error_index = int(preprocessing["error_feature_index"])
        invalid_error = np.isfinite(features[:, error_index]) & (features[:, error_index] <= 0)
        flag(invalid_error, "invalid_input", "Error must be strictly positive for log10 scaling.")
        input_valid &= ~invalid_error
        transformed[input_valid, error_index] = np.log10(features[input_valid, error_index])

    standardized = np.full(features.shape, np.nan, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        standardized[input_valid] = (transformed[input_valid] - means[1:]) / stds[1:]
    overflowed_input = input_valid & ~np.isfinite(standardized).all(axis=1)
    flag(overflowed_input, "invalid_input", "Features overflowed the model's float32 input range.")
    input_valid &= ~overflowed_input

    scaled_predictions = np.full(count, np.nan, dtype=np.float64)
    if input_valid.any():
        scaled_predictions[input_valid] = predict_in_batches(
            model, standardized[input_valid], device, batch_size
        )
    model_valid = input_valid & np.isfinite(scaled_predictions)
    flag(input_valid & ~model_valid, "invalid_model_output", "Model returned a non-finite prediction.")

    predictions = np.full(count, np.nan, dtype=np.float64)
    log_predictions = np.full(count, np.nan, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        restored = scaled_predictions[model_valid] * stds[0] + means[0]
        if preprocessing is not None:
            log_predictions[model_valid] = restored
            predictions[model_valid] = np.power(10.0, restored)
        else:
            predictions[model_valid] = restored
            positive = np.isfinite(predictions) & (predictions > 0)
            log_predictions[positive] = np.log10(predictions[positive])

    positive_prediction = np.isfinite(predictions) & (predictions > 0)
    flag(model_valid & ~positive_prediction, "invalid_prediction",
         "Tolerance is non-positive or outside float64 range; check predicted_log10_tolerance.")
    target_valid = np.isfinite(targets) & (targets > 0)
    flag(~target_valid, "invalid_target", "True tolerance must be finite and strictly positive.")

    squared_errors = np.full(count, np.nan, dtype=np.float64)
    squared_log_errors = np.full(count, np.nan, dtype=np.float64)
    squared_standardized_log_errors = np.full(count, np.nan, dtype=np.float64)
    smooth_l1_errors = np.full(count, np.nan, dtype=np.float64)
    # Legacy negative predictions still have a meaningful raw MSE, but not log MSE.
    raw_mask = target_valid & np.isfinite(predictions)
    if preprocessing is not None:
        raw_mask &= positive_prediction
    log_mask = target_valid & np.isfinite(log_predictions)
    with np.errstate(over="ignore", invalid="ignore"):
        squared_errors[raw_mask] = np.square(predictions[raw_mask] - targets[raw_mask])
        squared_log_errors[log_mask] = np.square(log_predictions[log_mask] - np.log10(targets[log_mask]))
        if preprocessing is not None:
            if not np.isfinite(smooth_l1_beta) or smooth_l1_beta < 0:
                raise ValueError("smooth_l1_beta must be finite and non-negative.")
            # Score the model's standardized output directly against the target
            # standardized with the checkpoint's training statistics. Float64
            # keeps report arithmetic safe; the loss is PyTorch's NN objective.
            standardized_log_targets = (
                np.log10(targets[log_mask]) - means[0]
            ) / stds[0]
            squared_standardized_log_errors[log_mask] = np.square(
                scaled_predictions[log_mask] - standardized_log_targets
            )
            smooth_l1_errors[log_mask] = torch.nn.functional.smooth_l1_loss(
                torch.from_numpy(scaled_predictions[log_mask]),
                torch.from_numpy(standardized_log_targets),
                beta=smooth_l1_beta,
                reduction="none",
            ).numpy()
    flag(raw_mask & ~np.isfinite(squared_errors), "metric_overflow", "Squared tolerance error exceeded float64 range.")
    flag(log_mask & ~np.isfinite(squared_log_errors), "metric_overflow", "Squared log error exceeded float64 range.")
    if preprocessing is not None:
        flag(log_mask & ~np.isfinite(squared_standardized_log_errors), "metric_overflow", "Squared standardized log error exceeded float64 range.")
        flag(log_mask & ~np.isfinite(smooth_l1_errors), "metric_overflow", "Smooth L1 error exceeded float64 range.")
    standardized_log_mask = log_mask & np.isfinite(squared_standardized_log_errors)
    smooth_l1_mask = log_mask & np.isfinite(smooth_l1_errors)
    raw_mask &= np.isfinite(squared_errors)
    log_mask &= np.isfinite(squared_log_errors)

    issues = [
        {"row_index": int(row), "csv_line": int(row + 2), "status": str(status[row]),
         "reason": "; ".join(reasons[row])}
        for row in np.flatnonzero(status != "ok")
    ]
    rows = {
        "true_tolerance": targets,
        "predicted_tolerance": np.where(np.isfinite(predictions), predictions, np.nan),
        "predicted_log10_tolerance": np.where(np.isfinite(log_predictions), log_predictions, np.nan),
        "squared_error": np.where(raw_mask, squared_errors, np.nan),
        "squared_log10_error": np.where(log_mask, squared_log_errors, np.nan),
        "squared_standardized_log10_error": np.where(
            standardized_log_mask, squared_standardized_log_errors, np.nan
        ),
        "smooth_l1_standardized_log10_error": np.where(
            smooth_l1_mask, smooth_l1_errors, np.nan
        ),
        "mse_included": raw_mask, "log10_mse_included": log_mask,
        "standardized_log10_mse_included": standardized_log_mask,
        "smooth_l1_included": smooth_l1_mask,
        "status": status, "reason": ["; ".join(items) for items in reasons],
    }
    metrics = {
        "mse": _finite_mean(squared_errors),
        "log10_mse": _finite_mean(squared_log_errors),
        "standardized_log10_mse": _finite_mean(squared_standardized_log_errors),
        "smooth_l1_standardized_log10": _finite_mean(smooth_l1_errors),
        "smooth_l1_beta": float(smooth_l1_beta),
        "num_examples": count,
        "num_predictions_valid": int(positive_prediction.sum()),
        "num_invalid_predictions": int((~positive_prediction).sum()),
        "num_mse_examples": int(raw_mask.sum()),
        "num_log10_mse_examples": int(log_mask.sum()),
        "num_standardized_log10_mse_examples": int(standardized_log_mask.sum()),
        "num_smooth_l1_examples": int(smooth_l1_mask.sum()),
        "num_rows_with_issues": len(issues),
        "status": "complete" if not issues else ("partial" if raw_mask.any() or log_mask.any() else "failed"),
        "metric_scope": "Each metric uses only rows marked included in the approach's predictions CSV; excluded rows are listed in row_issues.",
        "metric_notes": {
            "mse": f"Scored {int(raw_mask.sum())}/{count} rows; null only when no finite squared errors are available.",
            "log10_mse": f"Scored {int(log_mask.sum())}/{count} rows; null only when no finite squared log errors are available.",
        },
        "row_issues": issues,
    }
    return rows, metrics


def failed_evaluation(targets, error):
    """Export a clear failure instead of leaving stale predictions for one model."""
    count = len(targets)
    rows = {
        "true_tolerance": targets,
        **{name: np.full(count, np.nan) for name in (
            "predicted_tolerance", "predicted_log10_tolerance",
            "squared_error", "squared_log10_error",
            "squared_standardized_log10_error",
            "smooth_l1_standardized_log10_error",
        )},
        "mse_included": np.zeros(count, dtype=bool),
        "log10_mse_included": np.zeros(count, dtype=bool),
        "standardized_log10_mse_included": np.zeros(count, dtype=bool),
        "smooth_l1_included": np.zeros(count, dtype=bool),
        "status": ["model_error"] * count,
        "reason": [error] * count,
    }
    metrics = {
        "mse": None, "log10_mse": None, "standardized_log10_mse": None,
        "smooth_l1_standardized_log10": None,
        "num_examples": count, "num_predictions_valid": 0,
        "num_invalid_predictions": count, "num_mse_examples": 0,
        "num_log10_mse_examples": 0, "num_smooth_l1_examples": 0,
        "num_standardized_log10_mse_examples": 0,
        "num_rows_with_issues": count,
        "status": "failed", "error": error,
    }
    return rows, metrics


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    data = pd.read_csv(args.dataset)
    if data.empty:
        raise ValueError(f"Evaluation dataset is empty: {args.dataset}")
    if args.target_column not in data.columns:
        raise ValueError(
            f"Target column {args.target_column!r} is missing from {args.dataset}."
        )

    if args.id_column in data.columns:
        ids = data[args.id_column].copy()
    else:
        ids = pd.Series(data.index, index=data.index)

    excluded_columns = {args.target_column, args.id_column}
    feature_columns = [
        column for column in data.columns if column not in excluded_columns
    ]
    targets = pd.to_numeric(data[args.target_column], errors="coerce").to_numpy(dtype=np.float64)
    features = data[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    training_splits = load_training_splits(
        args.training_dataset, args.target_column, args.id_column, feature_columns,
        args.split_seed, args.validation_fraction, args.rows_per_problem,
    )

    results = {}
    # Deduplicate while preserving the requested output order.
    for approach in dict.fromkeys(args.models):
        path = checkpoint_path(args, approach)
        print(f"Evaluating {approach.upper()}: {path}")
        metadata = {"checkpoint": str(path.resolve())}
        try:
            device = select_device(args.device) if approach == "nn" else torch.device("cpu")
            model, means, stds, preprocessing, metadata = load_approach(
                approach, path, device, args.standardization, feature_columns, args.gp_point_estimate,
                args.smooth_l1_beta,
            )
            split_check = validate_reconstructed_split(
                training_splits["training"], preprocessing
            )
            rows, metrics = evaluate_rows(
                model, features, targets, feature_columns, means, stds,
                preprocessing, device, args.batch_size, metadata["smooth_l1_beta"],
            )
            for split_name, split in training_splits.items():
                _, split_metrics = evaluate_rows(
                    model, split["features"], split["targets"], feature_columns,
                    means, stds, preprocessing, device, args.batch_size,
                    metadata["smooth_l1_beta"],
                )
                metrics[split_name] = short_metrics(split_metrics)
            metadata["split_check"] = split_check
            saved_loss = metadata.get("saved_validation_loss")
            saved_loss_comparable = (
                preprocessing is not None
                and metadata["training_loss"] == "smooth_l1"
                and not (approach == "gp" and args.gp_point_estimate == "mean")
            )
            metadata["saved_validation_loss_comparable"] = saved_loss_comparable
            if saved_loss is not None and saved_loss_comparable:
                metrics["recomputed_minus_saved_validation_loss"] = (
                    metrics["validation"]["smooth_l1_standardized_log10"] - saved_loss
                )
        except Exception as exc:
            # A missing/corrupt checkpoint must not prevent evaluating the rest.
            error = f"{type(exc).__name__}: {exc}"
            rows, metrics = failed_evaluation(targets, error)
            print(f"WARNING: {approach.upper()} failed: {error}")
        metrics.update(metadata)
        results[approach] = (rows, metrics)

    issues = {
        name: metrics["num_rows_with_issues"]
        for name, (_, metrics) in results.items() if metrics["num_rows_with_issues"]
    }
    if args.strict and issues:
        raise ValueError(
            f"Strict evaluation failed; no reports were written. Rows with issues per approach: {issues}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    statuses = [metrics["status"] for _, metrics in results.values()]
    overall_status = (
        "complete" if all(status == "complete" for status in statuses)
        else "failed" if all(status == "failed" for status in statuses)
        else "partial"
    )
    summary = {
        "dataset": str(args.dataset.resolve()),
        "training_dataset": str(args.training_dataset.resolve()),
        "num_examples": len(data),
        "split": {
            "seed": args.split_seed,
            "validation_fraction": args.validation_fraction,
            "rows_per_problem": args.rows_per_problem,
            "num_training_examples": len(training_splits["training"]["targets"]),
            "num_validation_examples": len(training_splits["validation"]["targets"]),
        },
        "status": overall_status,
        "metric_definitions": {
            "mse": "Mean squared error in original tolerance units.",
            "log10_mse": "Mean squared difference between unstandardized log10(predicted tolerance) and log10(true tolerance).",
            "standardized_log10_mse": (
                "Mean squared residual on z = (log10(tolerance) - target_log10_mean) / target_log10_std, "
                "using the checkpoint's training statistics. Equals log10_mse / target_log10_std**2 "
                "when scored rows match."
            ),
            "smooth_l1_standardized_log10": (
                "Mean PyTorch Smooth L1 on standardized log10 tolerance z, with smooth_l1_beta. "
                "Matches the NN objective, with float64 report arithmetic. At beta=0.5 it is <= "
                "standardized_log10_mse on the same rows; this bound does not apply to log10_mse."
            ),
        },
        "metric_scope": (
            "All approaches use the same dataset, each with its own saved preprocessing. "
            "Each metric uses only rows marked included in that approach's CSV; "
            "null means no scorable rows. Check coverage before comparing models. "
            "Per-row diagnostics are in the status/reason columns."
        ),
        "models": {},
    }
    for approach, (rows, metrics) in results.items():
        predictions_path = args.output_dir / f"{approach}_predictions.csv"
        pd.DataFrame({"id": ids.to_numpy(), **rows}).to_csv(predictions_path, index=False)
        # Keep the shared JSON short; row-level diagnostics live in each CSV.
        summary["models"][approach] = {
            key: value for key, value in metrics.items()
            if key not in ("row_issues", "metric_notes", "metric_scope")
        }
        summary["models"][approach]["predictions"] = str(predictions_path.resolve())
        print(f"{approach.upper()} metrics:")
        metric_sets = {
            "training": metrics.get("training", {}),
            "validation": metrics.get("validation", {}),
            "evaluation": metrics,
        }
        for split_name, split_metrics in metric_sets.items():
            values = []
            for key, title in (
                ("mse", "MSE"), ("log10_mse", "log10 MSE"),
                ("standardized_log10_mse", "standardized log10 MSE"),
                ("smooth_l1_standardized_log10", "Smooth L1 (standardized log10)"),
            ):
                value = split_metrics.get(key)
                values.append(f"{title}={value:.10g}" if value is not None else f"{title}=unavailable")
            print(f"  {split_name}: " + ", ".join(values))
        if metrics["num_rows_with_issues"]:
            print(f"WARNING: {approach.upper()} has {metrics['num_rows_with_issues']} rows with issues; see {predictions_path.name}.")

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
        file.write("\n")
    print(f"Predictions saved to: {args.output_dir} (<model>_predictions.csv)")
    print(f"Metrics saved to: {metrics_path}")
    return 1 if overall_status == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
