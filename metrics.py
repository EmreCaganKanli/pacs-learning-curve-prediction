import hashlib

import numpy as np
import torch
from scipy.spatial.distance import cdist

from config import DEVICE, DISTANCE_SAMPLE_LIMIT


@torch.no_grad()
def extract_features(model, loader):
    model.eval()

    feature_batches = []
    label_batches = []

    for images, labels in loader:
        images = images.to(DEVICE)
        _, features = model(images, return_features=True)

        feature_batches.append(features.cpu())
        label_batches.append(labels)

    return (torch.cat(feature_batches).numpy(), torch.cat(label_batches).numpy(),)


def l2_normalize_rows(features):
    norms = np.linalg.norm(features, axis = 1, keepdims = True,)

    return features / np.maximum(norms, 1e-12)


def deterministic_subsample(features, limit, key):
    if len(features) <= limit:
        return features

    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()

    seed = int(digest[:8], 16)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(features), size = limit, replace = False,)

    return features[indices]


def median_rbf_gamma(source_features, target_features):
    combined = np.concatenate([source_features, target_features], axis = 0,)

    squared_distances = cdist(combined, combined, metric = "sqeuclidean",)

    nonzero_distances = squared_distances[squared_distances > 0]

    if len(nonzero_distances) == 0:
        return 1.0

    median_squared_distance = np.median(nonzero_distances)

    return 1.0 / max(2.0 * median_squared_distance, 1e-12,)


def rbf_mmd_squared(source_features, target_features):
    gamma = median_rbf_gamma(source_features, target_features,)

    source_source_squared = cdist(source_features, source_features, metric = "sqeuclidean",)

    target_target_squared = cdist(target_features, target_features, metric = "sqeuclidean",)

    source_target_squared = cdist(source_features, target_features, metric = "sqeuclidean",)

    kernel_source_source = np.exp(-gamma * source_source_squared)

    kernel_target_target = np.exp(-gamma * target_target_squared)

    kernel_source_target = np.exp(-gamma * source_target_squared)

    mmd_squared = (
        kernel_source_source.mean()
        + kernel_target_target.mean()
        - 2.0 * kernel_source_target.mean()
    )

    return max(float(mmd_squared), 0.0), float(gamma)


def centroid_metrics(source_features, target_features):
    source_centroid = source_features.mean(axis=0)
    target_centroid = target_features.mean(axis=0)

    centroid_l2 = np.linalg.norm(source_centroid - target_centroid)

    denominator = max(np.linalg.norm(source_centroid) * np.linalg.norm(target_centroid), 1e-12,)

    cosine_similarity = (np.dot(source_centroid, target_centroid) / denominator)

    return {
        "centroid_l2": float(centroid_l2),
        "centroid_cosine_distance": float(1.0 - cosine_similarity),
    }


def covariance_metrics(source_features, target_features):
    source_covariance = np.cov(source_features, rowvar = False,)

    target_covariance = np.cov(target_features, rowvar = False,)

    difference = source_covariance - target_covariance

    covariance_frobenius = np.linalg.norm(difference, ord = "fro",)

    feature_dimension = source_features.shape[1]

    source_trace = np.trace(source_covariance)
    target_trace = np.trace(target_covariance)

    return {
        "covariance_frobenius": float(covariance_frobenius),
        "covariance_frobenius_normalized": float(covariance_frobenius / max(feature_dimension, 1)),
        "source_feature_variance": float(source_trace),
        "target_feature_variance": float(target_trace),
        "feature_variance_difference": float(abs(source_trace - target_trace)),
    }


def mean_std_metrics(source_features, target_features):
    source_mean = source_features.mean(axis=0)
    target_mean = target_features.mean(axis=0)

    source_std = source_features.std(axis=0)
    target_std = target_features.std(axis=0)

    return {
        "feature_mean_absolute_difference": float(np.mean(np.abs(source_mean - target_mean))),
        "feature_std_absolute_difference": float(np.mean(np.abs(source_std - target_std))),
    }


def calculate_distribution_features(source_features, target_features, case_id,):
    source_normalized = l2_normalize_rows(source_features.astype(np.float64))

    target_normalized = l2_normalize_rows(target_features.astype(np.float64))

    source_sample = deterministic_subsample(
        source_normalized,
        DISTANCE_SAMPLE_LIMIT,
        f"{case_id}_source",
    )

    target_sample = deterministic_subsample(
        target_normalized,
        DISTANCE_SAMPLE_LIMIT,
        f"{case_id}_target",
    )

    mmd_squared, rbf_gamma = rbf_mmd_squared(source_sample, target_sample,)

    metrics = {
        "mmd_rbf_squared": mmd_squared,
        "mmd_rbf": float(np.sqrt(mmd_squared)),
        "mmd_rbf_gamma": rbf_gamma,
    }

    metrics.update(centroid_metrics(source_normalized, target_normalized,))

    metrics.update(covariance_metrics(source_normalized, target_normalized,))

    metrics.update(mean_std_metrics(source_normalized, target_normalized,))

    return metrics