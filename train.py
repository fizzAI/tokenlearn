import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from model2vec import StaticModel
from model2vec.distill import distill
from sklearn.decomposition import PCA

from tokenlearn.train import TextDataset, train_supervised
from tokenlearn.utils import collect_means_and_texts, count_tokens

logging.basicConfig(level=logging.INFO)


def train_model(model_name: str, train_txt: list[str], train_vec: np.ndarray, device: str = "cpu") -> StaticModel:
    """
    Train a tokenlearn model.

    :param model_name: The sentence transformer model name for distillation.
    :param train_txt: List of texts to train on.
    :param train_vec: List of vectors to train on.
    :param device: Device to run the training on.
    :return: The trained model.
    """
    model = distill(model_name)
    train_data = TextDataset(train_txt, torch.from_numpy(train_vec), model.tokenizer)

    # Train the model
    model, _ = train_supervised(train_dataset=train_data, model=model, device=device)

    return model


def weight_model(model: StaticModel, text: list[str], pca_dims: int, alpha: float = 1e-3) -> StaticModel:
    """
    Function to weight the model.

    :param model: The model to weight.
    :param text: The text to use for weighting.
    :param pca_dims: The number of PCA dimensions to use.
    :param alpha: The alpha value for SIF weighting. Words with probabilities above this value will be downweighted.
    :return: The weighted model.
    """
    logging.info("Applying reweighting and PCA to the model.")
    count_vector = count_tokens(model.tokenizer, text)

    w = model.embedding
    w = np.nan_to_num(w)

    # Apply PCA
    p = PCA(n_components=pca_dims)
    w = p.fit_transform(w)

    # Apply SIF weighting
    f = alpha / (alpha + count_vector)
    w *= f[:, None]
    model.embedding = w
    model.normalize = True

    return model


def main(args: Any) -> None:
    """Main function."""
    # Collect paths for training
    paths = sorted(Path(args.data_path).glob("*.json"))
    train_txt, train_vec = collect_means_and_texts(paths)

    model = train_model(args.model_name, train_txt, train_vec, device=args.device)
    model.save_pretrained(args.save_path)
    model = weight_model(model, train_txt, 256)
    weighted_name = f"{args.save_path}_weighted"
    model.save_pretrained(weighted_name)


if __name__ == "__main__":
    # Define CLI arguments
    parser = argparse.ArgumentParser(description="Train a Model2Vec using tokenlearn.")

    # Training arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="baai/bge-base-en-v1.5",
        help="The model name for distillation (e.g., 'baai/bge-base-en-v1.5').",
    )
    parser.add_argument(
        "--data-path", type=str, default="data/fineweb_bgebase", help="Path to the directory containing the dataset."
    )
    parser.add_argument("--save-path", type=str, help="Path to save the trained model.")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run the training on (e.g., 'cpu', 'cuda')."
    )

    args = parser.parse_args()

    main(args)
