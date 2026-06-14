"""PyTorch recommender models for CineMatch-AI notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ModelType = Literal["cf", "content", "hybrid"]


@dataclass
class IdMappings:
    user_ids: np.ndarray
    movie_ids: np.ndarray
    user_to_idx: dict
    movie_to_idx: dict

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_movies(self) -> int:
        return len(self.movie_ids)


def build_id_mappings(train_df) -> IdMappings:
    user_ids = np.sort(train_df["userId"].unique())
    movie_ids = np.sort(train_df["movieId"].unique())
    return IdMappings(
        user_ids=user_ids,
        movie_ids=movie_ids,
        user_to_idx={u: i for i, u in enumerate(user_ids)},
        movie_to_idx={m: i for i, m in enumerate(movie_ids)},
    )


def build_content_lookup(
    mappings: IdMappings,
    movie_features,
) -> np.ndarray:
    """movie_features: DataFrame indexed by movieId."""
    dim = movie_features.shape[1]
    lookup = np.zeros((mappings.n_movies, dim), dtype=np.float32)
    for movie_id, idx in mappings.movie_to_idx.items():
        if movie_id in movie_features.index:
            lookup[idx] = movie_features.loc[movie_id].to_numpy(dtype=np.float32)
    return lookup


class RatingDataset(Dataset):
    def __init__(
        self,
        df,
        mappings: IdMappings,
        content_lookup: np.ndarray | None = None,
    ):
        self.user_idx = df["userId"].map(mappings.user_to_idx).to_numpy(dtype=np.int64)
        self.movie_idx = df["movieId"].map(mappings.movie_to_idx).to_numpy(dtype=np.int64)
        self.ratings = df["rating"].to_numpy(dtype=np.float32)
        self.content_lookup = content_lookup

        if (self.user_idx < 0).any() or (self.movie_idx < 0).any():
            raise ValueError("rows contain users/movies not in train mappings")

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, i: int):
        u = self.user_idx[i]
        m = self.movie_idx[i]
        y = self.ratings[i]
        if self.content_lookup is None:
            return u, m, y
        return u, m, self.content_lookup[m], y


def _forward_batch(
    model: nn.Module,
    batch,
    model_type: ModelType,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if model_type == "hybrid":
        u, m, content, y = batch
        pred = model(u.to(device), m.to(device), content.to(device))
    elif model_type == "content":
        u, m, content, y = batch
        pred = model(u.to(device), content.to(device))
    else:
        u, m, y = batch
        pred = model(u.to(device), m.to(device))
    return pred, y.to(device)


class GMF(nn.Module):
    """Generalized Matrix Factorization — dot product of user & movie embeddings."""

    def __init__(self, n_users: int, n_movies: int, embed_dim: int = 64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.movie_emb = nn.Embedding(n_movies, embed_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.movie_emb.weight, std=0.01)

    def forward(self, user_idx: torch.Tensor, movie_idx: torch.Tensor) -> torch.Tensor:
        return (self.user_emb(user_idx) * self.movie_emb(movie_idx)).sum(dim=-1)


class NeuralCF(nn.Module):
    """MLP on concatenated user & movie embeddings (non-linear CF)."""

    def __init__(
        self,
        n_users: int,
        n_movies: int,
        embed_dim: int = 64,
        hidden: tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.movie_emb = nn.Embedding(n_movies, embed_dim)

        layers: list[nn.Module] = []
        in_dim = embed_dim * 2
        for size in hidden:
            layers.extend([nn.Linear(in_dim, size), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = size
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.movie_emb.weight, std=0.01)

    def forward(self, user_idx: torch.Tensor, movie_idx: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.user_emb(user_idx), self.movie_emb(movie_idx)], dim=-1)
        return self.mlp(x).squeeze(-1)


class NeuMF(nn.Module):
    """NeuMF — GMF (linear) + MLP (non-linear) branches combined (He et al., 2017)."""

    def __init__(
        self,
        n_users: int,
        n_movies: int,
        gmf_dim: int = 32,
        mlp_dim: int = 32,
        hidden: tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gmf_user = nn.Embedding(n_users, gmf_dim)
        self.gmf_movie = nn.Embedding(n_movies, gmf_dim)
        self.mlp_user = nn.Embedding(n_users, mlp_dim)
        self.mlp_movie = nn.Embedding(n_movies, mlp_dim)

        layers: list[nn.Module] = []
        in_dim = mlp_dim * 2
        for size in hidden:
            layers.extend([nn.Linear(in_dim, size), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = size
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(gmf_dim + in_dim, 1)

        for emb in (self.gmf_user, self.gmf_movie, self.mlp_user, self.mlp_movie):
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, user_idx: torch.Tensor, movie_idx: torch.Tensor) -> torch.Tensor:
        gmf = self.gmf_user(user_idx) * self.gmf_movie(movie_idx)
        mlp_x = torch.cat([self.mlp_user(user_idx), self.mlp_movie(movie_idx)], dim=-1)
        mlp_h = self.mlp(mlp_x)
        return self.out(torch.cat([gmf, mlp_h], dim=-1)).squeeze(-1)


class ContentNet(nn.Module):
    """User embedding + movie content only — no movie ID embedding (helps new/obscure titles)."""

    def __init__(
        self,
        n_users: int,
        content_dim: int,
        embed_dim: int = 64,
        hidden: tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)
        layers: list[nn.Module] = []
        in_dim = embed_dim + content_dim
        for size in hidden:
            layers.extend([nn.Linear(in_dim, size), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = size
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)
        nn.init.normal_(self.user_emb.weight, std=0.01)

    def forward(self, user_idx: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.user_emb(user_idx), content], dim=-1)
        return self.mlp(x).squeeze(-1)


class HybridNet(nn.Module):
    """Collaborative embeddings + movie content → MLP (main CineMatch hybrid)."""

    def __init__(
        self,
        n_users: int,
        n_movies: int,
        content_dim: int,
        embed_dim: int = 64,
        hidden: tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.movie_emb = nn.Embedding(n_movies, embed_dim)

        layers: list[nn.Module] = []
        in_dim = embed_dim * 2 + content_dim
        for size in hidden:
            layers.extend([nn.Linear(in_dim, size), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = size
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.movie_emb.weight, std=0.01)

    def forward(
        self,
        user_idx: torch.Tensor,
        movie_idx: torch.Tensor,
        content: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([self.user_emb(user_idx), self.movie_emb(movie_idx), content], dim=-1)
        return self.mlp(x).squeeze(-1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    val_loader: DataLoader | None = None,
    epochs: int = 6,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
    model_type: ModelType = "cf",
    epoch_checkpoint: Path | None = None,
    resume: bool = True,
) -> list[dict]:
    """Train with MSE loss; return per-epoch train/val RMSE history."""
    device = torch.device(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    history: list[dict] = []
    start_epoch = 1

    if resume and epoch_checkpoint is not None and epoch_checkpoint.exists():
        ckpt = torch.load(epoch_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        history = list(ckpt.get("history", []))
        start_epoch = int(ckpt["epoch"]) + 1
        print(f"    Resumed from epoch {start_epoch - 1}/{epochs}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            pred, y = _forward_batch(model, batch, model_type, device)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        row = {"epoch": epoch, "train_rmse": float(np.sqrt(np.mean(train_losses)))}
        if val_loader is not None:
            row["val_rmse"] = predict_rmse(
                model, val_loader, device=device, model_type=model_type
            )
        history.append(row)

        if epoch_checkpoint is not None:
            epoch_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                },
                epoch_checkpoint,
            )

    return history


@torch.no_grad()
def predict_rmse(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: str | torch.device = "cpu",
    model_type: ModelType = "cf",
) -> float:
    device = torch.device(device)
    model.eval()
    sq_errors = []
    for batch in loader:
        pred, y = _forward_batch(model, batch, model_type, device)
        sq_errors.append(((pred - y) ** 2).cpu().numpy())
    return float(np.sqrt(np.mean(np.concatenate(sq_errors))))


@torch.no_grad()
def predict_ratings(
    model: nn.Module,
    df,
    mappings: IdMappings,
    *,
    content_lookup: np.ndarray | None = None,
    device: str | torch.device = "cpu",
    model_type: ModelType = "cf",
    batch_size: int = 4096,
) -> np.ndarray:
    if "rating" not in df.columns:
        df = df.copy()
        df["rating"] = 0.0
    dataset = RatingDataset(df, mappings, content_lookup)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = torch.device(device)
    model.eval()
    preds: list[np.ndarray] = []
    for batch in loader:
        pred, _ = _forward_batch(model, batch, model_type, device)
        preds.append(pred.cpu().numpy())
    return np.concatenate(preds)


def evaluate_model(
    name: str,
    model: nn.Module,
    train_df,
    val_df,
    mappings: IdMappings,
    *,
    model_type: ModelType,
    content_lookup: np.ndarray | None = None,
    batch_size: int = 4096,
    epochs: int = 6,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
    epoch_checkpoint: Path | None = None,
    resume: bool = True,
) -> tuple[dict, list[dict]]:
    """Train, predict on val, return metrics row + history."""
    if model_type in ("content", "hybrid"):
        train_ds = RatingDataset(train_df, mappings, content_lookup)
        val_ds = RatingDataset(val_df, mappings, content_lookup)
    else:
        train_ds = RatingDataset(train_df, mappings)
        val_ds = RatingDataset(val_df, mappings)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    history = train_model(
        model,
        train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        model_type=model_type,
        epoch_checkpoint=epoch_checkpoint,
        resume=resume,
    )
    preds = predict_ratings(
        model,
        val_df,
        mappings,
        content_lookup=content_lookup,
        device=device,
        model_type=model_type,
        batch_size=batch_size,
    )

    y_true = val_df["rating"].to_numpy()
    rmse = float(np.sqrt(np.mean((y_true - preds) ** 2)))
    mae = float(np.mean(np.abs(y_true - preds)))
    return {"model": name, "RMSE": rmse, "MAE": mae}, history


def build_model_from_checkpoint(ckpt: dict) -> nn.Module:
    """Reconstruct a saved model from a train_pipeline checkpoint."""
    cfg = ckpt["config"]
    cls = ckpt["model_class"]
    n_users = int(ckpt["n_users"])
    n_movies = int(ckpt["n_movies"])
    content_dim = int(ckpt["content_dim"])
    hidden = tuple(cfg.get("hidden", (128, 64)))
    dropout = float(cfg.get("dropout", 0.2))
    embed_dim = int(cfg.get("embed_dim", 64))

    if cls == "GMF":
        model = GMF(n_users, n_movies, embed_dim=embed_dim)
    elif cls == "NeuMF":
        model = NeuMF(
            n_users,
            n_movies,
            gmf_dim=int(cfg.get("gmf_dim", 32)),
            mlp_dim=int(cfg.get("mlp_dim", 32)),
            hidden=hidden,
            dropout=dropout,
        )
    elif cls == "NeuralCF":
        model = NeuralCF(n_users, n_movies, embed_dim=embed_dim, hidden=hidden, dropout=dropout)
    elif cls == "ContentNet":
        model = ContentNet(n_users, content_dim, embed_dim=embed_dim, hidden=hidden, dropout=dropout)
    elif cls == "HybridNet":
        model = HybridNet(
            n_users,
            n_movies,
            content_dim,
            embed_dim=embed_dim,
            hidden=hidden,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model class in checkpoint: {cls}")

    model.load_state_dict(ckpt["model_state_dict"])
    return model


def id_mappings_from_checkpoint(ckpt: dict) -> IdMappings:
    user_ids = np.asarray(ckpt["user_ids"])
    movie_ids = np.asarray(ckpt["movie_ids"])
    return IdMappings(
        user_ids=user_ids,
        movie_ids=movie_ids,
        user_to_idx={int(u): i for i, u in enumerate(user_ids)},
        movie_to_idx={int(m): i for i, m in enumerate(movie_ids)},
    )
