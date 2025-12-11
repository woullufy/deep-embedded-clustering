import random

import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from umap import UMAP


def plot_all_reconstructions(images_dict):
    """
    Plot the reconstructions over epochs.
    Row: True image : [Reconstructed images]
    """

    n_images = len(images_dict)
    n_recons = max(len(v["reconstructions"]) for v in images_dict.values())

    fig, axes = plt.subplots(
        n_images,
        n_recons + 1,
        figsize=(3 * (n_recons + 1), 3 * n_images)
    )

    if n_images == 1:
        axes = axes.reshape(1, -1)

    for row_idx, (img_index, data) in enumerate(images_dict.items()):

        # Original image
        original = data["original"].detach().cpu().squeeze()
        original = original.view(28, 28)
        axes[row_idx, 0].imshow(original, cmap="gray")
        axes[row_idx, 0].set_title(f"Original (idx={img_index})")
        axes[row_idx, 0].axis("off")

        # All the reconstructions
        for col_idx in range(n_recons):
            ax = axes[row_idx, col_idx + 1]

            if col_idx < len(data["reconstructions"]):
                x_hat = data["reconstructions"][col_idx]

                if x_hat.dim() == 2:
                    img = x_hat.view(28, 28)
                else:
                    img = x_hat.squeeze()

                img = img.detach().cpu()

                ax.imshow(img, cmap="gray")
                ax.set_title(f"Epoch {col_idx + 1}")
            else:
                ax.axis("off")

            ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_ae_losses(ae_losses):
    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, len(ae_losses) + 1),
        ae_losses,
        linewidth=2,
    )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Autoencoder Training Loss", fontsize=14)

    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()


def plot_idec_losses(losses, kl_losses, mse_losses):
    epochs = range(1, len(losses) + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(epochs, losses, label="Total Loss", linewidth=2.5, color="black")
    plt.plot(epochs, kl_losses, label="Clustering Loss (KL)", linewidth=2, color="red")
    plt.plot(epochs, mse_losses, label="Reconstruction Loss", linewidth=2, color="blue")

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("IDEC Training Losses", fontsize=14)

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_losses(
        ae_losses=None,
        dec_losses=None,
        idec_total=None,
        idec_kl=None,
        idec_recon=None,
        title=None,
):
    plt.figure(figsize=(10, 6))

    if ae_losses is not None:
        plt.plot(
            range(1, len(ae_losses) + 1),
            ae_losses,
            label="AE Loss",
            linewidth=2,
            color="blue",
        )

    if dec_losses is not None:
        plt.plot(
            range(1, len(dec_losses) + 1),
            dec_losses,
            label="DEC KL Loss",
            linewidth=2,
            color="purple",
        )

    if idec_total is not None:
        plt.plot(
            range(1, len(idec_total) + 1),
            idec_total,
            label="IDEC Total Loss",
            linewidth=2.5,
            color="black",
        )
    if idec_kl is not None:
        plt.plot(
            range(1, len(idec_kl) + 1),
            idec_kl,
            label="IDEC Clustering Loss (KL)",
            linewidth=2,
            color="red",
        )
    if idec_recon is not None:
        plt.plot(
            range(1, len(idec_recon) + 1),
            idec_recon,
            label="IDEC Reconstruction Loss",
            linewidth=2,
            color="green",
        )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)

    if title:
        plt.title(title, fontsize=14)
    else:
        plt.title("Training Losses", fontsize=15)

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_classes_reconstruction(model, dataset, device=None):
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    label_to_indices = {i: [] for i in range(10)}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label_to_indices[label].append(idx)

    chosen_indices = [random.choice(label_to_indices[label]) for label in range(10)]

    originals = []
    recons = []

    with torch.no_grad():
        for idx in chosen_indices:
            img, _ = dataset[idx]
            img_batch = img.unsqueeze(0).to(device)

            x_hat, _ = model(img_batch)

            if x_hat.dim() == 2:
                rec = x_hat.view(1, 1, 28, 28)

            originals.append(img.squeeze())
            recons.append(rec.squeeze().cpu())

    rows = 2
    cols = 10
    plt.figure(figsize=(20, 4))

    for i in range(10):
        ax = plt.subplot(rows, cols, i + 1)
        plt.imshow(originals[i], cmap="gray")
        ax.set_title(f"Label {i}")
        plt.axis("off")

    for i in range(10):
        ax = plt.subplot(rows, cols, cols + i + 1)
        plt.imshow(recons[i], cmap="gray")
        ax.set_title("Reconstructed")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_dec_centers(dec, ae):
    dec.eval()
    ae.eval()

    with torch.no_grad():
        z = dec.cluster_centers
        centers_hat = ae.decoder(z)

    centers_img = centers_hat.cpu().numpy().reshape(-1, 28, 28)
    # centers_img = centers_hat.cpu().numpy().reshape(-1, 8, 8)

    n_clusters = len(centers_img)
    fig, axes = plt.subplots(1, n_clusters, figsize=(15, 3))

    for i in range(n_clusters):
        ax = axes[i]
        ax.imshow(centers_img[i], cmap='gray')
        ax.axis('off')
        ax.set_title(f'Cluster {i}')

    plt.tight_layout()
    plt.show()


def plot_pca(input, latent, labels, figsize=(12, 5), title=None):
    pca_input = PCA(n_components=2)
    pca_latent = PCA(n_components=2)

    input_pca = pca_input.fit_transform(input)
    latent_pca = pca_latent.fit_transform(latent)

    plot_input_latent(figsize, input_pca, labels, latent_pca, title)


def plot_umap(input, latent, labels, figsize=(12, 5), title=None, n_neighbors=15, min_dist=0.1):
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist)

    input_umap = reducer.fit_transform(input)
    latent_umap = reducer.fit_transform(latent)

    plot_input_latent(figsize, input_umap, labels, latent_umap, title)


def plot_input_latent(figsize, input_umap, labels, latent_umap, title):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    sc1 = axes[0].scatter(input_umap[:, 0], input_umap[:, 1], c=labels, cmap="tab10", s=5)
    axes[0].set_title("Raw Data")
    sc2 = axes[1].scatter(latent_umap[:, 0], latent_umap[:, 1], c=labels, cmap="tab10", s=5)
    axes[1].set_title("Latent Space")
    fig.colorbar(sc1, ax=axes, location='right', fraction=0.025, pad=0.02)
    if title:
        fig.suptitle(title)
    plt.show()
