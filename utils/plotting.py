import torch
import matplotlib.pyplot as plt


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


def plot_ae_reconstructions(model, dataset, device="cpu", n=10, indices=None):
    """
    Plot original vs reconstructed images.
    Top row: originals
    Bottom row: reconstructions
    """

    model.eval()
    if indices is None:
        indices = torch.randint(0, len(dataset), size=(n,))
    else:
        n = len(indices)

    originals = []
    recons = []

    with torch.no_grad():
        for idx in indices:
            img, _ = dataset[idx]
            img_batch = img.unsqueeze(0).to(device)

            x_hat, _ = model(img_batch)

            if x_hat.dim() == 2:
                rec = x_hat.view(1, 1, 28, 28)
            else:
                rec = x_hat

            originals.append(img.squeeze())
            recons.append(rec.squeeze().cpu())

    rows = 2
    cols = n
    plt.figure(figsize=(2 * n, 4))

    # Original images
    for i in range(n):
        ax = plt.subplot(rows, cols, i + 1)
        plt.imshow(originals[i], cmap="gray")
        ax.set_title("Original")
        plt.axis("off")

    # Reconstructed images
    for i in range(n):
        ax = plt.subplot(rows, cols, n + i + 1)
        plt.imshow(recons[i], cmap="gray")
        ax.set_title("Reconstructed")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
