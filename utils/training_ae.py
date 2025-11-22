import torch


def train_autoencoder(
        model,
        train_loader,
        optimizer,
        loss_fn,
        epochs=10,
        image_indices=None,
        device="cpu"
):
    model.to(device)
    model.train()
    losses = []
    images = {}

    if image_indices is not None:
        for idx in image_indices:
            img = train_loader.dataset[idx][0]  # shape: (1, 28, 28)
            images[idx] = {"original": img.unsqueeze(0).to(device), "reconstructions": []}

    for epoch in range(1, epochs + 1):
        epoch_loss = 0

        for x, _ in train_loader:
            x = x.to(device)

            x_hat, z = model(x)
            loss = loss_fn(x_hat, x.view(x.size(0), -1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)

        epoch_loss /= len(train_loader.dataset)
        losses.append(epoch_loss)

        if image_indices is not None:
            model.eval()
            with torch.no_grad():
                for k, v in images.items():
                    original = v["original"]  # shape: (1, 1, 28, 28)
                    x_hat, _ = model(original)  # shape: (1, 784)
                    images[k]["reconstructions"].append(x_hat)

            model.train()

        print(f"Epoch {epoch}/{epochs}: average loss = {epoch_loss:.4f}")

    if image_indices is not None:
        return losses, images

    return losses
