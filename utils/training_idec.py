import torch


def train_idec(
        model,
        train_loader,
        optimizer,
        kl_loss_fn,
        mse_loss_fn,
        tensor_x,  # TODO solve this nicely
        epochs=10,
        gamma=0.1,
        device="cpu",
):
    model.to(device)
    model.initialize_centers(train_loader, device)

    model.train()
    total_losses = []
    kl_losses = []
    mse_losses = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        total_kl_loss = 0
        total_mse_loss = 0

        # Update Target Distribution P using FULL dataset
        with torch.no_grad():
            q_full, _, _ = model(tensor_x)
            p_full = target_distribution(q_full)

        for batch_idx, (inputs, idxs) in enumerate(train_loader):
            q, _, x_reconstructed = model(inputs)
            p_batch = p_full[idxs]

            # Calculate loss
            kl_loss = kl_loss_fn(q.log(), p_batch)
            mse_loss = mse_loss_fn(x_reconstructed, inputs)
            # loss = kl_loss + gamma * mse_loss
            loss = mse_loss + gamma * kl_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_kl_loss += kl_loss.item() * inputs.size(0)
            total_mse_loss += mse_loss.item() * inputs.size(0)
            epoch_loss += loss.item() * inputs.size(0)

        epoch_loss /= len(train_loader.dataset)
        total_kl_loss /= len(train_loader.dataset)
        total_mse_loss /= len(train_loader.dataset)

        total_losses.append(epoch_loss)
        kl_losses.append(total_kl_loss)
        mse_losses.append(total_mse_loss)

        print(f"Epoch {epoch}/{epochs}: average kl loss = {total_kl_loss:.4f} mse loss = {total_mse_loss:.4f}")

    return total_losses, kl_losses, mse_losses

def target_distribution(q):
    weight = q ** 2 / torch.sum(q, dim=0)
    return (weight.t() / torch.sum(weight, dim=1)).t()
