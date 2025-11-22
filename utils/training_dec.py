import torch


def train_dec(
        model,
        train_loader,
        optimizer,
        loss_fn,
        tensor_x,  # TODO solve this nicely
        epochs=10,
        device="cpu"
):
    model.to(device)
    model.train()
    losses = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0

        # In the original paper target distribution P is updated every T iterations
        # Here we are updating target distribution P on full dataset, once per epoch
        with torch.no_grad():
            q_full, _ = model(tensor_x)
            p_full = target_distribution(q_full)

        for batch_idx, (inputs, idxs) in enumerate(train_loader):
            q, _ = model(inputs)
            p_batch = p_full[idxs]
            loss = loss_fn(q.log(), p_batch)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)

        epoch_loss /= len(train_loader.dataset)
        losses.append(epoch_loss)

        print(f"Epoch {epoch}/{epochs}: average loss = {epoch_loss:.4f}")

    return losses


def target_distribution(q):
    weight = q ** 2 / torch.sum(q, dim=0)
    return (weight.t() / torch.sum(weight, dim=1)).t()
