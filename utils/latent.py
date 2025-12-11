import torch
from sklearn.cluster import KMeans


def get_latent_kmeans(model, dataset, n_clusters=10, device="cpu"):
    model.eval()

    X = dataset.data.float().reshape(len(dataset), -1) / 255.0
    X = X.to(device)

    with torch.no_grad():
        _, latent = model(X)

    latent = latent.cpu().numpy()
    predictions = KMeans(n_clusters=n_clusters).fit_predict(latent)

    return latent, predictions
