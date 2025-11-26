import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans


class DEC(nn.Module):
    def __init__(self, autoencoder, num_clusters, latent_dim, alpha=1.0):
        super(DEC, self).__init__()
        self.encoder = autoencoder.encoder
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.cluster_centers = nn.Parameter(torch.Tensor(num_clusters, latent_dim))

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        z = self.encoder(x)

        # Efficient calculation of squared Euclidean distance
        # ||z - u||^2 = ||z||^2 + ||u||^2 - 2*z*uT
        z_norm_sq = torch.sum(z ** 2, dim=1, keepdim=True)
        u_norm_sq = torch.sum(self.cluster_centers ** 2, dim=1).unsqueeze(0)
        dist_sq = z_norm_sq + u_norm_sq - 2 * torch.matmul(z, self.cluster_centers.t())

        # Student's t-distribution)
        q = torch.pow(1.0 + dist_sq / self.alpha, -(self.alpha + 1.0) / 2.0)

        # Normalize for the probability dist
        q = q / torch.sum(q, dim=1, keepdim=True)

        return q, z

    def initialize_centers(self, dataloader, device):
        self.eval()
        features = []

        with torch.no_grad():
            for x, y in dataloader:
                # x = batch[0].to(device)
                x = x.to(device)
                z = self.encoder(x)
                features.append(z.cpu().numpy())

        features = np.concatenate(features)

        kmeans = KMeans(n_clusters=self.num_clusters, n_init=20)
        kmeans.fit(features)

        self.cluster_centers.data = torch.tensor(kmeans.cluster_centers_).to(device)


def target_distribution(q):
    weight = q ** 2 / torch.sum(q, dim=0)
    return (weight.t() / torch.sum(weight, dim=1)).t()
