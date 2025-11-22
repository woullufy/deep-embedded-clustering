import torch
import torch.nn as nn
import torch.nn.functional as F


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


def target_distribution(q):
    weight = q ** 2 / torch.sum(q, dim=0)
    return (weight.t() / torch.sum(weight, dim=1)).t()


def soft_assign(self, z):
    diff = z.unsqueeze(1) - self.cluster_centers.unsqueeze(0)
    dist_sq = torch.sum(diff ** 2, dim=2)

    # Student-t kernel
    numerator = (1.0 + dist_sq / self.alpha) ** (-(self.alpha + 1) / 2)
    q = numerator / torch.sum(numerator, dim=1, keepdim=True)

    return q


def init_cluster_centers(self, z):
    """
    Initialize cluster centers using k-means results.
    z: numpy array of latent vectors (N x latent_dim)
    """
    assert z.ndim == 2
    self.cluster_centers.data.copy_(torch.tensor(z, dtype=torch.float32))
