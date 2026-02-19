import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        distance = nn.functional.pairwise_distance(output1, output2)
        loss = (1 - label) * torch.pow(distance, 2) + \
               (label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        return loss.mean()

class ClusteringLoss(nn.Module):
    def __init__(self):
        super(ClusteringLoss, self).__init__()

    def forward(self, embeddings, cluster_centers, labels):
        distances = torch.cdist(embeddings, cluster_centers)
        loss = nn.functional.cross_entropy(distances, labels)
        return loss

class CustomLoss(nn.Module):
    def __init__(self, contrastive_weight=1.0, clustering_weight=1.0):
        super(CustomLoss, self).__init__()
        self.contrastive_loss = ContrastiveLoss()
        self.clustering_loss = ClusteringLoss()
        self.contrastive_weight = contrastive_weight
        self.clustering_weight = clustering_weight

    def forward(self, output1, output2, labels, embeddings, cluster_centers):
        loss_contrastive = self.contrastive_loss(output1, output2, labels)
        loss_clustering = self.clustering_loss(embeddings, cluster_centers, labels)
        return self.contrastive_weight * loss_contrastive + self.clustering_weight * loss_clustering