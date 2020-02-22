import torch.nn as nn


class CNN1DNet(nn.Module):
    """CNN1D net"""

    def __init__(self, input_size, num_classes):
        super().__init__()

        # feature extractor, 1D conv layers
        self.features = nn.Sequential(
            nn.Conv1d(input_size, 64, 5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(64, 64, 5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(64, 64, 5),
            nn.ReLU(),
        )
        # classifier with FC layers
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(5632, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), 5632)
        out = self.classifier(x)

        return out
