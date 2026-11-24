import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class ImageEncoder(nn.Module):
    def __init__(self, output_dim=256, pretrained=True, dropout=0.3):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        net = efficientnet_b0(weights=weights)
        in_features = net.classifier[1].in_features
        self.features = net.features
        self.pool = net.avgpool
        self.projection = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_features, output_dim), nn.ReLU())

    def forward(self, x):
        return self.projection(self.pool(self.features(x)))

    def freeze_backbone(self, frozen=True):
        for p in self.features.parameters():
            p.requires_grad = not frozen


class MetadataEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=128, dropout=0.3, linear=False):
        super().__init__()
        if linear:
            self.net = nn.Identity()
            self.output_dim = input_dim
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(128, output_dim), nn.ReLU(),
            )
            self.output_dim = output_dim

    def forward(self, x):
        return self.net(x)


class MetadataClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=8, hidden_dim=128, dropout=0.3, linear=False):
        super().__init__()
        self.encoder = MetadataEncoder(input_dim, hidden_dim, dropout, linear)
        self.classifier = nn.Linear(self.encoder.output_dim, num_classes)

    def forward(self, image, metadata):
        features = self.encoder(metadata)
        return {"logits": self.classifier(features), "metadata_embedding": features}


class ImageClassifier(nn.Module):
    def __init__(self, num_classes=8, image_dim=256, pretrained=True, dropout=0.3):
        super().__init__()
        self.image_encoder = ImageEncoder(image_dim, pretrained, dropout)
        self.classifier = nn.Linear(image_dim, num_classes)

    def forward(self, image, metadata):
        features = self.image_encoder(image)
        return {"logits": self.classifier(features), "image_embedding": features}


class MultimodalClassifier(nn.Module):
    def __init__(self, metadata_input_dim, num_classes=8, fusion="concat", image_dim=256,
                 metadata_dim=128, projection_dim=128, pretrained=True, dropout=0.3,
                 attention_heads=4, decision_alpha=0.8):
        super().__init__()
        self.fusion = fusion
        self.decision_alpha = decision_alpha
        self.image_encoder = ImageEncoder(image_dim, pretrained, dropout)
        self.metadata_encoder = MetadataEncoder(metadata_input_dim, metadata_dim, dropout)
        self.image_projection = nn.Linear(image_dim, projection_dim)
        self.metadata_projection = nn.Linear(metadata_dim, projection_dim)
        if fusion == "concat":
            self.head = nn.Sequential(nn.Linear(2 * projection_dim, projection_dim), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(projection_dim, num_classes))
        elif fusion == "gated":
            self.gate = nn.Linear(2 * projection_dim, projection_dim)
            self.head = nn.Linear(projection_dim, num_classes)
        elif fusion == "decision":
            self.image_head = nn.Linear(projection_dim, num_classes)
            self.metadata_head = nn.Linear(projection_dim, num_classes)
        elif fusion == "attention":
            if projection_dim % attention_heads:
                raise ValueError("projection_dim must be divisible by attention_heads")
            self.attention = nn.MultiheadAttention(projection_dim, attention_heads, dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(projection_dim)
            self.head = nn.Linear(projection_dim, num_classes)
        else:
            raise ValueError(f"Unknown fusion method: {fusion}")

    def forward(self, image, metadata):
        image_raw = self.image_encoder(image)
        metadata_raw = self.metadata_encoder(metadata)
        zi = torch.nn.functional.normalize(self.image_projection(image_raw), dim=1)
        zm = torch.nn.functional.normalize(self.metadata_projection(metadata_raw), dim=1)
        output = {"image_embedding": zi, "metadata_embedding": zm}
        if self.fusion == "concat":
            logits = self.head(torch.cat([zi, zm], dim=1))
        elif self.fusion == "gated":
            gate = torch.sigmoid(self.gate(torch.cat([zi, zm], dim=1)))
            output["gate"] = gate
            logits = self.head(gate * zi + (1.0 - gate) * zm)
        elif self.fusion == "decision":
            li, lm = self.image_head(zi), self.metadata_head(zm)
            probability = self.decision_alpha * li.softmax(1) + (1.0 - self.decision_alpha) * lm.softmax(1)
            logits = probability.clamp_min(1e-8).log()
            output.update({"image_logits": li, "metadata_logits": lm})
        else:
            tokens = torch.stack([zi, zm], dim=1)
            attended, attention_weights = self.attention(tokens, tokens, tokens, need_weights=True)
            fused = self.norm(tokens + attended).mean(dim=1)
            output["attention_weights"] = attention_weights
            logits = self.head(fused)
        output["logits"] = logits
        return output


def build_model(config, metadata_input_dim, num_classes=8):
    m = config["model"]
    model_type = m["type"]
    if model_type in ("metadata_linear", "metadata_mlp"):
        return MetadataClassifier(metadata_input_dim, num_classes, m["metadata_dim"], m["dropout"],
                                  linear=model_type == "metadata_linear")
    if model_type == "image":
        return ImageClassifier(num_classes, m["image_dim"], m["pretrained"], m["dropout"])
    if model_type == "multimodal":
        return MultimodalClassifier(
            metadata_input_dim, num_classes, m["fusion"], m["image_dim"], m["metadata_dim"],
            m["projection_dim"], m["pretrained"], m["dropout"], m["attention_heads"],
            m["decision_alpha"],
        )
    raise ValueError(f"Unknown model type: {model_type}")

