import torch
import torch.nn as nn
import torch.nn.functional as F
from models.embedding_layers import Embedding, UciEmbedding, DecodingEmbedding, DecodingUciEmbedding
from models.mask_generators import MaskGenerator

class MaskedTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask):
        B, num_features, D = x.shape
        
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        Q = Q.view(B, num_features, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, num_features, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, num_features, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        mask_expanded = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        scores = scores + mask_expanded
        attn_weights = F.softmax(scores, dim=-1)
        
        attended = torch.matmul(attn_weights, V)
        
        attended = attended.transpose(1, 2).contiguous().view(B, num_features, D)
        
        attended = self.out_proj(attended)
        
        x = x + self.dropout(attended)
        x = self.layernorm1(x)

        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.layernorm2(x)

        return x

class CausalPredictorOld(nn.Module):
    """Causal prediction module (old version), using label embedding and causally weighted other features"""
    def __init__(self, d_model, num_features, target_type=1, num_classes=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_features = num_features
        self.target_type = target_type  # 0: regression, 1: classification
        self.num_classes = num_classes
        
        self.predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes if target_type == 1 else 1)
        )
        
        self.causal_layernorm = nn.LayerNorm(d_model)
        
    def forward(self, encoded_features, mask, target_idx):
        """
        Args:
            encoded_features: [B, F, D] encoded features
            mask: [B, F, F] causal structure mask
            target_idx: int target feature index
        Returns:
            prediction: [B, num_classes] or [B, 1] prediction result
        """
        batch_size = encoded_features.size(0)
        
        label_embedding = encoded_features[:, target_idx, :]
        
        causal_weights = mask[:, target_idx, :]
        
        weighted_features = encoded_features * causal_weights.unsqueeze(-1)
        causal_aggregated = weighted_features.sum(dim=1)
        
        causal_aggregated = self.causal_layernorm(causal_aggregated)
        
        combined_features = torch.cat([label_embedding, causal_aggregated], dim=-1)
        
        prediction = self.predictor(combined_features)
        
        return prediction

class CausalPredictor(nn.Module):
    """Causal prediction module (new version), using F trainable parameters to weight sum F embedding rows"""
    def __init__(self, d_model, num_features, target_type=1, num_classes=2, dropout=0.1, target_idx=None):
        super().__init__()
        self.d_model = d_model
        self.num_features = num_features
        self.target_type = target_type  # 0: regression, 1: classification
        self.num_classes = num_classes
        self.target_idx = target_idx
        
        self.feature_weights = nn.Parameter(torch.abs(torch.randn(num_features)) * 0.1)
        
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_classes if target_type == 1 else 1)
        )
        
        self.weighted_layernorm = nn.LayerNorm(d_model)
        
    def forward(self, encoded_features, mask, target_idx):
        """
        Args:
            encoded_features: [B, F, D] encoded features
            mask: [B, F, F] causal structure mask (not used in new version)
            target_idx: int target feature index (not used in new version)
        Returns:
            prediction: [B, num_classes] or [B, 1] prediction result
        """
        batch_size = encoded_features.size(0)
        
        if self.target_idx is not None and self.target_idx < self.num_features:
            feature_weights = self.feature_weights.clone()
            mask = torch.ones(self.num_features, dtype=torch.bool, device=feature_weights.device)
            mask[self.target_idx] = False
            feature_weights[mask] = torch.relu(feature_weights[mask])
            feature_weights[self.target_idx] = self.feature_weights[self.target_idx]
        else:
            feature_weights = torch.relu(self.feature_weights)
        
        weighted_features = encoded_features * feature_weights.unsqueeze(0).unsqueeze(-1)
        
        aggregated_features = weighted_features.sum(dim=1)
        
        aggregated_features = self.weighted_layernorm(aggregated_features)
        
        prediction = self.predictor(aggregated_features)
        
        return prediction

class CausalAttentionMskModel(nn.Module):
    def __init__(self, v, num_classes_dict, d_model=64, num_heads=4, num_layers=1, dropout=0.1, share_embedding=False, mask_generator=None, target_idx=None):
        super().__init__()
        self.v = v
        self.num_features = len(v)
        self.cont_idx = (v == 0).nonzero(as_tuple=True)[0]
        self.cat_idx = (v == 1).nonzero(as_tuple=True)[0]
        self.num_cont = len(self.cont_idx)
        self.num_cat = len(self.cat_idx)
        self.num_classes_dict = num_classes_dict
        self.share_embedding = share_embedding

        if self.num_cont > 0:
            self.cont_embedding = UciEmbedding(self.num_cont, d_model)
        else:
            self.cont_embedding = None
            
        if self.num_cat > 0:
            total_cat_classes = sum(num_classes_dict[idx.item()] for idx in self.cat_idx)
            self.cat_embedding = Embedding(total_cat_classes, d_model)
            self.cat_offsets = {}
            current_offset = 0
            for idx in self.cat_idx:
                self.cat_offsets[idx.item()] = current_offset
                current_offset += num_classes_dict[idx.item()]
            
            self.cat_offset_tensor = torch.tensor([self.cat_offsets[idx.item()] for idx in self.cat_idx], 
                                                dtype=torch.long)
        else:
            self.cat_embedding = None
            self.cat_offsets = {}
            self.cat_offset_tensor = None

        self.mask_embedding = nn.Parameter(torch.randn(1, 1, d_model))

        if self.share_embedding:
            if self.num_cont > 0:
                self.cont_decoder = DecodingUciEmbedding(self.cont_embedding)
            else:
                self.cont_decoder = None
                
            if self.num_cat > 0:
                self.cat_decoder = DecodingEmbedding(self.cat_embedding, num_classes_dict, self.cat_idx, self.cat_offsets)
            else:
                self.cat_decoder = None
        else:
            if self.num_cont > 0:
                self.cont_decoders = nn.ModuleList([
                    nn.Linear(d_model, 1) for _ in range(self.num_cont)
                ])
            else:
                self.cont_decoders = None
                
            if self.num_cat > 0:
                self.cat_decoders = nn.ModuleList([
                    nn.Linear(d_model, num_classes_dict.get(idx.item(), 10)) for idx in self.cat_idx
                ])
            else:
                self.cat_decoders = None

        if target_idx is None:
            target_idx = len(v) - 1
        
        if target_idx < len(v):
            target_type = v[target_idx].item()
            if target_type == 0:
                num_classes = 1
            else:
                num_classes = num_classes_dict.get(target_idx, 2)
        else:
            target_type = 1
            num_classes = 2
        
        self.predictor = CausalPredictor(d_model, self.num_features, target_type=target_type, num_classes=num_classes, dropout=dropout, target_idx=target_idx)
        if mask_generator is None:
            from models.mask_generators import SigmoidMaskGenerator
            self.mask_generator = SigmoidMaskGenerator(self.num_features)
        else:
            self.mask_generator = mask_generator
            
        self.mask_logits = nn.Parameter(
            self.mask_generator.initialize_mask_logits(device=next(self.parameters()).device)
        )
        self.encoder = nn.ModuleList([
            MaskedTransformerBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

    def update_parameters(self):
        self.mask_generator.update_parameters()
        
    def get_parameters(self):
        params = self.mask_generator.get_parameters()
        params['mask_logits'] = self.mask_logits.detach().cpu().numpy()
        return params

    def get_embeddings(self, x, mask_positions=None):
        B, F = x.shape
        feat_embed = torch.zeros(B, F, self.encoder[0].q_proj.out_features, device=x.device)
        
        if self.num_cont > 0:
            x_cont = x[:, self.cont_idx]
            cont_embed = self.cont_embedding(x_cont)
            feat_embed[:, self.cont_idx, :] = cont_embed
        
        if self.num_cat > 0:
            x_cat = x[:, self.cat_idx]
            x_cat_ids = x_cat.long()
            x_cat_vals = torch.ones_like(x_cat, dtype=torch.float32)
            
            adjusted_ids = x_cat_ids + self.cat_offset_tensor.to(x.device).unsqueeze(0)
            
            cat_embed = self.cat_embedding({'ids': adjusted_ids, 'vals': x_cat_vals})
            feat_embed[:, self.cat_idx, :] = cat_embed
        
        if mask_positions is not None:
            mask_pos = mask_positions[0, 0].item()
            feat_embed[:, mask_pos] = self.mask_embedding.squeeze(0).squeeze(0)
        
        return feat_embed

    def forward(self, x, mask_positions=None, target_idx=None, task_type='reconstruction'):
        batch_size, num_features = x.shape
        
        if task_type == 'prediction' and target_idx is not None:
            if mask_positions is None:
                mask_positions = torch.zeros(batch_size, 1, dtype=torch.long, device=x.device)
                mask_positions[:, 0] = target_idx
        
        feats = self.get_embeddings(x, mask_positions)
        
        mask = self.mask_generator(self.mask_logits, batch_size)
        
        encoded = feats
        for layer in self.encoder:
            encoded = layer(encoded, mask)

        outputs = [None] * num_features
        
        if mask_positions is not None and mask_positions.numel() > 0:
            mask_idx = mask_positions[0, 0].item()
            
            if mask_idx in self.cont_idx:
                cont_idx_in_cont = (self.cont_idx == mask_idx).nonzero(as_tuple=True)[0].item()
                cont_encoded = encoded[:, mask_idx:mask_idx+1, :]
                
                if self.share_embedding:
                    full_cont_encoded = encoded[:, self.cont_idx, :]
                    cont_outputs = self.cont_decoder(full_cont_encoded)
                    outputs[mask_idx] = cont_outputs[:, cont_idx_in_cont]
                else:
                    outputs[mask_idx] = self.cont_decoders[cont_idx_in_cont](cont_encoded[:, 0, :]).squeeze(-1)
                    
            elif mask_idx in self.cat_idx:
                cat_idx_in_cat = (self.cat_idx == mask_idx).nonzero(as_tuple=True)[0].item()
                cat_encoded = encoded[:, mask_idx:mask_idx+1, :]
                
                if self.share_embedding:
                    full_cat_encoded = encoded[:, self.cat_idx, :]
                    cat_outputs = self.cat_decoder(full_cat_encoded)
                    outputs[mask_idx] = cat_outputs[cat_idx_in_cat]
                else:
                    outputs[mask_idx] = self.cat_decoders[cat_idx_in_cat](cat_encoded[:, 0, :])

        if target_idx is None:
            target_idx = self.num_features - 1
        prediction = self.predictor(encoded, mask, target_idx)

        return outputs, prediction, mask
        
    def get_causal_mask(self):
        causal_mask = self.mask_generator.get_causal_mask(self.mask_logits)
        return causal_mask.detach().cpu().numpy() 