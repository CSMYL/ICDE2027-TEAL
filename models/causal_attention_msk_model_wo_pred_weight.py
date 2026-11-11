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
        self.d_k = d_model // num_heads  # Dimension per head
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Multi-head QKV linear projection layers
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Output projection layer
        self.out_proj = nn.Linear(d_model, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask):
        B, num_features, D = x.shape
        
        # Generate Q, K, V
        Q = self.q_proj(x)  # [B, num_features, D]
        K = self.k_proj(x)  # [B, num_features, D]
        V = self.v_proj(x)  # [B, num_features, D]
        
        # Reshape to multi-head format
        Q = Q.view(B, num_features, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, num_features, d_k]
        K = K.view(B, num_features, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, num_features, d_k]
        V = V.view(B, num_features, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, num_features, d_k]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # [B, H, num_features, num_features]
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)  # [B, H, num_features, num_features]
        
        # Apply causal mask (to each head)
        mask_expanded = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # [B, H, num_features, num_features]
        attn_weights = attn_weights * mask_expanded
        
        # Apply attention to V
        attended = torch.matmul(attn_weights, V)  # [B, H, num_features, d_k]
        
        # Reshape back to original format
        attended = attended.transpose(1, 2).contiguous().view(B, num_features, D)  # [B, num_features, D]
        
        # Apply output projection
        attended = self.out_proj(attended)
        
        # Add residual connection
        x = x + self.dropout(attended)
        x = self.layernorm1(x)

        # FFN
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
        self.num_classes = num_classes  # Number of classes for classification
        
        # Prediction network - using label embedding + causally weighted other features
        self.predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # label embedding + causally weighted features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes if target_type == 1 else 1)  # Classification outputs num_classes, regression outputs 1
        )
        
        # LayerNorm for causally weighted features
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
        
        # 1. Get label embedding (target_idx row)
        label_embedding = encoded_features[:, target_idx, :]  # [B, D]
        
        # 2. Causally weight other features according to mask
        # mask[target_idx, :] represents causal weights from target feature to other features
        causal_weights = mask[:, target_idx, :]  # [B, F] - causal weights from target to other features
        
        # Weighted sum of other features
        weighted_features = encoded_features * causal_weights.unsqueeze(-1)  # [B, F, D]
        causal_aggregated = weighted_features.sum(dim=1)  # [B, D] - causally weighted other features
        
        # LayerNorm for causally weighted features
        causal_aggregated = self.causal_layernorm(causal_aggregated)  # [B, D]
        
        # 3. Concatenate label embedding and causally weighted features
        combined_features = torch.cat([label_embedding, causal_aggregated], dim=-1)  # [B, 2*D]
        
        # 4. Prediction
        prediction = self.predictor(combined_features)  # [B, num_classes] or [B, 1]
        
        return prediction

class CausalPredictor(nn.Module):
    """Causal prediction module (w/o pred weight), using simple mean pooling instead of learned adaptive weights.

    This is the ablation variant: CAIM w/o pred weight (eliminating the adaptive weights a_i).
    All columns are treated equally via mean pooling — no hierarchy-aware weighting.
    """
    def __init__(self, d_model, num_features, target_type=1, num_classes=2, dropout=0.1, target_idx=None):
        super().__init__()
        self.d_model = d_model
        self.num_features = num_features
        self.target_type = target_type
        self.num_classes = num_classes
        self.target_idx = target_idx

        # NO feature_weights — uses simple mean pooling

        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_classes if target_type == 1 else 1)
        )

        self.pooled_layernorm = nn.LayerNorm(d_model)

    def forward(self, encoded_features, mask, target_idx):
        # Simple mean pooling across all features (no learned weights)
        aggregated_features = encoded_features.mean(dim=1)  # [B, D]

        aggregated_features = self.pooled_layernorm(aggregated_features)

        prediction = self.predictor(aggregated_features)  # [B, num_classes] or [B, 1]

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

        # Optimization: use single embedding for all continuous features
        if self.num_cont > 0:
            self.cont_embedding = UciEmbedding(self.num_cont, d_model)
        else:
            self.cont_embedding = None
            
        # Optimization: use single embedding for all categorical features
        if self.num_cat > 0:
            # Calculate total categorical classes
            total_cat_classes = sum(num_classes_dict[idx.item()] for idx in self.cat_idx)
            self.cat_embedding = Embedding(total_cat_classes, d_model)
            # Record offset for each categorical feature
            self.cat_offsets = {}
            current_offset = 0
            for idx in self.cat_idx:
                self.cat_offsets[idx.item()] = current_offset
                current_offset += num_classes_dict[idx.item()]
            
            # Pre-compute categorical feature offset matrix for efficient matrix operations
            self.cat_offset_tensor = torch.tensor([self.cat_offsets[idx.item()] for idx in self.cat_idx], 
                                                dtype=torch.long)
        else:
            self.cat_embedding = None
            self.cat_offsets = {}
            self.cat_offset_tensor = None

        # [msk] embedding
        self.mask_embedding = nn.Parameter(torch.randn(1, 1, d_model))

        # Decoder part: each column independent, whether to share parameters controlled by share_embedding
        if self.share_embedding:
            # Each column encoder/decoder shares parameters
            if self.num_cont > 0:
                self.cont_decoder = DecodingUciEmbedding(self.cont_embedding)
            else:
                self.cont_decoder = None
                
            if self.num_cat > 0:
                self.cat_decoder = DecodingEmbedding(self.cat_embedding, num_classes_dict, self.cat_idx, self.cat_offsets)
            else:
                self.cat_decoder = None
        else:
            # Each column encoder/decoder completely independent
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

        # Predictor - initialize according to target feature type
        # Dynamically determine target feature type and class count based on v
        # If target_idx not specified, default to last column
        if target_idx is None:
            target_idx = len(v) - 1  # Default last column as target
        
        if target_idx < len(v):
            target_type = v[target_idx].item()  # 0: regression, 1: classification
            if target_type == 0:
                # Regression task
                num_classes = 1
            else:
                # Classification task, get class count from num_classes_dict
                num_classes = num_classes_dict.get(target_idx, 2)  # Default binary classification
        else:
            # Default classification
            target_type = 1
            num_classes = 2
        
        self.predictor = CausalPredictor(d_model, self.num_features, target_type=target_type, num_classes=num_classes, dropout=dropout, target_idx=target_idx)
        # Mask generator - use SigmoidMaskGenerator as default implementation
        if mask_generator is None:
            from models.mask_generators import SigmoidMaskGenerator
            self.mask_generator = SigmoidMaskGenerator(self.num_features)
        else:
            self.mask_generator = mask_generator
            
        self.mask_logits = nn.Parameter(
            self.mask_generator.initialize_mask_logits(device=next(self.parameters()).device)
        )
        # Transformer encoder - support multi-layer multi-head attention
        self.encoder = nn.ModuleList([
            MaskedTransformerBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

    def update_parameters(self):
        """Update model parameters"""
        self.mask_generator.update_parameters()
        
    def get_parameters(self):
        """Get current parameter state"""
        params = self.mask_generator.get_parameters()
        params['mask_logits'] = self.mask_logits.detach().cpu().numpy()
        return params

    def get_embeddings(self, x, mask_positions=None):
        """
        Get feature embeddings, support mask functionality
        
        Args:
            x: Input data [B, F]
            mask_positions: Positions to mask [B, num_masked], None means no mask
        """
        B, F = x.shape
        feat_embed = torch.zeros(B, F, self.encoder[0].q_proj.out_features, device=x.device)
        
        # Optimization: process all continuous features at once
        if self.num_cont > 0:
            x_cont = x[:, self.cont_idx]  # [B, F_cont]
            cont_embed = self.cont_embedding(x_cont)  # [B, F_cont, D]
            # Use matrix indexing to put continuous feature embeddings back to original positions at once
            feat_embed[:, self.cont_idx, :] = cont_embed
        
        # Optimization: process all categorical features at once
        if self.num_cat > 0:
            x_cat = x[:, self.cat_idx]  # [B, F_cat]
            x_cat_ids = x_cat.long()
            x_cat_vals = torch.ones_like(x_cat, dtype=torch.float32)
            
            # Use pre-computed offset matrix for efficient matrix operations
            adjusted_ids = x_cat_ids + self.cat_offset_tensor.to(x.device).unsqueeze(0)
            
            # Process all categorical features at once
            cat_embed = self.cat_embedding({'ids': adjusted_ids, 'vals': x_cat_vals})  # [B, F_cat, D]
            # Use matrix indexing to put categorical feature embeddings back to original positions at once
            feat_embed[:, self.cat_idx, :] = cat_embed
        
        # Apply mask
        if mask_positions is not None:
            # Optimization: mask positions are same for each batch, directly use first batch position
            mask_pos = mask_positions[0, 0].item()  # Get mask position
            feat_embed[:, mask_pos] = self.mask_embedding.squeeze(0).squeeze(0)  # Apply to all batches
        
        return feat_embed  # [B, F, D]

    def forward(self, x, mask_positions=None, target_idx=None, task_type='reconstruction'):
        """
        Forward pass
        
        Args:
            x: Input data [B, F]
            mask_positions: Positions to mask [B, num_masked], None means no mask
            target_idx: Prediction target index, only used when task_type='prediction'
            task_type: Task type, 'reconstruction' or 'prediction'
        """
        batch_size, num_features = x.shape
        
        # Determine mask strategy based on task type
        if task_type == 'prediction' and target_idx is not None:
            # Prediction task: mask target column
            if mask_positions is None:
                mask_positions = torch.zeros(batch_size, 1, dtype=torch.long, device=x.device)
                mask_positions[:, 0] = target_idx
        
        # Get feature embeddings (apply mask)
        feats = self.get_embeddings(x, mask_positions)  # [B, F, D]
        
        # Generate mask using MaskGenerator
        mask = self.mask_generator(self.mask_logits, batch_size)
        
        # Encode - multi-layer Transformer
        encoded = feats
        for layer in self.encoder:
            encoded = layer(encoded, mask)

        # Reconstruction part - lazy update strategy: only compute masked columns
        outputs = [None] * num_features
        
        # If mask_positions provided, only compute masked columns
        if mask_positions is not None and mask_positions.numel() > 0:
            # Assume each batch has only one mask position and it's the same
            mask_idx = mask_positions[0, 0].item()  # Get mask position of first batch
            
            # Determine if masked is continuous or categorical feature
            if mask_idx in self.cont_idx:
                # Masked is continuous feature
                cont_idx_in_cont = (self.cont_idx == mask_idx).nonzero(as_tuple=True)[0].item()
                cont_encoded = encoded[:, mask_idx:mask_idx+1, :]  # [B, 1, D]
                
                if self.share_embedding:
                    # Use shared parameter decoder
                    full_cont_encoded = encoded[:, self.cont_idx, :]  # [B, F_cont, D]
                    cont_outputs = self.cont_decoder(full_cont_encoded)  # [B, F_cont]
                    # Find position of masked continuous feature in cont_outputs
                    outputs[mask_idx] = cont_outputs[:, cont_idx_in_cont]
                else:
                    # Use independent decoder
                    outputs[mask_idx] = self.cont_decoders[cont_idx_in_cont](cont_encoded[:, 0, :]).squeeze(-1)  # [B]
                    
            elif mask_idx in self.cat_idx:
                # Masked is categorical feature
                cat_idx_in_cat = (self.cat_idx == mask_idx).nonzero(as_tuple=True)[0].item()
                cat_encoded = encoded[:, mask_idx:mask_idx+1, :]  # [B, 1, D]
                
                if self.share_embedding:
                    # Use shared parameter decoder
                    full_cat_encoded = encoded[:, self.cat_idx, :]  # [B, F_cat, D]
                    cat_outputs = self.cat_decoder(full_cat_encoded)  # List of [B, n_class]
                    # Find position of masked categorical feature in cat_outputs
                    outputs[mask_idx] = cat_outputs[cat_idx_in_cat]
                else:
                    # Use independent decoder
                    outputs[mask_idx] = self.cat_decoders[cat_idx_in_cat](cat_encoded[:, 0, :])  # [B, n_class]


        # Prediction part
        if target_idx is None:
            target_idx = self.num_features - 1  # Default last column as target
        prediction = self.predictor(encoded, mask, target_idx)

        # Return reconstruction outputs for all features, prediction result and mask
        return outputs, prediction, mask
        
    def get_causal_mask(self):
        """Get final causal structure mask
        
        Returns:
            numpy.ndarray: Binary causal structure matrix with shape [num_features, num_features]
        """
        causal_mask = self.mask_generator.get_causal_mask(self.mask_logits)
        return causal_mask.detach().cpu().numpy() 