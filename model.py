import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureInteractionLayer(nn.Module):
    """
    Bi-Directional Cross-Attention (Implemented as Self-Attention on Joint Feature Space).
    Allows Recipe and Constraints to 'negotiate' the final score.
    """
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x shape: [Batch, Seq_Len=2, Dim] (Sequence of Recipe_Emb, Constraint_Emb)
        attn_out, _ = self.multihead_attn(x, x, x)
        return self.norm(x + attn_out) # Residual Connection (Critical for deep networks)

class FRESH_Network(nn.Module):
    def __init__(self, embedding_dim=384):
        super(FRESH_Network, self).__init__()
        
        # --- TOWER 1: TASTE ENCODER (Recipe + User Bias) ---
        # We fuse Recipe(384) + User_Vector(384) to model pure preference
        self.taste_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256), # Modified to accept User Context
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
        # --- TOWER 2: CONSTRAINT ENCODER (Medical RAG) ---
        self.constraint_encoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
        # --- TOWER 3: WIDE CONTEXT ---
        self.wide_encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU()
        )
        
        # --- INTERACTION LAYER ---
        # Treats Recipe and Constraint as a sequence of length 2
        self.interaction = FeatureInteractionLayer(dim=128, num_heads=4)
        
        # --- FINAL HEAD ---
        self.head = nn.Sequential(
            nn.Linear(128 * 2 + 16 + 1, 64), 
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid() 
        )

    def forward(self, recipe_vecs, user_vecs, constraint_vecs, explicit_feats, align_score):
        """
        user_vecs: [Batch, 384] - The dynamic user preference vector (u_t)
        """
        # 1. Taste Modeling (Recipe || User)
        # This matches your Eq: N(v_r, u_t)
        taste_input = torch.cat([recipe_vecs, user_vecs], dim=1) 
        taste_emb = self.taste_encoder(taste_input)       # [Batch, 128]
        
        # 2. Constraint Modeling
        constraint_emb = self.constraint_encoder(constraint_vecs) # [Batch, 128]
        
        # 3. Neural Interaction (Cross-Attention)
        # Stack them as a sequence: [Batch, 2, 128]
        seq_input = torch.stack([taste_emb, constraint_emb], dim=1)
        interacted = self.interaction(seq_input)
        
        # Flatten for the head: [Batch, 256]
        flat_features = interacted.view(interacted.size(0), -1)
        
        # 4. Wide Tower
        wide_out = self.wide_encoder(explicit_feats)

        # 5. Neuro-Symbolic Fusion
        final_input = torch.cat([flat_features, wide_out, align_score], dim=1)
        
        return self.head(final_input)