import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .common import MLPBlock


class DNA_Module(nn.Module):
    def __init__(self, embed_dim, sam_orig_dim, cnn_orig_dim, num_heads,
                 NA_method='fix', NA_kernel_size=3, max_dilation=3):
        super(DNA_Module, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** 0.5

        self.rpb = nn.Parameter(torch.zeros(self.num_heads, NA_kernel_size, NA_kernel_size))
        self.kernel_size = NA_kernel_size
        
        self.NA_method = NA_method
        if NA_method == 'fix':
            self.dilation = max_dilation

        elif NA_method == "per-query":
            self.dilation_candidates = list(range(1, max_dilation + 1))
            self.dilation_predictor = nn.ModuleDict({
                'SS': nn.Linear(self.head_dim, len(self.dilation_candidates)),
                'SC': nn.Linear(self.head_dim, len(self.dilation_candidates)),
                'CC': nn.Linear(self.head_dim, len(self.dilation_candidates)),
                'CS': nn.Linear(self.head_dim, len(self.dilation_candidates))
            })

        # Linear layers for Query, Key, Value, Out
        self.q_proj_ss = nn.Linear(sam_orig_dim, embed_dim)
        self.k_proj_ss = nn.Linear(sam_orig_dim, embed_dim)
        self.v_proj_ss = nn.Linear(sam_orig_dim, embed_dim)
        self.q_proj_sc = nn.Linear(sam_orig_dim, embed_dim)
        self.k_proj_sc = nn.Linear(cnn_orig_dim, embed_dim)
        self.v_proj_sc = nn.Linear(cnn_orig_dim, embed_dim)
        self.q_proj_cc = nn.Linear(cnn_orig_dim, embed_dim)
        self.k_proj_cc = nn.Linear(cnn_orig_dim, embed_dim)
        self.v_proj_cc = nn.Linear(cnn_orig_dim, embed_dim)       
        self.q_proj_cs = nn.Linear(cnn_orig_dim, embed_dim)
        self.k_proj_cs = nn.Linear(sam_orig_dim, embed_dim)
        self.v_proj_cs = nn.Linear(sam_orig_dim, embed_dim)
        self.out_proj_sam_ss = nn.Linear(embed_dim, sam_orig_dim)
        self.out_proj_sam_sc = nn.Linear(embed_dim, sam_orig_dim)
        self.out_proj_cnn_cc = nn.Linear(embed_dim, cnn_orig_dim)
        self.out_proj_cnn_cs = nn.Linear(embed_dim, cnn_orig_dim)
        self.fuse_proj_sam = nn.Linear(2*sam_orig_dim, sam_orig_dim)
        self.mlp_sam = MLPBlock(sam_orig_dim, sam_orig_dim*4)
        self.ln_sam = nn.LayerNorm(sam_orig_dim)
        self.fuse_proj_cnn = nn.Linear(2*cnn_orig_dim, cnn_orig_dim)
        self.mlp_cnn = MLPBlock(cnn_orig_dim, cnn_orig_dim*4)
        self.ln_cnn = nn.LayerNorm(cnn_orig_dim)
        
        # Normalization layers
        self.norm_sam = nn.LayerNorm(sam_orig_dim)
        self.norm_cnn = nn.LayerNorm(cnn_orig_dim)
    
    def split_heads(self, x):
        B, N, C = x.shape
        x = x.view(B, N, self.num_heads, self.head_dim)
        return x.transpose(1, 2)
    
    def combine_heads(self, x):
        B, H, N, D = x.shape
        x = x.transpose(1, 2).contiguous().view(B, N, H * D)
        return x

    def StandardAttention(self, Q, K, V):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_probs = F.sigmoid(attn_scores)
        attn_output = torch.matmul(attn_probs, V)
        return attn_output
    
    def NeighborhoodAttention(self, Q, K, V, mode):
        B, _, N, _ = Q.shape
        Hs = Ws = int(math.sqrt(N))
        
        Q_flat = Q.reshape(B, self.num_heads, Hs*Ws, self.head_dim)
        K = K.reshape(B, self.num_heads, Hs, Ws, self.head_dim)
        V = V.reshape(B, self.num_heads, Hs, Ws, self.head_dim)
        
        if self.NA_method == "fix":
            attn_output = self.compute_na_attention(Q_flat, K, V, self.dilation)
            
        elif self.NA_method == "per-query":
            logits = self.dilation_predictor[mode](Q_flat)
            weights = F.softmax(logits, dim=-1)
            
            attn_output = torch.zeros_like(Q_flat)
            for i, dilation in enumerate(self.dilation_candidates):
                attn_output_i = self.compute_na_attention(Q_flat, K, V, dilation)
                attn_output += attn_output_i * weights[..., i].unsqueeze(-1)
                
        return attn_output
        
    def compute_na_attention(self, Q, K, V, dilation):
        B, H, Hs, Ws, C = K.shape
        k = self.kernel_size
        
        pad = dilation * (self.kernel_size - 1) // 2

        # --- Extract local neighborhoods from Key ---
        K_padded = F.pad(
            K.permute(0, 1, 4, 2, 3).reshape(B*H, C, Hs, Ws), (pad, pad, pad, pad)
        )        
        K_unfold = F.unfold(K_padded, kernel_size=k, dilation=dilation, padding=0, stride=1)
        K_unfold = K_unfold.transpose(1, 2).reshape(B, H, Hs*Ws, k*k, C)

        # --- Attention scores ---
        attn_scores = (Q.unsqueeze(3) * K_unfold).sum(-1) / self.scale
        rpb_flat = self.rpb.reshape(H, -1)
        attn_scores = attn_scores + rpb_flat.unsqueeze(0).unsqueeze(2)
        attn_probs = torch.sigmoid(attn_scores)

        # --- Extract local neighborhoods from Value ---
        V_padded = F.pad(
            V.permute(0, 1, 4, 2, 3).reshape(B*H, C, Hs, Ws),
            (pad, pad, pad, pad)
        )
        V_unfold = F.unfold(V_padded, kernel_size=k, dilation=dilation, padding=0, stride=1)
        V_unfold = V_unfold.transpose(1, 2).reshape(B, H, Hs*Ws, k*k, C)

        attn_output = (attn_probs.unsqueeze(-1) * V_unfold).sum(dim=-2)
        return attn_output
    
    def forward(self, image_embeddings, cnn_embeddings):
        B, Hs, Ws, Cs = image_embeddings.shape
        B, Cc, Hc, Wc = cnn_embeddings.shape
        orig_image_embeddings = image_embeddings.clone()
        orig_cnn_embeddings = cnn_embeddings.clone()
        
        # Align SAM and CNN embeddings
        image_embeddings = image_embeddings.permute(0, 3, 1, 2)
        if (Hc, Wc) != (Hs, Ws):
            cnn_embeddings = F.interpolate(cnn_embeddings, size=(Hs, Ws), mode='bilinear', align_corners=False)
        
        # Reshape to (B, Hs*Ws, C) for attention computation
        image_embeddings = image_embeddings.flatten(2).permute(0, 2, 1)
        cnn_embeddings = cnn_embeddings.flatten(2).permute(0, 2, 1)
        
        # ----- S -> S Attention -----
        Q = self.split_heads(self.q_proj_ss(image_embeddings))
        K = self.split_heads(self.k_proj_ss(image_embeddings))
        V = self.split_heads(self.v_proj_ss(image_embeddings))
        
        attn_output_ss = self.NeighborhoodAttention(Q, K, V, mode='SS')
        
        attn_output_ss = self.combine_heads(attn_output_ss)
        sam_attn_output_ss = self.out_proj_sam_ss(attn_output_ss)
        
        # ----- S -> C Attention -----
        Q = self.split_heads(self.q_proj_sc(image_embeddings))
        K = self.split_heads(self.k_proj_sc(cnn_embeddings))
        V = self.split_heads(self.v_proj_sc(cnn_embeddings))
        
        attn_output_sc = self.NeighborhoodAttention(Q, K, V, mode='SC')
        
        attn_output_sc = self.combine_heads(attn_output_sc)
        sam_attn_output_sc = self.out_proj_sam_sc(attn_output_sc)
        
        # ----- C -> C Attention -----
        Q = self.split_heads(self.q_proj_cc(cnn_embeddings))
        K = self.split_heads(self.k_proj_cc(cnn_embeddings))
        V = self.split_heads(self.v_proj_cc(cnn_embeddings))
        
        attn_output_cc = self.NeighborhoodAttention(Q, K, V, mode='CC')
        
        attn_output_cc = self.combine_heads(attn_output_cc)
        cnn_attn_output_cc = self.out_proj_cnn_cc(attn_output_cc)
                
        # ----- C -> S Attention -----
        Q = self.split_heads(self.q_proj_cs(cnn_embeddings))
        K = self.split_heads(self.k_proj_cs(image_embeddings))
        V = self.split_heads(self.v_proj_cs(image_embeddings))
        
        attn_output_cs = self.NeighborhoodAttention(Q, K, V, mode='CS')
        
        attn_output_cs = self.combine_heads(attn_output_cs)
        cnn_attn_output_cs = self.out_proj_cnn_cs(attn_output_cs)
        
        # ----- SAM, CNN Fusion -----       
        sam_fused_attn_output = torch.cat([sam_attn_output_ss, sam_attn_output_sc], dim=2)
        sam_fused_attn_output = self.fuse_proj_sam(sam_fused_attn_output)
        sam_fused_attn_output = self.mlp_sam(sam_fused_attn_output)
        sam_fused_attn_output = self.ln_sam(sam_fused_attn_output)
        
        cnn_fused_attn_output = torch.cat([cnn_attn_output_cc, cnn_attn_output_cs], dim=2)
        cnn_fused_attn_output = self.fuse_proj_cnn(cnn_fused_attn_output)
        cnn_fused_attn_output = self.mlp_cnn(cnn_fused_attn_output)
        cnn_fused_attn_output = self.ln_cnn(cnn_fused_attn_output)
        
        # Reshape features to original resolution
        sam_fused_attn_output = sam_fused_attn_output.reshape(B, Hs, Ws, Cs)
        cnn_fused_attn_output = cnn_fused_attn_output.permute(0, 2, 1).reshape(B, Cc, Hs, Ws)
        if (Hc, Wc) != (Hs, Ws):
            cnn_fused_attn_output = F.interpolate(cnn_fused_attn_output, size=(Hc, Wc), mode='bilinear', align_corners=False)

        # Add and Norm
        image_embeddings = orig_image_embeddings + sam_fused_attn_output
        accentuated_image_embeddings = self.norm_sam(image_embeddings)
        cnn_embeddings = orig_cnn_embeddings + cnn_fused_attn_output
        accentuated_cnn_embeddings = self.norm_cnn(cnn_embeddings.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            
        return accentuated_image_embeddings, accentuated_cnn_embeddings
 