import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block


class DynamicGraphAttention(GPT2Attention):
    """Dynamic graph learning attention mechanism extending GPT2Attention
    
    Adjusts attention weights using a learned adjacency matrix
    """
    
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention, layer_idx)
        
        # Initialize dimensions
        self.embed_dim = config.hidden_size
        self.head_dim = self.embed_dim // config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.num_heads = config.num_attention_heads
        
        # Head grouping parameters
        self.num_head_groups = getattr(config, "num_head_groups", 4)
        self.num_head_groups = min(self.num_head_groups, self.num_heads)
        self.heads_per_group = self.num_heads // self.num_head_groups
        
        # Graph generator network
        self.graph_generator = nn.Sequential(
            nn.Linear(self.head_dim * 2, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, 1),
            nn.Sigmoid()
        )
        
        # Regularization parameters
        self.sparsity_lambda = getattr(config, "sparsity_lambda", 0.001)
        self.use_dynamic_graph = getattr(config, "use_dynamic_graph", True)
        
        # Current adjacency matrix for regularization
        self.current_adjacency = None
        
        # Generation mode flag
        self.generation_mode = False
    
    def _compute_adjacency_matrix(self, query_states, key_states):
        """Compute adjacency matrix from query and key states
        
        Args:
            query_states: Query features [batch_size, seq_len_q, head_dim]
            key_states: Key features [batch_size, seq_len_k, head_dim]
            
        Returns:
            adjacency_matrix: Learned adjacency matrix [batch_size, seq_len_q, seq_len_k]
        """
        batch_size, seq_len_q, _ = query_states.size()
        _, seq_len_k, _ = key_states.size()
        
        # Create token pairs
        q_i = query_states.unsqueeze(2).expand(-1, -1, seq_len_k, -1)
        k_j = key_states.unsqueeze(1).expand(-1, seq_len_q, -1, -1)
        
        token_pairs = torch.cat([q_i, k_j], dim=-1)
        
        # Reshape for processing
        reshaped_pairs = token_pairs.view(-1, token_pairs.size(-1))
        
        # Compute edge weights
        edge_weights = self.graph_generator(reshaped_pairs).view(batch_size, seq_len_q, seq_len_k)
        
        return edge_weights
    
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        """Extended attention mechanism with graph structure integration"""
        # Original attention scores
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        
        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )
        
        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx and self.layer_idx is not None:
            attn_weights = attn_weights / float(self.layer_idx + 1)
        
        if not self.is_cross_attention:
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        
        # Apply dynamic graph if enabled
        if self.use_dynamic_graph:
            batch_size, num_heads, seq_length_q, head_dim = query.size()
            _, _, seq_length_k, _ = key.size()
            
            # Simplified calculation in generation mode with sequence length 1
            if self.generation_mode and seq_length_q == 1:
                group_adjacency_matrices = []
                
                for g in range(self.num_head_groups):
                    start_head_idx = g * self.heads_per_group
                    end_head_idx = min((g + 1) * self.heads_per_group, num_heads)
                    
                    q_group = query[:, start_head_idx:end_head_idx].mean(dim=1)
                    k_group = key[:, start_head_idx:end_head_idx].mean(dim=1)
                    
                    group_adj = self._compute_adjacency_matrix(q_group, k_group)
                    group_adjacency_matrices.append(group_adj)
                
                head_adjacency_matrices = []
                for h in range(num_heads):
                    group_idx = min(h // self.heads_per_group, self.num_head_groups - 1)
                    head_adjacency_matrices.append(group_adjacency_matrices[group_idx])
                
                adjacency_matrix = torch.stack(head_adjacency_matrices, dim=1)
                self.current_adjacency = torch.stack(group_adjacency_matrices, dim=0).mean(dim=0)
                
            else:
                # Standard mode (training or multi-token generation)
                group_adjacency_matrices = []
                
                for g in range(self.num_head_groups):
                    start_head_idx = g * self.heads_per_group
                    end_head_idx = min((g + 1) * self.heads_per_group, num_heads)
                    
                    q_group = query[:, start_head_idx:end_head_idx].mean(dim=1)
                    k_group = key[:, start_head_idx:end_head_idx].mean(dim=1)
                    
                    group_adj = self._compute_adjacency_matrix(q_group, k_group)
                    group_adjacency_matrices.append(group_adj)
                
                head_adjacency_matrices = []
                for h in range(num_heads):
                    group_idx = min(h // self.heads_per_group, self.num_head_groups - 1)
                    head_adjacency_matrices.append(group_adjacency_matrices[group_idx])
                
                adjacency_matrix = torch.stack(head_adjacency_matrices, dim=1)
                self.current_adjacency = torch.stack(group_adjacency_matrices, dim=0).mean(dim=0)
            
            # Apply graph influence to attention weights
            attn_weights = attn_weights + torch.log(adjacency_matrix + 1e-6)
        
        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Attention softmax
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply head mask
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        
        # Compute attention output
        attn_output = torch.matmul(attn_weights, value)
        
        return attn_output, attn_weights


def compute_fd_loss(adjacency_matrix, feature_to_tokens_list, fd_list, alpha=0.5, column_names=None):
    """Compute functional dependency loss for graph regularization
    
    Args:
        adjacency_matrix: Attention graph matrix [batch_size, seq_len, seq_len]
        feature_to_tokens_list: List of feature-to-token mappings for each sample
        fd_list: List of functional dependencies [[[left_feats], [right_feats]], ...]
        alpha: Expected minimum edge weight
        column_names: Column names list for mapping indices to features
    
    Returns:
        fd_loss: Functional dependency regularization loss
    """
    if not fd_list or adjacency_matrix is None:
        return torch.tensor(0.0, device=adjacency_matrix.device if adjacency_matrix is not None else "cpu")
    
    batch_size = adjacency_matrix.shape[0]
    fd_loss = torch.tensor(0.0, device=adjacency_matrix.device)
    
    # Handle batch size mismatch
    if len(feature_to_tokens_list) != batch_size:
        batch_size = min(batch_size, len(feature_to_tokens_list))
    
    # Handle multi-head attention
    if len(adjacency_matrix.shape) >= 4:
        adjacency_matrix = adjacency_matrix.mean(dim=1) if adjacency_matrix.shape[0] == batch_size else adjacency_matrix.mean(dim=0)
    
    # Process each FD
    for fd in fd_list:
        left_feat_indices, right_feat_indices = fd[0], fd[1]
        fd_sample_losses = []
        
        # Process each sample
        for batch_idx in range(batch_size):
            feature_to_tokens = feature_to_tokens_list[batch_idx]
            sample_adj = adjacency_matrix[batch_idx]
            
            # Collect tokens for left and right features
            left_tokens = []
            right_tokens = []
            
            # Map indices to feature names
            if column_names is not None:
                for idx in left_feat_indices:
                    if 0 <= idx < len(column_names):
                        feat_name = column_names[idx]
                        if feat_name in feature_to_tokens:
                            left_tokens.extend(feature_to_tokens[feat_name])
                
                for idx in right_feat_indices:
                    if 0 <= idx < len(column_names):
                        feat_name = column_names[idx]
                        if feat_name in feature_to_tokens:
                            right_tokens.extend(feature_to_tokens[feat_name])
            else:
                # Try direct index mapping
                for feat in left_feat_indices:
                    feat_key = feat if feat in feature_to_tokens else str(feat)
                    if feat_key in feature_to_tokens:
                        left_tokens.extend(feature_to_tokens[feat_key])
                
                for feat in right_feat_indices:
                    feat_key = feat if feat in feature_to_tokens else str(feat)
                    if feat_key in feature_to_tokens:
                        right_tokens.extend(feature_to_tokens[feat_key])
            
            # Skip if no tokens found
            if not left_tokens or not right_tokens:
                continue
            
            # Limit indices to matrix dimensions
            max_idx = sample_adj.shape[0] - 1
            left_tokens = [min(idx, max_idx) for idx in left_tokens if idx <= max_idx]
            right_tokens = [min(idx, max_idx) for idx in right_tokens if idx <= max_idx]
            
            if not left_tokens or not right_tokens:
                continue
            
            # Calculate average edge weight
            fd_edges = sample_adj[right_tokens][:, left_tokens]
            avg_weight = fd_edges.mean()
            
            # Compute loss using soft constraint (Softplus)
            sample_loss = torch.log(1 + torch.exp(alpha - avg_weight))
            fd_sample_losses.append(sample_loss)
        
        # Average loss across samples
        if fd_sample_losses:
            fd_loss += torch.stack(fd_sample_losses).mean()
    
    # Average across all FDs
    fd_loss = fd_loss / len(fd_list) if fd_list else fd_loss
    
    return fd_loss

class DynamicGraphGPT2Block(GPT2Block):
    """GPT2 block with dynamic graph attention"""
    
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        # Replace standard attention with dynamic graph attention
        self.attn = DynamicGraphAttention(config, layer_idx=layer_idx)


class TabDynamicGraphGPT2(GPT2LMHeadModel):
    """GPT2 model with dynamic graph learning for tabular data
    
    Uses dynamic graph mechanism to learn token dependencies and capture 
    structural information in tabular data with FD constraints.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Replace standard blocks with dynamic graph blocks
        for i, _ in enumerate(self.transformer.h):
            self.transformer.h[i] = DynamicGraphGPT2Block(config, layer_idx=i)
        
        # FD parameters
        self.fd_lambda = getattr(config, "fd_lambda", 0.1)
        self.fd_alpha = getattr(config, "fd_alpha", 0.5)
        self.fd_list = getattr(config, "fd_list", [])
        
        # Initialize weights
        self.post_init()
    
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        feature_to_tokens=None,
        column_names=None,
    ):
        # Call parent forward to get base outputs
        outputs = super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Add regularization losses during training
        if self.training and labels is not None:
            # 1. Graph sparsity regularization
            graph_loss = self._calculate_graph_regularization()
            sparsity_lambda = getattr(self.config, "sparsity_lambda", 0.001)
            
            # 2. FD regularization loss
            fd_loss = torch.tensor(0.0, device=outputs.loss.device)
            if self.fd_list and feature_to_tokens is not None:
                # Calculate FD loss from each layer's adjacency matrix
                fd_losses = []
                for layer in self.transformer.h:
                    if hasattr(layer.attn, "current_adjacency") and layer.attn.current_adjacency is not None:
                        layer_fd_loss = compute_fd_loss(
                            layer.attn.current_adjacency, 
                            feature_to_tokens, 
                            self.fd_list, 
                            self.fd_alpha,
                            column_names=column_names
                        )
                        fd_losses.append(layer_fd_loss)
                
                if fd_losses:
                    fd_loss = torch.stack(fd_losses).mean()
            
            # 3. Update total loss
            outputs.loss = outputs.loss + sparsity_lambda * graph_loss + self.fd_lambda * fd_loss
        
        return outputs
    
    def _calculate_graph_regularization(self):
        """Calculate graph regularization loss to promote sparsity"""
        regularization = 0.0
        count = 0
        
        for layer in self.transformer.h:
            if hasattr(layer.attn, "current_adjacency") and layer.attn.current_adjacency is not None:
                regularization += torch.mean(torch.abs(layer.attn.current_adjacency))
                count += 1
        
        return regularization / count if count > 0 else regularization
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """Prepare inputs for generation process"""
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, **kwargs
        )
        
        # Set generation mode flag
        for block in self.transformer.h:
            if hasattr(block.attn, "generation_mode"):
                block.attn.generation_mode = True
        
        # Pass through feature mappings
        if 'feature_to_tokens' in kwargs:
            model_inputs['feature_to_tokens'] = kwargs['feature_to_tokens']
        
        if 'column_names' in kwargs:
            model_inputs['column_names'] = kwargs['column_names']
        
        return model_inputs
    
