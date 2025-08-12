import torch
import torch.nn as nn
from einops import rearrange  # optional, but handy
from diffusers.models.embeddings import get_2d_sincos_pos_embed

class ConcatPatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding that processes two hidden states and concatenates them.
    Maintains exact compatibility with original SD3 model weights.
    
    Args:
        height (`int`, defaults to `224`): The height of the image.
        width (`int`, defaults to `224`): The width of the image.
        patch_size (`int`, defaults to `16`): The size of the patches.
        in_channels (`int`, defaults to `3`): The number of input channels.
        embed_dim (`int`, defaults to `768`): The output dimension of the embedding.
        layer_norm (`bool`, defaults to `False`): Whether or not to use layer normalization.
        flatten (`bool`, defaults to `True`): Whether or not to flatten the output.
        bias (`bool`, defaults to `True`): Whether or not to use bias.
        interpolation_scale (`float`, defaults to `1`): The scale of the interpolation.
        pos_embed_type (`str`, defaults to `"sincos"`): The type of positional embedding.
        pos_embed_max_size (`int`, defaults to `None`): The maximum size of the positional embedding.
    """

    def __init__(
        self,
        height=224,
        width=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
        interpolation_scale=1,
        pos_embed_type="sincos",
        pos_embed_max_size=None,
    ):
        super().__init__()

        num_patches = (height // patch_size) * (width // patch_size)
        self.flatten = flatten
        self.layer_norm = layer_norm
        self.pos_embed_max_size = pos_embed_max_size

        # Single projection layer for both hidden states - same as original
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
        )
        
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None

        self.patch_size = patch_size
        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = height // patch_size
        self.interpolation_scale = interpolation_scale

        # Initialize positional embeddings EXACTLY like the original SD3
        # This ensures weight compatibility
        if pos_embed_max_size:
            grid_size = pos_embed_max_size
        else:
            grid_size = int(num_patches**0.5)

        if pos_embed_type is None:
            self.pos_embed = None
            self.pos_embed_dual = None
        elif pos_embed_type == "sincos":
            # Create standard positional embeddings exactly like original
            pos_embed = get_2d_sincos_pos_embed(
                embed_dim,
                grid_size,
                base_size=self.base_size,
                interpolation_scale=self.interpolation_scale,
                output_type="pt",
            )
            persistent = True if pos_embed_max_size else False
            self.register_buffer("pos_embed", pos_embed.float().unsqueeze(0), persistent=persistent)
            
            # Pre-compute the tiled version for dual inputs
            # This is created from pos_embed, so it updates when pos_embed is loaded
            self._create_dual_pos_embed()
        else:
            raise ValueError(f"Unsupported pos_embed_type: {pos_embed_type}")
    
    def _create_dual_pos_embed(self):
        """Creates the dual (tiled) version of positional embeddings."""
        if self.pos_embed is None:
            self.pos_embed_dual = None
            return
            
        # Get dimensions
        batch_size = self.pos_embed.shape[0]
        num_positions = self.pos_embed.shape[1]
        embed_dim = self.pos_embed.shape[-1]
        
        # Calculate grid size
        grid_size = int(num_positions ** 0.5)
        
        # Reshape to 2D grid
        pos_embed_2d = self.pos_embed.reshape(batch_size, grid_size, grid_size, embed_dim)
        
        # Tile horizontally for dual images
        pos_embed_dual = torch.cat([pos_embed_2d, pos_embed_2d], dim=2)  # [B, H, W*2, D]
        pos_embed_dual = pos_embed_dual.reshape(batch_size, -1, embed_dim)
        
        # Register as a non-persistent buffer (computed from pos_embed)
        self.register_buffer("pos_embed_dual", pos_embed_dual, persistent=False)

    def cropped_pos_embed_dual(self, height, width):
        """Crops positional embeddings for dual input."""
        if self.pos_embed_max_size is None:
            raise ValueError("`pos_embed_max_size` must be set for cropping.")

        height = height // self.patch_size
        width = width // self.patch_size
        
        if height > self.pos_embed_max_size:
            raise ValueError(
                f"Height ({height}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )
        if width > self.pos_embed_max_size:
            raise ValueError(
                f"Width ({width}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )

        # Crop from the single image positional embeddings first
        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2
        
        spatial_pos_embed = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
        spatial_pos_embed_cropped = spatial_pos_embed[:, top : top + height, left : left + width, :]
        
        # Tile horizontally for dual images
        spatial_pos_embed_dual = torch.cat([spatial_pos_embed_cropped, spatial_pos_embed_cropped], dim=2)
        spatial_pos_embed_dual = spatial_pos_embed_dual.reshape(1, -1, spatial_pos_embed_dual.shape[-1])
        
        return spatial_pos_embed_dual

    def forward(self, hidden_states_list):
        """
        Args:
            hidden_states_list: List of two tensors, each of shape [B, C, H, W]
                               OR a single tensor [B, C, H, W] for compatibility
        
        Returns:
            Combined embedded patches with shape [B, 2*num_patches, embed_dim] for dual input
            or [B, num_patches, embed_dim] for single input
        """
        # Handle both dual input (list) and single input (tensor) for compatibility
        if isinstance(hidden_states_list, list):
            assert len(hidden_states_list) == 2, "Expected exactly 2 hidden states"
            hidden_states_1, hidden_states_2 = hidden_states_list
            assert hidden_states_1.shape == hidden_states_2.shape, "Both hidden states must have the same shape"
            is_dual = True
            reference_hidden_states = hidden_states_1
        else:
            # Single input mode for compatibility with original SD3
            hidden_states_1 = hidden_states_list
            hidden_states_2 = None
            is_dual = False
            reference_hidden_states = hidden_states_1
        
        if self.pos_embed_max_size is not None:
            height, width = reference_hidden_states.shape[-2:]
        else:
            height, width = reference_hidden_states.shape[-2] // self.patch_size, reference_hidden_states.shape[-1] // self.patch_size
        
        # Project hidden states
        latent_1 = self.proj(hidden_states_1)
        if self.flatten:
            latent_1 = latent_1.flatten(2).transpose(1, 2)  # BCHW -> BNC
        
        if is_dual:
            latent_2 = self.proj(hidden_states_2)
            if self.flatten:
                latent_2 = latent_2.flatten(2).transpose(1, 2)  # BCHW -> BNC
            # Concatenate along the sequence dimension
            latent = torch.cat([latent_1, latent_2], dim=1)  # [B, 2*num_patches, embed_dim]
        else:
            latent = latent_1  # [B, num_patches, embed_dim]
        
        if self.layer_norm:
            latent = self.norm(latent)
        
        if self.pos_embed is None:
            return latent.to(latent.dtype)
        
        # Handle positional embeddings
        if is_dual:
            # Use dual (tiled) positional embeddings
            if self.pos_embed_dual is None:
                self._create_dual_pos_embed()
            
            if self.pos_embed_max_size:
                pos_embed = self.cropped_pos_embed_dual(height, width)
            else:
                if self.height != height or self.width != width:
                    # Generate new positional embeddings dynamically
                    pos_embed_single = get_2d_sincos_pos_embed(
                        embed_dim=self.pos_embed.shape[-1],
                        grid_size=(height, width),
                        base_size=self.base_size,
                        interpolation_scale=self.interpolation_scale,
                        device=latent.device,
                        output_type="pt",
                    )
                    # Tile for dual
                    pos_embed_single_2d = pos_embed_single.reshape(1, height, width, -1)
                    pos_embed_dual_2d = torch.cat([pos_embed_single_2d, pos_embed_single_2d], dim=2)
                    pos_embed = pos_embed_dual_2d.reshape(1, -1, pos_embed_dual_2d.shape[-1])
                    pos_embed = pos_embed.float()
                else:
                    pos_embed = self.pos_embed_dual
        else:
            # Single input mode - use original positional embeddings
            if self.pos_embed_max_size:
                # Original cropping logic from SD3
                height_patches = height // self.patch_size
                width_patches = width // self.patch_size
                top = (self.pos_embed_max_size - height_patches) // 2
                left = (self.pos_embed_max_size - width_patches) // 2
                spatial_pos_embed = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
                spatial_pos_embed = spatial_pos_embed[:, top : top + height_patches, left : left + width_patches, :]
                pos_embed = spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])
            else:
                if self.height != height or self.width != width:
                    pos_embed = get_2d_sincos_pos_embed(
                        embed_dim=self.pos_embed.shape[-1],
                        grid_size=(height, width),
                        base_size=self.base_size,
                        interpolation_scale=self.interpolation_scale,
                        device=latent.device,
                        output_type="pt",
                    )
                    pos_embed = pos_embed.float().unsqueeze(0)
                else:
                    pos_embed = self.pos_embed

        return (latent + pos_embed).to(latent.dtype)
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Override to recreate dual pos_embed when loading weights."""
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        # After loading, recreate the dual version
        if hasattr(self, 'pos_embed') and self.pos_embed is not None:
            self._create_dual_pos_embed()
