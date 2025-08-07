import torch
import torch.nn as nn
from einops import rearrange  # optional, but handy
from diffusers.models.embeddings import get_2d_sincos_pos_embed

class ConcatPatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding with support for SD3 cropping,
    now accepting multiple latents and concatenating along the patch dimension.
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
        pos_embed_max_size=None,  # For SD3 cropping
    ):
        super().__init__()

        self.patch_size = patch_size
        num_patches = (height // patch_size) * (width // patch_size)

        self.flatten = flatten
        self.layer_norm = layer_norm
        self.pos_embed_max_size = pos_embed_max_size

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias
        )
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None

        # store original grid dims for single-latent case
        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = self.height
        self.interpolation_scale = interpolation_scale

        # build buffer for positional embeddings
        if pos_embed_max_size:
            grid_size = pos_embed_max_size
        else:
            grid_size = int(num_patches**0.5)

        if pos_embed_type is None:
            self.pos_embed = None
        elif pos_embed_type == "sincos":
            pe = get_2d_sincos_pos_embed(
                embed_dim, grid_size,
                base_size=self.base_size,
                interpolation_scale=self.interpolation_scale,
                output_type="pt"
            )
            persistent = True if pos_embed_max_size else False
            self.register_buffer("pos_embed", pe.unsqueeze(0).float(), persistent=persistent)
        else:
            raise ValueError(f"Unsupported pos_embed_type: {pos_embed_type}")

    def cropped_pos_embed(self, height, width):
        """Crop a larger sin-cos grid down to H×W patches."""
        if self.pos_embed_max_size is None:
            raise ValueError("`pos_embed_max_size` must be set for cropping.")

        h_patches = height // self.patch_size
        w_patches = width // self.patch_size
        if h_patches > self.pos_embed_max_size or w_patches > self.pos_embed_max_size:
            raise ValueError(f"Crop too large: {h_patches}×{w_patches} > max {self.pos_embed_max_size}")

        top = (self.pos_embed_max_size - h_patches) // 2
        left = (self.pos_embed_max_size - w_patches) // 2

        pe = self.pos_embed.reshape(
            1, self.pos_embed_max_size, self.pos_embed_max_size, -1
        )
        pe = pe[:, top : top + h_patches, left : left + w_patches, :]
        return pe.reshape(1, -1, pe.shape[-1])

    def forward(self, *latents: torch.Tensor):
        """
        Accepts one or more images of shape [B, C, H, W].
        Patchify each, optionally normalize, then concatenate
        along the patch-sequence axis before adding pos-emb.
        """
        processed = []
        for x in latents:
            # project to patches
            p = self.proj(x)  # [B, E, H/ps, W/ps]
            if self.flatten:
                p = p.flatten(2).transpose(1, 2)  # → [B, N, E]
            if self.layer_norm:
                p = self.norm(p)
            processed.append(p)

        # cat along the N (patch) dimension
        x = torch.cat(processed, dim=1)  # [B, n_latents * N, E]

        # if no pos-embed, return immediately
        if self.pos_embed is None:
            return x.to(x.dtype)

        # compute one block of pos_embed for a single latent
        # figure out H/W of patches per-latent
        sample = latents[0]
        if self.pos_embed_max_size:
            h, w = sample.shape[-2], sample.shape[-1]
            single_pe = self.cropped_pos_embed(h, w)
        else:
            h_patches = sample.shape[-2] // self.patch_size
            w_patches = sample.shape[-1] // self.patch_size
            if (h_patches, w_patches) != (self.height, self.width):
                pe = get_2d_sincos_pos_embed(
                    embed_dim=self.pos_embed.shape[-1],
                    grid_size=(h_patches, w_patches),
                    base_size=self.base_size,
                    interpolation_scale=self.interpolation_scale,
                    device=x.device,
                    output_type="pt",
                )
                single_pe = pe.unsqueeze(0).float()
            else:
                single_pe = self.pos_embed

        # tile pos_embed for each latent before concat
        n = len(latents)
        pe_full = single_pe.repeat(1, n, 1)  # [1, n*N, E]

        return (x + pe_full).to(x.dtype)
