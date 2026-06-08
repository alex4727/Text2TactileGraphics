import numpy as np
from PIL import Image

from tactilegen.config import DEFAULT_TILED_DIFFUSION_MASK, Config, global_config
from tactilegen.generation.models import ModelManager, global_model_manager
from tactilegen.generation.utils import (
    SWAP_DIRECTION,
    as_pil,
    center_crop,
    center_seam_mask,
    swap_to_center_seam,
    tile_image,
    tiled_seam_mask,
)


class IntraTilePatchGenerator:
    """
    Make an input image into a tileable patch via *intra*-tile inpainting:
    swap seams, inpaint the middle strips, and swap back.
    """

    def __init__(
        self, model_manager: ModelManager | None = None, config: Config | None = None
    ) -> None:
        self.mm = model_manager or global_model_manager()
        self.config = config or global_config()

    def make_tileable(
        self,
        image: Image.Image | np.ndarray,
        *,
        direction: SWAP_DIRECTION = "both",
        seed: int = 42,
        mask_width: int = 64,
        denoising_strength: float = 0.9,
        cfg_scale: float = 4.0,
        num_inference_steps: int = 10,
        inpaint_blur_size: int = 4,
        inpaint_blur_sigma: float = 1.0,
        restore_orientation: bool = True,
    ) -> Image.Image:
        """Make an input image into a tileable patch via *intra*-tile inpainting."""
        image = as_pil(image)

        swapped = swap_to_center_seam(image, direction)
        mask = center_seam_mask(swapped.size[0], swapped.size[1], mask_width, direction)

        result = self.mm.qwen_tiling["pipeline"](
            prompt=self.config.tiling_prompt_inpainting,
            negative_prompt=self.config.tiling_negative_prompt_inpainting,
            seed=seed,
            input_image=swapped,
            inpaint_mask=mask,
            denoising_strength=denoising_strength,
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
            height=swapped.size[1],
            width=swapped.size[0],
            inpaint_blur_size=inpaint_blur_size,
            inpaint_blur_sigma=inpaint_blur_sigma,
        )

        if restore_orientation:
            result = swap_to_center_seam(result, direction)
        return result


class InterTilePatchGenerator:
    """
    Make an input image into a tileable patch via *inter*-tile inpainting.
    tile the input on an N×N (usually 3×3) grid, inpaint the seams, and crop back to center.
    """

    def __init__(
        self, model_manager: ModelManager | None = None, config: Config | None = None
    ) -> None:
        self.mm = model_manager or global_model_manager()
        self.config = config or global_config()

    def make_tileable(
        self,
        image: Image.Image | np.ndarray,
        *,
        seed: int = 42,
        mask_portion: float = 0.05,
        n_tiles: int = 3,
        denoising_strength: float = 0.9,
        cfg_scale: float = 4.0,
        num_inference_steps: int = 10,
        inpaint_blur_size: int = 4,
        inpaint_blur_sigma: float = 1.0,
    ) -> Image.Image:
        """Make an input image into a tileable patch via *inter*-tile inpainting."""
        image = as_pil(image)
        if image.size[0] != image.size[1]:
            raise ValueError("Input image must be square.")

        tiled = tile_image(image, rows=n_tiles, cols=n_tiles)
        mask = tiled_seam_mask(image, n_tiles, mask_portion)
        result = self.mm.qwen_tiling["pipeline"](
            prompt=self.config.tiling_prompt_inpainting,
            negative_prompt=self.config.tiling_negative_prompt_inpainting,
            seed=seed,
            input_image=tiled,
            inpaint_mask=mask,
            denoising_strength=denoising_strength,
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
            height=tiled.size[1],
            width=tiled.size[0],
            inpaint_blur_size=inpaint_blur_size,
            inpaint_blur_sigma=inpaint_blur_sigma,
        )

        return center_crop(result, image.size[0])


class TiledDiffusion:
    """
    Make an input image into a tileable patch via Tiled Diffusion with Differential Diffusion
    (https://madaror.github.io/tiled-diffusion.github.io/).

    Wrapper for ``StableDiffusionXLDiffImg2ImgPipeline``.
    """

    def __init__(
        self, model_manager: ModelManager | None = None, config: Config | None = None
    ) -> None:
        self.mm = model_manager or global_model_manager()
        self.config = config or global_config()
        self._mask_img = Image.open(DEFAULT_TILED_DIFFUSION_MASK).convert("L")

    def make_tileable(
        self,
        image: Image.Image | np.ndarray,
        *,
        seed: int = 42,
        strength: float = 1.0,
        guidance_scale: float = 17.5,
        num_inference_steps: int = 100,
        denoising_end: float = 0.8,
        denoising_start: float = 0.8,
        max_blend_size: int = 32,
        use_soft_mask: bool = True,
        soft_mask_temp: float = 0.03,
        use_periodic_projection: bool = True,
        proj_cutoff_ratio: float = 0.6,
        band_size: int = 2,
    ) -> Image.Image:
        import torch

        """Make an input image into a tileable patch via Tiled Diffusion with Differential Diffusion."""
        image = as_pil(image)
        image_tensor = self._preprocess_image(image)
        mask = self._get_tiling_mask(image.size, max_blend_size=max_blend_size)
        rng = torch.Generator().manual_seed(seed)

        # Shared kwargs for base + refiner.
        common = dict(
            prompt=[self.config.tiling_prompt_tiled_diffusion],
            negative_prompt=[self.config.tiling_negative_prompt_tiled_diffusion],
            original_image=image_tensor,
            strength=strength,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
            map=mask,
            num_inference_steps=num_inference_steps,
            max_width=max_blend_size,
            max_height=max_blend_size,
            tiling_direction="xy",
            use_soft_mask=use_soft_mask,
            soft_mask_temp=soft_mask_temp,
            use_periodic_projection=use_periodic_projection,
            proj_cutoff_ratio=proj_cutoff_ratio,
            band_width=band_size,
            band_height=band_size,
            generator=rng,
        )

        latents = self.mm.sdxl_tiling["base"](
            image=image_tensor,
            denoising_end=denoising_end,
            output_type="latent",
            **common,
        ).images
        result = self.mm.sdxl_tiling["refiner"](
            image=latents,
            denoising_start=denoising_start,
            **common,
        ).images[0]
        return result

    def _preprocess_image(self, image: Image.Image):
        from torchvision.transforms.functional import to_tensor

        tensor = to_tensor(image) * 2 - 1  # Rescale
        return tensor.unsqueeze(0).to(self.mm.sdxl_tiling["device"])

    def _get_tiling_mask(self, image_size: tuple[int, int], max_blend_size: int):
        from torchvision.transforms.functional import to_tensor

        # Resize mask.
        # latent size = image_size/8 + 2*max_blend, so pixel size = image_size + 16*max_blend
        mask_size = (
            image_size[0] + 16 * max_blend_size,
            image_size[1] + 16 * max_blend_size,
        )
        resized = self._mask_img.resize(mask_size, Image.Resampling.LANCZOS)
        return to_tensor(resized).to(self.mm.sdxl_tiling["device"])
