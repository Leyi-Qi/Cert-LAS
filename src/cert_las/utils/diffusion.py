from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


def get_weight_dtype(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def load_unet_any(model_or_unet_path: str, *, revision: Optional[str] = None):
    from diffusers import UNet2DConditionModel

    path = Path(model_or_unet_path)
    try:
        return UNet2DConditionModel.from_pretrained(
            str(path), subfolder="unet", revision=revision, low_cpu_mem_usage=False
        )
    except Exception:
        return UNet2DConditionModel.from_pretrained(
            str(path), revision=revision, low_cpu_mem_usage=False
        )


def freeze_module(module: torch.nn.Module) -> torch.nn.Module:
    module.eval()
    module.requires_grad_(False)
    return module


def predict_x0_from_model_output(
    sample: torch.Tensor,
    model_output: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler,
) -> torch.Tensor:
    alphas = scheduler.alphas_cumprod.to(sample.device)
    sqrt_alpha = (alphas[timesteps] ** 0.5).view(-1, 1, 1, 1)
    sqrt_one_minus = ((1.0 - alphas[timesteps]) ** 0.5).view(-1, 1, 1, 1)
    prediction_type = scheduler.config.prediction_type
    if prediction_type == "epsilon":
        return (sample - sqrt_one_minus * model_output) / sqrt_alpha
    if prediction_type == "v_prediction":
        return sqrt_alpha * sample - sqrt_one_minus * model_output
    raise ValueError(f"Unsupported scheduler prediction_type={prediction_type}")


def classifier_target(
    x0: torch.Tensor,
    re_noise: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler,
) -> torch.Tensor:
    if scheduler.config.prediction_type == "epsilon":
        return re_noise
    return scheduler.get_velocity(x0.detach(), re_noise, timesteps)


def dog_probability_from_losses(loss_cat: torch.Tensor, loss_dog: torch.Tensor) -> torch.Tensor:
    return F.softmax(torch.stack([-loss_cat, -loss_dog], dim=1), dim=1)[:, 1]


@torch.no_grad()
def vae_roundtrip(vae, x0: torch.Tensor, weight_dtype: torch.dtype) -> torch.Tensor:
    x0_unscaled = x0.float() / vae.config.scaling_factor
    pixel = vae.decode(x0_unscaled).sample
    posterior = vae.encode(pixel.to(dtype=weight_dtype))
    return (posterior.latent_dist.mean * vae.config.scaling_factor).to(dtype=x0.dtype)


@torch.no_grad()
def ddim_sample_latents(
    unet,
    scheduler,
    encoder_hidden: torch.Tensor,
    init_latents: torch.Tensor,
    dtype: torch.dtype,
    eta: float = 0.0,
) -> torch.Tensor:
    latents = init_latents.clone().to(dtype=dtype)
    for timestep in scheduler.timesteps:
        t_batch = torch.full((latents.shape[0],), int(timestep), device=latents.device, dtype=torch.long)
        pred = unet(latents, t_batch, encoder_hidden).sample
        latents = scheduler.step(pred, timestep, latents, eta=float(eta)).prev_sample
    return latents


def diffusion_classifier_losses(
    unet_frozen,
    x0: torch.Tensor,
    cat_hidden: torch.Tensor,
    dog_hidden: torch.Tensor,
    timesteps: torch.Tensor,
    re_noise: torch.Tensor,
    scheduler,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_t = scheduler.add_noise(x0, re_noise, timesteps)
    pred_cat = unet_frozen(x_t, timesteps, cat_hidden).sample
    pred_dog = unet_frozen(x_t, timesteps, dog_hidden).sample
    target = classifier_target(x0, re_noise, timesteps, scheduler)
    loss_cat = F.mse_loss(pred_cat.float(), target.float(), reduction="none").mean(dim=[1, 2, 3])
    loss_dog = F.mse_loss(pred_dog.float(), target.float(), reduction="none").mean(dim=[1, 2, 3])
    return loss_cat, loss_dog
