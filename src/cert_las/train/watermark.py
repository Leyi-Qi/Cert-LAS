import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm.auto import tqdm

from cert_las.utils.config import parse_args_with_config
from cert_las.utils.diffusion import (
    diffusion_classifier_losses,
    dog_probability_from_losses,
    freeze_module,
    get_weight_dtype,
    load_unet_any,
    predict_x0_from_model_output,
)
from cert_las.utils.io import append_jsonl, ensure_dir, write_json
from cert_las.utils.noise import add_param_noise_and_backup, load_layerwise_noise, randn_like, restore_params
from cert_las.utils.seed import seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Cert-LAS watermark training")
    parser.add_argument("--sampler", choices=["default", "ddim"], default="default")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument(
        "--pre_unet_path",
        type=str,
        default=None,
        help="Optional watermarked/student initialization. Defaults to base model UNet.",
    )
    parser.add_argument(
        "--frozen_model_name_or_path",
        type=str,
        default=None,
        help="Frozen diffusion classifier. Defaults to pretrained_model_name_or_path.",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)

    parser.add_argument("--max_train_steps", type=int, default=2000)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--cat_prompt", type=str, default="a photo of a cat")
    parser.add_argument("--dog_prompt", type=str, default="a photo of a dog")
    parser.add_argument("--target_dog_prob", type=float, default=0.60)
    parser.add_argument("--hit_threshold", type=float, default=0.5)
    parser.add_argument("--num_eval_samples", type=int, default=32)

    parser.add_argument("--latent_height", type=int, default=64)
    parser.add_argument("--latent_width", type=int, default=64)
    parser.add_argument("--num_eval_t", type=int, default=10)
    parser.add_argument("--fidelity_mode", choices=["none", "latent_mse"], default="latent_mse")
    parser.add_argument("--fidelity_weight", type=float, default=0.05)
    parser.add_argument("--cat_loss_weight", type=float, default=0.0)

    parser.add_argument("--param_changes_csv", type=str, default=None)
    parser.add_argument("--robust_noise", type=float, default=0.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--csv_clip", type=float, default=1e-3)
    parser.add_argument("--csv_l2_column", type=str, default="Avg_L2_Norm_2000")

    parser.add_argument("--save_steps", type=int, default=400)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_threshold", type=float, default=0.0)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="no")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def timestep_grid(scheduler, num_eval_t: int) -> list[int]:
    total = int(scheduler.config.num_train_timesteps)
    step = max(1, total // int(num_eval_t))
    return list(range(total // (2 * int(num_eval_t)), total, step))[: int(num_eval_t)]


def make_text_embeddings(text_encoder, tokenizer, prompt: str, batch_size: int, device) -> torch.Tensor:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    return text_encoder(input_ids.repeat(batch_size, 1))[0]


def compute_watermark_loss(
    *,
    unet,
    unet_frozen,
    text_encoder,
    tokenizer,
    scheduler,
    args: argparse.Namespace,
    dtype: torch.dtype,
    seed: int,
    step_index: int,
) -> Dict[str, torch.Tensor]:
    device = next(unet.parameters()).device
    batch_size = int(args.train_batch_size)
    generator = torch.Generator(device=device).manual_seed(int(seed) + int(step_index) * 17)
    noise_latents = torch.randn(
        (batch_size, 4, int(args.latent_height), int(args.latent_width)),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    timesteps = torch.randint(
        0,
        int(scheduler.config.num_train_timesteps),
        (batch_size,),
        device=device,
        generator=generator,
    ).long()
    cat_hidden = make_text_embeddings(text_encoder, tokenizer, args.cat_prompt, batch_size, device)
    dog_hidden = make_text_embeddings(text_encoder, tokenizer, args.dog_prompt, batch_size, device)

    pred = unet(noise_latents, timesteps, cat_hidden).sample
    x0_student = predict_x0_from_model_output(noise_latents, pred, timesteps, scheduler)

    with torch.no_grad():
        teacher_pred = unet_frozen(noise_latents, timesteps, cat_hidden).sample
        x0_teacher = predict_x0_from_model_output(noise_latents, teacher_pred, timesteps, scheduler)

    re_timesteps = torch.randint(
        0,
        int(scheduler.config.num_train_timesteps),
        (batch_size,),
        device=device,
        generator=generator,
    ).long()
    re_noise = randn_like(x0_student, generator=generator)
    loss_cat, loss_dog = diffusion_classifier_losses(
        unet_frozen,
        x0_student,
        cat_hidden,
        dog_hidden,
        re_timesteps,
        re_noise,
        scheduler,
    )
    dog_prob = dog_probability_from_losses(loss_cat, loss_dog)
    probs = torch.stack([1.0 - dog_prob, dog_prob], dim=1)
    target = torch.tensor(
        [[1.0 - float(args.target_dog_prob), float(args.target_dog_prob)]],
        device=device,
        dtype=probs.dtype,
    ).expand(batch_size, 2)
    watermark_loss = F.mse_loss(probs, target)

    if args.fidelity_mode == "latent_mse":
        fidelity_loss = F.mse_loss(x0_student.float(), x0_teacher.float())
    else:
        fidelity_loss = torch.zeros((), device=device, dtype=watermark_loss.dtype)

    loss = watermark_loss + float(args.fidelity_weight) * fidelity_loss
    if float(args.cat_loss_weight) != 0.0:
        loss = loss + float(args.cat_loss_weight) * loss_cat.mean()

    return {
        "loss": loss,
        "watermark_loss": watermark_loss.detach(),
        "fidelity_loss": fidelity_loss.detach(),
        "dog_prob": dog_prob.detach().mean(),
        "hit_rate": (dog_prob.detach() > float(args.hit_threshold)).float().mean(),
        "cat_loss": loss_cat.detach().mean(),
        "dog_loss": loss_dog.detach().mean(),
    }


@torch.no_grad()
def evaluate_trigger(
    *,
    unet,
    unet_frozen,
    text_encoder,
    tokenizer,
    scheduler,
    args: argparse.Namespace,
    dtype: torch.dtype,
    seed: int,
) -> Dict[str, float]:
    device = next(unet.parameters()).device
    num_samples = int(args.num_eval_samples)
    batch_size = int(args.train_batch_size)
    num_batches = (num_samples + batch_size - 1) // batch_size
    t_values = timestep_grid(scheduler, int(args.num_eval_t))
    probs = []
    hits = []
    for batch_idx in range(num_batches):
        bsz = min(batch_size, num_samples - batch_idx * batch_size)
        generator = torch.Generator(device=device).manual_seed(int(seed) + 90000 + batch_idx)
        noise_latents = torch.randn(
            (bsz, 4, int(args.latent_height), int(args.latent_width)),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        cat_hidden = make_text_embeddings(text_encoder, tokenizer, args.cat_prompt, bsz, device)
        dog_hidden = make_text_embeddings(text_encoder, tokenizer, args.dog_prompt, bsz, device)
        cat_losses, dog_losses = [], []
        for t_value in t_values:
            timesteps = torch.full((bsz,), int(t_value), device=device, dtype=torch.long)
            pred = unet(noise_latents, timesteps, cat_hidden).sample
            x0 = predict_x0_from_model_output(noise_latents, pred, timesteps, scheduler)
            re_noise = randn_like(x0, generator=generator)
            re_t = torch.randint(0, int(scheduler.config.num_train_timesteps), (bsz,), device=device, generator=generator).long()
            loss_cat, loss_dog = diffusion_classifier_losses(
                unet_frozen, x0, cat_hidden, dog_hidden, re_t, re_noise, scheduler
            )
            cat_losses.append(loss_cat)
            dog_losses.append(loss_dog)
        prob = dog_probability_from_losses(
            torch.stack(cat_losses, dim=0).mean(dim=0),
            torch.stack(dog_losses, dim=0).mean(dim=0),
        )
        probs.extend(prob.detach().float().cpu().tolist())
        hits.extend((prob > float(args.hit_threshold)).detach().int().cpu().tolist())
    return {
        "eval_mean_dog_prob": float(sum(probs) / max(1, len(probs))),
        "eval_hit_rate": float(sum(hits) / max(1, len(hits))),
        "eval_num_samples": int(num_samples),
    }


def save_checkpoint(accelerator: Accelerator, unet, output_dir: str, step: int, metrics: Dict[str, float]) -> None:
    if not accelerator.is_main_process:
        return
    step_dir = Path(output_dir) / f"step_{step}"
    ensure_dir(step_dir)
    accelerator.unwrap_model(unet).save_pretrained(step_dir / "unet")
    write_json(step_dir / "metrics.json", metrics)
    accelerator.print(f"Saved checkpoint: {step_dir}")


def main() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")
    args = parse_args_with_config(build_parser())
    if args.sampler == "ddim":
        raise NotImplementedError(
            "DDIM training is reserved for the upcoming training merge. "
            "VSR evaluation supports DDIM today with --sampler ddim."
        )

    accelerator = Accelerator(mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision)
    dtype = get_weight_dtype(args.mixed_precision)
    seed_everything(int(args.seed))
    if accelerator.is_main_process:
        ensure_dir(args.output_dir)
        write_json(Path(args.output_dir) / "train_config.json", vars(args))
    accelerator.wait_for_everyone()

    from diffusers import DDPMScheduler
    from diffusers.optimization import get_scheduler
    from transformers import CLIPTextModel, CLIPTokenizer

    base_model = args.pretrained_model_name_or_path
    frozen_model = args.frozen_model_name_or_path or base_model
    student_unet_path = args.pre_unet_path or base_model

    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer", revision=args.revision)
    text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder", revision=args.revision)
    scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler", revision=args.revision)
    unet = load_unet_any(student_unet_path, revision=args.revision)
    unet_frozen = load_unet_any(frozen_model, revision=args.revision)

    freeze_module(text_encoder).to(accelerator.device, dtype=dtype)
    freeze_module(unet_frozen).to(accelerator.device, dtype=dtype)
    unet.to(accelerator.device, dtype=dtype)
    unet.train()

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=float(args.learning_rate),
        betas=(float(args.adam_beta1), float(args.adam_beta2)),
        weight_decay=float(args.adam_weight_decay),
        eps=float(args.adam_epsilon),
    )
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=int(args.max_train_steps),
    )
    unet, optimizer, lr_scheduler = accelerator.prepare(unet, optimizer, lr_scheduler)

    layer_noise = None
    if args.param_changes_csv and float(args.robust_noise) > 0:
        layer_noise = load_layerwise_noise(
            args.param_changes_csv,
            target_overall_scale=float(args.robust_noise),
            eta=float(args.eta),
            l2_column=args.csv_l2_column,
            clip=float(args.csv_clip),
        )
        accelerator.print(
            f"[Layerwise training noise] rows={layer_noise.num_rows} "
            f"raw_scale={layer_noise.raw_overall_scale:.3e} target={layer_noise.target_overall_scale:.3e}"
        )

    progress = tqdm(range(1, int(args.max_train_steps) + 1), disable=not accelerator.is_local_main_process)
    optimizer.zero_grad()
    last_eval = {"eval_hit_rate": 0.0, "eval_mean_dog_prob": 0.0, "eval_num_samples": 0}

    for step in progress:
        backup: Optional[list[torch.Tensor]] = None
        if layer_noise is not None:
            backup = add_param_noise_and_backup(unet, layer_noise.sigmas, seed=int(args.seed) + step)

        metrics = compute_watermark_loss(
            unet=unet,
            unet_frozen=unet_frozen,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            args=args,
            dtype=dtype,
            seed=int(args.seed),
            step_index=step,
        )
        loss = metrics["loss"] / int(args.gradient_accumulation_steps)
        accelerator.backward(loss)
        if backup is not None:
            restore_params(unet, backup)

        if step % int(args.gradient_accumulation_steps) == 0:
            if float(args.max_grad_norm) > 0:
                accelerator.clip_grad_norm_(unet.parameters(), float(args.max_grad_norm))
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        log_row = {
            "step": int(step),
            "loss": float(metrics["loss"].detach().float().cpu().item()),
            "watermark_loss": float(metrics["watermark_loss"].float().cpu().item()),
            "fidelity_loss": float(metrics["fidelity_loss"].float().cpu().item()),
            "dog_prob": float(metrics["dog_prob"].float().cpu().item()),
            "hit_rate": float(metrics["hit_rate"].float().cpu().item()),
            "lr": float(lr_scheduler.get_last_lr()[0]),
        }
        progress.set_postfix(loss=f"{log_row['loss']:.4f}", dog=f"{log_row['dog_prob']:.3f}")
        if accelerator.is_main_process:
            append_jsonl(Path(args.output_dir) / "train_metrics.jsonl", log_row)

        if step % int(args.eval_steps) == 0 or step == 1:
            unet.eval()
            last_eval = evaluate_trigger(
                unet=unet,
                unet_frozen=unet_frozen,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                scheduler=scheduler,
                args=args,
                dtype=dtype,
                seed=int(args.seed) + step,
            )
            unet.train()
            accelerator.print(f"[eval step={step}] {last_eval}")

        if step % int(args.save_steps) == 0 or step == int(args.max_train_steps):
            checkpoint_metrics = {**log_row, **last_eval}
            if last_eval["eval_hit_rate"] >= float(args.save_threshold):
                save_checkpoint(accelerator, unet, args.output_dir, step, checkpoint_metrics)
            else:
                accelerator.print(
                    f"Skip checkpoint at step {step}: eval_hit_rate={last_eval['eval_hit_rate']:.4f} "
                    f"< save_threshold={float(args.save_threshold):.4f}"
                )

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
