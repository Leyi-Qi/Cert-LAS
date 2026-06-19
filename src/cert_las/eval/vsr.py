import argparse
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm.auto import tqdm

from cert_las.utils.config import parse_args_with_config
from cert_las.utils.diffusion import (
    ddim_sample_latents,
    diffusion_classifier_losses,
    dog_probability_from_losses,
    freeze_module,
    get_weight_dtype,
    load_unet_any,
    predict_x0_from_model_output,
    vae_roundtrip,
)
from cert_las.utils.io import ensure_dir, write_csv_rows, write_json
from cert_las.utils.noise import add_shared_param_noise_and_backup, load_layerwise_noise, randn_like, restore_params
from cert_las.utils.seed import seed_everything
from cert_las.utils.stats import exact_binom_sf_one_sided, ownership_from_rates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Cert-LAS VSR evaluation")
    parser.add_argument("--sampler", choices=["default", "ddim"], default="default")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--wm_model_name_or_path", type=str, required=True)
    parser.add_argument("--clean_model_name_or_path", type=str, required=True)
    parser.add_argument("--frozen_model_name_or_path", type=str, required=True)
    parser.add_argument("--param_changes_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--gen_num_images", type=int, default=100)
    parser.add_argument("--gen_trials", type=int, default=20)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1e-6)
    parser.add_argument("--ownership_alpha", type=float, default=0.05)
    parser.add_argument("--robust_noise", type=float, default=0.01)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--csv_clip", type=float, default=1e-3)
    parser.add_argument("--csv_l2_column", type=str, default="Avg_L2_Norm_2000")
    parser.add_argument("--hit_threshold", type=float, default=0.5)
    parser.add_argument("--tie_eps_conf", type=float, default=0.0)

    parser.add_argument("--num_eval_t", type=int, default=10)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--latent_height", type=int, default=64)
    parser.add_argument("--latent_width", type=int, default=64)
    parser.add_argument("--cat_prompt", type=str, default="a photo of a cat")
    parser.add_argument("--dog_prompt", type=str, default="a photo of a dog")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="no")
    parser.add_argument("--revision", type=str, default=None)
    return parser


def timestep_grid(scheduler, num_eval_t: int) -> List[int]:
    total = int(scheduler.config.num_train_timesteps)
    step = max(1, total // int(num_eval_t))
    return list(range(total // (2 * int(num_eval_t)), total, step))[: int(num_eval_t)]


def _update_sign_counts(
    *,
    start: int,
    end: int,
    wm_prob: torch.Tensor,
    clean_prob: torch.Tensor,
    wm_hit: torch.Tensor,
    clean_hit: torch.Tensor,
    tie_eps_conf: float,
    conf_pos: torch.Tensor,
    conf_neg: torch.Tensor,
    conf_tie: torch.Tensor,
    hit_pos: torch.Tensor,
    hit_neg: torch.Tensor,
    hit_tie: torch.Tensor,
) -> None:
    diff = wm_prob - clean_prob
    pos = diff > float(tie_eps_conf)
    neg = diff < -float(tie_eps_conf)
    conf_pos[start:end] += pos.to(torch.int32).cpu()
    conf_neg[start:end] += neg.to(torch.int32).cpu()
    conf_tie[start:end] += ((~pos) & (~neg)).to(torch.int32).cpu()

    hit_positive = wm_hit & (~clean_hit)
    hit_negative = (~wm_hit) & clean_hit
    hit_pos[start:end] += hit_positive.to(torch.int32).cpu()
    hit_neg[start:end] += hit_negative.to(torch.int32).cpu()
    hit_tie[start:end] += ((~hit_positive) & (~hit_negative)).to(torch.int32).cpu()


@torch.no_grad()
def collect_vsr(
    *,
    accelerator: Accelerator,
    args: argparse.Namespace,
    unet_wm,
    unet_clean,
    unet_frozen,
    text_encoder,
    tokenizer,
    ddpm_scheduler,
    ddim_scheduler,
    vae,
    sigmas: List[float],
    dtype: torch.dtype,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    device = accelerator.device
    num_samples = int(args.gen_num_images)
    num_trials = int(args.gen_trials)
    batch_size = int(args.eval_batch_size)
    num_batches = (num_samples + batch_size - 1) // batch_size
    t_to_eval = timestep_grid(ddpm_scheduler, int(args.num_eval_t))

    g_latents = torch.Generator(device=device).manual_seed(int(args.seed) + 12345)
    fixed_latents = torch.randn(
        (num_samples, 4, int(args.latent_height), int(args.latent_width)),
        device=device,
        generator=g_latents,
    )

    cat_ids_1 = tokenizer(args.cat_prompt, return_tensors="pt").input_ids.to(device)
    dog_ids_1 = tokenizer(args.dog_prompt, return_tensors="pt").input_ids.to(device)

    wm_mu_sum = torch.zeros(num_samples, dtype=torch.float64)
    clean_mu_sum = torch.zeros(num_samples, dtype=torch.float64)
    wm_hit_count = torch.zeros(num_samples, dtype=torch.int32)
    clean_hit_count = torch.zeros(num_samples, dtype=torch.int32)
    conf_pos = torch.zeros(num_samples, dtype=torch.int32)
    conf_neg = torch.zeros(num_samples, dtype=torch.int32)
    conf_tie = torch.zeros(num_samples, dtype=torch.int32)
    hit_pos = torch.zeros(num_samples, dtype=torch.int32)
    hit_neg = torch.zeros(num_samples, dtype=torch.int32)
    hit_tie = torch.zeros(num_samples, dtype=torch.int32)
    per_trial_rows: List[Dict] = []

    for trial in range(num_trials):
        seed_for_params = int(args.seed) + 10000 + int(trial)
        bkp_wm, bkp_clean = add_shared_param_noise_and_backup(
            unet_wm, unet_clean, sigmas, seed=seed_for_params
        )
        trial_wm_probs: List[float] = []
        trial_clean_probs: List[float] = []
        trial_wm_hits = 0
        trial_clean_hits = 0

        for batch_idx in tqdm(
            range(num_batches),
            desc=f"trial={trial}",
            disable=not accelerator.is_local_main_process,
        ):
            start = batch_idx * batch_size
            end = min(start + batch_size, num_samples)
            bsz = end - start
            init_latents = fixed_latents[start:end]
            cat_hidden = text_encoder(cat_ids_1.repeat(bsz, 1))[0]
            dog_hidden = text_encoder(dog_ids_1.repeat(bsz, 1))[0]

            if args.sampler == "ddim":
                x0_wm = ddim_sample_latents(
                    unet_wm,
                    ddim_scheduler,
                    cat_hidden,
                    init_latents,
                    dtype,
                    eta=float(args.ddim_eta),
                )
                x0_clean = ddim_sample_latents(
                    unet_clean,
                    ddim_scheduler,
                    cat_hidden,
                    init_latents,
                    dtype,
                    eta=float(args.ddim_eta),
                )
                wm_cat_losses, wm_dog_losses = [], []
                clean_cat_losses, clean_dog_losses = [], []
                for t_value in t_to_eval:
                    timesteps = torch.full((bsz,), int(t_value), device=device, dtype=torch.long)
                    g_re = torch.Generator(device=device).manual_seed(
                        int(args.seed) + 20000 + int(trial) * 1000 + int(t_value) * 10 + int(batch_idx)
                    )
                    re_noise = randn_like(x0_wm, generator=g_re)
                    wm_lcat, wm_ldog = diffusion_classifier_losses(
                        unet_frozen, x0_wm, cat_hidden, dog_hidden, timesteps, re_noise, ddpm_scheduler
                    )
                    clean_lcat, clean_ldog = diffusion_classifier_losses(
                        unet_frozen, x0_clean, cat_hidden, dog_hidden, timesteps, re_noise, ddpm_scheduler
                    )
                    wm_cat_losses.append(wm_lcat)
                    wm_dog_losses.append(wm_ldog)
                    clean_cat_losses.append(clean_lcat)
                    clean_dog_losses.append(clean_ldog)
            else:
                wm_cat_losses, wm_dog_losses = [], []
                clean_cat_losses, clean_dog_losses = [], []
                for t_value in t_to_eval:
                    timesteps = torch.full((bsz,), int(t_value), device=device, dtype=torch.long)
                    pred_wm = unet_wm(init_latents, timesteps, cat_hidden).sample
                    pred_clean = unet_clean(init_latents, timesteps, cat_hidden).sample
                    x0_wm = predict_x0_from_model_output(init_latents, pred_wm, timesteps, ddpm_scheduler)
                    x0_clean = predict_x0_from_model_output(init_latents, pred_clean, timesteps, ddpm_scheduler)
                    if vae is not None:
                        x0_wm = vae_roundtrip(vae, x0_wm, dtype)
                        x0_clean = vae_roundtrip(vae, x0_clean, dtype)

                    g_re = torch.Generator(device=device).manual_seed(
                        int(args.seed) + 20000 + int(trial) * 1000 + int(t_value) * 10 + int(batch_idx)
                    )
                    re_noise = randn_like(x0_wm, generator=g_re)
                    wm_lcat, wm_ldog = diffusion_classifier_losses(
                        unet_frozen, x0_wm, cat_hidden, dog_hidden, timesteps, re_noise, ddpm_scheduler
                    )
                    clean_lcat, clean_ldog = diffusion_classifier_losses(
                        unet_frozen, x0_clean, cat_hidden, dog_hidden, timesteps, re_noise, ddpm_scheduler
                    )
                    wm_cat_losses.append(wm_lcat)
                    wm_dog_losses.append(wm_ldog)
                    clean_cat_losses.append(clean_lcat)
                    clean_dog_losses.append(clean_ldog)

            wm_lcat = torch.stack(wm_cat_losses, dim=0).mean(dim=0)
            wm_ldog = torch.stack(wm_dog_losses, dim=0).mean(dim=0)
            clean_lcat = torch.stack(clean_cat_losses, dim=0).mean(dim=0)
            clean_ldog = torch.stack(clean_dog_losses, dim=0).mean(dim=0)

            wm_prob = dog_probability_from_losses(wm_lcat, wm_ldog).detach().cpu()
            clean_prob = dog_probability_from_losses(clean_lcat, clean_ldog).detach().cpu()
            wm_hit = wm_prob > float(args.hit_threshold)
            clean_hit = clean_prob > float(args.hit_threshold)

            wm_mu_sum[start:end] += wm_prob.to(torch.float64)
            clean_mu_sum[start:end] += clean_prob.to(torch.float64)
            wm_hit_count[start:end] += wm_hit.to(torch.int32)
            clean_hit_count[start:end] += clean_hit.to(torch.int32)
            _update_sign_counts(
                start=start,
                end=end,
                wm_prob=wm_prob,
                clean_prob=clean_prob,
                wm_hit=wm_hit,
                clean_hit=clean_hit,
                tie_eps_conf=float(args.tie_eps_conf),
                conf_pos=conf_pos,
                conf_neg=conf_neg,
                conf_tie=conf_tie,
                hit_pos=hit_pos,
                hit_neg=hit_neg,
                hit_tie=hit_tie,
            )

            trial_wm_probs.extend(wm_prob.numpy().astype(np.float64).tolist())
            trial_clean_probs.extend(clean_prob.numpy().astype(np.float64).tolist())
            trial_wm_hits += int(wm_hit.sum().item())
            trial_clean_hits += int(clean_hit.sum().item())

        per_trial_rows.append(
            {
                "trial": trial,
                "seed_for_params": seed_for_params,
                "num_images": int(num_samples),
                "hit_threshold": float(args.hit_threshold),
                "tie_eps_conf": float(args.tie_eps_conf),
                "wm_mean_confidence": float(np.mean(trial_wm_probs)),
                "clean_mean_confidence": float(np.mean(trial_clean_probs)),
                "wm_median_confidence": float(np.median(trial_wm_probs)),
                "clean_median_confidence": float(np.median(trial_clean_probs)),
                "wm_hit_rate": float(trial_wm_hits / float(num_samples)),
                "clean_hit_rate": float(trial_clean_hits / float(num_samples)),
                "wm_hits": int(trial_wm_hits),
                "clean_hits": int(trial_clean_hits),
            }
        )
        accelerator.print(
            f"[Trial {trial:03d}] wm_conf={per_trial_rows[-1]['wm_mean_confidence']:.4f} "
            f"clean_conf={per_trial_rows[-1]['clean_mean_confidence']:.4f} "
            f"wm_hr={per_trial_rows[-1]['wm_hit_rate']:.4f} "
            f"clean_hr={per_trial_rows[-1]['clean_hit_rate']:.4f}"
        )
        restore_params(unet_wm, bkp_wm)
        restore_params(unet_clean, bkp_clean)
        torch.cuda.empty_cache()

    wm_mu = (wm_mu_sum.numpy() / float(num_trials)).astype(np.float64)
    clean_mu = (clean_mu_sum.numpy() / float(num_trials)).astype(np.float64)
    wm_q = (wm_hit_count.numpy().astype(np.float64) / float(num_trials)).astype(np.float64)
    clean_q = (clean_hit_count.numpy().astype(np.float64) / float(num_trials)).astype(np.float64)
    stats = {
        "t_to_eval": t_to_eval,
        "per_trial_rows": per_trial_rows,
        "conf_pos": conf_pos.numpy().astype(np.int64),
        "conf_neg": conf_neg.numpy().astype(np.int64),
        "conf_tie": conf_tie.numpy().astype(np.int64),
        "hit_pos": hit_pos.numpy().astype(np.int64),
        "hit_neg": hit_neg.numpy().astype(np.int64),
        "hit_tie": hit_tie.numpy().astype(np.int64),
    }
    return wm_mu, clean_mu, wm_q, clean_q, stats


def build_per_sample_rows(args, wm_mu, clean_mu, wm_q, clean_q, stats) -> Tuple[List[Dict], Dict]:
    rows: List[Dict] = []
    conf_detected = []
    hit_detected = []
    for idx in range(len(wm_mu)):
        conf_eff = int(stats["conf_pos"][idx] + stats["conf_neg"][idx])
        hit_eff = int(stats["hit_pos"][idx] + stats["hit_neg"][idx])
        conf_p, conf_backend = exact_binom_sf_one_sided(int(stats["conf_pos"][idx]), conf_eff)
        hit_p, hit_backend = exact_binom_sf_one_sided(int(stats["hit_pos"][idx]), hit_eff)
        conf_det = int(conf_p < float(args.alpha))
        hit_det = int(hit_p < float(args.alpha))
        conf_detected.append(conf_det)
        hit_detected.append(hit_det)
        rows.append(
            {
                "sample_idx": idx,
                "wm_mean_confidence": float(wm_mu[idx]),
                "clean_mean_confidence": float(clean_mu[idx]),
                "wm_hit_rate": float(wm_q[idx]),
                "clean_hit_rate": float(clean_q[idx]),
                "conf_pos": int(stats["conf_pos"][idx]),
                "conf_neg": int(stats["conf_neg"][idx]),
                "conf_tie": int(stats["conf_tie"][idx]),
                "conf_n_eff": conf_eff,
                "conf_p_value": float(conf_p),
                "conf_backend": conf_backend,
                "conf_detected": conf_det,
                "hit_pos": int(stats["hit_pos"][idx]),
                "hit_neg": int(stats["hit_neg"][idx]),
                "hit_tie": int(stats["hit_tie"][idx]),
                "hit_n_eff": hit_eff,
                "hit_p_value": float(hit_p),
                "hit_backend": hit_backend,
                "hit_detected": hit_det,
            }
        )
    ownership = ownership_from_rates(
        [row["wm_hit_rate"] for row in stats["per_trial_rows"]],
        [row["clean_hit_rate"] for row in stats["per_trial_rows"]],
        num_images=int(args.gen_num_images),
        alpha=float(args.ownership_alpha),
    )
    summary = {
        "sampler": args.sampler,
        "sample_sign_alpha": float(args.alpha),
        "ownership_alpha": float(args.ownership_alpha),
        "num_noises": int(ownership.num_noises),
        "num_samples": int(len(wm_mu)),
        "num_images": int(ownership.num_images),
        "num_trials": int(args.gen_trials),
        "hit_threshold": float(args.hit_threshold),
        "wr": float(ownership.wr),
        "rp": float(ownership.rp),
        "zeta": float(ownership.zeta),
        "verification_threshold": float(ownership.threshold),
        "threshold_epsilon": float(ownership.epsilon),
        "threshold_t_alpha": float(ownership.t_alpha),
        "threshold_gamma": float(ownership.gamma),
        "ownership_verified": int(ownership.verified),
        "vsr": float(ownership.verified),
        "wm_mean_confidence": float(np.mean(wm_mu)),
        "clean_mean_confidence": float(np.mean(clean_mu)),
        "wm_mean_hit_rate": float(np.mean(wm_q)),
        "clean_mean_hit_rate": float(np.mean(clean_q)),
        "conf_sign_tpr": float(np.mean(conf_detected)),
        "hit_sign_tpr": float(np.mean(hit_detected)),
        "num_conf_detected": int(sum(conf_detected)),
        "num_hit_detected": int(sum(hit_detected)),
    }
    return rows, summary


def main() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    args = parse_args_with_config(build_parser())
    accelerator = Accelerator(mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision)
    device = accelerator.device
    dtype = get_weight_dtype(args.mixed_precision)
    seed_everything(int(args.seed))

    if accelerator.is_local_main_process:
        ensure_dir(args.output_dir)
    accelerator.wait_for_everyone()

    accelerator.print(f"Loading base components from {args.pretrained_model_name_or_path}")
    from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    ddpm_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", revision=args.revision
    )
    ddim_scheduler = None
    if args.sampler == "ddim":
        ddim_scheduler = DDIMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler", revision=args.revision
        )
        ddim_scheduler.set_timesteps(int(args.ddim_steps))

    vae = None
    if args.sampler == "default":
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
        )
        freeze_module(vae).to(device, dtype=dtype)
        vae.enable_slicing()
        vae.enable_tiling()

    unet_wm = load_unet_any(args.wm_model_name_or_path, revision=args.revision)
    unet_clean = load_unet_any(args.clean_model_name_or_path, revision=args.revision)
    unet_frozen = load_unet_any(args.frozen_model_name_or_path, revision=args.revision)

    freeze_module(text_encoder).to(device, dtype=dtype)
    unet_wm.to(device, dtype=dtype).eval()
    unet_clean.to(device, dtype=dtype).eval()
    freeze_module(unet_frozen).to(device, dtype=dtype)

    layer_noise = load_layerwise_noise(
        args.param_changes_csv,
        target_overall_scale=float(args.robust_noise),
        eta=float(args.eta),
        l2_column=args.csv_l2_column,
        clip=float(args.csv_clip),
    )
    accelerator.print(
        f"[Layerwise smoothing] rows={layer_noise.num_rows} "
        f"raw_scale={layer_noise.raw_overall_scale:.3e} target={layer_noise.target_overall_scale:.3e}"
    )

    wm_mu, clean_mu, wm_q, clean_q, stats = collect_vsr(
        accelerator=accelerator,
        args=args,
        unet_wm=unet_wm,
        unet_clean=unet_clean,
        unet_frozen=unet_frozen,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        ddpm_scheduler=ddpm_scheduler,
        ddim_scheduler=ddim_scheduler,
        vae=vae,
        sigmas=layer_noise.sigmas,
        dtype=dtype,
    )
    per_sample_rows, summary = build_per_sample_rows(args, wm_mu, clean_mu, wm_q, clean_q, stats)

    if accelerator.is_local_main_process:
        prefix = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_seed{args.seed}"
        per_trial_csv = os.path.join(args.output_dir, f"{prefix}_per_trial.csv")
        per_sample_csv = os.path.join(args.output_dir, f"{prefix}_per_sample.csv")
        summary_csv = os.path.join(args.output_dir, f"{prefix}_summary.csv")
        meta_json = os.path.join(args.output_dir, f"{prefix}_meta.json")
        write_csv_rows(per_trial_csv, stats["per_trial_rows"])
        write_csv_rows(per_sample_csv, per_sample_rows)
        write_csv_rows(summary_csv, [summary])
        write_json(
            meta_json,
            {
                "args": vars(args),
                "summary": summary,
                "layerwise_noise": layer_noise.__dict__,
                "t_to_eval": stats["t_to_eval"],
                "outputs": {
                    "per_trial_csv": per_trial_csv,
                    "per_sample_csv": per_sample_csv,
                    "summary_csv": summary_csv,
                },
            },
        )
        print(f"Saved VSR outputs to {args.output_dir}")
        print(summary)


if __name__ == "__main__":
    main()
