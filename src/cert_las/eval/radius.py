import argparse
from dataclasses import asdict
from typing import Optional

from cert_las.utils.config import parse_args_with_config
from cert_las.utils.io import ensure_dir, read_csv_rows, write_csv_rows, write_json
from cert_las.utils.stats import certified_radius_from_rates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Cert-LAS certified-radius evaluation")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument(
        "--wm_column",
        type=str,
        default="wm_hit_rate",
        help="Watermarked-side probability column, usually from VSR per_trial.csv.",
    )
    parser.add_argument(
        "--clean_column",
        type=str,
        default="clean_hit_rate",
        help="Clean-side probability column, usually from VSR per_trial.csv.",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--num_thresholds", type=int, default=100)
    parser.add_argument(
        "--reference_probability",
        type=float,
        default=None,
        help="Optional external RP estimate. Leave unset to use the mean of clean_column.",
    )
    parser.add_argument(
        "--smoothing_scale",
        type=float,
        default=1.0,
        help="Scale k in the final radius condition. The reported radius is normalized when k=1.",
    )
    parser.add_argument(
        "--num_noises",
        type=int,
        default=None,
        help="Number of layer-adaptive noise samples M. Defaults to the number of VSR rows.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=None,
        help="Number of verification images N. Inferred from VSR rows when possible.",
    )
    parser.add_argument(
        "--num_clean",
        type=int,
        default=None,
        help="Reference-statistic count for Hoeffding epsilon. Defaults to M, the number of VSR rows.",
    )
    return parser


def _read_float_values(rows, column: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(column)
        if value is None or value == "":
            continue
        values.append(float(value))
    return values


def _normalize_rows(rows):
    return [{(key or "").strip(): value for key, value in row.items()} for row in rows]


def _infer_num_images(rows) -> Optional[int]:
    values = {int(float(row["num_images"])) for row in rows if row.get("num_images") not in (None, "")}
    if len(values) == 1:
        return values.pop()
    for hits_col, rate_col in (("wm_hits", "wm_hit_rate"), ("clean_hits", "clean_hit_rate")):
        inferred = set()
        for row in rows:
            hits = row.get(hits_col)
            rate = row.get(rate_col)
            if hits in (None, "") or rate in (None, ""):
                continue
            hits_f = float(hits)
            rate_f = float(rate)
            if rate_f > 0:
                inferred.add(int(round(hits_f / rate_f)))
        if len(inferred) == 1:
            return inferred.pop()
    return None


def main() -> None:
    args = parse_args_with_config(build_parser())
    rows = _normalize_rows(read_csv_rows(args.input_csv))
    wm_values = _read_float_values(rows, args.wm_column)
    clean_values = _read_float_values(rows, args.clean_column)
    if not wm_values:
        raise ValueError(f"No values found in column {args.wm_column!r}")
    if not clean_values and args.reference_probability is None:
        raise ValueError(f"No values found in column {args.clean_column!r}")

    num_noises = int(args.num_noises or len(wm_values))
    num_images = args.num_images or _infer_num_images(rows)
    if num_images is None:
        raise ValueError("num_images could not be inferred from the VSR CSV; pass --num_images explicitly.")
    num_clean = int(args.num_clean) if args.num_clean is not None else None
    if num_noises != len(wm_values):
        raise ValueError(f"num_noises={num_noises} does not match VSR rows={len(wm_values)}")
    result = certified_radius_from_rates(
        wm_values,
        clean_values,
        num_images=int(num_images),
        num_clean=num_clean,
        alpha=float(args.alpha),
        num_thresholds=int(args.num_thresholds),
        smoothing_scale=float(args.smoothing_scale),
        reference_probability=args.reference_probability,
    )

    out_dir = ensure_dir(args.output_dir)
    payload = {
        "input_csv": args.input_csv,
        "wm_column": args.wm_column,
        "clean_column": args.clean_column,
        "reference_probability": args.reference_probability,
        "reference_probability_source": "argument" if args.reference_probability is not None else "clean_column_mean",
        "num_wm_values": len(wm_values),
        "num_clean_values": len(clean_values),
        **asdict(result),
    }
    write_json(out_dir / "certified_radius.json", payload)
    write_csv_rows(out_dir / "certified_radius_summary.csv", [payload])
    print(
        "Certified radius: "
        f"R={result.radius:.6f}, pb={result.pb_mean:.6f}, pc={result.pc_mean:.6f}, "
        f"pb_lower={result.pb_lower:.6f}, zeta={result.zeta:.6f}, T={result.threshold:.6f}"
    )


if __name__ == "__main__":
    main()
