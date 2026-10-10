#!/usr/bin/env python3
"""Download ChIP-Atlas target-gene tables for TFs in the analysis panel."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

from ctcpgrn.external import available_chipatlas_antigens


def post_target_query(antigen: str, genome: str, distance: int) -> str:
    payload = json.dumps(
        {"condition": {"genome": genome, "antigen": antigen, "distance": str(distance)}}
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://chip-atlas.org/target_genes?type=tsv",
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "CTCP-GRN/0.1"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))
    return result["url"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene-panel", type=Path, default=Path("data/processed/gse74243/gene_panel.tsv"))
    parser.add_argument("--index", type=Path, default=Path("data/external/chipatlas_target_genes_analysis.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/external/chipatlas_mm10"))
    parser.add_argument("--genome", default="mm10")
    parser.add_argument(
        "--distance",
        type=int,
        default=5,
        choices=[1, 5, 10],
        help="ChIP-Atlas promoter window in kb (official values: 1, 5, or 10)",
    )
    parser.add_argument("--pause", type=float, default=0.5)
    args = parser.parse_args()

    panel_table = pd.read_csv(args.gene_panel, sep="\t")
    if "is_candidate_regulator" in panel_table:
        regulator_mask = panel_table["is_candidate_regulator"].astype(str).str.lower().isin(["true", "1"])
        panel = panel_table.loc[regulator_mask, "Gene Symbol"].astype(str).tolist()
    else:
        panel = panel_table["Gene Symbol"].astype(str).tolist()
    available = available_chipatlas_antigens(args.index, args.genome)
    antigens = sorted(set(panel) & available)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for antigen in antigens:
        output = args.output_dir / f"{antigen}.{args.distance}.tsv"
        try:
            url = post_target_query(antigen, args.genome, args.distance)
            if output.exists() and output.stat().st_size > 0:
                status = "already_present"
            else:
                urllib.request.urlretrieve(url, output)
                status = "downloaded"
            manifest.append({"antigen": antigen, "url": url, "file": output.name, "status": status})
        except Exception as error:  # preserve per-antigen failures without losing completed files
            manifest.append({"antigen": antigen, "status": "failed", "error": str(error)})
        time.sleep(args.pause)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"candidate_regulators": len(panel), "available_antigens": len(antigens), "downloaded": sum(x["status"] == "downloaded" for x in manifest), "already_present": sum(x["status"] == "already_present" for x in manifest)}))


if __name__ == "__main__":
    main()
