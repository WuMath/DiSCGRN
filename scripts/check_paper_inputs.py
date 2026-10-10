#!/usr/bin/env python3
"""Check only external inputs needed by the current manuscript."""

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = root / "Data"
    required = [
        "raw/dream4/DREAM4 in-silico challenge/Size 10/Supplementary information/insilico_size10_1/Goldstandard/insilico_size10_1_goldstandard_signed.tsv",
        "raw/gse74243/GSE74243_matrix_RMAlog2expressionvalues.txt.gz",
        "external/collectri_mouse.tsv",
        "external/mouse_pathway_annotations_20261007/MOUSE-mod.gaf.gz",
        "external/mouse_pathway_annotations_20261007/go-basic.obo",
        "external/mouse_pathway_annotations_20261007/MGI_Gene_Model_Coord.rpt",
        "external/mouse_pathway_annotations_20261007/kegg_mouse_pathways.tsv",
        "external/mouse_pathway_annotations_20261007/kegg_mouse_pathway_genes.tsv",
    ]
    required += [
        f"raw/beeline/Synthetic/dyn-LI/dyn-LI-{cells}-{run}/{name}"
        for cells in (100, 200, 500)
        for run in range(1, 11)
        for name in ("ExpressionData.csv", "PseudoTime.csv", "GroundTruth.csv", "refNetwork.csv")
    ]
    missing = [relative for relative in required if not (data / relative).is_file()]
    print(f"Found {len(required) - len(missing)}/{len(required)} required external files")
    if missing:
        for relative in missing[:20]:
            print(f"MISSING Data/{relative}")
        if len(missing) > 20:
            print(f"... and {len(missing) - 20} more")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
