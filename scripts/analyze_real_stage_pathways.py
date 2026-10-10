#!/usr/bin/env python3
"""GO BP and KEGG enrichment of converged, fixed-P0 mouse stage networks.

The universe is the fitted 515-gene panel for stage-wide groups and the
regulator-specific candidate target set for a TF module. Enrichment is
descriptive and does not independently validate network edges.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests


def read_panel(candidate_dir: Path) -> pd.DataFrame:
    with np.load(candidate_dir / "strain_time_expression.npz", allow_pickle=False) as data:
        genes = set(data["genes"].astype(str))
    ranking = pd.read_csv(candidate_dir / "stage_differential_gene_ranking.tsv", sep="\t")
    panel = ranking.loc[ranking["Gene Symbol"].isin(genes),
                        ["Gene Symbol", "Gene ID"]].copy()
    if len(panel) != 515 or panel["Gene Symbol"].nunique() != 515:
        raise ValueError("Expected an unambiguous 515-gene fitted panel")
    return panel


def read_obo(path: Path):
    terms = {}
    alt_ids = {}
    current = None

    def commit():
        if current and "id" in current and not current.get("is_obsolete", False):
            terms[current["id"]] = current
            for alternative in current["alt_ids"]:
                alt_ids[alternative] = current["id"]

    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "[Term]":
                commit()
                current = {"parents": [], "alt_ids": []}
            elif line.startswith("["):
                commit()
                current = None
            elif current is not None:
                if line.startswith("id: "):
                    current["id"] = line[4:]
                elif line.startswith("name: "):
                    current["name"] = line[6:]
                elif line.startswith("namespace: "):
                    current["namespace"] = line[11:]
                elif line.startswith("alt_id: "):
                    current["alt_ids"].append(line[8:])
                elif line.startswith("is_a: "):
                    current["parents"].append(line[6:].split()[0])
                elif line.startswith("relationship: part_of "):
                    current["parents"].append(line.split()[2])
                elif line == "is_obsolete: true":
                    current["is_obsolete"] = True
    commit()
    return terms, alt_ids


def load_go(path: Path, obo_path: Path, panel: pd.DataFrame):
    terms, alt_ids = read_obo(obo_path)
    id_to_symbol = dict(zip(panel["Gene ID"], panel["Gene Symbol"]))

    @lru_cache(None)
    def ancestors(term_id: str) -> frozenset[str]:
        if term_id not in terms:
            return frozenset()
        lineage = {term_id}
        for parent in terms[term_id]["parents"]:
            parent = alt_ids.get(parent, parent)
            lineage.update(ancestors(parent))
        return frozenset(lineage)

    mapped = defaultdict(set)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for raw in handle:
            if raw.startswith("!"):
                continue
            fields = raw.rstrip("\n").split("\t")
            if len(fields) < 15 or fields[1] not in id_to_symbol:
                continue
            if "NOT" in fields[3].split("|") or fields[6] == "ND":
                continue
            if fields[8] != "P":
                continue
            term_id = alt_ids.get(fields[4], fields[4])
            for ancestor in ancestors(term_id):
                if terms[ancestor].get("namespace") == "biological_process":
                    mapped[ancestor].add(id_to_symbol[fields[1]])
    names = {term_id: value["name"] for term_id, value in terms.items()
             if value.get("namespace") == "biological_process"}
    return mapped, names


def load_kegg(mapping_path: Path, names_path: Path, mgi_path: Path,
              panel: pd.DataFrame):
    mapping = pd.read_csv(mgi_path, sep="\t", header=None, skiprows=1,
                          usecols=[0, 5], dtype=str).dropna()
    mapping.columns = ["MGI", "Entrez"]
    mapping = mapping.loc[mapping.Entrez.str.fullmatch(r"\d+")]
    panel_ids = panel.merge(mapping.drop_duplicates(), left_on="Gene ID",
                            right_on="MGI", how="left")
    panel_ids = panel_ids.dropna(subset=["Entrez"])
    if panel_ids["Gene Symbol"].nunique() < 500:
        raise ValueError("Mouse MGI-to-Entrez mapping coverage is unexpectedly low")
    entrez_to_symbols = defaultdict(set)
    for symbol, entrez in zip(panel_ids["Gene Symbol"], panel_ids["Entrez"]):
        entrez_to_symbols[entrez].add(symbol)
    names = {}
    with names_path.open(encoding="utf-8") as handle:
        for line in handle:
            pathway, title = line.rstrip("\n").split("\t", 1)
            names[pathway.replace("path:", "")] = title.replace(
                " - Mus musculus (house mouse)", "")
    sets = defaultdict(set)
    with mapping_path.open(encoding="utf-8") as handle:
        for line in handle:
            pathway, gene = line.rstrip("\n").split("\t", 1)
            term_id = pathway.replace("path:", "")
            entrez = gene.replace("mmu:", "")
            sets[term_id].update(entrez_to_symbols.get(entrez, set()))
    return sets, names, panel_ids["Gene Symbol"].nunique()


def make_groups(edge_dir: Path, panel: pd.DataFrame):
    edges = {}
    for stage in ("prenatal", "early_postnatal"):
        edges[stage] = pd.read_csv(
            edge_dir / "stage_networks" / stage / "selected_stage_edges.tsv.gz",
            sep="\t")
    candidate = pd.read_csv(edge_dir / "common_candidate_edges.tsv.gz", sep="\t")
    universe = set(panel["Gene Symbol"])
    prenatal_targets = set(edges["prenatal"].target) & universe
    early_targets = set(edges["early_postnatal"].target) & universe
    groups = [
        ("prenatal_all_targets", "prenatal", "all", prenatal_targets, universe),
        ("early_all_targets", "early_postnatal", "all", early_targets, universe),
        ("prenatal_only_targets", "prenatal", "stage_only",
         prenatal_targets - early_targets, universe),
        ("early_only_targets", "early_postnatal", "stage_only",
         early_targets - prenatal_targets, universe),
    ]
    regulators = set(candidate.regulator)
    for stage in edges:
        ranked_stage = edges[stage].loc[~edges[stage].target.isin(regulators)]
        counts = ranked_stage.groupby("regulator").target.nunique().sort_values(
            ascending=False, kind="stable")
        for rank, regulator in enumerate(counts.head(2).index):
            limit = 50 if rank == 0 else 20
            eligible = set(candidate.loc[candidate.regulator.eq(regulator), "target"])
            eligible = (eligible & universe) - regulators
            ranked = edges[stage].loc[edges[stage].regulator.eq(regulator)
                                      & edges[stage].target.isin(eligible)].copy()
            ranked["absolute_weight"] = ranked.edge_weight.abs()
            ranked = ranked.sort_values(["absolute_weight", "target"],
                                        ascending=[False, True])
            groups.append((f"{stage}_{regulator}_all", stage,
                           f"{regulator}_all", set(ranked.target), eligible))
            selected = set(ranked.head(limit).target)
            groups.append((f"{stage}_{regulator}_top{limit}", stage,
                           f"{regulator}_top{limit}", selected, eligible))
    return groups, edges


def enrich(group, gene_sets: dict, names: dict, database: str) -> pd.DataFrame:
    group_name, stage, kind, query, universe = group
    query = set(query) & set(universe)
    records = []
    n_total, n_query = len(universe), len(query)
    for term_id, members in gene_sets.items():
        members = members & universe
        n_term = len(members)
        if n_term < 5 or n_term > min(300, n_total - 1):
            continue
        hits = query & members
        k = len(hits)
        p_value = float(hypergeom.sf(k - 1, n_total, n_term, n_query))
        odds = ((k + 0.5) / (n_query - k + 0.5)) / (
            (n_term - k + 0.5) / (n_total - n_term - n_query + k + 0.5))
        records.append({"group": group_name, "stage": stage, "set_rule": kind,
                        "database": database, "term_id": term_id,
                        "term_name": names.get(term_id, term_id),
                        "universe_size": n_total, "query_size": n_query,
                        "term_size_in_universe": n_term, "overlap_count": k,
                        "fold_enrichment": (k / n_query) / (n_term / n_total)
                        if n_query else np.nan,
                        "odds_ratio_haldane": odds,
                        "p_value": p_value, "overlap_genes": ",".join(sorted(hits))})
    result = pd.DataFrame(records)
    if result.empty:
        return result
    result["q_value"] = multipletests(result.p_value, method="fdr_bh")[1]
    return result.sort_values(["q_value", "p_value", "term_id"],
                              kind="stable").reset_index(drop=True)


def plot_significant(results: pd.DataFrame, output_dir: Path, database: str) -> None:
    plotted = []
    for group in results.group.unique():
        relevant = results.loc[(results.group == group) & (results.q_value < 0.05)
                               & (results.overlap_count >= 3)]
        relevant = relevant.drop_duplicates("overlap_genes").head(4)
        plotted.append(relevant)
    plotted = pd.concat(plotted, ignore_index=True)
    if plotted.empty:
        return
    plotted = plotted.sort_values("q_value", ascending=False)
    group_labels = {
        "prenatal_only_targets": "E11.5–P0-only targets",
        "early_all_targets": "P0–P11 network",
        "early_postnatal_Zbtb20_all": "P0–P11 Zbtb20 targets",
    }
    labels = [f"{group_labels.get(row.group, row.group)}: {row.term_name}"
              for row in plotted.itertuples()]
    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.48 * len(plotted) + 1.5)),
                           constrained_layout=True)
    scores = -np.log10(plotted.q_value.clip(lower=1e-300))
    colors = ["#3274A1" if row.stage == "prenatal" else "#DC753F"
              for row in plotted.itertuples()]
    ax.barh(labels, scores, color=colors)
    ax.set(xlabel=r"$-\log_{10}$(BH-adjusted p)",
           title="GO biological processes in stage-network targets"
           if database == "GO_BP" else "KEGG pathways in stage-network targets")
    ax.axvline(-np.log10(0.05), color="#777777", lw=1, ls="--")
    fig.savefig(output_dir / f"{database}_significant_terms.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{database}_significant_terms.png", dpi=220,
                bbox_inches="tight")
    plt.close(fig)


def plot_kegg_overview(results: pd.DataFrame, output_dir: Path) -> None:
    """Show nominally ranked KEGG annotations without implying FDR significance."""
    display = []
    group_labels = {
        "prenatal_only_targets": "E11.5–P0-only targets",
        "early_all_targets": "P0–P11 network",
    }
    for group in group_labels:
        subset = results.loc[(results.group == group)
                             & (results.overlap_count >= 3)].copy()
        display.append(subset.sort_values(["p_value", "term_id"]).head(4))
    display = pd.concat(display, ignore_index=True)
    if display.empty:
        return
    display = display.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.2, 5.5), constrained_layout=True)
    y = np.arange(len(display))
    colors = ["#3274A1" if row.stage == "prenatal" else "#DC753F"
              for row in display.itertuples()]
    ax.barh(y, display.fold_enrichment, color=colors, alpha=0.88)
    ax.axvline(1.0, ls="--", lw=1, color="#777777")
    ax.set_yticks(y, [f"{group_labels[row.group]}: {row.term_name}"
                      for row in display.itertuples()])
    ax.set_xlim(0, max(2.25, float(display.fold_enrichment.max()) * 1.23))
    ax.set_xlabel("Observed / expected target proportion")
    ax.set_title("KEGG annotations of stage-network targets (all BH q ≥ 0.05)")
    for pos, row in enumerate(display.itertuples()):
        ax.text(row.fold_enrichment + 0.025, pos,
                f"{row.overlap_count}/{row.term_size_in_universe}; q={row.q_value:.3f}",
                va="center", ha="left", fontsize=8)
    fig.savefig(output_dir / "KEGG_descriptive_pathways.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "KEGG_descriptive_pathways.png", dpi=220,
                bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path,
                        default=Path("data/processed/three_stage_global_500_30tf_20261007"))
    parser.add_argument("--edge-dir", type=Path,
                        default=Path("results/real_p0_rho_profile_common_candidates_20261007"))
    parser.add_argument("--annotation-dir", type=Path,
                        default=Path("data/external/mouse_pathway_annotations_20261007"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/real_p0_rho015_go_kegg_20261007"))
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = read_panel(args.candidate_dir)
    groups, edges = make_groups(args.edge_dir, panel)
    go_sets, go_names = load_go(args.annotation_dir / "MOUSE-mod.gaf.gz",
                                args.annotation_dir / "go-basic.obo", panel)
    kegg_sets, kegg_names, n_mapped = load_kegg(
        args.annotation_dir / "kegg_mouse_pathway_genes.tsv",
        args.annotation_dir / "kegg_mouse_pathways.tsv",
        args.annotation_dir / "MGI_Gene_Model_Coord.rpt", panel)
    pd.DataFrame([{"group": name, "stage": stage, "set_rule": kind,
                   "query_size": len(query), "universe_size": len(universe),
                   "query_genes": ",".join(sorted(query))}
                  for name, stage, kind, query, universe in groups]).to_csv(
                      args.output_dir / "gene_set_definitions.tsv", sep="\t", index=False)
    for database, gene_sets, names in [("GO_BP", go_sets, go_names),
                                        ("KEGG", kegg_sets, kegg_names)]:
        frame = pd.concat([enrich(group, gene_sets, names, database)
                           for group in groups], ignore_index=True)
        frame.to_csv(args.output_dir / f"{database}_all_tests.tsv.gz", sep="\t",
                     index=False, compression="gzip")
        significant = frame.loc[(frame.q_value < 0.05)
                                & (frame.overlap_count >= 3)].copy()
        significant.to_csv(args.output_dir / f"{database}_significant.tsv", sep="\t",
                           index=False)
        linked_edges = []
        for term in significant.itertuples():
            hits = set(term.overlap_genes.split(","))
            part = edges[term.stage].loc[edges[term.stage].target.isin(hits)]
            for edge in part.itertuples():
                linked_edges.append({"group": term.group, "term_id": term.term_id,
                                     "term_name": term.term_name,
                                     "q_value": term.q_value, "stage": term.stage,
                                     "regulator": edge.regulator,
                                     "target": edge.target,
                                     "edge_weight": edge.edge_weight})
        pd.DataFrame(linked_edges, columns=["group", "term_id", "term_name",
                                            "q_value", "stage", "regulator",
                                            "target", "edge_weight"]).to_csv(
            args.output_dir / f"{database}_significant_term_edges.tsv.gz",
            sep="\t", index=False, compression="gzip")
        plot_significant(frame, args.output_dir, database)
        if database == "KEGG":
            plot_kegg_overview(frame, args.output_dir)
    summary = {"panel_genes": len(panel), "mgi_entrez_mapped": n_mapped,
               "annotation_sources": json.loads((args.annotation_dir / "manifest.json").read_text()),
               "background_rule": "515 fitted panel genes for stage-wide lists; regulator-specific common candidate targets minus the 30 candidate regulators for TF modules",
               "groups": {name: {"stage": stage, "rule": kind,
                                  "query_size": len(query), "universe_size": len(universe)}
                          for name, stage, kind, query, universe in groups},
               "statistic": "one-sided hypergeometric survival; BH FDR separately for each database and query group, over all terms with 5-300 background genes; all mouse GO BP annotations except NOT and ND, propagated through is_a and part_of",
               "interpretation": "Enrichment describes annotated functions among fitted candidate targets; it is not independent validation of regulatory edges or cell-type-specific activity"}
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"groups": summary["groups"],
                      "mgi_entrez_mapped": n_mapped}, indent=2), flush=True)


if __name__ == "__main__":
    main()
