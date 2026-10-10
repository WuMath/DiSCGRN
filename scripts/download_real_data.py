#!/usr/bin/env python3
"""Download public data for a real-data DiSCGRN study.

Standard-library only. Run ``python scripts/download_real_data.py --help`` from
the project root. Files are never replaced unless --force is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import html.parser
import json
import os
import shutil
import subprocess
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


USER_AGENT = "DiSCGRN-data-downloader/1.0 (public research data)"
GEO_HOST = "https://ftp.ncbi.nlm.nih.gov/geo/series"
COLLECTRI_URL = (
    "https://omnipathdb.org/interactions?datasets=tf_target&resources=CollecTRI"
    "&organisms=10090&format=tsv&fields=sources,references&genesymbols=1"
)
CHIP_INDEX_URL = "https://chip-atlas.org/data/target_genes_analysis.json"
CHIP_QUERY_URL = "https://chip-atlas.org/target_genes?type=tsv"
MGI_ORTHOLOGY_URL = "https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt"
TRANSPORT = "auto"


def geo_base(accession: str) -> str:
    """NCBI GEO uses GSE74nnn/GSE74243 for accession GSE74243."""
    if not accession.startswith("GSE") or not accession[3:].isdigit():
        raise ValueError(f"Invalid GEO Series accession: {accession}")
    bucket = accession[:-3] + "nnn"
    return f"{GEO_HOST}/{bucket}/{accession}"


def geo_url(accession: str, kind: str, filename: str) -> str:
    return f"{geo_base(accession)}/{kind}/{urllib.parse.quote(filename)}"


class Links(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


def request(url: str, *, data: bytes | None = None, headers: dict[str, str] | None = None):
    return urllib.request.Request(
        url,
        data=data,
        headers={"User-Agent": USER_AGENT, **(headers or {})},
        method="POST" if data is not None else "GET",
    )


def remote_files(accession: str, kind: str) -> list[str]:
    """List only filenames belonging to this accession, never parent links."""
    url = f"{geo_base(accession)}/{kind}/"
    with urllib.request.urlopen(request(url), timeout=60) as response:
        page = response.read(2_000_000).decode("utf-8", errors="replace")
    parser = Links()
    parser.feed(page)
    names = set()
    for href in parser.hrefs:
        name = urllib.parse.unquote(urllib.parse.urlsplit(href).path.rsplit("/", 1)[-1])
        if name.startswith(accession + "_") or name.startswith(accession + "-"):
            names.add(name)
    return sorted(names)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_download(path: Path, target: Path) -> None:
    """Reject error pages and malformed archives before publishing a download."""
    if path.stat().st_size == 0:
        raise RuntimeError("Empty download")
    with path.open("rb") as handle:
        prefix = handle.read(64).lstrip().lower()
    if prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html"):
        raise RuntimeError("The server returned an HTML page, not a data file")
    if target.name.endswith(".gz") and not prefix.startswith(b"\x1f\x8b"):
        raise RuntimeError("Expected a gzip file")
    if target.name.endswith(".tar"):
        try:
            with tarfile.open(path, "r:") as archive:
                archive.getmembers()
        except tarfile.TarError as exc:
            raise RuntimeError(f"Incomplete or invalid TAR archive: {exc}") from exc
    if target.name.endswith(".xlsx"):
        if not zipfile.is_zipfile(path):
            raise RuntimeError("Expected an XLSX/ZIP file")
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            if bad_member:
                raise RuntimeError(f"Corrupt XLSX member: {bad_member}")


def fetch_urllib(url: str, part: Path) -> None:
    offset = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with urllib.request.urlopen(request(url, headers=headers), timeout=120) as response:
        append = offset > 0 and response.status == 206
        with part.open("ab" if append else "wb") as output:
            while chunk := response.read(4 * 1024 * 1024):
                output.write(chunk)


def fetch_curl(url: str, part: Path) -> None:
    executable = shutil.which("curl")
    if executable is None:
        raise RuntimeError("curl is not installed; try --transport urllib")
    command = [
        executable, "--fail", "--location", "--http1.1", "--silent", "--show-error",
        "--retry", "6", "--retry-delay", "3",
        "--connect-timeout", "30", "--continue-at", "-", "--output", str(part), url,
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"curl exit {result.returncode}: {result.stderr.strip()}")


def download(url: str, target: Path, *, force: bool, retries: int = 4) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 0 and not force:
        validate_download(target, target)
        return {"path": str(target), "url": url, "status": "existing",
                "bytes": target.stat().st_size, "sha256": sha256(target)}
    part = target.with_name(target.name + ".part")
    error = None
    transports = ("urllib", "curl") if TRANSPORT == "auto" else (TRANSPORT,)
    for transport in transports:
        for attempt in range(retries if transport == "urllib" else 1):
            try:
                if transport == "urllib":
                    fetch_urllib(url, part)
                else:
                    fetch_curl(url, part)
                validate_download(part, target)
                result = {"path": str(target), "url": url, "status": "downloaded",
                          "bytes": part.stat().st_size, "sha256": sha256(part),
                          "transport": transport}
                os.replace(part, target)
                return result
            except (OSError, urllib.error.URLError, RuntimeError) as exc:
                error = exc
                print(f"[retry] {target.name}: {transport}: {exc}")
                if attempt + 1 < retries and transport == "urllib":
                    time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Failed to download {url} to {target}: {error}")


def geo_matrix(accession: str, root: Path, *, force: bool, dry_run: bool) -> list[dict]:
    """Fetch Series Matrix files (including multiple-platform Series)."""
    if dry_run:
        print(f"[plan] {accession}: discover Series Matrix files at {geo_base(accession)}/matrix/")
        return []
    try:
        names = [n for n in remote_files(accession, "matrix") if n.endswith("_series_matrix.txt.gz")]
    except Exception as exc:
        print(f"[warning] {accession}: matrix listing failed ({exc}); trying standard filename")
        names = [f"{accession}_series_matrix.txt.gz"]
    if not names:
        print(f"[warning] {accession}: no matrix files listed; trying standard filename")
        names = [f"{accession}_series_matrix.txt.gz"]
    records = []
    for name in names:
        records.append(download(geo_url(accession, "matrix", name),
                                root / "raw" / accession.lower() / name, force=force))
    return records


def geo_supplement(accession: str, filename: str, root: Path,
                   *, force: bool, dry_run: bool) -> list[dict]:
    url = geo_url(accession, "suppl", filename)
    target = root / "raw" / accession.lower() / filename
    if dry_run:
        print(f"[plan] {target} <- {url}")
        return []
    return [download(url, target, force=force)]


def direct(url: str, target: Path, *, force: bool, dry_run: bool) -> list[dict]:
    if dry_run:
        print(f"[plan] {target} <- {url}")
        return []
    return [download(url, target, force=force)]


def collectri(root: Path, *, force: bool, dry_run: bool) -> list[dict]:
    return direct(COLLECTRI_URL, root / "external" / "collectri_mouse.tsv",
                  force=force, dry_run=dry_run)


def chip_index(root: Path, *, force: bool, dry_run: bool) -> list[dict]:
    return direct(CHIP_INDEX_URL, root / "external" / "chipatlas_target_genes_analysis.json",
                  force=force, dry_run=dry_run)


def chip_target(root: Path, tf: str, distance: int, *, force: bool, dry_run: bool) -> list[dict]:
    target = root / "external" / "chipatlas_mm10" / f"{tf}.{distance}.tsv"
    if dry_run:
        print(f"[plan] {target} <- ChIP-Atlas mm10 {tf}, TSS window +/-{distance} kb")
        return []
    index = root / "external" / "chipatlas_target_genes_analysis.json"
    if not index.exists():
        raise FileNotFoundError(f"First download the ChIP-Atlas index: {index}")
    available = json.loads(index.read_text(encoding="utf-8")).get("mm10", {})
    if tf not in available:
        raise ValueError(f"TF {tf!r} is not an mm10 ChIP-Atlas antigen; check spelling/case")
    payload = json.dumps({"condition": {"genome": "mm10", "antigen": tf,
                                        "distance": str(distance)}}).encode("utf-8")
    with urllib.request.urlopen(
        request(CHIP_QUERY_URL, data=payload, headers={"Content-Type": "application/json"}),
        timeout=120,
    ) as response:
        result = json.loads(response.read().decode("utf-8"))
    url = urllib.parse.urljoin(CHIP_QUERY_URL, result["url"])
    record = download(url, target, force=force)
    record.update({"tf": tf, "genome": "mm10", "promoter_window_kb": distance,
                   "query_url": CHIP_QUERY_URL})
    return [record]


def download_dataset(name: str, root: Path, *, force: bool, dry_run: bool,
                     raw_cel: bool, large: bool) -> list[dict]:
    records: list[dict] = []
    if name == "gse74243":
        for filename in (
            "GSE74243_matrix_RMAlog2expressionvalues.txt.gz",
            "GSE74243_Probes_BultLung2015.tsv.gz",
        ):
            records += geo_supplement("GSE74243", filename, root, force=force, dry_run=dry_run)
        records += geo_matrix("GSE74243", root, force=force, dry_run=dry_run)
        if raw_cel:
            records += geo_supplement("GSE74243", "GSE74243_RAW.tar", root,
                                      force=force, dry_run=dry_run)
    elif name == "gse129628":
        # The SuperSeries archive contains scRNA-seq and ChIP-seq files. The
        # bulk knockout/control table is a separate GSE129583 supplement.
        records += geo_supplement("GSE129628", "GSE129628_RAW.tar", root,
                                  force=force, dry_run=dry_run)
        records += geo_supplement(
            "GSE129583", "GSE129583_processeddata_matrix_Nkx2-1_RNAseq.xlsx",
            root, force=force, dry_run=dry_run,
        )
        for accession in ("GSE129583", "GSE129584", "GSE129627"):
            records += geo_matrix(accession, root, force=force, dry_run=dry_run)
        # Separate ChIP peak tables are small and preserve direct provenance.
        records += geo_supplement("GSE129627", "GSE129627_RAW.tar", root,
                                  force=force, dry_run=dry_run)
    elif name == "gse149563":
        if not large:
            raise ValueError("GSE149563_RAW.tar is about 2.2 GB; rerun with --allow-large")
        records += geo_supplement("GSE149563", "GSE149563_RAW.tar", root,
                                  force=force, dry_run=dry_run)
        records += geo_matrix("GSE149563", root, force=force, dry_run=dry_run)
    elif name == "gse14334":
        records += geo_matrix("GSE14334", root, force=force, dry_run=dry_run)
        if raw_cel:
            records += geo_supplement("GSE14334", "GSE14334_RAW.tar", root,
                                      force=force, dry_run=dry_run)
    elif name == "gse23043":
        records += geo_supplement("GSE23043", "GSE23043_merged.txt.gz", root,
                                  force=force, dry_run=dry_run)
        records += geo_matrix("GSE23043", root, force=force, dry_run=dry_run)
        if raw_cel:
            records += geo_supplement("GSE23043", "GSE23043_RAW.tar", root,
                                      force=force, dry_run=dry_run)
    elif name == "orthology":
        records += direct(MGI_ORTHOLOGY_URL,
                          root / "external" / "HOM_MouseHumanSequence.rpt",
                          force=force, dry_run=dry_run)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return records


def main() -> None:
    global TRANSPORT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"),
                        help="Root containing raw/ and external/ (default: data)")
    parser.add_argument("--profile", choices=["core", "validation", "optional", "all"],
                        help="core=GSE74243+CollecTRI+ChIP index; "
                             "validation=GSE129628 plus GSE129583 knockout/control matrix; "
                             "optional=GSE149563+GSE14334+GSE23043+mouse-human orthology")
    parser.add_argument("--dataset", action="append",
                        choices=["gse74243", "collectri", "chipatlas", "gse129628",
                                 "gse149563", "gse14334", "gse23043",
                                 "orthology"],
                        help="Download one dataset; repeat for several")
    parser.add_argument("--chip-tf", action="append", default=[],
                        help="Download an mm10 ChIP-Atlas target table for this TF; repeatable")
    parser.add_argument("--chip-distance", type=int, choices=[1, 5, 10], default=5,
                        help="ChIP-Atlas TSS window in kb (default: 5)")
    parser.add_argument("--allow-large", action="store_true",
                        help="Explicitly permit the ~2.2 GB GSE149563 archive")
    parser.add_argument("--raw-cel", action="store_true",
                        help="Also fetch optional GEO raw CEL/ChIP-chip archives")
    parser.add_argument("--force", action="store_true", help="Replace existing files")
    parser.add_argument("--transport", choices=["auto", "urllib", "curl"], default="auto",
                        help="Download transport; auto tries urllib then curl (default: auto)")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without downloading")
    args = parser.parse_args()
    TRANSPORT = args.transport
    if not args.profile and not args.dataset and not args.chip_tf:
        parser.error("Choose --profile, --dataset, or --chip-tf")

    choices = set(args.dataset or [])
    profiles = {
        "core": {"gse74243", "collectri", "chipatlas"},
        "validation": {"gse129628"},
        "optional": {"gse149563", "gse14334", "gse23043", "orthology"},
    }
    if args.profile == "all":
        choices.update(set.union(*profiles.values()))
    elif args.profile:
        choices.update(profiles[args.profile])
    if args.chip_tf:
        choices.add("chipatlas")

    root = args.data_root
    records = []
    failures = []
    for name in sorted(choices):
        try:
            if name == "collectri":
                records += collectri(root, force=args.force, dry_run=args.dry_run)
            elif name == "chipatlas":
                records += chip_index(root, force=args.force, dry_run=args.dry_run)
            else:
                records += download_dataset(name, root, force=args.force,
                                            dry_run=args.dry_run, raw_cel=args.raw_cel,
                                            large=args.allow_large)
            print(f"[ok] {name}")
        except Exception as exc:
            failures.append({"dataset": name, "error": str(exc)})
            print(f"[failed] {name}: {exc}")
    for tf in args.chip_tf:
        try:
            records += chip_target(root, tf, args.chip_distance,
                                   force=args.force, dry_run=args.dry_run)
            print(f"[ok] ChIP-Atlas {tf}")
        except Exception as exc:
            failures.append({"dataset": f"chipatlas:{tf}", "error": str(exc)})
            print(f"[failed] ChIP-Atlas {tf}: {exc}")
    if not args.dry_run:
        manifest_path = root / "real_data_download_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        by_path = {record["path"]: record for record in old.get("files", [])}
        by_path.update({record["path"]: record for record in records})
        payload = {"files": sorted(by_path.values(), key=lambda x: x["path"]),
                   "last_attempt_failures": failures}
        temporary = manifest_path.with_name(manifest_path.name + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        os.replace(temporary, manifest_path)
        print(f"Manifest: {manifest_path} ({len(payload['files'])} files)")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
