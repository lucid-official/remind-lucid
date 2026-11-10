"""Fetch the released weights into ``checkpoints/``.

    python -m remind_lucid.download            # everything
    python -m remind_lucid.download --status   # per-file status only

* The two ReMind files are downloaded whole and checked against the
  release's ``SHA256SUMS``.
* The UMT5 encoder and the Wan2.2 VAE are ``torch.save`` archives; they are
  read remotely, tensor by tensor, by a torch-free reader that refuses any
  pickle global a state dict does not need, and written as SafeTensors in
  the dtype they were released in.
* The fp32 Wan2.2 DiT shards (20 GB) are compared header-to-header with the
  ReMind generator; only base tensors the generator does not supply would be
  fetched (cast to bf16), which for the released files is none.
"""

from __future__ import annotations

import argparse
import shutil

from remind_lucid import catalog, hub
from remind_lucid.io.torch_zip import to_safetensors


def published_sha256() -> dict[str, str]:
    sums = {}
    for line in hub.fetch_text(catalog.REMIND_REPO, "SHA256SUMS").splitlines():
        if line.strip():
            digest, name = line.split()
            sums[name.lstrip("*")] = digest
    return sums


def fetch_remind(progress: hub.Progress | None = None) -> None:
    sums = published_sha256()
    for name, dest in (
        (catalog.GENERATOR_FILE, catalog.generator_path()),
        (catalog.LORA_FILE, catalog.lora_path()),
    ):
        hub.fetch_file(catalog.REMIND_REPO, name, dest, sha256=sums[name], progress=progress)  # fmt: skip


def fetch_torch_archive(filename: str, dest, progress: hub.Progress | None = None) -> None:  # fmt: skip
    if dest.exists():
        return
    remote = hub.open_remote(catalog.WAN_REPO, filename, progress)
    to_safetensors(remote, dest)


def fetch_base_extra(progress: hub.Progress | None = None) -> list[str]:
    """Fetch Wan base tensors the generator lacks. Returns their names (empty for the release)."""
    generator = hub.RemoteSafetensors(catalog.REMIND_REPO, catalog.GENERATOR_FILE)
    generator_keys = [t.name for t in generator.tensors]
    fetched: list[str] = []
    for shard in catalog.WAN_SHARDS:
        remote = hub.RemoteSafetensors(catalog.WAN_REPO, shard)
        need = set(catalog.base_keys_needed(generator_keys, [t.name for t in remote.tensors]))  # fmt: skip
        if need:
            dest = catalog.base_extra_path().with_name(f"wan_base_extra.{shard}")
            hub.fetch_cast(remote, dest, "BF16", select=need.__contains__, progress=progress)  # fmt: skip
            fetched += sorted(need)
    return fetched


def fetch_shared(progress: hub.Progress | None = None) -> None:
    for name in catalog.TOKENIZER_FILES:
        hub.fetch_file(catalog.WAN_REPO, f"google/umt5-xxl/{name}", catalog.tokenizer_dir() / name, progress=progress)  # fmt: skip
    hub.fetch_file(catalog.WAN_REPO, "config.json", catalog.wan_dir() / "config.json", progress=progress)  # fmt: skip
    fetch_torch_archive(catalog.VAE_FILE, catalog.vae_path(), progress)
    fetch_torch_archive(catalog.T5_FILE, catalog.t5_path(), progress)


def fetch_all(progress: hub.Progress | None = None) -> None:
    catalog.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    fetch_shared(progress)
    fetch_remind(progress)
    extra = fetch_base_extra(progress)
    if extra:
        print(
            f"  fetched {len(extra)} Wan base tensors the generator lacks", flush=True
        )


def print_status() -> int:
    """Print the per-file table; return the bytes still to download."""
    missing = 0
    for row in catalog.status():
        missing += (
            max(0, row["size"] - row["have"]) if row["state"] != "verified" else 0
        )
        print(
            f"  {row['state']:<10} {row['have'] / 1e9:6.2f}/{row['size'] / 1e9:5.2f} GB  "
            f"{row['label']:<32} <- {row['source']}"
        )
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    parser.add_argument("--status", action="store_true", help="print the per-file status and exit")  # fmt: skip
    args = parser.parse_args()
    catalog.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    print(f"checkpoints: {catalog.CHECKPOINTS}")
    missing = print_status()
    free = shutil.disk_usage(catalog.CHECKPOINTS).free
    print(f"  to download: {missing / 1e9:.1f} GB   free disk: {free / 1e9:.1f} GB")
    if args.status or missing == 0:
        return
    if free < missing + (2 << 30):
        raise SystemExit("not enough free disk space")
    fetch_all(hub.print_progress())
    print_status()


if __name__ == "__main__":
    main()
