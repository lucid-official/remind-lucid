"""Parity with upstream ReMind on the released weights (Metal, bf16).

The reference is ``tests/ref/remind_upstream.py``: upstream's own code on the
CPU, preset 01, with the initial noise and every re-noising draw recorded.
The same noise is injected here. Skipped when the weights or the fixture are
missing. Every measured number is also written to
``outputs/parity_report.json`` (the README's parity table is that file).

Set ``REMIND_LUCID_COMPILE=1`` to run the same checks through the
``lucid.compile`` programs.
"""

import json
import math
import os
import struct

import lucid
import pytest

from conftest import FIXTURES, ROOT, cosine, max_rel
from remind_lucid import catalog, presets
from remind_lucid.pipeline import InjectedNoise, Job, ReMindPipeline, load_vae
from remind_lucid.text.encoder import PromptEncoder

PRESET = "01_latte_occluder_recovery"
FIXTURE = FIXTURES / f"remind_upstream_{PRESET}_large.safetensors"
COMPILE = os.environ.get("REMIND_LUCID_COMPILE") == "1"
REPORT = (
    ROOT
    / "outputs"
    / ("parity_report_compiled.json" if COMPILE else "parity_report.json")
)

pytestmark = pytest.mark.skipif(
    not (FIXTURE.exists() and catalog.ready()),
    reason="needs the released weights and tests/ref/remind_upstream.py",
)


def _metadata(path) -> dict:
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(n)).get("__metadata__", {})


def psnr(a, b) -> float:
    """PSNR of two tensors on a [0, 1] scale."""
    mse = (a.float().to("cpu") - b.float().to("cpu")).square().mean().item()
    return 10 * math.log10(1.0 / mse) if mse > 0 else float("inf")


class State:
    report: dict = {}


@pytest.fixture(scope="module")
def ref():
    return lucid.load_safetensors(str(FIXTURE))


@pytest.fixture(scope="module")
def meta():
    return _metadata(FIXTURE)


@pytest.fixture(scope="module")
def job(meta):
    path, preset = presets.load_preset(PRESET)
    job = Job.from_preset(path, preset)
    assert job.prompts == json.loads(meta["schedule"])
    assert job.seed == int(meta["seed"])
    return job


@pytest.fixture(scope="module", autouse=True)
def write_report():
    yield
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(State.report, indent=2) + "\n")


def record(name: str, **values) -> None:
    State.report[name] = {
        k: (round(v, 6) if isinstance(v, float) else v) for k, v in values.items()
    }


def test_cameras(ref, job):
    assert max_rel(job.viewmats, ref["viewmats"]) == 0.0
    assert max_rel(job.intrinsics, ref["intrinsics"]) == 0.0


def test_umt5_features(ref, meta, job):
    """Our UMT5 (bf16, Metal) against upstream's (bf16, CPU) on the unique chunk prompts."""
    prompts = json.loads(meta["prompts"])
    encoder = PromptEncoder(use_cache=False)
    ours = encoder.encode(prompts)[0]
    encoder.release()
    worst_cos, worst_rel = 1.0, 0.0
    for i in range(len(prompts)):
        worst_cos = min(worst_cos, cosine(ours[i], ref[f"t5_{i}"]))
        worst_rel = max(worst_rel, max_rel(ours[i], ref[f"t5_{i}"]))
    record(
        "umt5_features", prompts=len(prompts), min_cosine=worst_cos, max_rel=worst_rel
    )
    State.features_ours = (
        lucid.stack([ours[prompts.index(p)] for p in job.prompts])
        .unsqueeze(0)
        .to("cpu")
    )
    assert worst_cos >= 0.999


def test_vae_first_frame(ref, job):
    pipe = ReMindPipeline()
    source = pipe.encode_source(job)
    State.source_ours = source.to("cpu")
    c, r = cosine(source, ref["source_prefix"]), max_rel(source, ref["source_prefix"])
    record("vae_encode_first_frame", cosine=c, max_rel=r)
    assert c >= 0.999


@pytest.fixture(scope="module")
def pipe():
    pipe = ReMindPipeline(accelerate=COMPILE)
    pipe.ensure_generator()
    return pipe


def test_lora_merge(ref, pipe):
    """Samples of the merged weights of blocks 0 and 29 against peft's merge on the CPU."""
    model = pipe.generator
    exact, total, worst_sum = 0, 0, 0.0
    for key in [k for k in ref if k.startswith("merged.") and k.endswith(".slice")]:
        name = key[len("merged.") : -len(".slice")]
        module = model
        for part in name.split("."):
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        ours = module.weight[:256, :256].float().to("cpu")
        theirs = ref[key].float()
        exact += int((ours == theirs).sum().item())
        total += theirs.numel()
        s_ours = module.weight.to("cpu").double().sum().item()
        s_ref = ref[f"merged.{name}.sum"].item()
        worst_sum = max(worst_sum, abs(s_ours - s_ref) / max(abs(s_ref), 1e-9))
    record(
        "lora_merge",
        sampled_values=total,
        bit_identical=exact,
        fraction=exact / total,
        max_rel_sum_error=worst_sum,
    )
    assert exact / total > 0.999


def _run(pipe, job, ref, meta, features, source, taps):
    chunks = int(meta["chunks"])
    noise = InjectedNoise(
        {k[len("noise.") :]: v for k, v in ref.items() if k.startswith("noise.")}
    )
    out = []
    for chunk in pipe.rollout(
        job, features.to("metal"), source.to("metal"), noise, taps=taps
    ):
        out.append(chunk.latents.to("cpu"))
        if chunk.index == 0:
            State_kv = {
                b: tuple(t.to("cpu") for t in pipe.cache.read(b)) for b in (0, 14, 29)
            }
            taps["kv"] = State_kv
        if chunk.index + 1 == chunks:
            break
    return lucid.cat(out, dim=1)


def test_dit_isolated(ref, meta, job, pipe):
    """The generator alone: upstream's own text features and first-frame latent, upstream's noise."""
    prompts = json.loads(meta["prompts"])
    features = lucid.stack(
        [ref[f"t5_{prompts.index(p)}"] for p in job.prompts]
    ).unsqueeze(0)
    taps = {}
    latents = _run(pipe, job, ref, meta, features, ref["source_prefix"][:, :1], taps)
    blocks = [cosine(taps[f"block_{i}"], ref[f"c0s0.block_{i}"]) for i in range(30)]
    record(
        "dit_blocks_chunk0_step0",
        min_cosine=min(blocks),
        last_block_cosine=blocks[-1],
        per_block=[round(b, 6) for b in blocks],
    )
    kv = {}
    for b in (0, 14, 29):
        k, v = taps["kv"][b]
        kv[f"k_{b}"] = cosine(k.permute(0, 2, 1, 3), ref[f"kv.k_{b}"])
        kv[f"v_{b}"] = cosine(v.permute(0, 2, 1, 3), ref[f"kv.v_{b}"])
    record("kv_cache_after_chunk0", **kv)
    per_chunk = [
        cosine(latents[:, 3 * c : 3 * c + 3], ref["latents"][:, 3 * c : 3 * c + 3])
        for c in range(latents.shape[1] // 3)
    ]
    record(
        "latents_dit_isolated",
        per_chunk_cosine=[round(c, 6) for c in per_chunk],
        min_cosine=min(per_chunk),
    )
    State.latents_isolated = latents
    assert min(blocks) >= 0.999
    assert min(per_chunk) >= 0.995


def test_end_to_end(ref, meta, job, pipe):
    """Everything ours — UMT5, VAE encode, generator — only the noise is upstream's."""
    latents = _run(pipe, job, ref, meta, State.features_ours, State.source_ours, {})
    per_chunk = [
        cosine(latents[:, 3 * c : 3 * c + 3], ref["latents"][:, 3 * c : 3 * c + 3])
        for c in range(latents.shape[1] // 3)
    ]
    record(
        "latents_end_to_end",
        per_chunk_cosine=[round(c, 6) for c in per_chunk],
        min_cosine=min(per_chunk),
    )
    State.latents_e2e = latents
    assert min(per_chunk) >= 0.995


def test_decode(ref, pipe):
    if "pixels" not in ref:
        pytest.skip("reference was run with --no-decode")
    theirs = ref["pixels"].float()  # (T, 3, H, W) in [0, 1]
    vae = pipe.ensure_vae()
    decode = pipe.decoder()

    def to_unit(latents):
        decode.reset()
        return (
            (decode(latents.to("metal")).float() * 0.5 + 0.5)
            .clamp(0, 1)
            .permute(1, 0, 2, 3)
            .to("cpu")
        )

    same_latents = psnr(to_unit(ref["latents"]), theirs)
    end_to_end = psnr(to_unit(State.latents_e2e), theirs)
    record(
        "decode",
        psnr_same_latents_db=same_latents,
        psnr_end_to_end_db=end_to_end,
        frames=int(theirs.shape[0]),
        vae_dtype=str(vae.dtype),
    )
    assert same_latents >= 35.0
    assert end_to_end >= 35.0
