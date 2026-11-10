"""Architecture fixtures: upstream modules at toy sizes with random fp32 weights.

    .venv-ref/bin/python tests/ref/tiny_modules.py

Writes ``tests/fixtures/tiny_{dit,vae,umt5}.safetensors`` and
``schedule_torch.safetensors``. No released weights are needed; these check
that every module is the same *function* as upstream (fp32, CPU and Metal),
independently of the bf16 end-to-end comparison.
"""

import _upstream  # noqa: F401  (must come first)
import torch
from safetensors.torch import save_file

torch.manual_seed(0)
FIX = _upstream.FIXTURES
FIX.mkdir(exist_ok=True)


def randomize(module, std=0.05):
    with torch.no_grad():
        for name, p in module.named_parameters():
            if p.ndim == 1 and ("norm" in name or "gamma" in name):
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            else:
                p.copy_(std * torch.randn_like(p) if p.ndim > 1 or "modulation" in name else 0.02 * torch.randn_like(p))


# --------------------------------------------------------------------------- #
def tiny_dit():
    _upstream.fp32_attention()
    from wan.modules.causal_model import CausalWanModel

    cfg = dict(model_type="ti2v", patch_size=(1, 2, 2), text_len=32, in_dim=48, dim=192, ffn_dim=256,
               freq_dim=256, text_dim=64, out_dim=48, num_heads=6, num_layers=2, cc_rope_mode="cc_full",
               prope_temporal_dim=8, cc_phase_slots=16)
    model = CausalWanModel(**cfg).eval()
    randomize(model)
    with torch.no_grad():
        for blk in model.blocks:  # zero-init upstream; random here so every branch is exercised
            blk.self_attn.camera_phase_mlp.proj.weight.copy_(0.3 * torch.randn_like(blk.self_attn.camera_phase_mlp.proj.weight))
    out = {f"state.{k}": v for k, v in model.state_dict().items()}

    F_, H, W = 3, 8, 12
    tpf = (H // 2) * (W // 2)
    heads, hd = 6, 32

    def cams(n):
        ang = torch.linspace(0, 0.4, n)
        view = torch.eye(4).repeat(1, n, 1, 1)
        view[0, :, 0, 0] = torch.cos(ang); view[0, :, 0, 2] = torch.sin(ang)
        view[0, :, 2, 0] = -torch.sin(ang); view[0, :, 2, 2] = torch.cos(ang)
        view[0, :, :3, 3] = torch.randn(n, 3) * 0.5
        K = torch.eye(3).repeat(1, n, 1, 1)
        K[0, :, 0, 0] = 700 + 10 * torch.arange(n); K[0, :, 1, 1] = 680.0
        K[0, :, 0, 2] = 250.0; K[0, :, 1, 2] = 140.0
        return view, K

    view, K = cams(6)
    context = torch.randn(1, 2, 20, 64)  # two chunk contexts, shorter than text_len (zero-padded inside)
    x0, x1 = torch.randn(1, 48, F_, H, W), torch.randn(1, 48, F_, H, W)
    t0 = torch.tensor([[0, 938, 938]]); t1 = torch.tensor([[625, 625, 625]])
    cache = [{"k": torch.zeros(1, 2 * F_ * tpf, heads, hd), "v": torch.zeros(1, 2 * F_ * tpf, heads, hd)} for _ in range(2)]
    taps = {}
    hooks = [blk.register_forward_hook(lambda m, a, o, i=i: taps.__setitem__(i, o.detach().clone())) for i, blk in enumerate(model.blocks)]
    grabbed = {}

    def pre(module, args, kwargs):
        for key in ("pose_c", "viewmats", "Ks", "freqs_x", "e", "context"):
            v = kwargs[key]
            grabbed[key] = torch.view_as_real(v).clone() if v.is_complex() else v.detach().clone()

    handle = model.blocks[0].register_forward_pre_hook(pre, with_kwargs=True)

    def run(name, x, t, chunk, kv_size):
        lo = chunk * F_
        with torch.no_grad():
            flow = model(x, t=t, context=context[:, chunk:chunk + 1], seq_len=10**6, kv_cache=cache, kv_size=kv_size,
                         freqs_offset=lo, viewmats=view[:, lo:lo + F_], Ks=K[:, lo:lo + F_])
        out[f"{name}.flow"] = flow
        for i, v in taps.items():
            out[f"{name}.block_{i}"] = v
        for key, v in grabbed.items():
            out[f"{name}.{key}"] = v.double() if key == "freqs_x" else v

    run("c0_denoise", x0, t0, 0, (0, 0))
    run("c0_commit", x0, torch.zeros_like(t0), 0, (0, -1))
    for i in range(2):
        out[f"c0_commit.k_{i}"] = cache[i]["k"][:, :F_ * tpf].clone()
        out[f"c0_commit.v_{i}"] = cache[i]["v"][:, :F_ * tpf].clone()
    run("c1_denoise", x1, t1, 1, (0, F_ * tpf))
    # A gap in time: the same chunk at temporal positions 40..42 (freqs_positions), history at 0..2.
    with torch.no_grad():
        out["c1_gap.flow"] = model(x1, t=t1, context=context[:, 1:2], seq_len=10**6, kv_cache=cache, kv_size=(0, F_ * tpf),
                                   freqs_positions=torch.tensor([40, 41, 42]), viewmats=view[:, 3:6], Ks=K[:, 3:6])
    out.update(x0=x0, x1=x1, context=context, viewmats=view, Ks=K)
    handle.remove(); [h.remove() for h in hooks]
    save_file({k: v.contiguous() for k, v in out.items()}, str(FIX / "tiny_dit.safetensors"))
    print("tiny_dit", len(out))


# --------------------------------------------------------------------------- #
def tiny_vae():
    from wan.modules.vae_wan22 import VideoVAE38_, WanVideoVAE38

    model = VideoVAE38_(dim=16, z_dim=48, dec_dim=16).eval()
    randomize(model, std=0.08)
    ref = WanVideoVAE38.__new__(WanVideoVAE38)  # only for its mean/std constants
    torch.nn.Module.__init__(ref)
    full = WanVideoVAE38
    import inspect, re
    src = inspect.getsource(full.__init__)
    nums = [float(v) for v in re.findall(r"-?\d+\.\d+", src.split("self.mean")[0])]
    mean, std = torch.tensor(nums[:48]), torch.tensor(nums[48:96])
    scale = [mean, 1.0 / std]
    video = torch.rand(1, 3, 9, 32, 48) * 2 - 1
    with torch.no_grad():
        z = model.encode(video, scale)
        z1 = model.encode(video[:, :, :1], scale)
        pix = model.decode(z, scale)
    out = {f"state.{k}": v for k, v in model.state_dict().items()}
    out.update(video=video, latent=z, latent_first=z1, pixels=pix, mean=mean, std=std)
    save_file({k: v.contiguous() for k, v in out.items()}, str(FIX / "tiny_vae.safetensors"))
    print("tiny_vae", tuple(z.shape), tuple(pix.shape))


# --------------------------------------------------------------------------- #
def tiny_umt5():
    from wan.modules.t5 import T5Encoder, T5RelativeEmbedding

    model = T5Encoder(vocab=100, dim=64, dim_attn=64, dim_ffn=128, num_heads=4, num_layers=2, num_buckets=32,
                      shared_pos=False, dropout=0.1).eval()
    randomize(model, std=0.2)
    ids = torch.randint(2, 100, (2, 40)); mask = torch.ones(2, 40, dtype=torch.long)
    ids[0, 25:] = 0; mask[0, 25:] = 0
    with torch.no_grad():
        y = model(ids, mask)
    rel = T5RelativeEmbedding(32, 64, bidirectional=True)
    pos = torch.arange(512).unsqueeze(0) - torch.arange(512).unsqueeze(1)
    out = {f"state.{k}": v for k, v in model.state_dict().items()}
    out.update(ids=ids, mask=mask, out=y, buckets=rel._relative_position_bucket(pos))
    save_file({k: v.contiguous() for k, v in out.items()}, str(FIX / "tiny_umt5.safetensors"))
    print("tiny_umt5", tuple(y.shape))


# --------------------------------------------------------------------------- #
def schedule():
    from utils.scheduler import FlowMatchScheduler

    s = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    s.set_timesteps(1000, training=True)
    x = torch.randn(3, 4, 5, 6).to(torch.bfloat16); n = torch.randn(3, 4, 5, 6).to(torch.bfloat16)
    t = torch.tensor([0, 833, 833])
    noisy = s.add_noise(x, n, t)
    idx = torch.argmin((s.timesteps.double().unsqueeze(0) - t.double().unsqueeze(1)).abs(), dim=1)
    x0 = (n.double() - s.sigmas.double()[idx].reshape(-1, 1, 1, 1) * x.double()).to(torch.bfloat16)
    probe = torch.tensor([1000, 938, 833, 625, 0])
    pidx = torch.argmin((s.timesteps.unsqueeze(0) - probe.unsqueeze(1)).abs(), dim=1)
    save_file(dict(sigmas=s.sigmas, timesteps=s.timesteps, x=x, noise=n, noisy=noisy, x0=x0, probe_index=pidx,
                   latent_index=torch.linspace(0, 80, 21).round().long()), str(FIX / "schedule_torch.safetensors"))
    print("schedule", s.sigmas[:3].tolist())


if __name__ == "__main__":
    schedule(); tiny_umt5(); tiny_vae(); tiny_dit(); tiny_archive()


def tiny_archive():
    """A small ``torch.save`` archive (fp32, bf16, int64 and a shared-storage view) for the torch-free reader."""
    base = torch.randn(4, 6)
    state = {"w": torch.randn(3, 5), "half": torch.randn(7).to(torch.bfloat16), "ids": torch.arange(9), "base": base, "row": base[2]}
    torch.save(state, str(FIX / "tiny_state.pth"))
    save_file({k: v.clone() for k, v in state.items()}, str(FIX / "tiny_state.safetensors"))
    print("tiny_archive", len(state))
