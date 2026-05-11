import CodeBlock from '../../../components/CodeBlock';

export default function Logging() {
  return (
    <div>
      <div className="mb-2">
        <span className="text-xs uppercase tracking-[0.25em] text-neutral-500">Guides</span>
      </div>
      <h1 className="text-4xl font-semibold tracking-tight mb-4">Logging & Debugging</h1>
      <p className="text-lg leading-8 text-neutral-300 mb-10">
        OMA gives you a single call —{' '}
        <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">self.oma_log()</code> — that works
        like <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">print()</code> for training.
        Drop it anywhere, pass any value, and it routes to the right backend automatically.
        You can also tap into predefined methods without touching their source.
      </p>

      {/* ── Three lanes ───────────────────────────────────────── */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-4">How OMA logging is structured</h2>
        <p className="text-neutral-300 leading-7 mb-5">
          Three independent lanes coexist without conflict. Knowing which lane to use tells you exactly
          where to put a log call.
        </p>
        <div className="rounded-2xl border border-white/10 bg-white/5 divide-y divide-white/10 overflow-hidden mb-4">
          {[
            {
              lane: 'Lane 1',
              api: 'attach_metric(state, key, value)',
              who: 'Method framework',
              what: 'Built-in scalars the method always emits. Goes through _log_metrics → Lightning → CSV + TB + W&B. Nothing for you to do.',
            },
            {
              lane: 'Lane 2',
              api: 'self.oma_log(key, value, ...)',
              who: 'You — anywhere in your code',
              what: 'Ad-hoc logging of anything: scalars, images, histograms. Full control over dest, frequency, stage, hooks, and progress bar.',
            },
            {
              lane: 'Lane 3',
              api: 'attach_artifact(state, key, value)',
              who: 'Method framework',
              what: 'Raw tensors for EvaluatorManager (saving images to disk, computing metrics). Not logging — data routing.',
            },
          ].map(({ lane, api, who, what }) => (
            <div key={lane} className="flex gap-4 px-5 py-4">
              <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-white/15 bg-white/5 text-xs font-semibold text-white">
                {lane.split(' ')[1]}
              </span>
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2 mb-0.5">
                  <code className="text-xs bg-white/10 px-1.5 py-0.5 rounded font-mono text-blue-300">{api}</code>
                  <span className="text-xs text-neutral-500">— {who}</span>
                </div>
                <div className="text-sm text-neutral-400 leading-6">{what}</div>
              </div>
            </div>
          ))}
        </div>
        <p className="text-sm text-neutral-500">
          This guide covers Lane 2. Lanes 1 and 3 are handled automatically by the method base class.
        </p>
      </section>

      {/* ── oma_log signature ─────────────────────────────────── */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">The <code className="font-mono text-blue-300 text-xl">oma_log</code> call</h2>
        <p className="text-neutral-300 leading-7 mb-4">
          Available on every OMA method. All parameters after{' '}
          <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">value</code> are optional —
          the simplest call is just a key and a value.
        </p>
        <CodeBlock code={`self.oma_log(
    key,                    # str  — e.g. "debug/t_hist"
    value,                  # anything — scalar, tensor, str
    kind=None,              # "scalar" | "image" | "histogram" | "text"
                            #   auto-detected from value when None
    dest="all",             # "all" | "local" | "remote" | ["wandb"] | ...
    every_n_steps=1,        # only log when global_step % every_n_steps == 0
    stages=None,            # ["val", "test"] — None means all stages
    hooks=None,             # per-call hooks; None = use global registry
    prog_bar=False,         # show in tqdm bar (scalars only)
    on_step=True,
    on_epoch=True,
)`} />
        <div className="mt-4 rounded-2xl border border-white/10 overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-white/5 text-neutral-500 text-xs uppercase tracking-wider">
              <tr>
                <th className="text-left px-4 py-3">Value type</th>
                <th className="text-left px-4 py-3">Auto-detected as</th>
                <th className="text-left px-4 py-3">Backends</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 text-neutral-300">
              {[
                ['float, int, 0-dim tensor', '"scalar"', 'Lightning → CSV + TB + W&B'],
                ['tensor (C,H,W) or (B,C,H,W), C ∈ {1,3,4}', '"image"', 'TB add_image / W&B Image'],
                ['1-D tensor', '"histogram"', 'TB add_histogram / W&B Histogram'],
                ['str', '"text"', 'TB add_text'],
                ['explicit kind= given', 'kind= wins', '—'],
              ].map(([val, kind, backend]) => (
                <tr key={val}>
                  <td className="px-4 py-3 font-mono text-xs text-blue-300">{val}</td>
                  <td className="px-4 py-3 font-mono text-xs">{kind}</td>
                  <td className="px-4 py-3 text-xs text-neutral-400">{backend}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* ── Debug scalars ─────────────────────────────────────── */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">Debugging with scalars</h2>
        <p className="text-neutral-300 leading-7 mb-4">
          The fastest way to start debugging — drop a scalar log anywhere and check the plot in TensorBoard.
          By default logs are silent (not in the progress bar). Opt into the bar explicitly.
        </p>
        <CodeBlock code={`def build_state(self, batch, stage, batch_idx, group=None):
    source, target = batch["source"], batch["target"]
    pred = self.generator(source)

    # ── Drop debug scalars anywhere, like print() ──────────────

    # Are discriminator scores separating? Log both distributions
    with torch.no_grad():
        fake_scores = self.discriminator(pred.detach())
        real_scores = self.discriminator(target)
    self.oma_log("debug/disc_fake_mean", fake_scores.mean())
    self.oma_log("debug/disc_real_mean", real_scores.mean())

    # Is the generator loss too low too fast? (mode collapse warning)
    g_loss = -fake_scores.mean()
    self.oma_log("debug/g_loss_raw", g_loss)

    # Watch this one number in the progress bar
    self.oma_log("train/loss", g_loss, prog_bar=True)

    # Val metric — epoch-end only, show in bar when it appears
    self.oma_log("val/psnr", compute_psnr(pred, target),
                 stages=["val"], on_step=False, on_epoch=True, prog_bar=True)

    return {...}`} />
        <p className="text-neutral-400 leading-7 mt-4 text-sm">
          Scalars go through Lightning's{' '}
          <code className="text-xs bg-white/10 px-1.5 py-0.5 rounded font-mono">self.log()</code> and
          reach all active loggers (CSV, TensorBoard, W&B) simultaneously.
          The CSV file is always written — it's your safety net even without a running TensorBoard.
        </p>
      </section>

      {/* ── Debug images ──────────────────────────────────────── */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">Debugging with images</h2>
        <p className="text-neutral-300 leading-7 mb-4">
          Image tensors are assembled into a grid automatically.
          Values outside [0, 1] are normalised. Images never appear in the progress bar.
          Use <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">stages=["val"]</code> and{' '}
          <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">every_n_steps=</code> to keep
          TensorBoard clean.
        </p>
        <CodeBlock code={`def build_state(self, batch, stage, batch_idx, group=None):
    source, target = batch["source"], batch["target"]
    pred = self.model(source)

    # ── Side-by-side comparison grid — val only, every 50 steps ──
    comparison = torch.cat([source[:4], pred[:4], target[:4]], dim=0)
    self.oma_log(
        "val/source_pred_target",
        comparison,
        kind="image",           # optional — auto-detected from shape
        stages=["val"],
        every_n_steps=50,
    )

    # ── Residual map — what is the model getting wrong? ──────────
    residual = (pred - target).abs()
    self.oma_log(
        "val/residual",
        residual[:4],
        kind="image",
        stages=["val"],
        every_n_steps=50,
    )

    # ── Noisy intermediate during diffusion (always available) ───
    self.oma_log(
        "debug/xt",
        state.get("xt", None)[:2],
        kind="image",
        every_n_steps=200,
    )

    return {...}`} />
        <div className="mt-5 rounded-2xl border border-white/10 bg-white/5 p-5">
          <p className="text-xs uppercase tracking-[0.2em] text-neutral-500 mb-3">Quick pattern reference</p>
          <div className="space-y-2 text-sm text-neutral-300">
            {[
              ['Pred vs target', 'torch.cat([pred[:4], target[:4]], dim=0)'],
              ['Source / pred / target', 'torch.cat([source[:4], pred[:4], target[:4]], dim=0)'],
              ['Residual', '(pred - target).abs()[:4]'],
              ['Diffusion intermediate', 'state["xt"][:2]'],
              ['Single sample', 'pred[[0]]   # keep batch dim'],
            ].map(([label, code]) => (
              <div key={label} className="flex gap-3">
                <span className="text-neutral-500 shrink-0 w-36">{label}</span>
                <code className="font-mono text-xs text-blue-300">{code}</code>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Debug histograms ──────────────────────────────────── */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">Debugging with histograms</h2>
        <p className="text-neutral-300 leading-7 mb-4">
          Histograms are the best tool for monitoring distributions that scalars flatten —
          discriminator scores, timestep sampling, latent codes, gradient magnitudes.
        </p>
        <CodeBlock code={`# Are discriminator scores bimodal? (real ≠ fake)
self.oma_log("debug/disc_fake", fake_scores.flatten())
self.oma_log("debug/disc_real", real_scores.flatten())

# Is the timestep sampler covering the full range?
self.oma_log("debug/t_dist", t.float(), every_n_steps=10)

# Latent code distribution — should it be close to N(0,1)?
self.oma_log("debug/z_dist", z.flatten(), kind="histogram")

# Force histogram for a 2-D or higher-dim tensor
self.oma_log("debug/features", features, kind="histogram", every_n_steps=50)`} />
      </section>

      {/* ── dest routing ──────────────────────────────────────── */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">Routing with <code className="font-mono text-blue-300 text-xl">dest=</code></h2>
        <p className="text-neutral-300 leading-7 mb-5">
          Keep debug noise out of your W&B dashboard or send a sweep metric only to W&B.
        </p>
        <div className="rounded-2xl border border-white/10 overflow-hidden mb-5">
          <table className="w-full text-sm">
            <thead className="bg-white/5 text-neutral-500 text-xs uppercase tracking-wider">
              <tr>
                <th className="text-left px-4 py-3">dest=</th>
                <th className="text-left px-4 py-3">Goes to</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 text-neutral-300">
              {[
                ['"all"  (default)', 'all active backends'],
                ['"local"', 'CSV only (disk)'],
                ['"remote"', 'TensorBoard + W&B'],
                ['["wandb"]', 'W&B only'],
                ['["tensorboard"]', 'TensorBoard only'],
              ].map(([dest, goes]) => (
                <tr key={dest}>
                  <td className="px-4 py-3 font-mono text-xs text-blue-300">{dest}</td>
                  <td className="px-4 py-3 text-sm text-neutral-400">{goes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <CodeBlock code={`# Debug iteration — on disk only, W&B stays clean
self.oma_log("debug/disc_scores", fake_scores.flatten(), dest="local")

# Image — remote only (CSV can't store images anyway)
self.oma_log("val/pred_grid", pred[:4], kind="image", dest="remote")

# Metric for W&B hyperparameter sweep comparison
self.oma_log("sweep/val_psnr", psnr, dest=["wandb"], stages=["val"])`} />
      </section>

      {/* ── Hooks inside method ───────────────────────────────── */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">Hooks — automatic stats on every log</h2>
        <p className="text-neutral-300 leading-7 mb-4">
          Hooks fire after a log call and emit additional entries via the same interface.
          Register them once on{' '}
          <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">self.oma_logger</code> and
          they apply to every matching call automatically. OMA ships three built-in hooks.
        </p>
        <div className="rounded-2xl border border-white/10 bg-white/5 divide-y divide-white/10 overflow-hidden mb-6">
          {[
            { name: 'ImageStatsHook', fires: 'kind == "image"', emits: '{key}/mean, {key}/std, {key}/min, {key}/max as scalars' },
            { name: 'TensorStatsHook', fires: 'any tensor', emits: '{key}/norm, {key}/mean, {key}/std as scalars' },
            { name: 'NaNDetectorHook', fires: 'any tensor  (use kind="*")', emits: '{key}/has_nan and {key}/has_inf flags (1.0 when found)' },
          ].map(({ name, fires, emits }) => (
            <div key={name} className="px-5 py-4">
              <code className="text-sm font-mono text-blue-300">{name}</code>
              <div className="mt-1 text-xs text-neutral-500">Fires on: {fires}</div>
              <div className="text-sm text-neutral-400 leading-6 mt-0.5">Emits: {emits}</div>
            </div>
          ))}
        </div>

        <h3 className="text-lg font-medium mb-1 text-neutral-200">Register in your method</h3>
        <CodeBlock code={`from oma.methods.base import GroupedLossMethod
from oma.logging import ImageStatsHook, NaNDetectorHook


class MyGANMethod(GroupedLossMethod):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Every image log now auto-emits mean/std/min/max as scalars
        self.oma_logger.register_hook("image", ImageStatsHook())

        # Catch NaN/Inf anywhere in the training loop
        self.oma_logger.register_hook("*", NaNDetectorHook())

    def build_state(self, batch, stage, batch_idx, group=None):
        ...
        # ImageStatsHook fires here automatically — no extra code needed
        self.oma_log("val/pred", pred[:4], kind="image", stages=["val"])
        ...`} />

        <h3 className="text-lg font-medium mt-6 mb-1 text-neutral-200">Per-call override</h3>
        <CodeBlock code={`from oma.logging import ImageStatsHook

# Pass hooks= to replace the global registry for just this call
self.oma_log("val/pred", pred[:4], kind="image",
             hooks=[ImageStatsHook()])

# hooks=[] disables all hooks for this call
self.oma_log("debug/fmap", fmap, kind="image", hooks=[])`} />

        <h3 className="text-lg font-medium mt-6 mb-1 text-neutral-200">Writing a custom hook</h3>
        <p className="text-neutral-300 leading-7 mb-4">
          Subclass{' '}
          <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">LoggingHook</code> and implement{' '}
          <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">__call__</code>.
          The <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">log_fn</code> argument
          is identical to <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">oma_log</code> —
          use it to emit extra entries.
        </p>
        <CodeBlock code={`from oma.logging import LoggingHook
import torch


class SNRHook(LoggingHook):
    """
    When a timestep tensor t is logged, also emit the
    signal-to-noise ratio at those timesteps.
    """

    def __init__(self, alphas_cumprod: torch.Tensor) -> None:
        self.alphas_cumprod = alphas_cumprod

    def __call__(self, key, value, kind, step, log_fn):
        if not torch.is_tensor(value):
            return
        t = value.long().clamp(0, len(self.alphas_cumprod) - 1)
        alpha = self.alphas_cumprod[t].float()
        snr   = alpha / (1.0 - alpha + 1e-8)
        log_fn(f"{key}/snr_mean", snr.mean(), kind="scalar")
        log_fn(f"{key}/snr_min",  snr.min(),  kind="scalar")


# Register once — fires on every log of the timestep tensor
self.oma_logger.register_hook("histogram", SNRHook(process.alphas_cumprod))

# Now this single call also logs snr_mean and snr_min automatically
self.oma_log("debug/t", t.float())`} />
      </section>

      {/* ── Logging without touching the method ──────────────── */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">Logging without touching the method</h2>
        <p className="text-neutral-300 leading-7 mb-4">
          When you use a predefined OMA method (like{' '}
          <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">GaussianDiffusionMethod</code>)
          and want to log something it doesn't expose by default — without editing its source — use a
          Lightning callback. All OMA predefined methods expose their internal tensors (
          <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">t</code>,{' '}
          <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">noise</code>,{' '}
          <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">xt</code>,{' '}
          <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">x0_pred</code>, etc.) in{' '}
          <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">outputs["state"]</code>.
        </p>

        <h3 className="text-lg font-medium mb-1 text-neutral-200">Read from state — no source changes</h3>
        <CodeBlock code={`import lightning as L
import torch


class TimestepDebugCallback(L.Callback):
    """
    Logs timestep distribution and SNR without touching
    GaussianDiffusionMethod source code.
    """

    def __init__(self, alphas_cumprod: torch.Tensor) -> None:
        self.alphas_cumprod = alphas_cumprod

    def on_train_batch_end(self, trainer, method, outputs, batch, batch_idx):
        state = outputs.get("state", {})
        t = state.get("t", None)
        if t is None:
            return

        # Use method.oma_log — same system, same backends
        method.oma_log("debug/t_hist", t.float(),
                       kind="histogram", every_n_steps=10)

        alpha = self.alphas_cumprod[t.long().cpu()].float()
        snr   = alpha / (1.0 - alpha + 1e-8)
        method.oma_log("debug/snr_mean", snr.mean())


# In the recipe — zero changes to the method
trainer = L.Trainer(
    callbacks=[TimestepDebugCallback(process.alphas_cumprod)],
)`} />

        <h3 className="text-lg font-medium mt-6 mb-1 text-neutral-200">Run a new forward pass with custom inputs</h3>
        <p className="text-neutral-300 leading-7 mb-4">
          Sometimes you want to log something that never happens in normal training — for example,
          what the model predicts at a fixed timestep{' '}
          <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">t=500</code> throughout
          training to visualise model evolution.
          A callback gets full access to the method object, so you can run any forward pass.
        </p>
        <CodeBlock code={`import lightning as L
import torch


class FixedSampleCallback(L.Callback):
    """
    Every N steps: run inference on a fixed held-out batch and
    log the output as an image. Visualises model evolution during training.

    This computation never happens in normal training — it is injected
    entirely from outside the method.
    """

    def __init__(self, fixed_source: torch.Tensor, every_n_steps: int = 200) -> None:
        self.fixed_source  = fixed_source   # (4, C, H, W) held-out batch
        self.every_n_steps = every_n_steps

    def on_train_batch_end(self, trainer, method, outputs, batch, batch_idx):
        if method.global_step % self.every_n_steps != 0:
            return

        device = next(method.parameters()).device
        src    = self.fixed_source.to(device)

        with torch.no_grad():
            pred = method.model(src)    # or method.infer(cond=src, ...)

        # Log via the same oma_log interface
        method.oma_log(
            "viz/fixed_evolution",
            pred,
            kind="image",
            hooks=[],           # no hooks for this — we control it entirely
        )


# Grab a held-out batch once before training starts
fixed_batch = next(iter(val_dataloader))

trainer = L.Trainer(
    callbacks=[
        FixedSampleCallback(fixed_batch["source"], every_n_steps=200),
    ],
)`} />
      </section>

      {/* ── Full method example ───────────────────────────────── */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">Complete example</h2>
        <p className="text-neutral-300 leading-7 mb-4">
          All three lanes and hooks working together inside a diffusion method.
        </p>
        <CodeBlock filename="my_diffusion_method.py" code={`from oma.methods.diffusion.gaussian import GaussianDiffusionMethod
from oma.logging import ImageStatsHook, NaNDetectorHook


class MyDiffusionMethod(GaussianDiffusionMethod):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Global hooks — apply to every matching oma_log call
        self.oma_logger.register_hook("image", ImageStatsHook())
        self.oma_logger.register_hook("*",     NaNDetectorHook())

    def build_diffusion_state(self, inputs, stage, batch_idx, group=None):
        x0   = inputs["x0"]
        cond = inputs.get("cond")

        t     = self.sample_time(x0.shape[0], x0.device, stage=stage, state=None)
        noise = self.sample_noise(x0)
        xt    = self.build_process_state(x0=x0, t=t, cond=cond, noise=noise)["xt"]
        pred  = self.forward_model(x=xt, t=t, cond=cond)

        # ── Lane 1: method scalar — always logged ─────────────────
        state = {"x0": x0, "xt": xt, "t": t, "noise": noise, "cond": cond}
        self.attach_metric(state, f"{stage}/t_mean", t.float().mean().detach())

        # ── Lane 2: ad-hoc debug logging ─────────────────────────
        # Timestep histogram — is the sampler covering the full range?
        self.oma_log("debug/t_dist", t.float(),
                     every_n_steps=10, dest="local")

        # Image comparison — val only, ImageStatsHook fires automatically
        self.oma_log("val/pred_grid",
                     torch.cat([cond[:4], pred[:4], x0[:4]], dim=0),
                     kind="image", stages=["val"], every_n_steps=50)

        # Watch training loss in progress bar
        self.oma_log("train/loss", (pred - noise).pow(2).mean(), prog_bar=True)

        # ── Lane 3: feed evaluators ───────────────────────────────
        self.attach_artifact(state, "pred",   pred)
        self.attach_artifact(state, "target", x0)
        self.attach_artifact(state, "source", cond)

        return state`} />
      </section>

      {/* ── Backend setup ─────────────────────────────────────── */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">Setting up backends</h2>
        <p className="text-neutral-300 leading-7 mb-4">
          Pass a list of loggers to the Trainer. CSV is always recommended as a local baseline.
        </p>
        <CodeBlock filename="recipes/my_recipe.py" code={`from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger
import lightning as L

trainer = L.Trainer(
    max_epochs=200,
    logger=[
        CSVLogger("logs/"),                           # always on — scalars to disk
        TensorBoardLogger("logs/", name="my_run"),    # local images + scalars
        # WandbLogger(project="oma", name="my_run"),  # cloud, opt-in
    ],
    log_every_n_steps=10,
)`} />
      </section>

      <div className="flex gap-4 pt-4 border-t border-white/10">
        <a href="#/docs/guides/custom-loss" className="text-sm text-neutral-400 hover:text-white transition">← Custom Loss Term</a>
        <a href="#/docs/guides/evaluation" className="text-sm text-blue-400 hover:text-blue-300 transition ml-auto">Next: Evaluation & Metrics →</a>
      </div>
    </div>
  );
}
