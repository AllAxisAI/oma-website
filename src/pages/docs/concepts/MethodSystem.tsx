import CodeBlock from '../../../components/CodeBlock';

export default function MethodSystem() {
  return (
    <div>
      <div className="mb-2">
        <span className="text-xs uppercase tracking-[0.25em] text-neutral-500">Core Concepts</span>
      </div>
      <h1 className="text-4xl font-semibold tracking-tight mb-4">Method System</h1>
      <p className="text-lg leading-8 text-neutral-300 mb-10">
        The Method is the central abstraction in OpenMedAxis. It owns the training algorithm — 
        what happens at each step, how losses are computed, and how optimizers are configured.
        Everything else (data, logging, distributed training) is handled by the framework.
      </p>

      {/* Overview */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-4">Overview</h2>
        <p className="text-neutral-300 leading-7 mb-4">
          <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">Method</code> is a subclass of PyTorch Lightning's{' '}
          <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">LightningModule</code>. It delegates
          infrastructure concerns (GPU placement, gradient accumulation, mixed precision, distributed training)
          to Lightning, while letting you focus on the algorithm.
        </p>
        <div className="rounded-2xl border border-white/10 bg-white/5 p-5 mb-4">
          <p className="text-xs uppercase tracking-[0.2em] text-neutral-500 mb-4">Class hierarchy</p>
          <div className="font-mono text-sm space-y-1.5 text-neutral-300">
            <div className="flex items-center gap-2">
              <span className="w-4 h-px bg-neutral-600 inline-block" />
              <span className="text-blue-400">Method</span>
              <span className="text-neutral-600 text-xs ml-1">(base — override step())</span>
            </div>
            <div className="flex items-center gap-2 ml-6">
              <span className="w-4 h-px bg-neutral-600 inline-block" />
              <span className="text-purple-400">GroupedLossMethod</span>
              <span className="text-neutral-600 text-xs ml-1">(override build_state())</span>
            </div>
            <div className="flex items-center gap-2 ml-12">
              <span className="w-4 h-px bg-neutral-600 inline-block" />
              <span className="text-emerald-400">TranslationMethod</span>
            </div>
            <div className="flex items-center gap-2 ml-12">
              <span className="w-4 h-px bg-neutral-600 inline-block" />
              <span className="text-emerald-400">AutoencoderKLMethod</span>
            </div>
            <div className="flex items-center gap-2 ml-12">
              <span className="w-4 h-px bg-neutral-600 inline-block" />
              <span className="text-emerald-400">DiffusionBridgeTranslationMethod</span>
            </div>
          </div>
        </div>
      </section>

      {/* Base Method */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">The base Method class</h2>
        <p className="text-neutral-300 leading-7 mb-4">
          The base <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">Method</code> class handles
          the training/validation/test loop boilerplate. Your job is to override{' '}
          <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">step()</code> with your algorithm.
        </p>

        <h3 className="text-lg font-medium mb-1 text-neutral-200">Constructor signature</h3>
        <CodeBlock code={`from oma.methods.base import Method

Method(
    model,                  # Your nn.Module (required)
    loss_fn=None,           # LossComposer or any callable
    optimizer_cfg=None,     # dict or pre-built optimizer
    scheduler_cfg=None,     # dict or pre-built scheduler
    metrics=None,           # dict of torchmetrics
    **kwargs                # passed to LightningModule
)`} />

        <h3 className="text-lg font-medium mb-1 text-neutral-200">The step() contract</h3>
        <p className="text-sm text-neutral-400 leading-6 mb-3">
          Override <code className="text-sm bg-white/10 px-1.5 py-0.5 rounded font-mono">step()</code> to define your algorithm.
          It is called automatically by <code className="text-sm bg-white/10 px-1.5 py-0.5 rounded font-mono">training_step()</code>,{' '}
          <code className="text-sm bg-white/10 px-1.5 py-0.5 rounded font-mono">validation_step()</code>, and{' '}
          <code className="text-sm bg-white/10 px-1.5 py-0.5 rounded font-mono">test_step()</code>.
        </p>
        <CodeBlock code={`def step(
    self,
    batch,
    stage: str,       # "train", "val", or "test"
    batch_idx: int,
) -> dict:
    # Must return:
    return {
        "loss": loss_tensor,       # scalar Tensor, used for backprop
        "metrics": {},             # dict of scalars, logged automatically
        "artifacts": {},           # images, predictions, etc. — for evaluators
    }`} />

        <h3 className="text-lg font-medium mb-1 text-neutral-200">Full minimal example</h3>
        <CodeBlock code={`import torch
from oma.methods.base import Method


class MyRegressionMethod(Method):
    def __init__(self, model, lr=1e-4):
        super().__init__(
            model=model,
            optimizer_cfg={"type": "Adam", "lr": lr},
        )

    def step(self, batch, stage, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = torch.nn.functional.l1_loss(pred, y)

        return {
            "loss": loss,
            "metrics": {"l1": loss.detach()},
            "artifacts": {"pred": pred.detach(), "target": y},
        }`} />
      </section>

      {/* Override points */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-4">Override points</h2>
        <div className="rounded-2xl border border-white/10 overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-white/5 text-neutral-400 text-xs uppercase tracking-wider">
              <tr>
                <th className="text-left px-4 py-3">Method</th>
                <th className="text-left px-4 py-3">Default behaviour</th>
                <th className="text-left px-4 py-3">When to override</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {[
                ['step()', 'Must override', 'Always — defines your algorithm'],
                ['parse_batch()', 'Returns batch as-is', 'When batch format needs unpacking'],
                ['infer()', 'Calls self.model(input)', 'When inference is more complex than a single forward pass'],
                ['compute_loss()', 'Calls self.loss_fn(state)', 'When loss needs custom state construction'],
                ['compute_metrics()', 'Returns empty dict', 'To add extra metrics beyond the loss'],
                ['configure_optimizers()', 'Reads optimizer_cfg dict', 'When you need custom optimizer logic'],
              ].map(([method, def_, when]) => (
                <tr key={method as string} className="text-neutral-300">
                  <td className="px-4 py-3 font-mono text-xs text-blue-300">{method}</td>
                  <td className="px-4 py-3 text-neutral-400 text-xs">{def_}</td>
                  <td className="px-4 py-3 text-xs">{when}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* GroupedLossMethod */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">GroupedLossMethod</h2>
        <p className="text-neutral-300 leading-7 mb-4">
          Use <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">GroupedLossMethod</code> when your
          training uses multiple optimizers — for example, generator + discriminator in adversarial training.
          It manages the optimizer alternation automatically.
        </p>
        <p className="text-neutral-300 leading-7 mb-4">
          Instead of overriding <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">step()</code>,
          you override <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">build_state()</code>
          to construct a shared state dict. The LossComposer then runs each LossTerm against the state
          and partitions the totals by loss group (<code className="text-sm bg-white/10 px-1 py-0.5 rounded font-mono">"main"</code>,{' '}
          <code className="text-sm bg-white/10 px-1 py-0.5 rounded font-mono">"disc"</code>, etc.).
        </p>

        <CodeBlock code={`from oma.methods.base import GroupedLossMethod
from oma.losses import LossComposer
from oma.losses.terms import L1LossTerm, KLLossTerm


class MyVAEMethod(GroupedLossMethod):
    def __init__(self, model, disc, lr=1e-4):
        loss_fn = LossComposer([
            L1LossTerm(weight=1.0, group="main"),
            KLLossTerm(weight=0.001, group="main"),
        ])
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            optimizer_cfg=[
                {"type": "Adam", "lr": lr, "params": "model"},
            ],
        )

    def build_state(self, batch, stage):
        # Construct the state dict — all loss terms read from this
        x = batch["image"]
        recon, posterior = self.model(x)
        return {
            "input": x,
            "recon": recon,
            "posterior": posterior,
            "split": stage,
            "global_step": self.global_step,
        }`} />
      </section>

      {/* Optimizer config */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">Optimizer configuration</h2>
        <p className="text-neutral-300 leading-7 mb-4">
          Pass an <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">optimizer_cfg</code> dict to avoid boilerplate.
          The framework resolves the optimizer class from{' '}
          <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">torch.optim</code> using the <code className="text-sm bg-white/10 px-1 py-0.5 rounded font-mono">"type"</code> key.
        </p>
        <CodeBlock code={`# Simple single-optimizer config
optimizer_cfg = {
    "type": "Adam",         # any class in torch.optim
    "lr": 1e-4,
    "betas": (0.9, 0.999),
    "weight_decay": 1e-5,
}

# With a learning rate scheduler
scheduler_cfg = {
    "type": "CosineAnnealingLR",
    "T_max": 100,
    "eta_min": 1e-6,
}

method = MyMethod(
    model=model,
    optimizer_cfg=optimizer_cfg,
    scheduler_cfg=scheduler_cfg,
)`} />
      </section>

      {/* Built-in methods */}
      <section className="mb-10">
        <h2 className="text-2xl font-semibold mb-4">Built-in methods</h2>
        <div className="space-y-4">
          {[
            {
              name: 'TranslationMethod',
              import_: 'from oma.methods import TranslationMethod',
              desc: 'Paired image-to-image translation. Expects batches with source and target images. Computes a pixel loss between the model output and the target.',
            },
            {
              name: 'DiffusionBridgeTranslationMethod',
              import_: 'from oma.methods import DiffusionBridgeTranslationMethod',
              desc: 'Non-adversarial diffusion bridge for paired translation. At each training step, samples a random timestep t, corrupts the target image along a bridge conditioned on the source, and trains the model to predict the clean target (x0 prediction).',
            },
            {
              name: 'AutoencoderKLMethod',
              import_: 'from oma.methods import AutoencoderKLMethod',
              desc: 'Variational autoencoder with KL regularization. Uses GroupedLossMethod with a "main" group (reconstruction + KL) and an optional "disc" group (adversarial loss with a patch discriminator).',
            },
          ].map((m) => (
            <div key={m.name} className="rounded-2xl border border-white/10 bg-white/5 p-5">
              <div className="font-mono text-sm text-blue-300 mb-1">{m.name}</div>
              <div className="text-xs text-neutral-500 font-mono mb-3">{m.import_}</div>
              <div className="text-sm text-neutral-300 leading-6">{m.desc}</div>
            </div>
          ))}
        </div>
      </section>

      <div className="flex gap-4 pt-4 border-t border-white/10">
        <a href="#/docs/concepts/loss-system" className="text-sm text-blue-400 hover:text-blue-300 transition">
          Next: Loss System →
        </a>
      </div>
    </div>
  );
}
