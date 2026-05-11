import CodeBlock from '../../../components/CodeBlock';

export default function Evaluation() {
  return (
    <div>
      <div className="mb-2">
        <span className="text-xs uppercase tracking-[0.25em] text-neutral-500">Guides</span>
      </div>
      <h1 className="text-4xl font-semibold tracking-tight mb-4">Evaluation & Metrics</h1>
      <p className="text-lg leading-8 text-neutral-300 mb-10">
        The evaluation system separates metric computation from the training loop. Wire evaluators into
        any method once — they save image grids automatically during validation, accumulate PSNR/SSIM
        across batches, and report epoch means to your logger. The same evaluators run offline on{' '}
        <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">.npy</code> files to compare
        external model outputs without a training run.
      </p>

      {/* ── Three roles ───────────────────────────────────────── */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-4">How the evaluation system works</h2>
        <p className="text-neutral-300 leading-7 mb-5">
          The system has three components that compose cleanly. You configure evaluators once; the base
          method class drives them automatically.
        </p>
        <div className="rounded-2xl border border-white/10 bg-white/5 divide-y divide-white/10 overflow-hidden mb-4">
          {[
            {
              role: '1',
              name: 'Evaluator',
              when: 'per-batch + epoch-end',
              what: 'The unit of evaluation logic. __call__ runs each batch (image saving, accumulation). reset() clears state at epoch start. compute() returns aggregated metrics at epoch end.',
            },
            {
              role: '2',
              name: 'EvaluatorManager',
              when: 'orchestration',
              what: 'Holds a dict of named evaluators. Calls reset() / update() / compute() in sequence. Namespaces all results as {stage}/{name}/{key}.',
            },
            {
              role: '3',
              name: 'Method base class',
              when: 'automatic',
              what: 'Drives the manager at the right moments: reset at epoch start, update each val/test batch, compute at epoch end and log scalars.',
            },
          ].map(({ role, name, when, what }) => (
            <div key={role} className="flex gap-4 px-5 py-4">
              <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-white/15 bg-white/5 text-xs font-semibold text-white">
                {role}
              </span>
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2 mb-0.5">
                  <code className="text-xs bg-white/10 px-1.5 py-0.5 rounded font-mono text-blue-300">{name}</code>
                  <span className="text-xs text-neutral-500">— {when}</span>
                </div>
                <div className="text-sm text-neutral-400 leading-6">{what}</div>
              </div>
            </div>
          ))}
        </div>
        <p className="text-sm text-neutral-500">
          Image evaluators participate in <code className="font-mono">update()</code> (save each batch's grid)
          and return nothing from <code className="font-mono">compute()</code>.
          Metric evaluators accumulate silently in <code className="font-mono">update()</code> and report
          in <code className="font-mono">compute()</code>. Both types compose freely in the same manager.
        </p>
      </section>

      {/* ── Wiring into a recipe ──────────────────────────────── */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">Wiring evaluators into a recipe</h2>
        <p className="text-neutral-300 leading-7 mb-4">
          Build an{' '}
          <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">EvaluatorManager</code> and
          pass it to any method via <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">evaluator_manager=</code>.
          No other changes needed — the base class handles the rest.
        </p>
        <CodeBlock filename="recipes/my_recipe.py" code={`from oma.evaluation import (
    EvaluatorManager,
    SaveImageEvaluatorGeneric,
    ReconstructionEvaluator,
)

evaluator_manager = EvaluatorManager({
    # Save image grids every 50 val batches
    "images": SaveImageEvaluatorGeneric(
        image_keys=["source", "pred", "target"],
        max_samples=4,
        save_every_n_steps=50,
    ),
    # Accumulate PSNR and SSIM across all val batches; report epoch mean
    "metrics": ReconstructionEvaluator(
        pred_key="pred",
        target_key="target",
    ),
})

method = MyMethod(
    model=model,
    loss_fn=loss_fn,
    evaluator_manager=evaluator_manager,   # ← one line
)`} />
        <p className="text-neutral-400 leading-7 mt-4 text-sm">
          The method's <code className="font-mono text-xs">artifacts</code> dict (whatever{' '}
          <code className="font-mono text-xs">step()</code> returns under the{' '}
          <code className="font-mono text-xs">"artifacts"</code> key) is forwarded to every evaluator.
          Make sure your method populates the keys that evaluators expect —{' '}
          <code className="font-mono text-xs">"pred"</code>, <code className="font-mono text-xs">"target"</code>,{' '}
          <code className="font-mono text-xs">"source"</code> by convention.
        </p>
      </section>

      {/* ── SaveImageEvaluatorGeneric ─────────────────────────── */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">
          <code className="font-mono text-blue-300 text-xl">SaveImageEvaluatorGeneric</code>
        </h2>
        <p className="text-neutral-300 leading-7 mb-4">
          Saves a grid of images to disk at the end of selected validation batches. Each call produces
          one <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">.png</code> containing
          all requested keys side-by-side for up to <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">max_samples</code> rows.
          Images are normalised to [0, 1] for display regardless of the original value range.
        </p>
        <CodeBlock code={`SaveImageEvaluatorGeneric(
    name="images",            # str   — used as the sub-folder name on disk
    image_keys=["pred"],      # list  — which artifact keys to include in the grid
    max_samples=4,            # int   — rows in the grid (clips to batch size)
    save_every_n_steps=1,     # int   — skip batches where batch_idx % N != 0
    dpi=150,                  # int   — figure DPI
    output_dir=None,          # str   — fallback path used only in standalone mode
                              #         (during training, logger.log_dir always wins)
)`} />

        <h3 className="text-lg font-medium mt-6 mb-3 text-neutral-200">Output path structure</h3>
        <p className="text-neutral-300 leading-7 mb-4">
          During training, files are written under the logger's directory. Each validation epoch gets
          its own folder named after the training step at which validation fired.
        </p>
        <div className="rounded-2xl border border-white/10 bg-black/30 p-5 font-mono text-sm text-neutral-300 mb-4">
          <div className="text-neutral-500 mb-2"># during training</div>
          <div>{'{logger.log_dir}'}<span className="text-blue-300">/{'{stage}'}</span><span className="text-emerald-400">/{'{name}'}</span><span className="text-yellow-400">/global_step_{'{N}'}</span><span className="text-orange-300">/step_{'{batch_idx}'}.png</span></div>
          <div className="mt-3 text-neutral-500 mb-2"># example</div>
          <div className="text-neutral-400">lightning_logs/version_0/<span className="text-blue-300">val</span>/<span className="text-emerald-400">images</span>/<span className="text-yellow-400">global_step_1000</span>/<span className="text-orange-300">step_0.png</span></div>
          <div className="text-neutral-400">lightning_logs/version_0/<span className="text-blue-300">val</span>/<span className="text-emerald-400">images</span>/<span className="text-yellow-400">global_step_1000</span>/<span className="text-orange-300">step_50.png</span></div>
          <div className="text-neutral-400">lightning_logs/version_0/<span className="text-blue-300">val</span>/<span className="text-emerald-400">images</span>/<span className="text-yellow-400">global_step_2000</span>/<span className="text-orange-300">step_0.png</span></div>
        </div>
        <div className="rounded-2xl border border-white/10 bg-white/5 divide-y divide-white/10 overflow-hidden text-sm">
          {[
            ['{stage}', 'val or test — from the method hook that fired'],
            ['{name}', 'the name= constructor argument (default "images")'],
            ['global_step_{N}', 'training step counter when validation fired — one folder per epoch'],
            ['step_{batch_idx}.png', 'batch index within the val epoch — controlled by save_every_n_steps'],
          ].map(([seg, desc]) => (
            <div key={seg} className="flex gap-4 px-4 py-3">
              <code className="font-mono text-xs text-blue-300 w-44 shrink-0">{seg}</code>
              <span className="text-neutral-400 text-xs leading-5">{desc}</span>
            </div>
          ))}
        </div>
      </section>

      {/* ── ReconstructionEvaluator ───────────────────────────── */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">
          <code className="font-mono text-blue-300 text-xl">ReconstructionEvaluator</code>
        </h2>
        <p className="text-neutral-300 leading-7 mb-4">
          Accumulates PSNR and SSIM per batch and reports the epoch mean at validation end.
          The values are logged automatically through Lightning to all active backends
          (CSV, TensorBoard, W&B).
        </p>
        <CodeBlock code={`ReconstructionEvaluator(
    name="reconstruction",       # str  — used as the namespace in result keys
    pred_key="pred",             # str  — key in artifacts dict for predictions
    target_key="target",         # str  — key in artifacts dict for ground truth
    mask=None,                   # ndarray | None — optional spatial mask
    norm="mean",                 # "mean" | "01" — normalisation before PSNR/SSIM
    compute_psnr=True,           # bool
    compute_ssim=True,           # bool
    ssim_multiply_by_100=True,   # bool — reports SSIM as 0–100, not 0–1
)`} />

        <h3 className="text-lg font-medium mt-6 mb-3 text-neutral-200">Logged metric keys</h3>
        <div className="rounded-2xl border border-white/10 overflow-hidden mb-4">
          <table className="w-full text-sm">
            <thead className="bg-white/5 text-neutral-500 text-xs uppercase tracking-wider">
              <tr>
                <th className="text-left px-4 py-3">Key in logger</th>
                <th className="text-left px-4 py-3">Description</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 text-neutral-300">
              {[
                ['val/metrics/psnr', 'Mean PSNR across all val batches (dB)'],
                ['val/metrics/ssim_x100', 'Mean SSIM × 100 across all val batches (ssim_multiply_by_100=True)'],
                ['val/metrics/ssim', 'Mean SSIM in [0, 1] (ssim_multiply_by_100=False)'],
                ['test/metrics/psnr', 'Same metrics logged when running trainer.test()'],
              ].map(([key, desc]) => (
                <tr key={key}>
                  <td className="px-4 py-3 font-mono text-xs text-blue-300">{key}</td>
                  <td className="px-4 py-3 text-xs text-neutral-400">{desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-sm text-neutral-500">
          The key format is <code className="font-mono">{'{stage}'}/{'{name}'}/{'{metric}'}</code> where
          <code className="font-mono"> name</code> is the dict key you used in{' '}
          <code className="font-mono">EvaluatorManager({'{"metrics": ReconstructionEvaluator(...)}'})}</code>.
        </p>
      </section>

      {/* ── Standalone evaluation ─────────────────────────────── */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">Standalone evaluation — <code className="font-mono text-blue-300 text-xl">evaluate_from_npy</code></h2>
        <p className="text-neutral-300 leading-7 mb-4">
          Evaluate any model's predictions saved as{' '}
          <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">.npy</code> files without
          a training loop. Useful for comparing SAM, nnU-Net, or any other external model against your
          trained model's ground truth.
        </p>
        <CodeBlock filename="scripts/eval_sam.py" code={`from oma.evaluation.runner import evaluate_from_npy

results = evaluate_from_npy(
    pred="outputs/sam_predictions.npy",   # (N, H, W) or (N, 1, H, W)
    target="data/ground_truth.npy",
    source="data/inputs.npy",             # optional — included in image grid
    output_dir="eval_results/sam_vs_gt",
    save_images=True,
    image_keys=["source", "pred", "target"],
    max_samples=8,
    compute_psnr=True,
    compute_ssim=True,
)

print(results["metrics"])
# {"test/reconstruction/psnr": 32.1, "test/reconstruction/ssim_x100": 87.4}

print(results["artifacts"])
# {"test/images/grid_path": "eval_results/sam_vs_gt/test/images/samples.png"}`} />

        <h3 className="text-lg font-medium mt-6 mb-3 text-neutral-200">All parameters</h3>
        <div className="rounded-2xl border border-white/10 overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-white/5 text-neutral-500 text-xs uppercase tracking-wider">
              <tr>
                <th className="text-left px-4 py-3">Parameter</th>
                <th className="text-left px-4 py-3">Default</th>
                <th className="text-left px-4 py-3">Description</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 text-neutral-300">
              {[
                ['pred', '—', 'Path to .npy or numpy array. Shape (N,H,W) or (N,1,H,W).'],
                ['target', '—', 'Ground-truth array or path. Same shape as pred.'],
                ['source', 'None', 'Optional input array for image grids.'],
                ['output_dir', '"eval_results"', 'Directory where grids are saved.'],
                ['stage', '"test"', 'Label used in result keys and sub-directories.'],
                ['compute_psnr', 'True', 'Whether to compute mean PSNR.'],
                ['compute_ssim', 'True', 'Whether to compute mean SSIM.'],
                ['mask', 'None', 'Optional spatial mask for PSNR/SSIM computation.'],
                ['norm', '"mean"', '"mean" or "01" — normalisation mode.'],
                ['ssim_multiply_by_100', 'True', 'Report SSIM as 0–100 to match training defaults.'],
                ['save_images', 'True', 'Whether to save an image grid.'],
                ['image_keys', 'auto', '["pred","target"] or ["source","pred","target"] when source given.'],
                ['max_samples', '8', 'Number of rows in the image grid.'],
                ['dpi', '150', 'Figure DPI for saved images.'],
              ].map(([param, def_, desc]) => (
                <tr key={param}>
                  <td className="px-4 py-3 font-mono text-xs text-blue-300 align-top">{param}</td>
                  <td className="px-4 py-3 font-mono text-xs text-neutral-500 align-top">{def_}</td>
                  <td className="px-4 py-3 text-xs text-neutral-400 leading-5">{desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="mt-5 rounded-2xl border border-amber-500/20 bg-amber-500/5 p-4 text-sm text-amber-200/80">
          <span className="font-semibold text-amber-300">Note — </span>
          Only <code className="font-mono text-xs">.npy</code> files are supported. For{' '}
          <code className="font-mono text-xs">.npz</code> archives, load the array first:{' '}
          <code className="font-mono text-xs">np.load("file.npz")["arr_0"]</code> and pass the array
          directly.
        </div>
      </section>

      {/* ── output_dir resolution ─────────────────────────────── */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">Where files are saved</h2>
        <p className="text-neutral-300 leading-7 mb-5">
          <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">output_dir</code> is
          resolved at call time, not at construction time. The priority is:
        </p>
        <div className="rounded-2xl border border-white/10 bg-white/5 divide-y divide-white/10 overflow-hidden mb-6 text-sm">
          {[
            { priority: '1st', source: 'logger.log_dir (call-time)', note: 'Always used during training. Passed automatically by the base class from the active Lightning logger.' },
            { priority: '2nd', source: 'output_dir= in constructor', note: 'Fallback. Only used if the call-time value is None — i.e. no logger configured, or standalone mode.' },
            { priority: 'Error', source: 'both None', note: 'Raises ValueError. Set output_dir in the constructor when using standalone mode without a logger.' },
          ].map(({ priority, source, note }) => (
            <div key={priority} className="flex gap-4 px-4 py-3">
              <span className={`text-xs font-mono w-12 shrink-0 pt-0.5 ${priority === 'Error' ? 'text-red-400' : 'text-neutral-500'}`}>{priority}</span>
              <div>
                <code className="text-xs font-mono text-blue-300">{source}</code>
                <p className="text-xs text-neutral-500 mt-0.5 leading-5">{note}</p>
              </div>
            </div>
          ))}
        </div>

        <div className="rounded-2xl border border-white/10 bg-black/30 p-5 font-mono text-sm text-neutral-300">
          <div className="text-neutral-500 mb-2"># training — files saved under the active logger</div>
          <div className="text-neutral-400">{'{logger.log_dir}/{stage}/{name}/global_step_{N}/step_{batch_idx}.png'}</div>
          <div className="mt-4 text-neutral-500 mb-2"># standalone (evaluate_from_npy) — no logger</div>
          <div className="text-neutral-400">{'{output_dir}/{stage}/{name}/samples.png'}</div>
        </div>
      </section>

      {/* ── Custom evaluator ──────────────────────────────────── */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">Writing a custom evaluator</h2>
        <p className="text-neutral-300 leading-7 mb-4">
          Subclass <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">Evaluator</code> and
          implement <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">__call__</code>.
          Override <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">reset()</code> and{' '}
          <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">compute()</code> only if
          your evaluator needs epoch-level aggregation.
        </p>

        <h3 className="text-lg font-medium mb-1 text-neutral-200">Stateless — runs per batch, no aggregation</h3>
        <p className="text-neutral-400 leading-7 mb-4 text-sm">
          For evaluators that produce something each batch and don't need to accumulate — like
          saving a custom plot or computing a simple per-batch stat.
        </p>
        <CodeBlock code={`from oma.evaluation.evaluators.base import Evaluator, EvaluatorOutput
from typing import Any, Mapping, Optional


class ErrorMapEvaluator(Evaluator):
    """Saves a heatmap of |pred - target| every N batches."""

    def __init__(self, name="error_map", save_every_n_steps=50):
        super().__init__(name=name)
        self.save_every_n_steps = save_every_n_steps

    def __call__(
        self,
        *,
        stage: str,
        outputs: Mapping[str, Any],
        output_dir: Optional[str] = None,
        step: Optional[int] = None,
        global_step: Optional[int] = None,
    ) -> EvaluatorOutput:
        if step is not None and step % self.save_every_n_steps != 0:
            return EvaluatorOutput()   # skip — no work, no output

        pred   = outputs["pred"]
        target = outputs["target"]
        error  = (pred - target).abs()

        # ... save error as heatmap to output_dir ...

        return EvaluatorOutput(artifacts={"error_map_path": "/path/to/file.png"})`} />

        <h3 className="text-lg font-medium mt-8 mb-1 text-neutral-200">Stateful — accumulate per batch, report at epoch end</h3>
        <p className="text-neutral-400 leading-7 mb-4 text-sm">
          For evaluators that need data from multiple batches before they can produce a result — like
          computing a metric that requires the full val set.
        </p>
        <CodeBlock code={`from oma.evaluation.evaluators.base import Evaluator, EvaluatorOutput
import numpy as np


class NMSEEvaluator(Evaluator):
    """Accumulates Normalized MSE over the epoch and reports the mean."""

    def __init__(self, name="nmse", pred_key="pred", target_key="target"):
        super().__init__(name=name)
        self.pred_key   = pred_key
        self.target_key = target_key
        self._values = []

    def reset(self) -> None:
        self._values = []                              # ← called at epoch start

    def __call__(self, *, stage, outputs, **kwargs) -> EvaluatorOutput:
        pred   = outputs[self.pred_key]
        target = outputs[self.target_key]

        nmse = ((pred - target) ** 2).sum() / (target ** 2).sum()
        self._values.append(float(nmse))

        return EvaluatorOutput()                       # ← nothing per batch

    def compute(self, **kwargs) -> EvaluatorOutput:   # ← called at epoch end
        if not self._values:
            return EvaluatorOutput()
        return EvaluatorOutput(metrics={"nmse": float(np.mean(self._values))})`} />

        <div className="mt-5 rounded-2xl border border-white/10 bg-white/5 p-5">
          <p className="text-xs uppercase tracking-[0.2em] text-neutral-500 mb-3">Interface summary</p>
          <div className="space-y-2 text-sm text-neutral-300">
            {[
              ['__call__', 'Required. Runs every batch. Returns EvaluatorOutput (can be empty).'],
              ['reset()', 'Optional. Called at epoch start. Clear any accumulated state here.'],
              ['compute()', 'Optional. Called at epoch end. Return EvaluatorOutput with aggregated metrics.'],
            ].map(([method, desc]) => (
              <div key={method} className="flex gap-3">
                <code className="font-mono text-xs text-blue-300 shrink-0 w-24">{method}</code>
                <span className="text-neutral-400 text-xs leading-5">{desc}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── nav ───────────────────────────────────────────────── */}
      <div className="flex gap-4 pt-4 border-t border-white/10">
        <a href="#/docs/guides/logging" className="text-sm text-neutral-400 hover:text-white transition">← Logging & Debugging</a>
        <a href="#/docs/recipes/diffusion-bridge" className="text-sm text-blue-400 hover:text-blue-300 transition ml-auto">Next: Diffusion Bridge Recipe →</a>
      </div>
    </div>
  );
}
