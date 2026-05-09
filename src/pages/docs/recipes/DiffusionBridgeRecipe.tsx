import CodeBlock from '../../../components/CodeBlock';
export default function DiffusionBridgeRecipe() {
  return (
    <div>
      <div className="mb-2">
        <span className="text-xs uppercase tracking-[0.25em] text-neutral-500">Recipes</span>
      </div>
      <h1 className="text-4xl font-semibold tracking-tight mb-4">Diffusion Bridge Translation</h1>
      <p className="text-lg leading-8 text-neutral-300 mb-6">
        A complete annotated walkthrough of the diffusion bridge recipe for paired MRI translation
        (IXI T1 → T2). This is the primary recipe in OpenMedAxis and demonstrates the full framework stack.
      </p>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-4">What is a diffusion bridge?</h2>
        <p className="text-neutral-300 leading-7 mb-4">
          Standard DDPM diffusion starts from pure Gaussian noise. A diffusion bridge defines a
          stochastic process between two images — source{' '}
          <code className="text-sm bg-white/10 px-1.5 py-0.5 rounded font-mono">x</code> and target{' '}
          <code className="text-sm bg-white/10 px-1.5 py-0.5 rounded font-mono">y</code>. The noisy sample at
          timestep <code className="text-sm bg-white/10 px-1.5 py-0.5 rounded font-mono">t</code> is:
        </p>
        <div className="rounded-2xl border border-white/10 bg-white/5 p-5 font-mono text-sm text-neutral-300 mb-4">
          <span className="text-blue-300">x_t</span>{' '}={' '}
          <span className="text-orange-300">sqrt(ᾱ_t)</span> · y +{' '}
          <span className="text-orange-300">sqrt(1 − ᾱ_t)</span> · (x + σ·ε)
          <div className="mt-3 text-neutral-500 text-xs">ε ~ N(0, I), x is the source, y is the target, σ controls bridge noise</div>
        </div>
        <p className="text-neutral-300 leading-7">
          The model learns to predict the clean target <code className="text-sm bg-white/10 px-1.5 py-0.5 rounded font-mono">y</code> (x0-prediction)
          from the bridge sample and source. At inference, recursive x0 predictions iteratively
          denoise toward the target domain without needing an adversarial discriminator.
        </p>
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">Training step — what happens at each iteration</h2>
        <p className="text-neutral-300 leading-7 mb-4">
          Understanding the training loop helps when debugging or extending the method.
        </p>
        <div className="rounded-2xl border border-white/10 bg-white/5 divide-y divide-white/10 overflow-hidden mb-6">
          {[
            ['1', 'Sample t', 'A random timestep t ∈ [0, T] is drawn uniformly for each batch element.'],
            ['2', 'q_sample', 'The target y is corrupted along the bridge: x_t = sqrt(ᾱ_t)·y + sqrt(1−ᾱ_t)·(x + σε).'],
            ['3', 'Model forward', 'The NCSNpp receives [x_t ‖ x] (concatenated along channels) and predicts y_hat (x0 prediction).'],
            ['4', 'Loss', 'L1 loss between y_hat and the true target y, scaled by lambda_rec_loss.'],
            ['5', 'Backprop', 'Standard Adam step. No discriminator, no adversarial training.'],
          ].map(([n, title, desc]) => (
            <div key={n} className="flex gap-4 px-5 py-4">
              <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-white/15 bg-white/5 text-xs font-semibold text-white">{n}</span>
              <div>
                <div className="text-sm font-medium text-white mb-0.5">{title}</div>
                <div className="text-sm text-neutral-400 leading-6">{desc}</div>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">Complete training script</h2>
        <CodeBlock filename="recipes/diffusion_bridge_translation/train.py" code={`from oma import Trainer
from oma.data import LSplitDataModule
from oma.data.datasets import NumpyDataset
from oma.data.recipes import IXIRecipe
from oma.methods import DiffusionBridgeTranslationMethod
from oma.models.networks.backbones import NCSNpp
from oma.models.diffusion import DiffusionBridge
from oma.evaluation import EvaluatorManager
from oma.evaluation.evaluators import SaveImageEvaluator

# ── 1. Prepare data ───────────────────────────────────────────────────────────
recipe = IXIRecipe(data_dir="/data/IXI")
recipe.prepare_data()
recipe.setup()

# ── 2. DataModule ─────────────────────────────────────────────────────────────
datamodule = LSplitDataModule(
    dataset_cls=NumpyDataset,
    dataset_kwargs={
        "data_dir": "/data/IXI",
        "source_modality": "T1",
        "target_modality": "T2",
        "image_size": 256,
    },
    train_dataloader_kwargs={"batch_size": 4, "num_workers": 4, "shuffle": True},
    val_dataloader_kwargs={"batch_size": 2, "num_workers": 2},
    manifest_path="/data/IXI/split_manifest.json",
)

# ── 3. Model ──────────────────────────────────────────────────────────────────
bridge_model = NCSNpp(
    image_size=256,
    in_channels=2,          # [x_t ‖ source] concatenated
    out_channels=1,         # predicted clean target y
    ch_mult=(1, 2, 4, 8),
    num_res_blocks=2,
    attn_resolutions=(16,),
    dropout=0.1,
)

diffusion = DiffusionBridge(
    num_timesteps=1000,
    noise_std=0.5,          # σ — bridge noise amplitude
)

# ── 4. Evaluation ─────────────────────────────────────────────────────────────
evaluator_manager = EvaluatorManager(
    evaluators={
        "val_images": SaveImageEvaluator(
            output_dir="outputs/images",
            save_every_n_epochs=5,
            n_samples=8,
        ),
    }
)

# ── 5. Method ─────────────────────────────────────────────────────────────────
method = DiffusionBridgeTranslationMethod(
    bridge_model=bridge_model,
    diffusion=diffusion,
    evaluator_manager=evaluator_manager,
    optimizer_cfg={
        "type": "Adam",
        "lr": 1e-4,
        "betas": (0.9, 0.999),
    },
    scheduler_cfg={
        "type": "CosineAnnealingLR",
        "T_max": 100,
        "eta_min": 1e-6,
    },
    lambda_rec_loss=1.0,    # weight for the L1 bridge loss
    n_recursions=1,         # recursive x0 refinement steps at inference
)

# ── 6. Trainer ────────────────────────────────────────────────────────────────
trainer = Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=1,
    log_every_n_steps=10,
    val_check_interval=1.0,
    default_root_dir="outputs/",
)

trainer.fit(method, datamodule)`} />
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">Key hyperparameters</h2>
        <div className="rounded-2xl border border-white/10 overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-white/5 text-neutral-500 text-xs uppercase tracking-wider">
              <tr>
                <th className="text-left px-4 py-3">Parameter</th>
                <th className="text-left px-4 py-3">Default</th>
                <th className="text-left px-4 py-3">Effect</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 text-neutral-300">
              {[
                ['noise_std (σ)', '0.5', 'Bridge noise amplitude. Higher = more stochastic, harder to train.'],
                ['num_timesteps', '1000', 'Diffusion chain length. Can reduce to 100–250 for faster training.'],
                ['lambda_rec_loss', '1.0', 'Weight on the L1 reconstruction loss.'],
                ['n_recursions', '1', 'Recursive x0 refinement steps at inference. 1–5 range.'],
                ['lr', '1e-4', 'Adam learning rate. 5e-5 for fine-tuning on small datasets.'],
                ['ch_mult', '(1,2,4,8)', 'UNet channel multipliers. Reduce to (1,2,4) for smaller GPUs.'],
              ].map(([p, d, e]) => (
                <tr key={p as string}>
                  <td className="px-4 py-3 font-mono text-xs text-blue-300">{p}</td>
                  <td className="px-4 py-3 font-mono text-xs text-neutral-500">{d}</td>
                  <td className="px-4 py-3 text-xs">{e}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="mb-10">
        <h2 className="text-2xl font-semibold mb-4">Extending the recipe</h2>
        <div className="space-y-4">
          {[
            {
              title: 'Different modality pair',
              desc: 'Change source_modality and target_modality in the dataset kwargs. The rest is identical.',
            },
            {
              title: 'Add perceptual loss',
              desc: 'Subclass DiffusionBridgeTranslationMethod, override compute_loss() to add an LPIPSLossTerm on the final prediction.',
            },
            {
              title: 'Multi-GPU training',
              desc: 'Set devices=N and strategy="ddp" in the Trainer. The Method and DataModule are Lightning-compatible out of the box.',
            },
            {
              title: '3D volumetric inputs',
              desc: 'Swap NCSNpp for a 3D-aware backbone and adjust in_channels. The diffusion process is architecture-agnostic.',
            },
          ].map((item) => (
            <div key={item.title} className="rounded-2xl border border-white/10 bg-white/5 p-5">
              <div className="font-medium text-white mb-1">{item.title}</div>
              <div className="text-sm text-neutral-400 leading-6">{item.desc}</div>
            </div>
          ))}
        </div>
      </section>

      <div className="flex gap-4 pt-4 border-t border-white/10">
        <a href="#/docs/guides/custom-loss" className="text-sm text-neutral-400 hover:text-white transition">← Custom Loss</a>
        <a href="https://github.com/AllAxisAI/OpenMedAxis" target="_blank" rel="noreferrer" className="text-sm text-blue-400 hover:text-blue-300 transition ml-auto">View on GitHub →</a>
      </div>
    </div>
  );
}
