import CodeBlock from '../../components/CodeBlock';

export default function GettingStarted() {
  return (
    <div>
      <div className="mb-2">
        <span className="text-xs uppercase tracking-[0.25em] text-neutral-500">Getting Started</span>
      </div>
      <h1 className="text-4xl font-semibold tracking-tight mb-4">Installation & Quickstart</h1>
      <p className="text-lg leading-8 text-neutral-300 mb-10">
        This guide walks through installing OpenMedAxis and running your first training experiment.
      </p>

      {/* Prerequisites */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-4">Prerequisites</h2>
        <div className="rounded-2xl border border-white/10 bg-white/5 divide-y divide-white/10 overflow-hidden">
          {[
            ['Python', '3.9 or later'],
            ['PyTorch', '2.0 or later (CUDA 11.8+ for GPU training)'],
            ['GPU', '8 GB+ VRAM recommended for full-resolution training'],
            ['OS', 'Linux or macOS (Windows via WSL2)'],
          ].map(([k, v]) => (
            <div key={k} className="flex items-baseline gap-4 px-5 py-3">
              <span className="font-mono text-sm text-neutral-400 w-24 shrink-0">{k}</span>
              <span className="text-sm text-neutral-200">{v}</span>
            </div>
          ))}
        </div>
      </section>

      {/* Installation */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">Installation</h2>
        <p className="text-neutral-300 leading-7 mb-4">
          The recommended install includes all dependencies: PyTorch Lightning, nibabel, SimpleITK,
          torchvision, and evaluation tools.
        </p>

        <h3 className="text-lg font-medium mb-1 text-neutral-200">Via pip</h3>
        <CodeBlock language="bash" code={`pip install "openmedaxis[full]"`} />

        <h3 className="text-lg font-medium mb-1 text-neutral-200">From source</h3>
        <p className="text-sm text-neutral-400 mb-3">
          Use this when you want to modify the framework or stay on the latest commits.
        </p>
        <CodeBlock language="bash" code={`git clone https://github.com/AllAxisAI/OpenMedAxis.git
cd OpenMedAxis
pip install -e ".[full]"`} />

        <h3 className="text-lg font-medium mb-1 text-neutral-200">Verify</h3>
        <CodeBlock code={`import oma
from oma import Trainer
from oma.methods import DiffusionBridgeTranslationMethod
from oma.losses import LossComposer

print("OpenMedAxis ready")`} />
      </section>

      {/* Quickstart */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">Quickstart — Diffusion Bridge Translation</h2>
        <p className="text-neutral-300 leading-7 mb-6">
          This complete example trains a diffusion bridge model to translate T1-weighted MRI to T2-weighted MRI
          on the IXI dataset. It covers every component in the framework: data, model, method, and trainer.
        </p>

        <h3 className="text-lg font-medium mb-1 text-neutral-200">Step 1 — Prepare the IXI dataset</h3>
        <p className="text-sm text-neutral-400 leading-6 mb-3">
          OpenMedAxis ships a data recipe that downloads and preprocesses the IXI dataset into
          numpy arrays organized by modality.
        </p>
        <CodeBlock code={`from oma.data.recipes import IXIRecipe

recipe = IXIRecipe(data_dir="/data/IXI")
recipe.prepare_data()   # Downloads and converts to numpy
recipe.setup()          # Creates train/val split manifest`} />

        <h3 className="text-lg font-medium mb-1 text-neutral-200">Step 2 — Build the data module</h3>
        <CodeBlock code={`from oma.data import LSplitDataModule
from oma.data.datasets import NumpyDataset

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
)`} />

        <h3 className="text-lg font-medium mb-1 text-neutral-200">Step 3 — Define the model</h3>
        <p className="text-sm text-neutral-400 leading-6 mb-3">
          The generator is an NCSNpp backbone (score-based model). The DiffusionBridge wraps
          the diffusion process math — noise scheduling, q_sample, and bridge sampling.
        </p>
        <CodeBlock code={`from oma.models.networks.backbones import NCSNpp
from oma.models.diffusion import DiffusionBridge

bridge_model = NCSNpp(
    image_size=256,
    in_channels=2,    # concatenation of source and noisy target
    out_channels=1,   # predicted clean target (x0)
    ch_mult=(1, 2, 4, 8),
    num_res_blocks=2,
    attn_resolutions=(16,),
    dropout=0.1,
)

diffusion = DiffusionBridge(
    num_timesteps=1000,
    noise_std=0.5,
)`} />

        <h3 className="text-lg font-medium mb-1 text-neutral-200">Step 4 — Configure evaluation</h3>
        <CodeBlock code={`from oma.evaluation import EvaluatorManager
from oma.evaluation.evaluators import SaveImageEvaluator

evaluator_manager = EvaluatorManager(
    evaluators={
        "val_images": SaveImageEvaluator(
            output_dir="outputs/images",
            save_every_n_epochs=5,
        ),
    }
)`} />

        <h3 className="text-lg font-medium mb-1 text-neutral-200">Step 5 — Create the method</h3>
        <p className="text-sm text-neutral-400 leading-6 mb-3">
          The Method owns the training algorithm. It wires the model, diffusion process, optimizer,
          and evaluator together.
        </p>
        <CodeBlock code={`from oma.methods import DiffusionBridgeTranslationMethod

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
    },
    lambda_rec_loss=1.0,
    n_recursions=1,
)`} />

        <h3 className="text-lg font-medium mb-1 text-neutral-200">Step 6 — Train</h3>
        <CodeBlock code={`from oma import Trainer

trainer = Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=1,
    log_every_n_steps=10,
    default_root_dir="outputs/",
)

trainer.fit(method, datamodule)`} />
      </section>

      {/* What's next */}
      <section>
        <h2 className="text-2xl font-semibold mb-4">What's next</h2>
        <div className="grid gap-4 sm:grid-cols-2">
          {[
            { path: '/docs/concepts/method-system', label: 'Method System', desc: 'Understand how training logic is structured.' },
            { path: '/docs/concepts/loss-system', label: 'Loss System', desc: 'Learn to compose pixel, perceptual, and adversarial losses.' },
            { path: '/docs/concepts/data-pipeline', label: 'Data Pipeline', desc: 'Connect your own dataset to the training loop.' },
            { path: '/docs/recipes/diffusion-bridge', label: 'Recipe deep-dive', desc: 'Full annotated walkthrough of the diffusion bridge recipe.' },
          ].map((item) => (
            <a key={item.path} href={`#${item.path}`} className="rounded-2xl border border-white/10 bg-white/5 p-5 transition hover:bg-white/10 group">
              <div className="font-medium text-white group-hover:text-blue-300 transition mb-1">{item.label}</div>
              <div className="text-sm text-neutral-400 leading-snug">{item.desc}</div>
            </a>
          ))}
        </div>
      </section>
    </div>
  );
}
