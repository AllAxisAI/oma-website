const sections = [
  {
    path: '/docs/getting-started',
    label: 'Installation & Quickstart',
    desc: 'Install OpenMedAxis and run your first training experiment in minutes.',
  },
  {
    path: '/docs/concepts/method-system',
    label: 'Method System',
    desc: 'Learn how the Method abstraction wraps training logic, optimizers, and metrics.',
  },
  {
    path: '/docs/concepts/loss-system',
    label: 'Loss System',
    desc: 'Compose losses from atomic LossTerm units using the LossComposer.',
  },
  {
    path: '/docs/concepts/data-pipeline',
    label: 'Data Pipeline',
    desc: 'Connect datasets and datamodules to your training loop in a standard way.',
  },
  {
    path: '/docs/guides/custom-method',
    label: 'Writing a Custom Method',
    desc: 'Step-by-step guide to building your own training method.',
  },
  {
    path: '/docs/guides/custom-loss',
    label: 'Writing a Custom Loss',
    desc: 'Create composable loss terms for any training objective.',
  },
  {
    path: '/docs/recipes/diffusion-bridge',
    label: 'Diffusion Bridge Recipe',
    desc: 'End-to-end walkthrough of paired MRI translation using a diffusion bridge.',
  },
];

const architecture = [
  { name: 'Trainer', role: 'Executes infrastructure (PyTorch Lightning)', color: 'border-blue-500/40 bg-blue-500/5' },
  { name: 'Method', role: 'Defines the algorithm — step(), losses, optimizers', color: 'border-purple-500/40 bg-purple-500/5' },
  { name: 'DataModule', role: 'Provides data — train/val/test dataloaders', color: 'border-emerald-500/40 bg-emerald-500/5' },
  { name: 'LossComposer', role: 'Aggregates LossTerm units into grouped totals', color: 'border-orange-500/40 bg-orange-500/5' },
];

export default function DocsHome() {
  return (
    <div>
      <div className="mb-2">
        <span className="text-xs uppercase tracking-[0.25em] text-neutral-500">Introduction</span>
      </div>
      <h1 className="text-4xl font-semibold tracking-tight mb-4">OpenMedAxis Documentation</h1>
      <p className="text-lg leading-8 text-neutral-300 mb-10">
        OpenMedAxis is a PyTorch Lightning-based framework for generative medical imaging research.
        It provides clean abstractions for training methods, composable losses, medical imaging datasets,
        and evaluation — without hiding the details that researchers care about.
      </p>

      {/* Architecture overview */}
      <div className="rounded-2xl border border-white/10 bg-white/5 p-6 mb-10">
        <p className="text-xs uppercase tracking-[0.2em] text-neutral-500 mb-5">Core architecture</p>
        <div className="grid gap-3 sm:grid-cols-2">
          {architecture.map((item) => (
            <div key={item.name} className={`rounded-xl border p-4 ${item.color}`}>
              <div className="font-mono text-sm font-semibold text-white mb-1">{item.name}</div>
              <div className="text-sm text-neutral-400 leading-snug">{item.role}</div>
            </div>
          ))}
        </div>
        <p className="mt-5 text-sm text-neutral-500 leading-6">
          The <span className="text-white">Trainer</span> drives training via Lightning.
          The <span className="text-white">Method</span> owns the algorithm.
          The <span className="text-white">DataModule</span> owns data.
          The <span className="text-white">LossComposer</span> aggregates atomic loss terms — enabling adversarial, perceptual, and reconstruction losses in a single composable system.
        </p>
      </div>

      {/* Navigation cards */}
      <h2 className="text-xl font-semibold mb-5">Browse the docs</h2>
      <div className="grid gap-4 sm:grid-cols-2">
        {sections.map((s) => (
          <a
            key={s.path}
            href={`#${s.path}`}
            className="rounded-2xl border border-white/10 bg-white/5 p-5 transition hover:bg-white/10 hover:border-white/20 group"
          >
            <div className="font-medium text-white group-hover:text-blue-300 transition mb-1">{s.label}</div>
            <div className="text-sm text-neutral-400 leading-snug">{s.desc}</div>
          </a>
        ))}
      </div>

      <div className="mt-10 rounded-2xl border border-white/10 bg-black/20 p-6">
        <p className="text-sm text-neutral-400 leading-7">
          <span className="text-white font-medium">Status:</span> OpenMedAxis is currently in pre-alpha (v0.0.5).
          The core training engine, loss system, data pipeline, and diffusion bridge method are functional.
          APIs may change before the stable release. Follow the{' '}
          <a href="https://github.com/AllAxisAI/OpenMedAxis" className="text-blue-400 hover:underline" target="_blank" rel="noreferrer">
            GitHub repository
          </a>{' '}
          for updates.
        </p>
      </div>
    </div>
  );
}
