export default function Landing() {
  const pillars = [
    {
      title: 'Research-First Design',
      text: 'Built for medical generative imaging research with modular components, clean abstractions, and minimal friction for experimentation.',
    },
    {
      title: 'Bring Your Own Models',
      text: 'Integrate custom architectures, losses, and training recipes without rewriting the whole framework around rigid assumptions.',
    },
    {
      title: '2D to 3D Extensibility',
      text: 'Designed to evolve from simple image pipelines to scalable volumetric workflows across datasets and model families.',
    },
  ];

  const roadmap = [
    'Core training engine and recipe system',
    'Dataset pipelines for common medical imaging benchmarks',
    'Autoencoder, GAN, and diffusion baselines',
    'Evaluation, visualization, and experiment management',
  ];

  const users = [
    'ML researchers building custom generative models',
    'Research engineers scaling experiments across compute',
    'Teams building production imaging pipelines',
  ];

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100">
      <header className="sticky top-0 z-20 border-b border-white/10 bg-neutral-950/80 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <div>
            <div className="text-sm uppercase tracking-[0.25em] text-neutral-400">AllAxisAI</div>
            <div className="text-xl font-semibold">OpenMedAxis</div>
          </div>
          <nav className="hidden gap-8 text-sm text-neutral-300 md:flex">
            <a href="#about" className="transition hover:text-white">About</a>
            <a href="#pillars" className="transition hover:text-white">Pillars</a>
            <a href="#roadmap" className="transition hover:text-white">Roadmap</a>
            <a href="#/docs" className="transition hover:text-white">Docs</a>
            <a href="https://github.com/AllAxisAI/OpenMedAxis" className="transition hover:text-white">GitHub</a>
          </nav>
        </div>
      </header>

      <main>
        <section className="relative overflow-hidden">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(59,130,246,0.18),transparent_35%),radial-gradient(circle_at_80%_20%,rgba(168,85,247,0.16),transparent_28%)]" />
          <div className="relative mx-auto grid max-w-7xl gap-12 px-6 py-24 md:grid-cols-[1.2fr_0.8fr] md:py-32">
            <div className="max-w-3xl">
              <div className="mb-4 inline-flex rounded-full border border-white/10 bg-white/5 px-4 py-1 text-xs uppercase tracking-[0.25em] text-neutral-300">
                Open-source medical generative imaging framework
              </div>
              <h1 className="text-5xl font-semibold leading-tight tracking-tight md:text-7xl">
                Elegant infrastructure for the next generation of medical imaging research.
              </h1>
              <p className="mt-6 max-w-2xl text-lg leading-8 text-neutral-300 md:text-xl">
                OpenMedAxis is a modular, researcher-friendly framework for building, training, and scaling generative medical imaging systems — from autoencoders and GANs to diffusion-based pipelines.
              </p>
              <div className="mt-10 flex flex-wrap gap-4">
                <a
                  href="#/docs/getting-started"
                  className="rounded-2xl bg-white px-6 py-3 text-sm font-medium text-neutral-950 transition hover:scale-[1.02]"
                >
                  Get started
                </a>
                <a
                  href="#/docs"
                  className="rounded-2xl border border-white/15 bg-white/5 px-6 py-3 text-sm font-medium text-white transition hover:bg-white/10"
                >
                  Read the docs
                </a>
              </div>
            </div>

            <div className="flex items-center">
              <div className="w-full rounded-[2rem] border border-white/10 bg-white/5 p-6 shadow-2xl shadow-black/30 backdrop-blur-sm">
                <div className="mb-4 flex items-center gap-2">
                  <div className="h-3 w-3 rounded-full bg-white/30" />
                  <div className="h-3 w-3 rounded-full bg-white/20" />
                  <div className="h-3 w-3 rounded-full bg-white/10" />
                </div>
                <div className="space-y-3 font-mono text-sm text-neutral-300">
                  <div className="rounded-xl border border-white/10 bg-black/20 p-4">
                    <span className="text-purple-300">from</span>{' '}
                    <span className="text-blue-300">oma</span>{' '}
                    <span className="text-purple-300">import</span> Trainer<br />
                    <span className="text-purple-300">from</span>{' '}
                    <span className="text-blue-300">oma.methods</span>{' '}
                    <span className="text-purple-300">import</span> DiffusionBridgeTranslationMethod<br />
                    <br />
                    trainer = Trainer(max_epochs=<span className="text-orange-300">100</span>, accelerator=<span className="text-emerald-400">"gpu"</span>)<br />
                    trainer.fit(method, datamodule)
                  </div>
                  <div className="rounded-xl border border-white/10 bg-black/20 p-4 text-neutral-400">
                    modular datasets · recipes · losses · logging · evaluation
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section id="about" className="mx-auto max-w-7xl px-6 py-20">
          <div className="grid gap-10 md:grid-cols-2">
            <div>
              <p className="text-sm uppercase tracking-[0.25em] text-neutral-400">Why OpenMedAxis</p>
              <h2 className="mt-4 text-3xl font-semibold tracking-tight md:text-4xl">
                Built to remove friction between ideas and experiments.
              </h2>
            </div>
            <div className="text-lg leading-8 text-neutral-300">
              Many existing frameworks are either too rigid for research or too fragmented for sustainable development. OpenMedAxis aims to provide a clean middle ground: a system that is structured enough for scale, yet flexible enough for real experimentation.
            </div>
          </div>
        </section>

        <section id="pillars" className="mx-auto max-w-7xl px-6 py-8">
          <div className="mb-10">
            <p className="text-sm uppercase tracking-[0.25em] text-neutral-400">Core pillars</p>
            <h2 className="mt-4 text-3xl font-semibold tracking-tight md:text-4xl">
              A framework designed for longevity.
            </h2>
          </div>
          <div className="grid gap-6 md:grid-cols-3">
            {pillars.map((pillar) => (
              <div key={pillar.title} className="rounded-[2rem] border border-white/10 bg-white/5 p-8">
                <h3 className="text-xl font-semibold">{pillar.title}</h3>
                <p className="mt-4 leading-7 text-neutral-300">{pillar.text}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="mx-auto max-w-7xl px-6 py-8">
          <div className="grid gap-6 md:grid-cols-[1fr_1fr]">
            <div className="rounded-[2rem] border border-white/10 bg-white/5 p-8">
              <p className="text-sm uppercase tracking-[0.25em] text-neutral-400">Who it is for</p>
              <h2 className="mt-4 text-3xl font-semibold tracking-tight">Built for people who want velocity without chaos.</h2>
              <div className="mt-6 space-y-4">
                {users.map((item, index) => (
                  <div key={item} className="flex items-start gap-4 rounded-2xl border border-white/10 bg-black/20 p-4">
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-white/15 bg-white/5 text-sm font-semibold text-white">
                      {index + 1}
                    </div>
                    <div className="pt-1 text-neutral-200">{item}</div>
                  </div>
                ))}
              </div>
            </div>

            <div id="quickstart" className="rounded-[2rem] border border-white/10 bg-white/5 p-8">
              <p className="text-sm uppercase tracking-[0.25em] text-neutral-400">Quickstart</p>
              <h2 className="mt-4 text-3xl font-semibold tracking-tight">Get started in seconds.</h2>
              <div className="mt-6 grid gap-6">
                <div className="rounded-2xl border border-white/10 bg-black/30 p-5 font-mono text-sm leading-7 text-neutral-200">
                  <div className="mb-3 text-xs uppercase tracking-[0.2em] text-neutral-500">Install</div>
                  pip install <span className="text-emerald-400">"openmedaxis[full]"</span>
                </div>
                <div className="rounded-2xl border border-white/10 bg-black/30 p-5 font-mono text-sm leading-7 text-neutral-200">
                  <div className="mb-3 text-xs uppercase tracking-[0.2em] text-neutral-500">From source</div>
                  git clone https://github.com/AllAxisAI/OpenMedAxis.git<br />
                  cd OpenMedAxis && pip install -e <span className="text-emerald-400">".[full]"</span>
                </div>
              </div>
              <a
                href="#/docs/getting-started"
                className="mt-6 inline-block text-sm text-blue-400 hover:text-blue-300 transition"
              >
                Full installation guide →
              </a>
            </div>
          </div>
        </section>

        <section id="roadmap" className="mx-auto max-w-7xl px-6 py-20">
          <div className="rounded-[2rem] border border-white/10 bg-gradient-to-br from-white/8 to-white/3 p-8 md:p-12">
            <p className="text-sm uppercase tracking-[0.25em] text-neutral-400">Roadmap</p>
            <div className="mt-6 grid gap-10 md:grid-cols-[0.9fr_1.1fr]">
              <div>
                <h2 className="text-3xl font-semibold tracking-tight md:text-4xl">
                  Starting with a strong foundation, then expanding into a full ecosystem.
                </h2>
                <p className="mt-5 text-lg leading-8 text-neutral-300">
                  The first phase focuses on building a polished training core and reusable research primitives. Future phases grow toward model zoo support, benchmarks, tutorials, and broader ecosystem tooling.
                </p>
              </div>
              <div className="space-y-4">
                {roadmap.map((item, index) => (
                  <div key={item} className="flex items-start gap-4 rounded-2xl border border-white/10 bg-black/20 p-4">
                    <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-white/15 bg-white/5 text-sm font-semibold text-white">
                      {index + 1}
                    </div>
                    <div className="pt-1 text-neutral-200">{item}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>
      </main>

      <footer id="contact" className="border-t border-white/10">
        <div className="mx-auto flex max-w-7xl flex-col gap-6 px-6 py-10 md:flex-row md:items-center md:justify-between">
          <div>
            <div className="text-lg font-semibold">OpenMedAxis</div>
            <div className="mt-2 text-sm text-neutral-400">Open-source infrastructure for medical generative imaging.</div>
          </div>
          <div className="flex flex-wrap gap-4 text-sm text-neutral-300">
            <a href="https://github.com/AllAxisAI/OpenMedAxis" className="rounded-xl border border-white/10 px-4 py-2 transition hover:bg-white/5">GitHub</a>
            <a href="#/docs" className="rounded-xl border border-white/10 px-4 py-2 transition hover:bg-white/5">Documentation</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
