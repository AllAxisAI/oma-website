import CodeBlock from '../../../components/CodeBlock';

export default function DiffusionSystem() {
  return (
    <div>
      <div className="mb-2">
        <span className="text-xs uppercase tracking-[0.25em] text-neutral-500">Core Concepts</span>
      </div>
      <h1 className="text-4xl font-semibold tracking-tight mb-4">Diffusion System</h1>
      <p className="text-lg leading-8 text-neutral-300 mb-10">
        OpenMedAxis treats diffusion as a set of composable, swappable components. You choose a
        process, an objective, a sampler, and a time sampler — then plug them into a method.
        Each component can be replaced independently without touching the others.
      </p>

      {/* Architecture */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-4">Component architecture</h2>
        <div className="rounded-2xl border border-white/10 bg-white/5 p-6 mb-6">
          <p className="text-xs uppercase tracking-[0.2em] text-neutral-500 mb-5">Four independent axes</p>
          <div className="grid gap-3 sm:grid-cols-2">
            {[
              {
                name: 'Process',
                color: 'border-blue-500/40 bg-blue-500/5',
                role: 'Forward corruption math',
                options: 'Gaussian · I2SB · SelfRDB · VESDE',
                desc: 'Defines how x_t is built from x0, time t, and noise. Owns the schedule buffers and forward_state().',
              },
              {
                name: 'Objective',
                color: 'border-purple-500/40 bg-purple-500/5',
                role: 'What the model predicts',
                options: 'ε · x0 · Residual · Velocity',
                desc: 'Maps model output to a supervision target. Populates pred_key and target_key in state for loss terms.',
              },
              {
                name: 'Sampler',
                color: 'border-emerald-500/40 bg-emerald-500/5',
                role: 'Reverse / inference procedure',
                options: 'DDPM · DDIM · Langevin · SingleStep',
                desc: 'Runs the full iterative reverse chain at inference time. Separate val_sampler and test_sampler are supported.',
              },
              {
                name: 'Time Sampler',
                color: 'border-orange-500/40 bg-orange-500/5',
                role: 'Timestep distribution during training',
                options: 'Uniform · Importance',
                desc: 'Draws the batch of timesteps t at each training step. Importance sampling can focus training on hard timesteps.',
              },
            ].map((c) => (
              <div key={c.name} className={`rounded-xl border p-4 ${c.color}`}>
                <div className="flex items-baseline gap-2 mb-1">
                  <span className="font-mono text-sm font-semibold text-white">{c.name}</span>
                  <span className="text-xs text-neutral-500">{c.role}</span>
                </div>
                <div className="text-xs text-neutral-400 font-mono mb-2">{c.options}</div>
                <div className="text-xs text-neutral-400 leading-5">{c.desc}</div>
              </div>
            ))}
          </div>
        </div>
        <p className="text-neutral-300 leading-7">
          These four components plug into <code className="text-sm bg-white/10 px-1.5 py-0.5 rounded font-mono">BaseDiffusionMethod</code> (or its subclasses).
          The method orchestrates them: sample time → corrupt x0 via process → call model → apply objective → compute loss.
        </p>
      </section>

      {/* Standard DDPM setup */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">Standard Gaussian DDPM</h2>
        <p className="text-neutral-300 leading-7 mb-4">
          The simplest configuration: linear schedule, epsilon prediction, ancestral DDPM sampling.
        </p>
        <CodeBlock code={`from oma.methods.diffusion import GaussianDiffusionMethod
from oma.methods.diffusion.components.processes import GaussianDiffusionProcess
from oma.methods.diffusion.components.objective import EpsilonObjective
from oma.methods.diffusion.components.samplers import DDPMSampler
from oma.methods.diffusion.components.time_samplers import UniformTimeSampler

process = GaussianDiffusionProcess(
    num_steps=1000,
    schedule="linear",
    beta_start=1e-4,
    beta_end=2e-2,
)

method = GaussianDiffusionMethod(
    model=model,
    process=process,
    objective=EpsilonObjective(),
    sampler=DDPMSampler(),
    time_sampler=UniformTimeSampler(num_steps=1000),
    optimizer_cfg={"type": "Adam", "lr": 2e-4},
)`} />
      </section>

      {/* Switching to DDIM */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">Faster inference with DDIM</h2>
        <p className="text-neutral-300 leading-7 mb-4">
          Keep the same training setup but switch to DDIM at inference for 10-20× fewer steps.
          Use <code className="text-sm bg-white/10 px-1.5 py-0.5 rounded font-mono">val_sampler</code> to
          use a different sampler during validation without affecting training.
        </p>
        <CodeBlock code={`from oma.methods.diffusion.components.samplers import DDIMSampler

method = GaussianDiffusionMethod(
    model=model,
    process=process,
    objective=EpsilonObjective(),

    # Training: not used (no sampler here, training doesn't sample)
    sampler=DDIMSampler(
        num_inference_steps=50,   # full reverse in 50 steps instead of 1000
        eta=0.0,                  # fully deterministic
    ),

    # Optionally use a different (faster) sampler just for validation
    val_sampler=DDIMSampler(
        num_inference_steps=20,
        eta=0.0,
    ),

    sample_on_val=True,           # generate samples at each validation epoch
    optimizer_cfg={"type": "Adam", "lr": 2e-4},
)`} />
      </section>

      {/* Switching objective */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">Switching prediction objective</h2>
        <p className="text-neutral-300 leading-7 mb-4">
          Change what the model predicts by swapping the objective. The process, sampler, and
          time sampler stay exactly the same.
        </p>

        <h3 className="text-lg font-medium mb-1 text-neutral-200">x0 prediction</h3>
        <p className="text-sm text-neutral-400 leading-6 mb-3">
          The model directly outputs the clean image. Often more stable for medical imaging.
        </p>
        <CodeBlock code={`from oma.methods.diffusion.components.objective import X0Objective

process = GaussianDiffusionProcess(num_steps=1000, schedule="cosine")

method = GaussianDiffusionMethod(
    model=model,
    process=process,
    objective=X0Objective(),    # model outputs x0 directly
    sampler=DDIMSampler(num_inference_steps=50),
    optimizer_cfg={"type": "AdamW", "lr": 1e-4, "weight_decay": 1e-2},
)`} />

        <h3 className="text-lg font-medium mb-1 text-neutral-200 mt-6">Velocity prediction</h3>
        <p className="text-sm text-neutral-400 leading-6 mb-3">
          Introduced in progressive distillation. Balances between ε and x0 objectives.
          The process computes velocity_target automatically.
        </p>
        <CodeBlock code={`from oma.methods.diffusion.components.objective import VelocityObjective

method = GaussianDiffusionMethod(
    model=model,
    process=GaussianDiffusionProcess(num_steps=1000, schedule="cosine"),
    objective=VelocityObjective(),   # model predicts v = √ᾱ·ε - √(1-ᾱ)·x0
    sampler=DDIMSampler(num_inference_steps=50, eta=0.0),
    optimizer_cfg={"type": "Adam", "lr": 1e-4},
)`} />

        <h3 className="text-lg font-medium mb-1 text-neutral-200 mt-6">Residual prediction</h3>
        <p className="text-sm text-neutral-400 leading-6 mb-3">
          The model predicts the difference between x0 and a reference (default: x_t).
          Useful when source and target are close in appearance.
        </p>
        <CodeBlock code={`from oma.methods.diffusion.components.objective import ResidualObjective

method = GaussianDiffusionMethod(
    model=model,
    process=process,
    objective=ResidualObjective(reference_key="xt"),  # model predicts x0 - xt
    optimizer_cfg={"type": "Adam", "lr": 1e-4},
)`} />
      </section>

      {/* Importance sampling */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">Importance-weighted time sampling</h2>
        <p className="text-neutral-300 leading-7 mb-4">
          By default, timesteps are sampled uniformly. With{' '}
          <code className="text-sm bg-white/10 px-1.5 py-0.5 rounded font-mono">ImportanceTimeSampler</code>,
          you can focus training on harder timesteps — for example, weighting by SNR or by empirical loss.
        </p>
        <CodeBlock code={`import torch
from oma.methods.diffusion.components.time_samplers import ImportanceTimeSampler

num_steps = 1000

# Min-SNR weighting: focus training on mid-range timesteps
# (high t = too noisy, low t = trivial; mid t is where gradients are informative)
t = torch.arange(num_steps, dtype=torch.float32)
snr = (t / num_steps).clamp(1e-5)          # proxy for SNR
min_snr = torch.minimum(snr, torch.tensor(5.0))  # min-SNR-5 clipping
weights = min_snr

time_sampler = ImportanceTimeSampler(weights=weights)

method = GaussianDiffusionMethod(
    model=model,
    process=GaussianDiffusionProcess(num_steps=num_steps),
    objective=EpsilonObjective(),
    sampler=DDIMSampler(num_inference_steps=50),
    time_sampler=time_sampler,             # replaces UniformTimeSampler
    optimizer_cfg={"type": "Adam", "lr": 2e-4},
)`} />
      </section>

      {/* Cosine schedule */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">Cosine vs linear schedule</h2>
        <p className="text-neutral-300 leading-7 mb-4">
          Switch the noise schedule in the process. Cosine is smoother at the boundaries
          and generally preferred for image quality.
        </p>
        <CodeBlock code={`# Linear schedule (original DDPM)
process = GaussianDiffusionProcess(
    num_steps=1000,
    schedule="linear",
    beta_start=1e-4,
    beta_end=2e-2,
)

# Cosine schedule (improved DDPM, generally better)
process = GaussianDiffusionProcess(
    num_steps=1000,
    schedule="cosine",
    cosine_s=0.008,         # small offset to avoid singularity near t=0
)

# Custom schedule: pass any callable → tensor function
def my_schedule(T: int) -> torch.Tensor:
    t = torch.linspace(0, 1, T)
    return 0.001 * t ** 2   # quadratic schedule

process = GaussianDiffusionProcess(
    num_steps=1000,
    schedule_fn=my_schedule,
)`} />
      </section>

      {/* VESDE + Langevin */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">VE-SDE with Langevin sampling</h2>
        <p className="text-neutral-300 leading-7 mb-4">
          For score-based models (NCSNpp style), use the Variance Exploding process with
          annealed Langevin dynamics. The mean is preserved — only variance grows.
        </p>
        <CodeBlock code={`from oma.methods.diffusion.components.processes import VESDEProcess
from oma.methods.diffusion.components.samplers import AnnealedLangevinSampler
from oma.methods.diffusion.components.objective import EpsilonObjective

process = VESDEProcess(
    num_levels=232,        # number of discrete noise levels
    sigma_min=0.01,        # lowest noise σ_0
    sigma_max=50.0,        # highest noise σ_{T-1}
    prediction_type="epsilon",
)

sampler = AnnealedLangevinSampler(
    n_steps=10,            # Langevin steps per noise level
    step_size=2e-5,        # base step size ε
    denoise_last=True,     # deterministic final step
    noise_level_order="descending",   # σ_max → σ_min
)

method = GaussianDiffusionMethod(
    model=ncsnpp_model,
    process=process,
    objective=EpsilonObjective(),
    sampler=sampler,
    time_sampler=UniformTimeSampler(num_steps=232),
    optimizer_cfg={"type": "Adam", "lr": 1e-4},
)`} />
      </section>

      {/* Diffusion Bridge */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">Diffusion bridge — paired translation</h2>
        <p className="text-neutral-300 leading-7 mb-4">
          For paired image-to-image translation (MRI T1 → T2, CT → MRI, etc.),
          use <code className="text-sm bg-white/10 px-1.5 py-0.5 rounded font-mono">DiffusionBridgeMethod</code> with
          a bridge process. The bridge connects source and target distributions
          rather than starting from pure noise.
        </p>

        <h3 className="text-lg font-medium mb-1 text-neutral-200">Self-consistent Recursive Diffusion Bridge (SelfRDB)</h3>
        <CodeBlock code={`from oma.methods.diffusion import DiffusionBridgeMethod
from oma.methods.diffusion.components.processes import SelfRDBProcess
from oma.methods.diffusion.components.objective import X0Objective

process = SelfRDBProcess(
    n_steps=10,                    # short chain — bridge needs fewer steps
    beta_start=0.1,
    beta_end=3.0,
    gamma=1.0,                     # noise magnitude scaling
    n_recursions=3,                # recursive x0 refinement per timestep
    consistency_threshold=0.01,    # early stop if refinements converge
)

method = DiffusionBridgeMethod(
    model=bridge_model,
    process=process,
    objective=X0Objective(),       # model predicts clean target directly
    lambda_rec=1.0,                # L1 reconstruction weight
    n_recursions=3,                # override process.n_recursions at inference
    optimizer_cfg={"type": "Adam", "lr": 1e-4},
    scheduler_cfg={"type": "CosineAnnealingLR", "T_max": 100},
)`} />

        <h3 className="text-lg font-medium mb-1 text-neutral-200 mt-6">Image-to-Image Schrödinger Bridge (I2SB)</h3>
        <p className="text-sm text-neutral-400 leading-6 mb-3">
          I2SB uses a principled Schrödinger bridge formulation with clean linear coefficients.
          Theoretically grounded, often produces sharper outputs.
        </p>
        <CodeBlock code={`from oma.methods.diffusion.components.processes import I2SBProcess

process = I2SBProcess(
    num_steps=1000,
    schedule="linear",
    beta_start=1e-4,
    beta_end=2e-2,
)

method = DiffusionBridgeMethod(
    model=bridge_model,
    process=process,
    objective=X0Objective(),
    lambda_rec=1.0,
    optimizer_cfg={"type": "Adam", "lr": 1e-4},
)`} />
      </section>

      {/* Custom process */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">Writing a custom process</h2>
        <p className="text-neutral-300 leading-7 mb-4">
          Subclass <code className="text-sm bg-white/10 px-1.5 py-0.5 rounded font-mono">BaseDiffusionProcess</code> and
          implement <code className="text-sm bg-white/10 px-1.5 py-0.5 rounded font-mono">forward_state()</code>.
          The method will call your process automatically — no other changes needed.
        </p>
        <CodeBlock code={`import torch
import torch.nn as nn
from oma.methods.diffusion.components.processes.base import BaseDiffusionProcess


class CyclicNoiseProcess(BaseDiffusionProcess, nn.Module):
    """
    A custom process where noise follows a sinusoidal schedule.
    Demonstrates how easy it is to plug in new process math.
    """

    def __init__(self, num_steps: int = 1000, max_sigma: float = 1.0):
        super().__init__()
        self.num_steps = num_steps
        t = torch.linspace(0, torch.pi, num_steps)
        # Sinusoidal noise: starts low, peaks in middle, returns low
        sigma = max_sigma * torch.sin(t)
        self.register_buffer("sigma", sigma)

    def sample_time(self, batch_size, device, stage, state=None):
        return torch.randint(0, self.num_steps, (batch_size,), device=device)

    def forward_state(self, *, x0, t, cond=None, noise=None, **kwargs):
        if noise is None:
            noise = torch.randn_like(x0)

        sigma_t = self.sigma[t].view(-1, 1, 1, 1)
        xt = x0 + sigma_t * noise   # additive noise, mean preserved

        return {
            "x0": x0,
            "xt": xt,
            "t": t,
            "noise": noise,
            "cond": cond,
            "process_aux": {"sigma_t": sigma_t},
        }

    def predict_x0(self, *, model_pred, xt, t, **kwargs):
        # If model predicts epsilon: x0 = xt - sigma_t * eps_pred
        sigma_t = self.sigma[t].view(-1, 1, 1, 1)
        return xt - sigma_t * model_pred


# Use it exactly like any built-in process:
method = GaussianDiffusionMethod(
    model=model,
    process=CyclicNoiseProcess(num_steps=1000, max_sigma=0.8),
    objective=EpsilonObjective(),
    sampler=DDIMSampler(num_inference_steps=50),
    optimizer_cfg={"type": "Adam", "lr": 1e-4},
)`} />
      </section>

      {/* Custom sampler */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">Writing a custom sampler</h2>
        <p className="text-neutral-300 leading-7 mb-4">
          Subclass <code className="text-sm bg-white/10 px-1.5 py-0.5 rounded font-mono">BaseDiffusionSampler</code> and
          implement <code className="text-sm bg-white/10 px-1.5 py-0.5 rounded font-mono">sample()</code>.
          Use <code className="text-sm bg-white/10 px-1.5 py-0.5 rounded font-mono">self.call_model()</code> and{' '}
          <code className="text-sm bg-white/10 px-1.5 py-0.5 rounded font-mono">self.reconstruct_clean()</code> for
          portable model calls that work with any method and process combination.
        </p>
        <CodeBlock code={`from oma.methods.diffusion.components.samplers.base import BaseDiffusionSampler
import torch


class StochasticDDIMSampler(BaseDiffusionSampler):
    """
    DDIM with configurable stochasticity and optional self-refinement.
    Demonstrates the BaseDiffusionSampler API.
    """

    def __init__(self, num_steps: int = 50, eta: float = 0.5, refine_steps: int = 0):
        self.num_steps = num_steps
        self.eta = eta           # 0 = deterministic, 1 = DDPM-like
        self.refine_steps = refine_steps   # extra refinement at t=0

    def sample(self, *, model, process, cond=None, shape=None,
               x_init=None, method=None, **kwargs):
        device, dtype = self.get_model_device_and_dtype(model)
        x = self.prepare_initial_state(shape=shape, x_init=x_init, device=device, dtype=dtype)
        B = x.shape[0]

        timesteps = list(range(process.num_steps - 1, 0, -process.num_steps // self.num_steps))

        for i, t_val in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            t = self.make_time_tensor(t_value=t_val, batch_size=B, device=device)
            t_p = self.make_time_tensor(t_value=t_prev, batch_size=B, device=device)

            # Call model — works with any objective/method combination
            model_pred = self.call_model(model=model, x=x, t=t, cond=cond, method=method)

            # Reconstruct x0 — works with any process/objective combination
            x0_pred = self.reconstruct_clean(
                model_pred=model_pred, x=x, t=t,
                cond=cond, process=process, method=method,
            )

            # DDIM update
            alpha_bar = process.alphas_cumprod[t_val]
            alpha_bar_prev = process.alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0)
            eps = (x - alpha_bar.sqrt() * x0_pred) / (1 - alpha_bar).sqrt()
            noise = torch.randn_like(x) * self.eta
            x = alpha_bar_prev.sqrt() * x0_pred + (1 - alpha_bar_prev).sqrt() * eps + noise

        return x


# Plug in as sampler or val_sampler:
method = GaussianDiffusionMethod(
    model=model,
    process=process,
    objective=X0Objective(),
    sampler=StochasticDDIMSampler(num_steps=50, eta=0.3),
    optimizer_cfg={"type": "Adam", "lr": 1e-4},
)`} />
      </section>

      {/* Reference table */}
      <section className="mb-10">
        <h2 className="text-2xl font-semibold mb-4">Component reference</h2>

        {[
          {
            title: 'Processes',
            rows: [
              ['GaussianDiffusionProcess', 'processes.gaussian', 'Standard DDPM. Linear or cosine schedule. Supports ε, x0, v prediction types.'],
              ['I2SBProcess', 'processes.i2sb', 'Schrödinger Bridge. Theoretically optimal transport between two image distributions.'],
              ['SelfRDBProcess', 'processes.selfrdb', 'Recursive Diffusion Bridge. Short chains (n_steps≈10), recursive x0 refinement with consistency stopping.'],
              ['VESDEProcess', 'processes.vesde', 'Variance Exploding SDE. Mean-preserving, geometric sigma schedule. For score-based / NCSNpp models.'],
              ['IdentityProcess', 'processes.base', 'No-op. xt = x0. Useful for pipeline smoke tests.'],
            ],
          },
          {
            title: 'Objectives',
            rows: [
              ['EpsilonObjective', 'objective', 'Model predicts added noise ε. Standard DDPM objective.'],
              ['X0Objective', 'objective', 'Model predicts clean image x0 directly. More interpretable outputs.'],
              ['ResidualObjective', 'objective', 'Model predicts x0 − reference. Good when source and target are close.'],
              ['VelocityObjective', 'objective', 'Model predicts v = √ᾱ·ε − √(1−ᾱ)·x0. Balanced SNR weighting.'],
            ],
          },
          {
            title: 'Samplers',
            rows: [
              ['DDPMSampler', 'samplers.ddpm', 'Ancestral stochastic sampling. Requires full T reverse steps.'],
              ['DDIMSampler', 'samplers.ddim', 'Deterministic sampling with step skipping. 10-20× faster. eta controls stochasticity [0, 1].'],
              ['AnnealedLangevinSampler', 'samplers.langevin', 'For VE-SDE. Iterates Langevin MCMC at each noise level.'],
              ['SingleStepSampler', 'samplers.base', 'One forward pass at fixed t. For smoke tests only.'],
            ],
          },
          {
            title: 'Time Samplers',
            rows: [
              ['UniformTimeSampler', 'time_samplers.uniform', 'Uniform distribution over [low, num_steps). Default.'],
              ['ImportanceTimeSampler', 'time_samplers.importance', 'Weighted multinomial sampling. Pass any weight tensor (e.g. min-SNR, loss-proportional).'],
            ],
          },
        ].map((group) => (
          <div key={group.title} className="mb-6">
            <h3 className="text-base font-medium text-neutral-300 mb-2">{group.title}</h3>
            <div className="rounded-2xl border border-white/10 overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-white/5 text-neutral-500 text-xs uppercase tracking-wider">
                  <tr>
                    <th className="text-left px-4 py-2.5">Class</th>
                    <th className="text-left px-4 py-2.5 hidden sm:table-cell">Import from</th>
                    <th className="text-left px-4 py-2.5">Description</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {group.rows.map(([cls, mod, desc]) => (
                    <tr key={cls}>
                      <td className="px-4 py-3 font-mono text-xs text-blue-300 whitespace-nowrap">{cls}</td>
                      <td className="px-4 py-3 font-mono text-xs text-neutral-600 hidden sm:table-cell">oma.methods.diffusion.components.{mod}</td>
                      <td className="px-4 py-3 text-xs text-neutral-400">{desc}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ))}
      </section>

      <div className="flex gap-4 pt-4 border-t border-white/10">
        <a href="#/docs/concepts/data-pipeline" className="text-sm text-neutral-400 hover:text-white transition">← Data Pipeline</a>
        <a href="#/docs/guides/custom-method" className="text-sm text-blue-400 hover:text-blue-300 transition ml-auto">Next: Custom Method →</a>
      </div>
    </div>
  );
}
