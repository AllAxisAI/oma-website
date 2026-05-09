import CodeBlock from '../../../components/CodeBlock';

export default function LossSystem() {
  return (
    <div>
      <div className="mb-2">
        <span className="text-xs uppercase tracking-[0.25em] text-neutral-500">Core Concepts</span>
      </div>
      <h1 className="text-4xl font-semibold tracking-tight mb-4">Loss System</h1>
      <p className="text-lg leading-8 text-neutral-300 mb-10">
        OpenMedAxis uses a composable loss architecture. You define individual{' '}
        <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">LossTerm</code> units and combine them
        into a <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">LossComposer</code> that handles
        grouping, weighting, and logging automatically.
      </p>

      {/* Architecture */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-4">Architecture</h2>
        <div className="rounded-2xl border border-white/10 bg-white/5 p-5 mb-6 font-mono text-sm">
          <div className="text-neutral-500 mb-3 text-xs uppercase tracking-wider">Data flow</div>
          <div className="space-y-2 text-neutral-300">
            <div><span className="text-neutral-500">step()</span> → builds <span className="text-blue-300">state dict</span></div>
            <div className="ml-4"><span className="text-blue-300">state dict</span> → passed to <span className="text-purple-300">LossComposer</span></div>
            <div className="ml-8"><span className="text-purple-300">LossComposer</span> → calls each <span className="text-emerald-300">LossTerm.forward(state)</span></div>
            <div className="ml-12"><span className="text-emerald-300">LossTerm</span> → returns <span className="text-orange-300">LossOutput</span> (scalar + logs)</div>
            <div className="ml-8"><span className="text-purple-300">LossComposer</span> → aggregates by <span className="text-yellow-300">group</span> → returns total per group</div>
          </div>
        </div>
        <p className="text-neutral-300 leading-7">
          The key design decision: loss terms read from a shared <strong className="text-white">state dict</strong> rather
          than receiving positional arguments. This makes it trivial to add a new term — just
          read whatever keys you need from state without modifying the method.
        </p>
      </section>

      {/* LossTerm */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">LossTerm</h2>
        <p className="text-neutral-300 leading-7 mb-4">
          A LossTerm is the atomic unit of loss computation. Subclass it and override{' '}
          <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">compute()</code>.
        </p>

        <h3 className="text-lg font-medium mb-1 text-neutral-200">Constructor parameters</h3>
        <div className="rounded-2xl border border-white/10 overflow-hidden mb-6">
          <table className="w-full text-sm">
            <thead className="bg-white/5 text-neutral-500 text-xs uppercase tracking-wider">
              <tr>
                <th className="text-left px-4 py-3">Parameter</th>
                <th className="text-left px-4 py-3">Type</th>
                <th className="text-left px-4 py-3">Description</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 text-neutral-300">
              {[
                ['weight', 'float', 'Scalar multiplier applied to the raw loss value'],
                ['name', 'str | None', 'Used in log keys. Defaults to the class name.'],
                ['group', 'str', 'Which optimizer group this loss belongs to. Default "main".'],
                ['log_keys', 'list[str]', 'Which state keys to include in per-step logs'],
              ].map(([p, t, d]) => (
                <tr key={p as string}>
                  <td className="px-4 py-3 font-mono text-xs text-blue-300">{p}</td>
                  <td className="px-4 py-3 font-mono text-xs text-neutral-500">{t}</td>
                  <td className="px-4 py-3 text-xs">{d}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <h3 className="text-lg font-medium mb-1 text-neutral-200">The compute() contract</h3>
        <CodeBlock code={`from oma.losses.base import LossTerm
import torch

class MyLossTerm(LossTerm):
    def compute(self, state: dict, **kwargs) -> torch.Tensor:
        # state is the shared dict built in step() / build_state()
        pred   = state["recon"]     # model output
        target = state["input"]     # ground truth
        return torch.nn.functional.l1_loss(pred, target)`} />

        <h3 className="text-lg font-medium mb-1 text-neutral-200">LossOutput</h3>
        <p className="text-sm text-neutral-400 leading-6 mb-3">
          <code className="text-sm bg-white/10 px-1.5 py-0.5 rounded font-mono">LossTerm.forward()</code> wraps the scalar returned by <code className="text-sm bg-white/10 px-1.5 py-0.5 rounded font-mono">compute()</code>
          into a <code className="text-sm bg-white/10 px-1.5 py-0.5 rounded font-mono">LossOutput</code> dataclass:
        </p>
        <CodeBlock code={`@dataclass
class LossOutput:
    loss: Tensor          # weighted loss (raw * weight)
    raw_loss: Tensor      # unweighted scalar
    weighted_loss: Tensor # same as loss
    logs: dict            # {name/metric: value, ...}
    group: str            # "main", "disc", etc.`} />
      </section>

      {/* StatefulLossTerm */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">StatefulLossTerm</h2>
        <p className="text-neutral-300 leading-7 mb-4">
          Use <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">StatefulLossTerm</code> when a loss
          term needs to write intermediate results back into the state dict for downstream terms to consume.
          The classic example is a discriminator loss that needs to pass the discriminator's features
          to a feature-matching loss.
        </p>
        <CodeBlock code={`from oma.losses.base import StatefulLossTerm

class DiscriminatorAdversarialTerm(StatefulLossTerm):
    state_write_keys = ["disc_real_features", "disc_fake_features"]

    def compute(self, state: dict, **kwargs) -> torch.Tensor:
        real_out, real_feats = self.disc(state["input"])
        fake_out, fake_feats = self.disc(state["recon"].detach())

        # Write to state so FeatureMatchingLossTerm can read them
        state["disc_real_features"] = real_feats
        state["disc_fake_features"] = fake_feats

        loss = hinge_loss(real_out, fake_out)
        return loss`} />
      </section>

      {/* LossComposer */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">LossComposer</h2>
        <p className="text-neutral-300 leading-7 mb-4">
          <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">LossComposer</code> takes a list of
          LossTerm instances and aggregates them into per-group totals. It returns a dict with three keys:
        </p>
        <CodeBlock code={`from oma.losses import LossComposer
from oma.losses.terms import L1LossTerm, KLLossTerm, LPIPSLossTerm

loss_fn = LossComposer([
    L1LossTerm(weight=1.0,   group="main"),
    KLLossTerm(weight=0.001, group="main"),
    LPIPSLossTerm(weight=0.1, group="main"),
])

# During training:
state = {"input": x, "recon": x_rec, "posterior": posterior, ...}
output = loss_fn(state)

# output is:
# {
#   "losses":       {"main": tensor(...)},
#   "logs":         {"l1_loss": ..., "kl_loss": ..., "lpips": ...},
#   "term_outputs": {"l1": LossOutput, "kl": LossOutput, ...},
# }`} />

        <h3 className="text-lg font-medium mb-1 text-neutral-200">Multi-group (adversarial) example</h3>
        <p className="text-sm text-neutral-400 leading-6 mb-3">
          Assign terms to different groups to use multiple optimizers. The{' '}
          <code className="text-sm bg-white/10 px-1.5 py-0.5 rounded font-mono">GroupedLossMethod</code> will call{' '}
          <code className="text-sm bg-white/10 px-1.5 py-0.5 rounded font-mono">optimizer.step()</code> separately for each group.
        </p>
        <CodeBlock code={`from oma.losses.recipes import build_ldm_autoencoder_loss

# Pre-built recipe: L1 + perceptual + adversarial + KL
loss_fn = build_ldm_autoencoder_loss(
    disc_model=discriminator,
    perceptual_weight=1.0,
    adversarial_weight=0.1,
    kl_weight=0.001,
)

# LossComposer returns two groups:
output = loss_fn(state)
# output["losses"] == {"main": ..., "disc": ...}`} />
      </section>

      {/* Built-in terms table */}
      <section className="mb-10">
        <h2 className="text-2xl font-semibold mb-4">Built-in loss terms</h2>
        <div className="rounded-2xl border border-white/10 overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-white/5 text-neutral-500 text-xs uppercase tracking-wider">
              <tr>
                <th className="text-left px-4 py-3">Class</th>
                <th className="text-left px-4 py-3">Module</th>
                <th className="text-left px-4 py-3">State keys required</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 text-neutral-300">
              {[
                ['L1LossTerm', 'losses.terms.pixel', '"input", "recon"'],
                ['L2LossTerm', 'losses.terms.pixel', '"input", "recon"'],
                ['CharbonnierLossTerm', 'losses.terms.pixel', '"input", "recon"'],
                ['HuberLossTerm', 'losses.terms.pixel', '"input", "recon"'],
                ['KLLossTerm', 'losses.terms.regularization', '"posterior"'],
                ['LatentL1LossTerm', 'losses.terms.regularization', '"latent"'],
                ['LPIPSLossTerm', 'losses.terms.perceptual', '"input", "recon"'],
                ['GeneratorAdversarialTerm', 'losses.terms.adversarial', '"recon", "disc_model"'],
                ['DiscriminatorAdversarialTerm', 'losses.terms.adversarial', '"input", "recon", "disc_model"'],
                ['FeatureMatchingLossTerm', 'losses.terms.adversarial', '"disc_real_features", "disc_fake_features"'],
              ].map(([cls, mod, keys]) => (
                <tr key={cls as string}>
                  <td className="px-4 py-3 font-mono text-xs text-blue-300">{cls}</td>
                  <td className="px-4 py-3 font-mono text-xs text-neutral-500">oma.{mod}</td>
                  <td className="px-4 py-3 text-xs text-neutral-400">{keys}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <div className="flex gap-4 pt-4 border-t border-white/10">
        <a href="#/docs/concepts/method-system" className="text-sm text-neutral-400 hover:text-white transition">← Method System</a>
        <a href="#/docs/concepts/data-pipeline" className="text-sm text-blue-400 hover:text-blue-300 transition ml-auto">Next: Data Pipeline →</a>
      </div>
    </div>
  );
}
