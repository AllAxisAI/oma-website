import CodeBlock from '../../../components/CodeBlock';

export default function CustomLoss() {
  return (
    <div>
      <div className="mb-2">
        <span className="text-xs uppercase tracking-[0.25em] text-neutral-500">Guides</span>
      </div>
      <h1 className="text-4xl font-semibold tracking-tight mb-4">Writing a Custom Loss Term</h1>
      <p className="text-lg leading-8 text-neutral-300 mb-10">
        Custom loss terms slot directly into the LossComposer. You implement one method —{' '}
        <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">compute()</code> — and the
        framework handles weighting, logging, and grouping.
      </p>

      <section className="mb-10">
        <h2 className="text-2xl font-semibold mb-2">A simple custom term</h2>
        <p className="text-neutral-300 leading-7 mb-4">
          This example implements a gradient-difference loss (GDL), useful for sharpening predictions in medical imaging.
        </p>
        <CodeBlock code={`import torch
import torch.nn.functional as F
from oma.losses.base import LossTerm


class GradientDifferenceLoss(LossTerm):
    """
    Penalises the difference in image gradients between prediction and target.
    Encourages sharper, structure-preserving outputs.

    Reads from state:
        "recon"  — model prediction, shape [B, C, H, W]
        "input"  — ground truth target, shape [B, C, H, W]
    """

    def compute(self, state: dict, **kwargs) -> torch.Tensor:
        pred   = state["recon"]
        target = state["input"]

        # Horizontal and vertical gradients
        pred_dx   = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy   = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        loss_dx = F.l1_loss(pred_dx, target_dx)
        loss_dy = F.l1_loss(pred_dy, target_dy)
        return (loss_dx + loss_dy) / 2`} />
      </section>

      <section className="mb-10">
        <h2 className="text-2xl font-semibold mb-2">Use it in a LossComposer</h2>
        <CodeBlock code={`from oma.losses import LossComposer
from oma.losses.terms import L1LossTerm, KLLossTerm

loss_fn = LossComposer([
    L1LossTerm(weight=1.0,  group="main"),
    GradientDifferenceLoss(weight=0.5, group="main", name="gdl"),
    KLLossTerm(weight=0.001, group="main"),
])

# GDL is now logged automatically as "gdl_loss" in TensorBoard / WandB`} />
      </section>

      <section className="mb-10">
        <h2 className="text-2xl font-semibold mb-2">A StatefulLossTerm example</h2>
        <p className="text-neutral-300 leading-7 mb-4">
          Use <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">StatefulLossTerm</code> when
          your term needs to write intermediate results back to state for other terms to consume.
          Here a frequency loss caches the Fourier transform so it is only computed once:
        </p>
        <CodeBlock code={`import torch
from oma.losses.base import StatefulLossTerm


class FrequencyDomainLoss(StatefulLossTerm):
    """Penalises amplitude spectrum differences in Fourier space."""

    state_write_keys = ["pred_fft", "target_fft"]

    def compute(self, state: dict, **kwargs) -> torch.Tensor:
        pred   = state["recon"]
        target = state["input"]

        pred_fft   = torch.fft.fft2(pred,   norm="ortho")
        target_fft = torch.fft.fft2(target, norm="ortho")

        # Write to state — other terms can reuse these
        state["pred_fft"]   = pred_fft
        state["target_fft"] = target_fft

        loss = torch.mean(torch.abs(pred_fft.abs() - target_fft.abs()))
        return loss`} />
      </section>

      <div className="flex gap-4 pt-4 border-t border-white/10">
        <a href="#/docs/guides/custom-method" className="text-sm text-neutral-400 hover:text-white transition">← Custom Method</a>
        <a href="#/docs/recipes/diffusion-bridge" className="text-sm text-blue-400 hover:text-blue-300 transition ml-auto">Next: Diffusion Bridge Recipe →</a>
      </div>
    </div>
  );
}
