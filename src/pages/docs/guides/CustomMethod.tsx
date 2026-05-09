import CodeBlock from '../../../components/CodeBlock';

export default function CustomMethod() {
  return (
    <div>
      <div className="mb-2">
        <span className="text-xs uppercase tracking-[0.25em] text-neutral-500">Guides</span>
      </div>
      <h1 className="text-4xl font-semibold tracking-tight mb-4">Writing a Custom Method</h1>
      <p className="text-lg leading-8 text-neutral-300 mb-10">
        This guide walks through building a custom training method from scratch —
        a paired image translation method with SSIM + L1 loss and PSNR metrics.
      </p>

      <section className="mb-10">
        <h2 className="text-2xl font-semibold mb-2">When to write a custom Method</h2>
        <div className="space-y-3">
          {[
            ['Use Method.step()', 'Single optimizer. You control exactly what happens at each step.'],
            ['Use GroupedLossMethod.build_state()', 'Multiple optimizers (generator + discriminator). Let LossComposer handle grouping.'],
            ['Extend a built-in method', 'When you only want to change one aspect (e.g. a different loss or inference strategy).'],
          ].map(([when, what]) => (
            <div key={when as string} className="flex gap-4 rounded-xl border border-white/10 bg-white/5 p-4">
              <span className="text-sm font-medium text-white w-56 shrink-0">{when}</span>
              <span className="text-sm text-neutral-400">{what}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="mb-10">
        <h2 className="text-2xl font-semibold mb-2">Step 1 — Start from Method</h2>
        <CodeBlock code={`import torch
import torch.nn.functional as F
from oma.methods.base import Method


class SSIMTranslationMethod(Method):
    """
    Paired image translation with L1 + SSIM loss.
    Expects batches: {"source": Tensor, "target": Tensor}
    """

    def __init__(self, model, ssim_weight=0.5, lr=1e-4):
        super().__init__(
            model=model,
            optimizer_cfg={"type": "Adam", "lr": lr},
            scheduler_cfg={"type": "CosineAnnealingLR", "T_max": 200},
        )
        self.ssim_weight = ssim_weight`} />
      </section>

      <section className="mb-10">
        <h2 className="text-2xl font-semibold mb-2">Step 2 — Implement step()</h2>
        <CodeBlock code={`    def step(self, batch, stage, batch_idx):
        source = batch["source"]   # [B, 1, H, W]
        target = batch["target"]   # [B, 1, H, W]

        # Forward pass
        pred = self.model(source)  # [B, 1, H, W]

        # Losses
        l1   = F.l1_loss(pred, target)
        ssim = self._ssim_loss(pred, target)
        loss = l1 + self.ssim_weight * ssim

        return {
            "loss": loss,
            "metrics": {
                "l1":   l1.detach(),
                "ssim": ssim.detach(),
                "psnr": self._psnr(pred.detach(), target),
            },
            "artifacts": {
                "source": source,
                "pred":   pred.detach(),
                "target": target,
            },
        }`} />
      </section>

      <section className="mb-10">
        <h2 className="text-2xl font-semibold mb-2">Step 3 — Add helpers</h2>
        <CodeBlock code={`    def _ssim_loss(self, pred, target, window_size=11):
        # Simplified single-scale SSIM loss
        C1, C2 = 0.01**2, 0.03**2
        mu_p = F.avg_pool2d(pred,   window_size, 1, window_size//2)
        mu_t = F.avg_pool2d(target, window_size, 1, window_size//2)
        mu_p2, mu_t2 = mu_p**2, mu_t**2
        sigma_p  = F.avg_pool2d(pred**2,   window_size, 1, window_size//2) - mu_p2
        sigma_t  = F.avg_pool2d(target**2, window_size, 1, window_size//2) - mu_t2
        sigma_pt = F.avg_pool2d(pred*target, window_size, 1, window_size//2) - mu_p*mu_t
        ssim = ((2*mu_p*mu_t + C1)*(2*sigma_pt + C2)) / \
               ((mu_p2 + mu_t2 + C1)*(sigma_p + sigma_t + C2))
        return 1.0 - ssim.mean()

    def _psnr(self, pred, target):
        mse = F.mse_loss(pred, target)
        return 10 * torch.log10(1.0 / (mse + 1e-8))`} />
      </section>

      <section className="mb-10">
        <h2 className="text-2xl font-semibold mb-2">Step 4 — Wire it up</h2>
        <CodeBlock code={`from oma import Trainer
from oma.data import LSplitDataModule
from oma.data.datasets import NumpyDataset

# Your model (any nn.Module)
model = UNet(in_channels=1, out_channels=1)

method = SSIMTranslationMethod(
    model=model,
    ssim_weight=0.5,
    lr=1e-4,
)

datamodule = LSplitDataModule(
    dataset_cls=NumpyDataset,
    dataset_kwargs={"data_dir": "/data/IXI", "source_modality": "T1", "target_modality": "T2", "image_size": 256},
    train_dataloader_kwargs={"batch_size": 4, "num_workers": 4, "shuffle": True},
    val_dataloader_kwargs={"batch_size": 2},
)

trainer = Trainer(max_epochs=200, accelerator="gpu", devices=1)
trainer.fit(method, datamodule)`} />
      </section>

      <div className="flex gap-4 pt-4 border-t border-white/10">
        <a href="#/docs/concepts/data-pipeline" className="text-sm text-neutral-400 hover:text-white transition">← Data Pipeline</a>
        <a href="#/docs/guides/custom-loss" className="text-sm text-blue-400 hover:text-blue-300 transition ml-auto">Next: Custom Loss →</a>
      </div>
    </div>
  );
}
