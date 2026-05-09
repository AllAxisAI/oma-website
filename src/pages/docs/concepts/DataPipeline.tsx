import CodeBlock from '../../../components/CodeBlock';

export default function DataPipeline() {
  return (
    <div>
      <div className="mb-2">
        <span className="text-xs uppercase tracking-[0.25em] text-neutral-500">Core Concepts</span>
      </div>
      <h1 className="text-4xl font-semibold tracking-tight mb-4">Data Pipeline</h1>
      <p className="text-lg leading-8 text-neutral-300 mb-10">
        OpenMedAxis separates dataset logic from datamodule lifecycle. Datasets handle how
        individual samples are loaded and transformed. DataModules handle splits, dataloaders,
        and preparation steps.
      </p>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-4">Architecture</h2>
        <div className="rounded-2xl border border-white/10 bg-white/5 p-5 font-mono text-sm mb-6">
          <div className="text-neutral-500 text-xs uppercase tracking-wider mb-4">Components</div>
          <div className="space-y-3 text-neutral-300">
            <div><span className="text-blue-300">BaseDataset</span> — abstract, subclass for your data format</div>
            <div><span className="text-blue-300">NumpyDataset</span> — loads .npy files by modality (IXI, BraTS style)</div>
            <div className="mt-2"><span className="text-purple-300">BaseDataModule</span> — abstract lifecycle: prepare → setup → dataloaders</div>
            <div><span className="text-purple-300">LSplitDataModule</span> — generic wrapper: pass any dataset class + split config</div>
            <div className="mt-2"><span className="text-emerald-300">IXIRecipe</span> — downloads IXI, converts to numpy, creates manifests</div>
          </div>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">BaseDataset</h2>
        <p className="text-neutral-300 leading-7 mb-4">
          Subclass <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">BaseDataset</code> to load your
          own data format. Implement <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">__len__</code> and{' '}
          <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">__getitem__</code>.
          The base class provides normalization helpers and padding utilities.
        </p>
        <CodeBlock code={`from oma.data.datasets.base import BaseDataset
import numpy as np

class MyDataset(BaseDataset):
    def __init__(self, data_dir: str, image_size: int = 256):
        super().__init__(image_size=image_size)
        self.samples = sorted(Path(data_dir).glob("*.npy"))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        image = np.load(self.samples[idx])  # shape: (H, W)

        # BaseDataset provides: normalize, pad_to_size
        image = self._normalize(image, min_val=-1, max_val=1)
        image = self._pad_to_size(image, self.image_size)

        return {
            "image": torch.from_numpy(image).float().unsqueeze(0),
            "path": str(self.samples[idx]),
        }`} />
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">NumpyDataset</h2>
        <p className="text-neutral-300 leading-7 mb-4">
          The built-in <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">NumpyDataset</code> handles
          paired translation datasets organized in modality subdirectories. It loads source and target
          volumes as pairs.
        </p>
        <CodeBlock code={`# Expected directory structure:
# data_dir/
#   T1/  subject_001.npy  subject_002.npy  ...
#   T2/  subject_001.npy  subject_002.npy  ...

from oma.data.datasets import NumpyDataset

dataset = NumpyDataset(
    data_dir="/data/IXI",
    source_modality="T1",
    target_modality="T2",
    image_size=256,
)

sample = dataset[0]
# sample == {
#   "source": Tensor[1, 256, 256],
#   "target": Tensor[1, 256, 256],
#   "source_path": "...",
#   "target_path": "...",
# }`} />
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">LSplitDataModule</h2>
        <p className="text-neutral-300 leading-7 mb-4">
          <code className="text-sm bg-white/10 px-2 py-0.5 rounded font-mono">LSplitDataModule</code> is a generic
          Lightning-compatible DataModule that accepts any dataset class and wraps it with
          train/val/test splitting and dataloader creation.
        </p>
        <CodeBlock code={`from oma.data import LSplitDataModule
from oma.data.datasets import NumpyDataset

datamodule = LSplitDataModule(
    # Dataset class and its init kwargs
    dataset_cls=NumpyDataset,
    dataset_kwargs={
        "data_dir": "/data/IXI",
        "source_modality": "T1",
        "target_modality": "T2",
        "image_size": 256,
    },

    # DataLoader kwargs per split
    train_dataloader_kwargs={
        "batch_size": 4,
        "num_workers": 4,
        "shuffle": True,
        "pin_memory": True,
    },
    val_dataloader_kwargs={
        "batch_size": 2,
        "num_workers": 2,
    },

    # Split manifest (optional — or use train_ratio)
    manifest_path="/data/IXI/split_manifest.json",
    # OR: train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
)`} />

        <h3 className="text-lg font-medium mb-1 text-neutral-200 mt-6">DataModule lifecycle</h3>
        <p className="text-sm text-neutral-400 leading-6 mb-3">
          You don't call these manually — the Trainer calls them in order:
        </p>
        <div className="rounded-2xl border border-white/10 bg-white/5 divide-y divide-white/10 overflow-hidden">
          {[
            ['prepare_data()', 'Download, convert, or verify data. Called once, not in parallel.'],
            ['setup(stage)', 'Split data, instantiate datasets. Called on each GPU.'],
            ['train_dataloader()', 'Returns the training DataLoader.'],
            ['val_dataloader()', 'Returns the validation DataLoader.'],
            ['test_dataloader()', 'Returns the test DataLoader.'],
          ].map(([method, desc]) => (
            <div key={method as string} className="flex gap-4 px-5 py-3">
              <code className="text-xs font-mono text-blue-300 w-44 shrink-0 pt-0.5">{method}</code>
              <span className="text-sm text-neutral-300">{desc}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-2">Data recipes</h2>
        <p className="text-neutral-300 leading-7 mb-4">
          Recipes handle the full data preparation pipeline for known datasets —
          downloading, format conversion, and manifest generation.
        </p>

        <h3 className="text-lg font-medium mb-1 text-neutral-200">IXI</h3>
        <CodeBlock code={`from oma.data.recipes import IXIRecipe

recipe = IXIRecipe(
    data_dir="/data/IXI",
    modalities=["T1", "T2", "PD"],   # which modalities to prepare
    image_size=256,
)

recipe.prepare_data()
# Downloads IXI zip archives, extracts NIfTI files,
# resamples to isotropic resolution, normalizes per-volume,
# saves axial slices as numpy arrays in data_dir/{modality}/

recipe.setup()
# Creates data_dir/split_manifest.json with subject-level train/val/test split
# (prevents slice-level leakage across splits)`} />

        <h3 className="text-lg font-medium mb-1 text-neutral-200 mt-6">Bring your own dataset</h3>
        <p className="text-sm text-neutral-400 leading-6 mb-3">
          For custom datasets, implement <code className="text-sm bg-white/10 px-1 py-0.5 rounded font-mono">BaseDataModule</code> directly:
        </p>
        <CodeBlock code={`from oma.data.datamodule.base import BaseDataModule
from torch.utils.data import DataLoader

class MyDataModule(BaseDataModule):
    def prepare_data(self):
        # download or verify — called once
        pass

    def setup(self, stage: str = "fit"):
        if stage == "fit":
            self.train_dataset = MyDataset(split="train")
            self.val_dataset   = MyDataset(split="val")
        if stage == "test":
            self.test_dataset  = MyDataset(split="test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=4, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=2)`} />
      </section>

      <div className="flex gap-4 pt-4 border-t border-white/10">
        <a href="#/docs/concepts/loss-system" className="text-sm text-neutral-400 hover:text-white transition">← Loss System</a>
        <a href="#/docs/guides/custom-method" className="text-sm text-blue-400 hover:text-blue-300 transition ml-auto">Next: Custom Method →</a>
      </div>
    </div>
  );
}
