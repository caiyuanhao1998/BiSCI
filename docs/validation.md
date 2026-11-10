# Validation — 2026-11-10

A fresh Python 3.10 Conda environment was created on Linux with an NVIDIA RTX A5000. Installation of the original requirements failed while building dependencies of OpenCV 4.4 on Python 3.10. The updated environment uses PyTorch 2.5.1+cu121, torchvision 0.20.1, NumPy 1.26.4, and SciPy 1.13.0; installation and `python -m pip check` passed.

## Executed checks

| Check | Coverage | Result |
| --- | --- | --- |
| HF weight downloads | Both README `hf download --include` commands and the single-BiSRNet download | Passed |
| Simulation testing | All 8 released checkpoints, each on all 10 original TSA test scenes | Passed |
| Real testing | All 4 released real checkpoints, each on all 5 original measurements | Passed |
| README testing commands | All 12 commands, with their documented paths and default GPU setting | Passed |
| Simulation training smoke checks | All 8 methods, one batch and one epoch, batch size 1, followed by evaluation and checkpoint save/reload | Passed |
| Real training smoke checks | All 8 methods, one 96×96 batch and one epoch, followed by checkpoint save/reload | Passed |
| Targeted regressions | 7 unit tests; zero-measurement/subset inference; retention of pretrained real-training weights | Passed |
| Parameter/FLOP example | README BiSRNet example with `stage=1` | Passed |

Testing loaded every checkpoint strictly. All simulation outputs have shape `10×256×256×28`; each real output has shape `660×660×28`. Outputs and trained parameters were checked for finite values. Every training smoke check performed a verified optimizer update with finite gradients.

The original simulation and real test data were downloaded through the public OneDrive links in the README. The simulation training smoke checks used one quantized test cube as a small training fixture. For real training, the CAVE/KAIST readers were substituted with small fixtures derived from a test cube; the original dataset class, measurement synthesis, training loop, optimizer, and checkpoint handling were exercised. These checks establish that the training code runs; they do not measure training convergence or reproduce paper metrics.

Full CAVE/KAIST training datasets and full training schedules were not run. MATLAB is unavailable on the validation machine, so MATLAB metrics and visualization commands were not executed. Their corrected paths and MAT field names were checked against the Python outputs.

## Fixes covered

- Replace incompatible legacy dependency pins with the tested Python 3.10 environment.
- Use portable dataset paths, a supported default simulation model, and GPU 0 by default.
- Match HF checkpoint subdirectories and normalize the Bi-Real filename without changing weight contents.
- Fix single-item batch augmentation, exact-size crops, and splicing images of different dimensions.
- Accept directory paths without trailing slashes and ignore non-MAT metadata in data loaders.
- Respect real testing scene counts and avoid division by zero for empty measurements.
- Respect real training sample counts, preserve supplied pretrained weights, and allow saving the best simulation checkpoint when initial PSNR is negative.
- Correct MATLAB visualization paths/output fields and keep metric calculation from overwriting reconstruction arrays.

Run the unit regressions from the repository root:

```bash
python -m unittest discover -s tests -v
```

These unit tests run on CPU. The training and inference checks above were executed separately on CUDA.
