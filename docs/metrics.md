# Quantitative evaluation and Octave validation — 2026-11-10

The eight released simulation checkpoints were evaluated on all ten original TSA simulation scenes. The four released real checkpoints were rerun on all five captured measurements. Inference used the README commands in a fresh Python 3.10 environment with PyTorch 2.5.1+cu121 and an NVIDIA RTX A5000.

## Simulation metrics

These results come from the repository's `Cal_quality_assessment.m` and `quality_assessment.m`, executed with GNU Octave 10.3.0 and the image 2.20.1 package. Each row averages the ten scene scores. CUDA convolutions and matrix multiplications use full FP32 precision with TF32 disabled.

| Method | PSNR (dB) ↑ | SSIM ↑ | RMSE ↓ | ERGAS ↓ | SAM (degrees) ↓ | UIQI ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BiConnect | 22.1891 | 0.574562 | 21.2859 | 115.1779 | 32.8746 | 0.343319 |
| BNN | 23.8842 | 0.593252 | 16.9513 | 97.8925 | 26.7027 | 0.430972 |
| Bi-Real **(abnormal)** | 16.2818 | 0.297884 | 43.0501 | 295.3399 | 26.7123 | 0.274436 |
| IRNet | 26.4837 | 0.685355 | 12.6481 | 65.9674 | 22.8355 | 0.495796 |
| ReActNet | 26.4850 | 0.670532 | 12.7219 | 66.7015 | 21.3923 | 0.519356 |
| BBCU | 26.5127 | 0.673793 | 12.4631 | 68.2215 | 23.8491 | 0.503986 |
| BTM | 27.2084 | 0.715682 | 11.6707 | 61.1778 | 24.3108 | 0.557092 |
| BiSRNet | 29.7669 | 0.833305 | 8.7791 | 45.0090 | 14.7521 | 0.661368 |

The [per-scene CSV](metrics-per-scene.csv) contains all 80 records and six metrics.

The metric protocol follows the original scripts: predictions are clipped to `[0,1]`, both reference and reconstruction are converted with `im2uint8`, and calculations use double precision. PSNR averages the per-band PSNR values. SSIM uses the existing 11×11 Gaussian window with sigma 1.5 and valid convolution. RMSE uses the 0–255 intensity scale, ERGAS uses ratio 1, and UIQI uses 32×32 windows. An independent Python implementation matched all six Octave metrics across all 80 scenes to absolute error below `1e-10`.

The training utility uses a different metric protocol: it rounds intensities after multiplying by 256 and computes SSIM with padded windows. Its scores should be recorded separately from the MATLAB evaluation scores above. For the same BiSRNet outputs, that utility reported 29.7208 dB / 0.836544, whereas the MATLAB script reported 29.7669 dB / 0.833305.

## Precision issue and remaining checkpoint limitation

TF32 changed reconstruction quality substantially in some binary networks. The training and testing entry points now disable it explicitly. Measurements before and after this fix are:

| Method | Default TF32 PSNR / SSIM | Full FP32 PSNR / SSIM |
| --- | ---: | ---: |
| Bi-Real | 6.5509 / 0.155663 | 16.2818 / 0.297884 |
| ReActNet | 25.4476 / 0.612617 | 26.4850 / 0.670532 |
| BiSRNet | 29.7531 / 0.833517 | 29.7669 / 0.833305 |

**Bi-Real remains abnormal:** its released checkpoint and current architecture produce approximately 16.28 dB / 0.298, compared with 26.26 dB / 0.664 in the project's published comparison table. Strict FP32 on CPU and GPU and a diagnostic FP64 run also showed poor quality on the first scene. The root cause remains unresolved. All eight simulation checkpoints were downloaded independently from the original Google Drive folder and had the same SHA256 values as their HF counterparts. The reported results use `model.eval()` with the supplied BatchNorm statistics and unchanged weights.

The four real models produce finite `660×660×28` reconstructions for all five measurements. Reference PSNR/SSIM are unavailable for these real scenes because the released files contain captured measurements without corresponding hyperspectral ground truth.

## MATLAB source scripts executed with Octave

- `Cal_quality_assessment.m`: all eight models, all ten scenes, six metrics checked.
- `show_simulation.m`: all ten BiSRNet reconstructions and 28 bands, **280 PNG files**.
- `show_real.m`: all five BiSRNet reconstructions and 28 bands, **140 PNG files**.
- `createfigure.m`: the legacy six-curve helper was smoke-tested using synthetic curves; its old comparison labels were retained. This helper output is a functionality check.

All 420 band PNG files were decoded successfully. Representative files from both datasets were checked for nontrivial color variation and visually inspected. The Octave compatibility fixes cover the unavailable `makecform`/`iptgate` interfaces, valid colormap indices, floating-point truecolor for the gnuplot backend, and figure cleanup during headless export. Curve labels now use `sprintf` to avoid the Mapping Toolbox's `roundn` dependency.

Native MATLAB was unavailable. These runs validate the `.m` source under Octave; the native MATLAB path remains untested.

## Reproduce the Octave runs

Install GNU Octave and its image package, prepare the datasets and weights, and run the README inference commands first. For the BiSRNet metrics, start from the repository root:

```bash
cd simulation/test_code/Quality_Metrics
cp ../exp/bisrnet/Test_result.mat results/Test_result.mat
octave --no-gui --quiet --eval "pkg load image; Cal_quality_assessment"
```

For band visualizations, copy the outputs into the locations described in the README, then start from the repository root:

```bash
cd visualization
octave --no-gui --quiet --eval "pkg load image; set(0,'defaultfigurevisible','off'); show_simulation; show_real"
```

The 12 README inference commands, all 16 one-batch training smoke checks, and all 7 Python unit regressions were rerun after the precision fix and passed. The Bi-Real quality limitation above is separate from successful execution. Full training convergence remains outside this validation.
