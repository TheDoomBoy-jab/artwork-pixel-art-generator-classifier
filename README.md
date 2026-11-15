# Image translation and generation experiments

A verified local conditional-GAN training path plus historical GAN/diffusion notebooks. The original artwork folder contained only four sketches and four unrelated pixel-art images. Sorting those collections is not valid paired supervision. The notebook now requires an explicit paired-data manifest for conditional GAN training.

## Fetch real paired data

```sh
python -m pip install pyarrow Pillow numpy tensorflow
python fetch_pairs.py --output data/facades --count 256
python train_paired.py --data data/facades --output results/facades --epochs 20
```

The download is the public [huggan/facades dataset](https://huggingface.co/datasets/huggan/facades), about 94 MB. Source columns imageA/imageB are used as paired translation domains. This is a different task from sketch-to-pixel-art. The images remain local and are ignored by Git; source URLs and the Parquet content hash are in `dataset-source.json`.

## Actual exploratory result

First 256 upstream rows, shuffled with seed 42 and exact duplicate pairs grouped: 179 training, 38 validation, 39 test. These are local subsets of upstream data, not an official benchmark split. The model is a compact U-Net conditional generator with a patch discriminator, adversarial loss and L1 loss. The best checkpoint was selected by validation MAE. Test metrics use [0,1] pixels.

- Test MAE: 0.15726; mean-training-image baseline MAE: 0.16126.
- Test PSNR: 14.04 dB.
- Test SSIM: 0.2475.
- Checkpoint reload reproduced predictions.

See `results/facades/metrics.json`, `split.json`, and `examples.png` for the measured run. This is a small proof of a functioning training/evaluation pipeline, with modest image quality. It does not establish perceptual realism, pixel-art translation quality or a meaningful FID score. The separate diffusion experiment has not been trained/evaluated in this review.
