# Pretrained Models

The naming convention for models uses the following format:
```
COVID-Net-US-<dataset version>(-<dataset subset if applicable>) (<minor dataset version if applicable>)
```
For example, a COVID-Net-US model trained on COVIDxUS-1 dataset's Convex subset's "A" variant would be called "COVID-Net-US-1-convex (1A)".

## COVID-Net CT-1 Models
These models are trained and tested on the COVIDxUS-1 dataset.

|        Model        | Type | Input Resolution | COVID-19 Sensitivity (%) | Accuracy (%) | # Params (K) | FLOPs (M) |
|:-------------------:|:----:|:----------------:|:------------------------:|:------------:|:------------:|:---------:|
|[COVID-Net-US-1-convex](https://bit.ly/covid-net-us-1-convex)| ckpt |     480 x 480    |           93.9           |     93.0     |    65   |    596   |
