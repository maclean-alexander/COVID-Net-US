# Training, Evaluation and Inference

The networks take as input an image of shape (N, 480, 480, 3) and output the softmax probabilities as (N, 2), where N is the number of images. For the TensorFlow checkpoints, here are some useful tensors:
* input tensor: `input_1:0`
* label tensor: `norm_dense_1_target:0`
* logits tensor: `norm_dense_1/final_dense:0`
* output confidence tensor: `softmax_tensor:0`
* output prediction tensor: `ArgMax:0`
* loss tensor: `add:0`
* training tensor: `is_training:0`

## Steps for training
1. We provide you with the TensorFlow training script, [train_tf.py](../train_tf.py)
2. Locate the TensorFlow checkpoint files (location of pretrained model)
3. To train from a pretrained model (e.g. COVID-Net-US Convex-1):
```
python train_tf.py \
    --weightspath models/COVID-Net-Us-1-convex \
    --metaname model.meta \
    --ckptname model-4780 \
    --datadir path_to_COVID_US_images_folder
```

By default, [labels files](../labels/) are present, but if a different split (either from a different COVID-US version or otherwise), then the below options can be used:
```
    --trainfile path_to_training_split.txt \
    --validfile path_to_validation_split.txt \
    --testfile path_to_testing_split.txt
```

For more options and information, `python train_tf.py --help`

## Steps for testing
1. We provide you with the TensorFlow testing script, [eval.py](../eval.py)
2. Locate the TensorFlow checkpoint files
3. To evaluate a TensorFlow checkpoint:
```
python eval.py \
    --weightspath models/COVID-Net-Us-1-convex \
    --metaname model.meta \
    --ckptname model-4780 \
    --datadir path_to_COVID_US_images_folder \
    --testfile path_to_testing_split.txt
```

The testing split file has a default value based on the current version, but can be modified here if desired.

For more options and information, `python eval.py --help`

## Steps for inference
**DISCLAIMER: Do not use this prediction for self-diagnosis. You should check with
your local authorities for the latest advice on seeking medical assistance.**

Inference may be run using via the following steps:
1. We provide you with the TensorFlow inference script, [inference.py](../inference.py)
2. Locate the TensorFlow checkpoint files
3. Locate the ultrasound image to be tested
4. To run inference,
```
python inference.py \
    --model_dir models/COVID-Net_CT-2_L \
    --meta_name model.meta \
    --ckpt_name model-4780 \
    --image_file assets/example_convex_covid.jpg
```
For more options and information, `python inference.py --help`