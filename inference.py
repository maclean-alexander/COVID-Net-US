import numpy as np
import tensorflow as tf
import os, argparse
import cv2

from dataset import process_image_file

# To remove TF Warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser(description='COVID-Net US Inference')
parser.add_argument('--weightspath', default='models/COVID-Net-US-1-convex', type=str, help='Path to model files, defaults to \'models/COVID-Net-US-1-convex\'')
parser.add_argument('--metaname', default='model.meta', type=str, help='Name of ckpt meta file')
parser.add_argument('--ckptname', default='model-1859', type=str, help='Name of model ckpts')
parser.add_argument('--n_classes', default=2, type=int, help='Number of detected classes (currently only n_classes=2 is implemented)')
parser.add_argument('--imagepath', default='assets/ex-covid.jpeg', type=str, help='Full path to image to be inferenced')
parser.add_argument('--in_tensorname', default='input_1:0', type=str, help='Name of input tensor to graph')
parser.add_argument('--out_tensorname', default='softmax_tensor:0', type=str, help='Name of output tensor from graph')
parser.add_argument('--input_size', default=480, type=int, help='Size of input (ex: if 480x480, --input_size 480)')

args = parser.parse_args()

if args.n_classes == 2:
    # For COVID-19 positive/negative detection
    mapping = {'negative': 0, 'positive': 1}
    inv_mapping = {0: 'negative', 1: 'positive'}
elif args.n_classes == 3:
    raise Exception('''COVID-Net currently only supports 2 class COVID-19 positive/negative detection''')
    # NOTE: no 3-class models developed as of yet
    # For detection of no pneumonia/non-COVID-19 pneumonia/COVID-19 pneumonia
    '''
    mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}
    inv_mapping = {0: 'normal', 1: 'pneumonia', 2: 'COVID-19'}'''
else:
    raise Exception('''COVID-Net currently only supports 2 class COVID-19 positive/negative detection''')
mapping_keys = list(mapping.keys())

sess = tf.Session()
tf.get_default_graph()
saver = tf.train.import_meta_graph(os.path.join(args.weightspath, args.metaname))
saver.restore(sess, os.path.join(args.weightspath, args.ckptname))

graph = tf.get_default_graph()

image_tensor = graph.get_tensor_by_name(args.in_tensorname)
pred_tensor = graph.get_tensor_by_name(args.out_tensorname)
is_training_tensor = graph.get_tensor_by_name('is_training:0')

x = process_image_file(args.imagepath, args.input_size)
x = x.astype('float32') / 255.0
pred = sess.run(pred_tensor, feed_dict={image_tensor: np.expand_dims(x, axis=0),
                                        is_training_tensor: False})

print('Prediction: {}'.format(inv_mapping[pred.argmax(axis=1)[0]]))
print('Confidence')
print(' '.join('{}: {:.3f}'.format(cls.capitalize(), pred[0][i]) for cls, i in mapping.items()))
print('**DISCLAIMER**')
print('Do not use this prediction for self-diagnosis. You should check with your local authorities for the latest advice on seeking medical assistance.')
