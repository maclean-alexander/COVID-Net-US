import tensorflow as tf
import os, argparse, pathlib

from eval import eval
from dataset import BalanceCovidDataset

# To remove TF Warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser(description='COVID-Net Training Script')
parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('--bs', default=8, type=int, help='Batch size')
parser.add_argument('--weightspath', default='models/COVID-Net-US-1-convex', type=str,
                    help='Path to model files, defaults to \'models/COVID-Net-US-1-convex\'')
parser.add_argument('--metaname', default='model.meta', type=str, help='Name of ckpt meta file')
parser.add_argument('--ckptname', default='model-1859', type=str, help='Name of model ckpts')
parser.add_argument('--n_classes', default=2, type=int, help='Number of detected classes, defaults to 2')
parser.add_argument('--trainfile', default='labels/train_COVIDUS1_convex.txt', type=str, help='Path to train file')
parser.add_argument('--validfile', default='labels/valid_COVIDUS1_convex.txt', type=str, help='Path to validation file')
parser.add_argument('--testfile', default='labels/test_COVIDUS1_convex.txt', type=str, help='Path to test file')
parser.add_argument('--name', default='COVID-Net-US', type=str, help='Name of folder to store training checkpoints')
parser.add_argument('--datadir', default='data', type=str, help='Path to data folder')
parser.add_argument('--covid_percent', default=0.5, type=float, help='Percentage of covid samples in batch')
parser.add_argument('--input_size', default=480, type=int, help='Size of input (ex: if 480x480, --input_size 480)')
parser.add_argument('--in_tensorname', default='input_1:0', type=str, help='Name of input tensor to graph')
parser.add_argument('--out_tensorname', default='softmax_tensor:0', type=str,
                    help='Name of output tensor from graph')
parser.add_argument('--logit_tensorname', default='norm_dense_1/final_dense:0', type=str,
                    help='Name of logit tensor for loss')
parser.add_argument('--label_tensorname', default='norm_dense_1_target:0', type=str,
                    help='Name of label tensor for loss')
parser.add_argument('--training_tensorname', default='is_training:0', type=str,
                    help='Name of training placeholder tensor')
args = parser.parse_args()

# Parameters
LEARNING_RATE = args.lr
BATCH_SIZE = args.bs
DISPLAY_STEP = 1

# output path
outputPath = './output/'
runID = args.name + '-lr' + str(LEARNING_RATE)
runPath = outputPath + runID
pathlib.Path(runPath).mkdir(parents=True, exist_ok=True)
print('Output: ' + runPath)

with open(args.trainfile) as f:
    trainfiles = f.readlines()
with open(args.validfile) as f:
    validfiles = f.readlines()
with open(args.testfile) as f:
    testfiles = f.readlines()

if args.n_classes == 2:
    # For COVID-19 positive/negative detection
    mapping = {
        'negative': 0,
        'positive': 1,
    }
else:
    raise Exception('''COVID-Net-US currently only supports 2 class COVID-19 positive/negative detection''')

generator = BalanceCovidDataset(data_dir=args.datadir,
                                csv_file=args.trainfile,
                                batch_size=BATCH_SIZE,
                                input_shape=(args.input_size, args.input_size),
                                n_classes=args.n_classes,
                                mapping=mapping,
                                covid_percent=args.covid_percent)

with tf.Session() as sess:
    tf.get_default_graph()
    saver = tf.train.import_meta_graph(os.path.join(args.weightspath, args.metaname))

    graph = tf.get_default_graph()

    image_tensor = graph.get_tensor_by_name(args.in_tensorname)
    labels_tensor = graph.get_tensor_by_name(args.label_tensorname)
    pred_tensor = graph.get_tensor_by_name(args.logit_tensorname)
    is_training = graph.get_tensor_by_name(args.training_tensorname)

    # Define loss and optimizer
    global_step = tf.train.get_or_create_global_step()
    loss = graph.get_tensor_by_name('add:0')
    decayed_lr = tf.train.exponential_decay(LEARNING_RATE,
                                            global_step, 1000,
                                            0.95, staircase=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=decayed_lr)
    grad_vars = optimizer.compute_gradients(loss)
    minimize_op = optimizer.apply_gradients(grad_vars, global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group(minimize_op, update_ops)

    # Initialize the variables
    init = tf.global_variables_initializer()

    # Run the initializer
    sess.run(init)

    # load weights
    saver.restore(sess, os.path.join(args.weightspath, args.ckptname))

    # save base model
    saver.save(sess, os.path.join(runPath, 'model'))
    print('Saved baseline checkpoint')
    print('Baseline eval:')
    eval(sess, graph, validfiles, os.path.join(args.datadir, 'all_images'),
       args.in_tensorname, args.out_tensorname, args.input_size, mapping)

    # Training cycle
    print('Training started')
    total_batch = len(generator)
    progbar = tf.keras.utils.Progbar(total_batch)
    for epoch in range(args.epochs):
        for i in range(total_batch):
            # Run optimization
            batch_x, batch_y, weights = next(generator)
            sess.run(train_op, feed_dict={image_tensor: batch_x,
                                          labels_tensor: batch_y,
                                          #sample_weights: weights,
                                          is_training: True})
            progbar.update(i+1)
        if epoch % DISPLAY_STEP == 0:
            pred = sess.run(pred_tensor, feed_dict={image_tensor:batch_x,
                                                    is_training:False})
            loss_val = sess.run(loss, feed_dict={pred_tensor: pred,
                                                labels_tensor: batch_y,
                                                #sample_weights: weights})
                                                is_training: False})
            print("Epoch:", '%04d' % (epoch + 1), "Minibatch loss=", "{:.9f}".format(loss_val))
            eval(sess, graph, validfiles,os.path.join(args.datadir, 'all_images'),
                 args.in_tensorname, args.out_tensorname, args.input_size, mapping)
            saver.save(sess, os.path.join(runPath, 'model'), global_step=epoch+1, write_meta_graph=False)
            print('Saving checkpoint at epoch {}'.format(epoch + 1))


print("Optimization Finished!")
