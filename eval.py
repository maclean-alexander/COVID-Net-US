from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np
import tensorflow as tf
import os, argparse, time
import cv2

from dataset import process_image_file

# To remove TF Warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def eval(sess, graph, testfile, testfolder, input_tensor, output_tensor, input_size, mapping, save_cm=False):
    image_tensor = graph.get_tensor_by_name(input_tensor)
    pred_tensor = graph.get_tensor_by_name(output_tensor)
    is_training_tensor = graph.get_tensor_by_name('is_training:0')

    y_test = []
    pred = []
    softmax = []
    progbar = tf.keras.utils.Progbar(len(testfile))
    for i in range(len(testfile)):
        line = testfile[i].split()
        x = process_image_file(os.path.join(testfolder, line[1]), input_size)
        x = x.astype('float32') / 255.0
        y_test.append(mapping[line[2]])
        model_output = np.array(sess.run(pred_tensor, feed_dict={image_tensor: np.expand_dims(x, axis=0),
                                                                 is_training_tensor: False}))
        softmax.append(model_output[0])
        pred.append(model_output.argmax(axis=1))
        if i % 25 == 24:
            progbar.update(i+1)
    y_test = np.array(y_test)
    pred = np.array(pred)
    softmax = np.array(softmax)

    matrix = confusion_matrix(y_test, pred)
    matrix = matrix.astype('float')
    print('Confusion Matrix:')
    print(matrix)
    class_acc = [matrix[i,i]/np.sum(matrix[i,:]) if np.sum(matrix[i,:]) else 0 for i in range(len(matrix))]
    if save_cm:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        disp = ConfusionMatrixDisplay(matrix, display_labels=['Negative', 'Positive'])
        disp.plot()
        timestr = time.strftime('%Y%m%d-%H%M%S')
        plt.savefig(f'covid-us-conf-mat-{timestr}.png')

    print('Sens:', ', '.join('{}: {:.3f}'.format(cls.capitalize(), class_acc[i]) for cls, i in mapping.items()))
    ppvs = [matrix[i,i]/np.sum(matrix[:,i]) if np.sum(matrix[:,i]) else 0 for i in range(len(matrix))]
    print('PPV:', ', '.join('{}: {:.3f}'.format(cls.capitalize(), ppvs[i]) for cls, i in mapping.items()))
    print('AUC:', roc_auc_score(y_test, softmax[:, 1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='COVID-Net US Evaluation')
    parser.add_argument('--weightspath', default='models/COVID-Net-US-1-convex', type=str, help='Path to model files, defaults to \'models/COVID-Net-US-1-convex\'')
    parser.add_argument('--metaname', default='model.meta', type=str, help='Name of ckpt meta file')
    parser.add_argument('--ckptname', default='model-2390', type=str, help='Name of model ckpts')
    parser.add_argument('--n_classes', default=2, type=int, help='Number of detected classes (currently only n_classes=2 is implemented)')
    parser.add_argument('--testfile', default='labels/test_COVIDUS1_convex.txt', type=str, help='Name of testfile')
    parser.add_argument('--testfolder', default='data/test', type=str, help='Folder where test data is located')
    parser.add_argument('--in_tensorname', default='input_1:0', type=str, help='Name of input tensor to graph')
    parser.add_argument('--out_tensorname', default='softmax_tensor:0', type=str, help='Name of output tensor from graph')
    parser.add_argument('--input_size', default=480, type=int, help='Size of input (ex: if 480x480, --input_size 480)')

    args = parser.parse_args()

    sess = tf.Session()
    tf.get_default_graph()
    saver = tf.train.import_meta_graph(os.path.join(args.weightspath, args.metaname))
    saver.restore(sess, os.path.join(args.weightspath, args.ckptname))

    graph = tf.get_default_graph()

    file = open(args.testfile, 'r')
    testfile = file.readlines()

    if args.n_classes == 2:
        # For COVID-19 positive/negative detection
        mapping = {
            'negative': 0,
            'positive': 1,
        }
    elif args.n_classes == 3:
        raise Exception('''COVID-Net currently only supports 2 class COVID-19 positive/negative detection''')
        # NOTE: no 3-class models developed as of yet
        # For detection of no pneumonia/non-COVID-19 pneumonia/COVID-19 pneumonia
        '''
        mapping = {
            'normal': 0,
            'pneumonia': 1,
            'COVID-19': 2
        }'''
    else:
        raise Exception('''COVID-Net currently only supports 2 class COVID-19 positive/negative detection''')

    print('Beginning Evaluation...')
    eval(sess, graph, testfile, args.testfolder, args.in_tensorname,
         args.out_tensorname, args.input_size, mapping, save_cm=True)
