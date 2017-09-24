# import os, sys
# import time
# start = time.time()
# import tensorflow as tf
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# # Unpersists graph from file
# with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
#     tf.import_graph_def(graph_def, name='')
#
# # change this as you see fit
# image_paths = [sys.argv[1], sys.argv[2]]
#
# image_data = []
# for path in image_paths:
#     # Read in the image_data
#     image_data.append(tf.gfile.FastGFile(path, 'rb').read())
#
# # Loads label file, strips off carriage return
# label_lines = [line.rstrip() for line
#                    in tf.gfile.GFile("retrained_labels.txt")]
#
# with tf.Session() as sess:
#     print "Loading Took:", time.time() - start
#
#     # Feed the image_data as input to the graph and get first prediction
#     softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
#
#     for image in image_data:
#         predictions = sess.run(softmax_tensor, \
#                  {'DecodeJpeg/contents:0': image})
#
#         # Sort to show labels of first prediction in order of confidence
#         top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
#
#         for node_id in top_k:
#             human_string = label_lines[node_id]
#             score = predictions[0][node_id]
#             print('%s (score = %.5f)' % (human_string, score))
#
# print "Complete time took:", time.time() - start
import os, sys, time
import tensorflow as tf
import numpy as np

class TensorFlow:
    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.label_lines = None
        # print os.listdir(".")
        self.initialize()


    def initialize(self):
        with open("./tensor_flow/tf_files/retrained_graph.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

        # Loads label file, strips off carriage return
        self.label_lines = [line.rstrip() for line
                           in tf.gfile.GFile("./tensor_flow/tf_files/retrained_labels.txt")]

    def query(self, cropped_image):
        with tf.Session() as sess:
            # Feed the image_data as input to the graph and get first prediction
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

            image = np.array(cropped_image)#.reshape(1,720,1280,3)
            predictions = sess.run(softmax_tensor, \
                     {'DecodeJpeg:0': image})

            # Sort to show labels of first prediction in order of confidence
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

            for node_id in top_k:
                human_string = self.label_lines[node_id]
                score = predictions[0][node_id]
                return human_string, score
                # if score > 0.5:
                #     print('%s (score = %.5f)' % (human_string, score))


if __name__ == "__main__":
    tf = TensorFlow()
    tf.query(np.asarray([1,2,3,4]))
