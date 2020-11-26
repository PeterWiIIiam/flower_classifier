import tensorflow as tf
import os
from PIL import Image
import numpy as np
from data import *


flags = tf.app.flags
flags.DEFINE_string("mode", "train", "train or test")
flags.DEFINE_integer("partition_size", 1000, "the size of each tf record")
FLAGS = flags.FLAGS

PATH = os.path.join("./flowers", FLAGS.mode)
IMGNUM = FLAGS.partition_size



def compile_img_label_partitition_list():
    """
    This function returns the list of all image path from the
    repository folder and the list of the corresponding label.

    Parameters:

    Returns:
        imgs: a list of image paths
        label: a list of corresponding label (sunflower, dandelion, etc)
    """

    names = []
    imgs = []
    labels = []

    for name in os.listdir(PATH):
        if name[0] != ".":
            names.append(name)
            curr_dir = os.path.join(PATH, name)
            # train_imgs.append(curr_dir)

            for img in os.listdir(curr_dir):
                img_path = os.path.join(curr_dir, img)
                imgs.append(img_path)

    label_dict = dict(zip(names, list(range(len(names)))))
    for name in names:
        curr_dir = os.path.join(PATH, name)
        for img in os.listdir(curr_dir):
            labels.append(label_dict[name])

    return imgs, labels


def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert(image_paths, labels, out_path, size=(244,244)):
    # Args:
    # image_paths   List of file-paths for the images.
    # labels        Class-labels for the images.
    # out_path      File-path for the TFRecords output file.
    print("Converting: " + out_path)
    # Number of images. Used when printing the progress.
    num_images = len(image_paths)
    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(out_path) as writer:
        # Iterate over all the image-paths and class-labels.
        for i, (path, label) in enumerate(zip(image_paths, labels)):
            # Print the percentage-progress.
            print("count: {}, total: {}".format(i, num_images-1))
            # Load the image-file using matplotlib's imread function.
            img = Image.open(path)
            img = img.resize(size)
            img = np.array(img)
            # Convert the image to raw bytes.
            img_bytes = img.tostring()
            # Create a dict with the data we want to save in the
            # TFRecords file. You can add more relevant data here.
            data = \
                {
                    'image': wrap_bytes(img_bytes),
                    'label': wrap_int64(label)
                }
            # Wrap the data as TensorFlow Features.
            feature = tf.train.Features(feature=data)
            # Wrap again as a TensorFlow Example.
            example = tf.train.Example(features=feature)
            # Serialize the data.
            serialized = example.SerializeToString()
            # Write the serialized data to the TFRecords file.
            writer.write(serialized)



def main(_):

    # check if the image data is already split
    if not os.path.isdir("./flowers/{}".format(FLAGS.mode)):
        split_training_and_testing_sets()

    images, labels = compile_img_label_partitition_list()
    partitions = {} # key indicates which partition the image is going to put into
    # value is a two element list. The first element is a list of image paths
    # The second element is a list of labels

    for i in range(len(images)):
        partition_key = i // IMGNUM

        if partition_key not in partitions:
            partitions[partition_key] = [[], []]

        img_list = partitions[partition_key][0]
        label_list = partitions[partition_key][1]
        img_list.append(images[i])
        label_list.append(labels[i])

    for k, v in partitions.items():
        # print(k, v)
        img_list = v[0]
        label_list = v[1]

        convert(image_paths=img_list,
                labels=label_list,
                out_path="./flowers/{}-0{}-of-0{}.tfrecord".format(FLAGS.mode, k,
                            len(partitions)))

if __name__ == '__main__':
    tf.app.run()


