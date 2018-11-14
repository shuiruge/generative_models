from tensorflow.examples.tutorials.mnist import input_data


def get_dataset(data_dir):
    mnist = input_data.read_data_sets(
        data_dir,
        one_hot=True,
        source_url='http://yann.lecun.com/exdb/mnist/')
    return mnist
