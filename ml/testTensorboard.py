import tensorflow as tf
# Only use tensorflow's keras!
#from tensorflow.python import keras as tfkeras
from tensorflow import keras as tfkeras
#from tensorflow.python.training.rmsprop import RMSPropOptimizer
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import os

#tensorflow.enable_eager_execution()
from MyMScProj.jmod.onestage.yolov3.callbacks import CustomTensorBoard

print("executing egarly ? ",tf.executing_eagerly())
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#gpus = tf.config.list_physical_devices('GPU')
#for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu, True)
#gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
#tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True))

class MyModel(tfkeras.Model):
    def __init__(self, tensorboard_folder_path):
        super(MyModel, self).__init__()
        self.dense1 = tfkeras.layers.LSTM(units=6)
        self.dense2 = tfkeras.layers.Dense(units=4)
        self.graph_has_been_written = False
        self.tensorboard_folder_path = tensorboard_folder_path

    def call(self, input, **kwargs):
        print("input shape", input.shape)
        result = self.dense1(input)
        result = self.dense2(result)
        if not tf.executing_eagerly() and not self.graph_has_been_written:
            # In non eager mode and a graph is available which can be written to Tensorboard using the "old" FileWriter:
            model_graph = result.graph
            writer = tf.compat.v1.summary.FileWriter(logdir=self.tensorboard_folder_path, graph=model_graph)
            writer.flush()
            self.graph_has_been_written = True
            print("Wrote eager graph to", self.tensorboard_folder_path)
        return result



if __name__ == "__main__":
    print("Eager execution:", tf.executing_eagerly())
    # Create model and specify tensorboard folder:

    tensorboard_logs = "/tmp/tensorboardtest"
    tensorboard = CustomTensorBoard(
        log_dir=tensorboard_logs,
        write_graph=True,
        write_images=True,
    )

    model = MyModel(tensorboard_logs)
    optimizer = RMSprop(learning_rate=0.001)
    model.compile(optimizer, tf.losses.categorical_crossentropy, run_eagerly=True)
    # Build the model (this will invoke model.call in non-eager mode). If model.build is not called explicitly here, it
    # will be called by model.fit_generator implicitly when the first batch is about to be feed to the network.
    model.build((None, None, 5))
    # Can only be called after the model has been built:
    model.summary()

    # Two arbitrary batches with different batch size and different sequence length:
    x1 = np.array([[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]],
                  dtype=np.float32)
    y1 = np.array([[1, 0, 0, 0]], dtype=np.float32)
    print("x1 shape", x1.shape)
    print("y1 shape", y1.shape)

    x2 = np.array([[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
                   [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]], dtype=np.float32)
    y2 = np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32)
    print("x2 shape", x2.shape)
    print("y2 shape", y2.shape)

    # Simply yield the two batches alternately
    def iterator():
        switcher = False
        while 1:
            if switcher:
                yield x1, y1
            else:
                yield x2, y2
            switcher = not switcher


    model.fit_generator(iterator(), steps_per_epoch=10, epochs=5, callbacks = [tensorboard])






