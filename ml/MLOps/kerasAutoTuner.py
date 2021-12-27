import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt


'''
# This is baseline model with random assignment of total number of hidden layer (512), drop out value (0.2)
# etc hyperparameter's
'''
def b_model_def():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(512,activation='relu', name='dense_1'),
    tf.keras.layers.Dropout(rate=0.2, name='dropout_1'),
    tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("*** base model summary ****")
    print(model.summary())

    return model

def b_model_fit(x_train, y_train):
    b_model = b_model_def()
    b_model.fit(x_train, y_train, epochs=5)

    return b_model

def b_model_evaluate(b_model, x_test, y_test):
    b_eval_dict = b_model.evaluate(x_test, y_test, verbose=1,return_dict=True)
    return b_eval_dict

def print_res(model, model_name, eval_dict):
    '''
      Prints the values of the hyparameters to tune, and the results of model evaluation

      Args:
        model (Model) - Keras model to evaluate
        model_name (string) - arbitrary string to be used in identifying the model
        eval_dict (dict) -  results of model.evaluate
      '''
    print(model_name, ":")
    print("number of hidden units in 1st dense layer: {}".format(model.get_layer("dense_1").units))
    print(f'dropout rate for the optimizer: {model.get_layer("dropout_1").rate}')
    print("learning rate for the optimizer: {}".format(model.optimizer.lr.numpy()))

    for key, val in eval_dict.items():
        print(f'{key} : {val}')

'''
Usig keras tuner to see the optimal hyperparameters
'''
def model_builder(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))

    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 16-512
    hp_units = hp.Int('units', min_value=16, max_value=512, step=16)
    model.add(tf.keras.layers.Dense(units=hp_units, activation='relu'))

    # Add next layers for drop out
    # model.add(keras.layers.Dropout(0.2))
    hp_dropout = hp.Float('dropout_1', min_value=0.10, max_value=0.25, step=0.5)
    model.add(keras.layers.Dropout(rate=hp_dropout))

    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rt = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])


    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rt),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def tune_hps():
    tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy', max_epochs=10, factor=3,
                     directory='my_dir', project_name='intro_to_kt')
    print(tuner.search_space_summary())

    # early stopping
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(x_train, y_train, epochs=30, validation_split=0.2, callbacks=[stop_early])

    return tuner

def find_best_hps(tuner):
    tuner = tune_hps()
    best_hps = tuner.get_best_hyperparameters()[0]

    print(f"""
    The hyperparameter search is complete. The optimal number of units in the first densely-connected
    layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """)

    return (best_hps, tuner)

'''
Search: Running Trial #30

Hyperparameter    |Value             |Best Value So Far 
units             |16                |160               
tuner/epochs      |10                |10                
tuner/initial_e...|0                 |4                 
tuner/bracket     |0                 |2                 
tuner/round       |0                 |2   

'''

# Now we can finalize our baseline model with 160 neurons in dense aka hidden layer
# similar exercise colab nb C3_W1_Lab_1_Keras_Tuner.ipynb -> https://colab.research.google.com/drive/1o0Mxm0ppCOKvSEeoGMuhB06t1WzwVp-N#scrollTo=_leAIdFKAxAD

def h_tuned_model(best_hps, tuner):
    h_model = tuner.hypermodel.build(best_hps)
    print("*** tuned hypermeter model summary ***")
    print(h_model.summary())

    return h_model

if __name__ == '__main__':
    # load mnist
    mnist = tf.keras.datasets.mnist

    # derive train and test set
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # define and fit base model
    b_model = b_model_fit(x_train, y_train)
    b_eval_dict = b_model_evaluate(b_model, x_test, y_test)

    # define and create model based on tuned hyperparameter
    hp_tuned_model = tune_hps()
    best_hps, tuner = find_best_hps(hp_tuned_model)

    h_model = h_tuned_model(best_hps, tuner)

    # Train the hypertuned model
    h_model.fit(x_train, y_train, epochs=5, validation_split=0.2)

    # evalue hyper tuned model
    h_eval_dict = h_model.evaluate(x_test, y_test, return_dict=True)

    # Print results of the baseline and hypertuned model
    print_res(b_model, 'BASELINE MODEL', b_eval_dict)
    print_res(h_model, 'HYPERTUNED MODEL', h_eval_dict)











