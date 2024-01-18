from typing import Tuple

from tensorflow import keras
from tensorflow.keras import layers

# NOTE:
# 1) OMR architecture fixed according to:
# Jorge Calvo-Zaragoza, Alejandro H. Toselli, Enrique Vidal
# Handwritten Music Recognition for Mensural notation with convolutional recurrent neural networks
# 2) AMT architecture based on:
# Miguel A. Román, Antonio Pertusa, Jorge Calvo-Zaragoza
# Data representations for audio-to-score monophonic music transcription

INPUT_HEIGHT = {
    "omr": 64,
    "amt": 256,
}

POOLING_FACTORS = {
    "omr": {"height_reduction": 16, "width_reduction": 2},
    "amt": {"height_reduction": 4, "width_reduction": 2},
}


def build_models(task: str, num_labels: int) -> Tuple[keras.Model, keras.Model]:
    def ctc_loss_lambda(args):
        y_true, y_pred, input_length, label_length = args
        return keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

    # Input block
    image = keras.Input(
        shape=(INPUT_HEIGHT[task], None, 1),
        dtype="float32",
        name="image",
    )
    image_len = keras.Input(shape=(1,), dtype="int32", name="image_len")
    label = keras.Input(shape=(None,), dtype="int32", name="label")
    label_len = keras.Input(shape=(1,), dtype="int32", name="label_len")

    if task == "omr":
        # Convolutional block
        x = layers.Conv2D(64, 5, padding="same", use_bias=False)(image)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

        x = layers.Conv2D(64, 5, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1))(x)

        x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1))(x)

        x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1))(x)

    elif task == "amt":
        # Convolutional block
        x = layers.Conv2D(8, (10, 2), padding="same", use_bias=False)(image)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

        x = layers.Conv2D(8, (8, 5), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1))(x)

    # Intermediate block (preparation to enter the recurrent one)
    # [batch, height, width, channels] -> [batch, width, height, channels]
    x = layers.Permute((2, 1, 3))(x)
    # [batch, width, height, channels] -> [batch, width, height * channels]
    x = layers.Reshape((-1, x.shape[2] * x.shape[3]))(x)

    # Recurrent block
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.5))(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.5))(x)

    # Dense layer
    # num_classes -> represents "num_labels + 1" classes,
    # where num_labels is the number of true labels,
    # and the largest value  "(num_classes - 1) = (num_labels + 1 - 1) = num_labels"
    # is reserved for the blank label
    # Range of true labels -> [0, len(voc_size))
    # Therefore, len(voc_size) is the default value for the CTC-blank index
    output = layers.Dense(num_labels + 1, activation="softmax")(x)

    # CTC-loss computation
    # Keras does not currently support loss functions with extra parameters,
    # so CTC loss is implemented in a Lambda layer
    ctc_loss = layers.Lambda(
        function=ctc_loss_lambda,
        output_shape=(1,),
        name="ctc_loss",
    )([label, output, image_len, label_len])

    # Create training model and predicition model
    model = keras.Model([image, image_len, label, label_len], ctc_loss)
    # The loss calculation is already done, so use a dummy lambda function for the loss
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss={"ctc_loss": lambda y_true, y_pred: y_pred},
    )
    # At inference time, we only have the image as input and the softmax prediction as output
    prediction_model = keras.Model(model.get_layer("image").input, output)

    return model, prediction_model
