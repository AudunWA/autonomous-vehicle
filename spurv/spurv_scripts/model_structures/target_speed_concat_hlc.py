from keras import Input, Model
from keras.layers import TimeDistributed, Flatten, concatenate, Dense, CuDNNLSTM, Lambda

from keras_segmentation.models import pspnet


def get_segmentation_model(freeze):
    segmentation_model = pspnet.pspnet(8, 192, 192)
    x = segmentation_model.get_layer("activation_10").output
    # Explicitly define new model input and output by slicing out old model layers
    model_new = Model(inputs=segmentation_model.layers[0].input,
                      outputs=x)

    if freeze:
        for layer in model_new.layers:
            layer.trainable = False

    return model_new


def get_target_speed_concat_hlc(seq_length, sine_steering, freeze):
    hlc_input = Input(shape=(seq_length, 4), name="hlc_input")
    info_input = Input(shape=(seq_length, 3), name="info_input")

    segmentation_model = get_segmentation_model(freeze)
    [_, height, width, _] = segmentation_model.input.shape.dims
    forward_image_input = Input(shape=(seq_length, height.value, width.value, 3), name="forward_image_input")
    segmentation_output = TimeDistributed(segmentation_model)(forward_image_input)
    segmentation_output = TimeDistributed(Flatten())(segmentation_output)

    # segmentation_output = Dropout(0.1)(segmentation_output)
    # segmentation_output = BatchNormalization()(segmentation_output)
    # segmentation_output = Activation(activation="relu")(segmentation_output)

    x = concatenate([segmentation_output, hlc_input, info_input])
    # x = Dropout(0.2)(x)

    x = TimeDistributed(Dense(100, activation="relu"))(x)
    x = concatenate([x, hlc_input])
    x = CuDNNLSTM(10, return_sequences=False)(x)
    hlc_latest = Lambda(lambda x: x[:, -1, :])(hlc_input)
    x = concatenate([x, hlc_latest])

    steer_dim = 1 if not sine_steering else 10
    steer_pred = Dense(steer_dim, activation="tanh", name="steer_pred")(x)

    target_speed_pred = Dense(1, name="target_speed_pred", activation="sigmoid")(x)
    model = Model(inputs=[forward_image_input, hlc_input, info_input], outputs=[steer_pred, target_speed_pred])
    model.summary()

    return model