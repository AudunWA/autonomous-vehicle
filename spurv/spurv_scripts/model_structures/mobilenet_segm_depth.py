from keras import Input, Model
from keras.layers import TimeDistributed, Flatten, concatenate, Dense, CuDNNLSTM, Lambda, ZeroPadding2D, Conv2D, BatchNormalization, Activation, MaxPooling2D
# from tensorflow.python.keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, Activation, MaxPooling2D

from keras_segmentation.models import pspnet
from keras_segmentation.models.mobilenet import get_mobilenet_encoder
from unet_depth_segm import _unet_depth_segm


def get_segmentation_model(freeze):
    segmentation_model = _unet_depth_segm(5, input_height=224, input_width=224, encoder=get_mobilenet_encoder)
    # Concat segmentation and depth output
    segm_output_layer = segmentation_model.get_layer(name="conv2d_5").output
    depth_output_layer = segmentation_model.get_layer(name="conv2d_10").output

    x = concatenate([segm_output_layer, depth_output_layer], axis=3)
    # Explicitly define new model input and output by slicing out old model layers
    model_new = Model(inputs=segmentation_model.layers[0].input,
                      outputs=x)

    if freeze:
        for layer in model_new.layers:
            layer.trainable = False

    return model_new


def get_mobilenet_segm_depth(seq_length, sine_steering, freeze,asd):
    hlc_input = Input(shape=(seq_length, 4), name="hlc_input")
    info_input = Input(shape=(seq_length, 3), name="info_input")

    segmentation_model = get_segmentation_model(freeze)
    [_, height, width, _] = segmentation_model.input.shape.dims
    forward_image_input = Input(shape=(seq_length, height.value, width.value, 3), name="forward_image_input")
    segmentation_output = TimeDistributed(segmentation_model)(forward_image_input)

    # Vanilla encoder
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    x = segmentation_output
    x = TimeDistributed(ZeroPadding2D((pad, pad)))(x)
    x = TimeDistributed(Conv2D(filter_size, (kernel, kernel),
                               padding='valid'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)
    x = TimeDistributed(MaxPooling2D((pool_size, pool_size)))(x)

    x = TimeDistributed(ZeroPadding2D((pad, pad)))(x)
    x = TimeDistributed(Conv2D(128, (kernel, kernel),
                               padding='valid'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)
    x = TimeDistributed(MaxPooling2D((pool_size, pool_size)))(x)

    for _ in range(3):
        x = TimeDistributed(ZeroPadding2D((pad, pad)))(x)
        x = TimeDistributed(Conv2D(256, (kernel, kernel), padding='valid'))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Activation('relu'))(x)
        x = TimeDistributed(MaxPooling2D((pool_size, pool_size)))(x)

    segmentation_output = x
    segmentation_output = TimeDistributed(Flatten())(segmentation_output)


    x = concatenate([segmentation_output, hlc_input, info_input])
    # x = Dropout(0.2)(x)

    x = TimeDistributed(Dense(100, activation="relu"))(x)
    x = concatenate([x, hlc_input])
    x = CuDNNLSTM(10, return_sequences=False)(x)
    hlc_latest = Lambda(lambda x: x[:, -1, :])(hlc_input)
    x = concatenate([x, hlc_latest])

    if sine_steering:
        steer_pred = Dense(10, activation="tanh", name="steer_pred")(x)
    else:
        steer_pred = Dense(1, activation="relu", name="steer_pred")(x)

    target_speed_pred = Dense(1, name="target_speed_pred", activation="sigmoid")(x)
    model = Model(inputs=[forward_image_input, hlc_input, info_input], outputs=[steer_pred, target_speed_pred])
    model.summary()

    return model