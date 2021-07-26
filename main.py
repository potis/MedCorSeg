import logging
logger = logging.getLogger('Unet')

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Conv3DTranspose, concatenate, LeakyReLU, SpatialDropout3D, Add

from _instance_normalization import InstanceNormalization

def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3),
                             padding='same', strides=(1, 1, 1), instance_normalization=True):
    """

    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param padding:
    :return: layer
    """
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=-1)(layer)
    elif instance_normalization:
        layer = InstanceNormalization(axis=-1)(layer)
    return LeakyReLU()(layer)



def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        return Conv3DTranspose(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling3D(size=pool_size)

def create_localization_module(convlay, n_filters):
    convlay = create_convolution_block(convlay, n_filters)
    convlay = create_convolution_block(convlay, n_filters, kernel=(1, 1, 1))
    return convlay


def create_up_sampling_module(convlay, n_filters, size=(2, 2, 2)):
    up_sample = UpSampling3D(size=size)(convlay)
    convlay = create_convolution_block(up_sample, n_filters)
    return convlay


def create_context_module(input_layer, n_level_filters, dropout_rate=0.3):
    convlay = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout3D(rate=dropout_rate)(convlay)
    convlay = create_convolution_block(input_layer=dropout, n_filters=n_level_filters)
    return convlay




def KindeyMultiClass(input_shape=(128, 128, 128, 4), initial_filters=16, depth=5, dropout_rate=0.3,
                      n_segmentation_levels=3, classes=4, out_activation="softmax"):
    """

    :param input_shape:
    :param initial_filters:
    :param depth:
    :param spacial dropout_rate:
    :param n_segmentation_levels:
    :param classes:
    :param out_activation:
    :return:
    """
    inputs = Input(input_shape)
    current_layer = inputs

    inputsw=inputs



    level_output_layers = list()
    level_filters = list()
    for level_number in range(depth):
        n_level_filters = (2**level_number) * initial_filters
        level_filters.append(n_level_filters)

        if current_layer is inputsw:

            in_conv = create_convolution_block(current_layer, n_level_filters)
        else:
            in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2, 2))

        context_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

        sum_l = Add()([in_conv, context_layer])
        level_output_layers.append(sum_l)
        current_layer = sum_l

    segmentation_layers = list()
    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])
        concatenation_layer = concatenate([level_output_layers[level_number], up_sampling])
        localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, Conv3D(classes, (1, 1, 1))(current_layer))

    output_layer = None
    for level_number in reversed(range(n_segmentation_levels)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling3D(size=(2, 2, 2))(output_layer)

    activation_block = Activation(out_activation)(output_layer)
    model = Model(inputs=inputs, outputs=activation_block)
    return model








def papermodel(params={}):
    shape_with_channels = params.get('image_shape', (128,128,128,1))
    classes = params.get("classes", 5) # Account for Background
    initial_filters = params.get("filters ", 16)
    depth = params.get("depth", 6)
    dropout_rate= params.get("dropout_rate",0.25) # Spacial drop out
    out_activation = params.get("activation", 'softmax')
    logger.info('parameters used for this model')
    logger.info("shape_with_channels = {}".format(shape_with_channels))
    logger.info("number_of_classes = {}".format(classes))
    model = KindeyMultiClass(input_shape=shape_with_channels, classes=classes, initial_filters=initial_filters, depth=depth,
                      out_activation=out_activation)
    return model


print(papermodel().summary())