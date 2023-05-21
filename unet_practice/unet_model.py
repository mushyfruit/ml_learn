from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
def unet(pretrained_weights=None, input_size=(256, 256, 1)):
    # Used to instantiate Keras tensor.
    inputs = Input(input_size)
    conv_1 = Conv2D(64, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(inputs)
    conv_1 = Conv2D(64, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(conv_1)
    pool_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)

    # Second convolution chain
    conv_2 = Conv2D(128, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(pool_1)
    conv_2 = Conv2D(128, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(conv_2)
    pool_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)

    # Third convolution chain
    conv_3 = Conv2D(256, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(pool_2)
    conv_3 = Conv2D(256, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(conv_3)
    pool_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)

    # Fourth convolution chain
    conv_4 = Conv2D(256, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(pool_3)
    conv_4 = Conv2D(256, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(conv_4)
    pool_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)