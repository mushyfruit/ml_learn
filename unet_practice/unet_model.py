from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import UpSampling2D, Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam


def unet(pretrained_weights=None, input_size=(256, 256, 1)):
    # Used to instantiate Keras tensor.
    n_filters = 64
    inputs = Input(input_size)
    # original U-Net paper uses valid padding.
    # practical implementations now use same padding.
    conv_1 = Conv2D(n_filters, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(inputs)
    conv_1 = Conv2D(n_filters, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(conv_1)
    pool_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)

    # Second convolution chain
    conv_2 = Conv2D(n_filters*2, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(pool_1)
    conv_2 = Conv2D(n_filters*2, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(conv_2)
    pool_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)

    # Third convolution chain
    conv_3 = Conv2D(n_filters*4, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(pool_2)
    conv_3 = Conv2D(n_filters*4, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(conv_3)
    pool_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)

    # Fourth convolution chain
    conv_4 = Conv2D(n_filters*8, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(pool_3)
    conv_4 = Conv2D(n_filters*8, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(conv_4)
    drop_4 = Dropout(0.5)(conv_4)
    pool_4 = MaxPooling2D(pool_size=(2, 2))(drop_4)

    # Fifth convolution chain, just conv + drop
    conv_5 = Conv2D(n_filters*16, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(pool_4)
    conv_5 = Conv2D(n_filters*16, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(conv_5)
    drop_5 = Dropout(0.5)(conv_5)

    # UpSampling2D vs. Conv2D Transpose
    # Former is computationally less expensive.
    # (batch_size, height*2, width*2, channels)
    # upsample_1 = UpSampling2D(size=(2, 2))(drop_5)
    # up_6 = Conv2D(n_filters * 8, 2, activation='relu', padding='same',
    #              kernel_initializer='he_normal')(upsample_1)

    # upsample chain 1
    up_6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2),
                           padding='same')(drop_5)
    concat_6 = concatenate([drop_4, up_6], axis=3)
    # (e.g. -> 56x56x512 + 56x56x512 = 56x56x1024)

    # distilling high + low level features from concat
    conv_6 = Conv2D(n_filters * 8, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(concat_6)
    conv_6 = Conv2D(n_filters * 8, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(conv_6)

    # upsampling chain 2
    up_7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2),
                           padding='same')(conv_6)
    concat_7 = concatenate([conv_3, up_7], axis=3)
    conv_7 = Conv2D(n_filters * 4, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(concat_7)
    conv_7 = Conv2D(n_filters * 4, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(conv_7)

    # upsampling chain 3
    up_8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2),
                           padding='same')(conv_7)
    concat_8 = concatenate([conv_2, up_8], axis=3)
    conv_8 = Conv2D(n_filters * 2, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(concat_8)
    conv_8 = Conv2D(n_filters * 2, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(conv_8)

    # upsampling chain 4
    up_9 = Conv2DTranspose(n_filters, (3, 3), strides=(2, 2),
                           padding='same')(conv_8)
    concat_9 = concatenate([conv_1, up_9], axis=3)
    conv_9 = Conv2D(n_filters, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(concat_9)
    conv_9 = Conv2D(n_filters, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(conv_9)

    conv_9 = Conv2D(2, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(conv_9)
    conv_10 = Conv2D(1, 1, activation='sigmoid')(conv_9)

    model = Model(inputs=inputs, outputs=conv_10)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
