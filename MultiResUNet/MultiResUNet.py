from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add
from tensorflow.keras.models import Model#, model_from_json
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.layers.advanced_activations import ELU, LeakyReLU
# from keras.utils.vis_utils import plot_model


from tensorflow.keras.layers import Lambda
from tensorflow.keras.backend import dropout

def PermaDropout(rate):
    '''a permanent dropout layer, which can be used to provide stochastic outputs at test time'''
    
    return Lambda(lambda x: dropout(x, level=rate))


def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None,dropout_rate=None):
    '''
    2D Convolutional layers
    
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
    
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(1, 1)})
        activation {str} -- activation function (default: {'relu'})
        name {str} -- name of the layer (default: {None})
    
    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    if(activation == None):
        return x

    x = Activation(activation, name=name)(x)
    
    if dropout_rate is not None:
        x = PermaDropout(dropout_rate)(x)

    return x


def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None,dropout_rate=None):
    '''
    2D Transposed Convolutional layers
    
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
    
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(2, 2)})
        name {str} -- name of the layer (default: {None})
    
    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3, scale=False)(x)
    
    if dropout_rate is not None:
        x = PermaDropout(dropout_rate)(x)
        
    return x


def MultiResBlock(U, inp, alpha = 1.67,individual_dropout_rate=None,output_dropout_rate=None):
    '''
    MultiRes Block
    
    Arguments:
        U {int} -- Number of filters in a corrsponding UNet stage
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''

    W = alpha * U

    shortcut = inp

    shortcut = conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) +
                         int(W*0.5), 1, 1, activation=None, padding='same',dropout_rate=individual_dropout_rate)

    conv3x3 = conv2d_bn(inp, int(W*0.167), 3, 3,
                        activation='relu', padding='same',dropout_rate=individual_dropout_rate)

    conv5x5 = conv2d_bn(conv3x3, int(W*0.333), 3, 3,
                        activation='relu', padding='same',dropout_rate=individual_dropout_rate)

    conv7x7 = conv2d_bn(conv5x5, int(W*0.5), 3, 3,
                        activation='relu', padding='same',dropout_rate=individual_dropout_rate)

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    out = BatchNormalization(axis=3)(out)

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)
    
    if output_dropout_rate is not None:
        out = PermaDropout(output_dropout_rate)(out)

    return out


def ResPath(filters, length, inp, individual_dropout_rate=None, output_dropout_rate=None):
    '''
    ResPath
    
    Arguments:
        filters {int} -- [description]
        length {int} -- length of ResPath
        inp {keras layer} -- input layer
        individual_dropout_rate {float between 0 and 1} -- the dropout rate applied to every single conv layer
        output_dropout_rate {float between 0 and 1} -- the dropout rate applied to the end of each residual "block"
    
    Returns:
        [keras layer] -- [output layer]
    '''


    shortcut = inp
    shortcut = conv2d_bn(shortcut, filters, 1, 1,
                         activation=None, padding='same')

    out = conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same',dropout_rate=individual_dropout_rate)

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    #dropouts added after BN for each shortcut "block"
    if output_dropout_rate is not None:
        out = PermaDropout(output_dropout_rate)(out)           

    for i in range(length-1):

        shortcut = out
        shortcut = conv2d_bn(shortcut, filters, 1, 1,
                             activation=None, padding='same',dropout_rate=individual_dropout_rate)

        out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same',dropout_rate=individual_dropout_rate)

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=3)(out)
        
        #dropouts added after BN for each shortcut "block"
        if output_dropout_rate is not None:
            out = PermaDropout(output_dropout_rate)(out)           

    return out


def MultiResUnet(height, width, n_channels, layer_dropout_rate=None , block_dropout_rate=None):
    '''
    MultiResUNet
    
    Arguments:
        height {int} -- height of image 
        width {int} -- width of image 
        n_channels {int} -- number of channels in image
        layer_dropout_rate {float between 0 and 1} -- the rate at which permanent dropout is applied to every single conv layer
        block_dropout_rate {float between 0 and 1} -- the rate at which permanent dropout is applied to each res block
    
    Returns:
        [keras model] -- MultiResUNet model
    '''

    inputs = Input((height, width, n_channels))

    mresblock1 = MultiResBlock(32, inputs,individual_dropout_rate=layer_dropout_rate,output_dropout_rate=block_dropout_rate)
    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
    mresblock1 = ResPath(32, 4, mresblock1,individual_dropout_rate=layer_dropout_rate,output_dropout_rate=block_dropout_rate)

    mresblock2 = MultiResBlock(32*2, pool1,individual_dropout_rate=layer_dropout_rate,output_dropout_rate=block_dropout_rate)
    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
    mresblock2 = ResPath(32*2, 3, mresblock2,individual_dropout_rate=layer_dropout_rate,output_dropout_rate=block_dropout_rate)

    mresblock3 = MultiResBlock(32*4, pool2,individual_dropout_rate=layer_dropout_rate,output_dropout_rate=block_dropout_rate)
    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
    mresblock3 = ResPath(32*4, 2, mresblock3,individual_dropout_rate=layer_dropout_rate,output_dropout_rate=block_dropout_rate)

    mresblock4 = MultiResBlock(32*8, pool3,individual_dropout_rate=layer_dropout_rate,output_dropout_rate=block_dropout_rate)
    pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
    mresblock4 = ResPath(32*8, 1, mresblock4,individual_dropout_rate=layer_dropout_rate,output_dropout_rate=block_dropout_rate)

    mresblock5 = MultiResBlock(32*16, pool4,individual_dropout_rate=layer_dropout_rate,output_dropout_rate=block_dropout_rate)

    up6 = concatenate([Conv2DTranspose(
        32*8, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4], axis=3)
    mresblock6 = MultiResBlock(32*8, up6,individual_dropout_rate=layer_dropout_rate,output_dropout_rate=block_dropout_rate)

    up7 = concatenate([Conv2DTranspose(
        32*4, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3], axis=3)
    mresblock7 = MultiResBlock(32*4, up7,individual_dropout_rate=layer_dropout_rate,output_dropout_rate=block_dropout_rate)

    up8 = concatenate([Conv2DTranspose(
        32*2, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2], axis=3)
    mresblock8 = MultiResBlock(32*2, up8,individual_dropout_rate=layer_dropout_rate,output_dropout_rate=block_dropout_rate)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(
        2, 2), padding='same')(mresblock8), mresblock1], axis=3)
    mresblock9 = MultiResBlock(32, up9,individual_dropout_rate=layer_dropout_rate,output_dropout_rate=block_dropout_rate)

    conv10 = conv2d_bn(mresblock9, 1, 1, 1, activation='sigmoid',individual_dropout_rate=layer_dropout_rate,output_dropout_rate=block_dropout_rate)
    
    model = Model(inputs=[inputs], outputs=[conv10])

    return model
   

def main():

    # Define the model

    model = MultiResUnet(128, 128,3)
    print(model.summary())

if __name__ == '__main__':
    main()
