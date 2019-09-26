from detect.basic import BasicClassifier
from keras.layers import *
from keras.models import load_model
from keras_layer_normalization import LayerNormalization
from keras_self_attention import SeqSelfAttention

from detect.capsule_layers import CapsuleLayer, PrimaryCap, Length


class LSTMClassifier(BasicClassifier):
    """
    < Predicting domain generation algorithms with long short-term memory networks >
    """
    def __init__(self, embed_size=128, lstm_unit=128, dropout_r=0.5):
        super().__init__("LSTM")

        self.embed_size = embed_size   
        self.lstm_unit = lstm_unit
        self.dropout_r = dropout_r 

    def extract_features(self, x):
        x = Embedding(self.n_words, self.embed_size, input_length=self.input_size)(x)
        x = LSTM(self.lstm_unit)(x)
        x = Dropout(self.dropout_r)(x)
        return x 

class GRUClassifier(BasicClassifier):
    """
    < Automatic Detection of Malware-Generated Domains with Recurrent Neural Models >
    """
    def __init__(self, embed_size=128, gru_unit=128, dropout_r=0.5):
        super().__init__("GRU")

        self.embed_size = embed_size   
        self.gru_unit = gru_unit
        self.dropout_r = dropout_r 

    def extract_features(self, x):
        x = Embedding(self.n_words, self.embed_size, input_length=self.input_size)(x)
        x = GRU(self.gru_unit)(x)
        x = Dropout(self.dropout_r)(x)
        return x 

class BiLSTMClassifier(BasicClassifier):
    """
    < Tweet2vec: Character-based distributed representations for social media >
    """
    def __init__(self, embed_size=128, lstm_unit=64, dropout_r=0.5):
        super().__init__("BiLSTM")

        self.embed_size = embed_size   
        self.lstm_unit = lstm_unit
        self.dropout_r = dropout_r 

    def extract_features(self, x):
        x = Embedding(self.n_words, self.embed_size, input_length=self.input_size)(x)
        x = Bidirectional(LSTM(self.lstm_unit))(x)
        x = Dropout(self.dropout_r)(x)
        return x 

class EXposeClassifier(BasicClassifier):
    """
    < eXpose: A character-level convolutional neural network with embeddings 
        for detecting malicious urls, file paths and registry keys >
    """
    def __init__(self, embed_size=32, filters=256, kernels=(2,3,4,5), dropout_r=0.5, fc_unit=1024, fc_layers=3):
        super().__init__("eXpose")

        self.embed_size = embed_size
        self.filters = filters
        self.kernels = kernels
        self.dropout_r = dropout_r
        self.fc_unit = fc_unit
        self.fc_layers = fc_layers
    
    def extract_features(self, x):
        x = Embedding(self.n_words, self.embed_size, input_length=self.input_size)(x)

        convs = list()
        for k in self.kernels:
            conv = Conv1D(filters=self.filters, kernel_size=k, padding="same", activation="relu")(x)
            conv = Lambda(lambda x: K.sum(x, axis=1), output_shape=(self.filters,))(conv)
            conv = Dropout(self.dropout_r)(conv)
            convs.append(conv)
        x = Concatenate()(convs)
        x = LayerNormalization()(x)

        for i in range(self.fc_layers):
            x = Dense(self.fc_unit)(x)
            x = LayerNormalization()(x)
            x = Dropout(self.dropout_r)(x)
        
        return x
    
    def load_model(self, model_path):
        self.model = load_model(model_path, custom_objects={"LayerNormalization": LayerNormalization})
        
class CNNLSTMClassifier(BasicClassifier):
    """
    < Tweet2vec: Learning tweet embeddings using character-level cnn-lstm encoder-decoder >
    """
    def __init__(self, embed_size=128, filters=128, kernel_size=3, pool_size=2, lstm_unit=64):
        super().__init__("CNNLSTM")

        self.embed_size = embed_size
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.lstm_unit = lstm_unit

    def extract_features(self, x):
        x = Embedding(self.n_words, self.embed_size, input_length=self.input_size)(x)
        conv = Conv1D(filters=self.filters, kernel_size=self.kernel_size,
                      padding="same", activation="relu", strides=1)(x)
        max_pool = MaxPooling1D(pool_size=self.pool_size, padding="same")(conv)
        encode = LSTM(self.lstm_unit, return_sequences=False)(max_pool)

        return encode

class StackedCNNClassifier(BasicClassifier):
    """
    < Character-level convolutional networks for text classificatio >
    """
    def __init__(self, conv_layers = [[128, 3, 2], [128, 2, 2]], fc_layers = [1024], 
                 embed_size=128, threshold=1e-6, dropout_p=0.5):
        super().__init__("StackedCNN")

        self.embed_size = embed_size
        self.conv_layers = conv_layers
        self.fc_layers = fc_layers
        self.threshold = threshold
        self.dropout_p = dropout_p

    def extract_features(self, x):
        x = Embedding(self.n_words, self.embed_size, input_length=self.input_size)(x)
        #x = Reshape((self.embed_size*self.input_size, 1))(x)
       
        for cl in self.conv_layers:
            x = Conv1D(filters=cl[0], kernel_size=cl[1])(x)
            x = ThresholdedReLU(self.threshold)(x)
            if cl[2] != -1:
                x = MaxPooling1D(cl[2])(x)
        x = Flatten()(x)
       
        for fl in self.fc_layers:
            x = Dense(fl)(x)
            x = ThresholdedReLU(self.threshold)(x)
            x = Dropout(self.dropout_p)(x)

        return x

class TCNClassifier(BasicClassifier):
    def __init__(self, embed_size=128, dilations=None, filters=64, kernel_size=2, n_stacks=1, dropout_r=0.0):
        super().__init__("TCN")

        self.embed_size = embed_size
        self.dilations = dilations
        self.filters = filters
        self.kernel_size = kernel_size
        self.n_stacks = n_stacks
        self.dropout_r = dropout_r

    def extract_features(self, x):
        if self.dilations is None:
            self.dilations = [1, 2, 4, 8, 16, 32]

        x = Embedding(self.n_words, self.embed_size, input_length=self.input_size)(x)
        x = Conv1D(self.filters, 1, padding='same', name='initial_conv')(x)
        skip_connections = []
        for s in range(self.n_stacks):
            for i in self.dilations:
                x, skip_out = self.residual_block(x, s, i)
                skip_connections.append(skip_out)
        
        x = Add()(skip_connections)
        x = Activation('relu')(x)
        x = Flatten()(x)

        return x

    def residual_block(self, x, s, i):
        original_x = x
        conv = Conv1D(filters=self.filters, kernel_size=self.kernel_size,
                      dilation_rate=i, padding='causal',
                      name='dilated_conv_%d_tanh_s%d' % (i, s))(x)
        x = Activation('relu')(conv)
        x = Lambda(self.channel_normalization)(x)
        x = SpatialDropout1D(self.dropout_r, name='spatial_dropout1d_%d_s%d_%f' % (i, s, self.dropout_r))(x)
        x = Convolution1D(self.filters, 1, padding='same')(x) # 1x1 conv.
        res_x = Add()([original_x, x])
        return res_x, x
    
    @staticmethod
    def channel_normalization(x):
        max_values = K.max(K.abs(x), 2, keepdims=True) + 1e-5
        out = x / max_values
        return out

class AttentionClassifier(BasicClassifier):
    def __init__(self, embed_size=128, lstm_unit=64):
        super().__init__("Attention")
        self.embed_size = embed_size
        self.lstm_unit = lstm_unit

    def extract_features(self, x):
        x = Embedding(self.n_words, self.embed_size, input_length=self.input_size)(x)
        x = Bidirectional(LSTM(self.lstm_unit, return_sequences=True))(x)
        x = SeqSelfAttention(attention_activation='sigmoid')(x)
        x = Flatten()(x)
        return x

    def load_model(self, model_path):
        self.model = load_model(model_path, custom_objects=SeqSelfAttention.get_custom_objects())

class CapsuleClassifier(BasicClassifier):
    def __init__(self, embed_size=128):
        super().__init__("Capsule")
        self.embed_size = embed_size

    def extract_features(self, x):
        x = Embedding(self.n_words, self.embed_size, input_length=self.input_size)(x)
        x = Conv1D(filters=256, kernel_size=8, padding="valid")(x)
        x = SpatialDropout1D(0.2)(x)
        x = Conv1D(filters=512, kernel_size=4, padding="valid")(x)
        x = Dropout(0.7)(x)
        x = Conv1D(filters=256, kernel_size=4, padding="valid")(x)
        x = PrimaryCap(inputs=x, dim_capsule=8, n_channels=32, kernel_size=4, strides=2, padding='valid')
        x = CapsuleLayer(num_capsule=1, dim_capsule=16, routing=7)(x) 
        x = Length(0.85, 0.15)(x)
        return x

    def load_model(self, model_path):
        self.model = load_model(model_path, custom_objects={"CapsuleLayer":CapsuleLayer, "Length":Length})

