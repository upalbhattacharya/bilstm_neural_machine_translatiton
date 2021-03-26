import tensorflow as tf

"""
Creating a custom attention-based BiLSTM model from scratch(not using tf LSTM and tf Bidirectional) to create a neural machine translation model to translate text from English to German
"""

# Defining the custom layers


class EncoderLSTM(tf.keras.layers.Layer):

    def __init__(self, batch_size, units, **kwargs):
        super(EncoderLSTM, self).__init__()
        self.units = units
        self.batch_size = batch_size

    def build(self, input_shape):
        """
        Shapes

        inputs = (batch_size, seq_len, embed_dim)
        x_t = (batch_size, 1, embed_dim) = (batch_size, embed_dim)
        s_t = (batch_size, units)
        f_t = (batch_size, units)
        g_t = (batch_size, units)
        imm s_t = (batch_size, units)
        b = ( 1, units) (forget, imm. state and extinput)
        W^x = (embed_dim, units)
        h_t = (batch_size, units)
        W^h = (units, units)
        q_t = (batch_size, units)
        b_q = (1, units)
        W_q^x = (embed_dim, units)
        W_q^h = (units, units)

        While it is possible to have a separate hidden_dim for h_t, W^h, q_t, b_q, W_q^x and W_q^h such that their shapes are:
        h_t = (batch_size, hidden_dim)
        W^h = (hidden_dim, units)
        q_t = (batch_size, hidden_dim)
        b_q = (1, hidden_dim)
        W_q^x = (embed_dim, hidden_dim)
        W_q^h = (hidden_dim, hidden_dim)

        the Keras LSTM layer implementation does not use a separate hidden_dim"""
        # Forget gate
        self.forget_input_weight = self.add_weight("forget_input_weight",
                                                   shape=[int(input_shape[-1]), self.units], initializer='glorot_normal')
        self.forget_hidden_weight = self.add_weight("forget_hidden_weight",
                                                    shape=[self.units, self.units], initializer='glorot_normal')
        self.forget_bias = self.add_weight(
            "forget_bias", shape=[1, self.units], initializer='Zeros')

        # External Input gate
        self.extinput_input_weight = self.add_weight("extinput_input_weight",
                                                     shape=[int(input_shape[-1]), self.units], initializer='glorot_normal')
        self.extinput_hidden_weight = self.add_weight("extinput_hidden_weight",
                                                      shape=[self.units, self.units], initializer='glorot_normal')
        self.extinput_bias = self.add_weight(
            "extinput_bias", shape=[1, self.units], initializer='Zeros')

        # Output control gate
        self.output_input_weight = self.add_weight("output_input_weight",
                                                   shape=[int(input_shape[-1]), self.units], initializer='glorot_normal')
        self.output_hidden_weight = self.add_weight("output_hidden_weight",
                                                    shape=[self.units, self.units], initializer='glorot_normal')
        self.output_bias = self.add_weight(
            "output_bias", shape=[1, self.units], initializer='Zeros')

        # Immemdiate state
        self.imm_state_input_weight = self.add_weight("imm_state_input_weight",
                                                      shape=[int(input_shape[-1]), self.units], initializer='glorot_normal')
        self.imm_state_hidden_weight = self.add_weight("imm_state_hidden_weight",
                                                       shape=[self.units, self.units], initializer='glorot_normal')
        self.imm_state_bias = self.add_weight(
            "imm_state_bias", shape=[1, self.units], initializer='Zeros')

    def call(self, inputs):
        time_steps = tf.unstack(inputs, axis=1)
        hidden = tf.Variable(
            tf.zeros([self.batch_size, self.units]), trainable=False)
        state = tf.Variable(
            tf.zeros([self.batch_size, self.units]), trainable=False)
        hidden_states = []

        for time_step in time_steps:

            forget_gate = tf.matmul(time_step, self.forget_input_weight) + \
                tf.matmul(hidden, self.forget_hidden_weight) + \
                tf.tile(self.forget_bias, [self.batch_size, 1])
            forget_gate = tf.math.sigmoid(forget_gate)

            extinput_gate = tf.matmul(time_step, self.extinput_input_weight) + \
                tf.matmul(hidden, self.extinput_hidden_weight) + \
                tf.tile(self.extinput_bias, [self.batch_size, 1])
            extinput_gate = tf.math.sigmoid(extinput_gate)

            output_gate = tf.matmul(time_step, self.output_input_weight) + \
                tf.matmul(hidden, self.output_hidden_weight) + \
                tf.tile(self.output_bias, [self.batch_size, 1])
            output_gate = tf.math.sigmoid(output_gate)

            imm_state_gate = tf.matmul(time_step, self.imm_state_input_weight) + \
                tf.matmul(hidden, self.imm_state_hidden_weight) + \
                tf.tile(self.imm_state_bias, [self.batch_size, 1])
            imm_state_gate = tf.math.sigmoid(imm_state_gate)

            state = tf.multiply(forget_gate, state) + \
                tf.multiply(extinput_gate, imm_state_gate)

            hidden = tf.multiply(tf.math.tanh(state), output_gate)

            hidden_states.append(hidden)

        return tf.stack(hidden_states, axis=1)


class Attention(tf.keras.layers.Layer):
    def __init__(self, prev_state_shape, enc_hidden_shape, units):
        super(Attention, self).__init__()
        self.units = units
        self.prev_state_shape = prev_state_shape
        self.enc_hidden_shape = enc_hidden_shape

    def build(self, input_shape):
        self.query_weight = self.add_weight("query_weight", shape=[
                                            self.prev_state_shape[-1], self.units], initializer='glorot_normal')

        self.values_weight = self.add_weight("values_weight", shape=[
                                             self.enc_hidden_shape[-1], self.units], initializer='glorot_normal')

        self.add_weight = self.add_weight(
            "add_weight", shape=[self.units, 1], initializer='glorot_normal')

    def call(self, prev_state, enc_hidden):
        batch_size = enc_hidden.shape[0]
        seq_len = enc_hidden.shape[1]
        enc_units = enc_hidden.shape[-1]

        align = tf.matmul(enc_hidden, tf.tile(tf.expand_dims(
            self.values_weight, 0), [batch_size, 1, 1]))
        align += tf.tile(tf.expand_dims(tf.matmul(prev_state,
                                                  self.query_weight), 1), [1, seq_len, 1])
        align = tf.nn.tanh(align)
        align = tf.matmul(align, tf.tile(tf.expand_dims(
            self.add_weight, 0), [batch_size, 1, 1]))
        alpha = tf.nn.softmax(align, axis=1)
        context = tf.reduce_sum(tf.multiply(
            tf.tile(alpha, [1, 1, enc_units]), enc_hidden), axis=1, keepdims=True)

        return context


class DecoderLSTM(tf.keras.layers.Layer):

    def __init__(self, batch_size, units, output_dim, encoder_units, attn_units, **kwargs):
        super(DecoderLSTM, self).__init__()
        self.units = units
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.encoder_units = encoder_units
        self.attn_units = attn_units

    def build(self, input_shape):
        """
        Shapes

        inputs = (batch_size, seq_len, embed_dim)
        x_t = (batch_size, 1, embed_dim) = (batch_size, embed_dim)
        s_t = (batch_size, units)
        f_t = (batch_size, units)
        g_t = (batch_size, units)
        imm s_t = (batch_size, units)
        b = ( 1, units) (forget, imm. state and extinput)
        W^x = (embed_dim, units)
        h_t = (batch_size, units)
        W^h = (units, units)
        q_t = (batch_size, units)
        b_q = (1, units)
        W_q^x = (embed_dim, units)
        W_q^h = (units, units)

        While it is possible to have a separate hidden_dim for h_t, W^h, q_t, b_q, W_q^x and W_q^h such that their shapes are:
        h_t = (batch_size, hidden_dim)
        W^h = (hidden_dim, units)
        q_t = (batch_size, hidden_dim)
        b_q = (1, hidden_dim)
        W_q^x = (embed_dim, hidden_dim)
        W_q^h = (hidden_dim, hidden_dim)

        the Keras LSTM layer implementation does not use a separate hidden_dim"""
        # Forget gate
        self.forget_input_weight = self.add_weight("forget_input_weight",
                                                   shape=[self.output_dim, self.units], initializer='glorot_normal')
        self.forget_hidden_weight = self.add_weight("forget_hidden_weight",
                                                    shape=[self.units, self.units], initializer='glorot_normal')
        self.forget_bias = self.add_weight(
            "forget_bias", shape=[1, self.units], initializer='Zeros')

        self.forget_context_weight = self.add_weight("forget_context_weight",
                                                     shape=[self.encoder_units, self.units], initializer='glorot_normal')

        # External Input gate
        self.extinput_input_weight = self.add_weight("extinput_input_weight",
                                                     shape=[self.output_dim, self.units], initializer='glorot_normal')
        self.extinput_hidden_weight = self.add_weight("extinput_hidden_weight",
                                                      shape=[self.units, self.units], initializer='glorot_normal')
        self.extinput_bias = self.add_weight(
            "extinput_bias", shape=[1, self.units], initializer='Zeros')

        self.extinput_context_weight = self.add_weight("extinput_context_weight",
                                                       shape=[self.encoder_units, self.units], initializer='glorot_normal')

        # Output control gate
        self.output_input_weight = self.add_weight("output_input_weight",
                                                   shape=[self.output_dim, self.units], initializer='glorot_normal')
        self.output_hidden_weight = self.add_weight("output_hidden_weight",
                                                    shape=[self.units, self.units], initializer='glorot_normal')
        self.output_bias = self.add_weight(
            "output_bias", shape=[1, self.units], initializer='Zeros')

        self.output_context_weight = self.add_weight("output_context_weight",
                                                     shape=[self.encoder_units, self.units], initializer='glorot_normal')

        # Immemdiate state
        self.imm_state_input_weight = self.add_weight("imm_state_input_weight",
                                                      shape=[self.output_dim, self.units], initializer='glorot_normal')
        self.imm_state_hidden_weight = self.add_weight("imm_state_hidden_weight",
                                                       shape=[self.units, self.units], initializer='glorot_normal')
        self.imm_state_bias = self.add_weight(
            "imm_state_bias", shape=[1, self.units], initializer='Zeros')

        self.imm_state_context_weight = self.add_weight("imm_state_context_weight",
                                                        shape=[self.encoder_units, self.units], initializer='glorot_normal')

        # Prediction weights
        self.pred_weight = self.add_weight("pred_weight",
                                           shape=[self.units, self.output_dim], initializer='glorot_normal')
        self.pred_bias = self.add_weight(
            "pred_bias", shape=[1, self.output_dim], initializer='Zeros')

    def call(self, initial_y, enc_hidden_states):
        hidden = tf.Variable(
            tf.zeros([self.batch_size, self.units]), trainable=False)
        state = tf.Variable(
            tf.zeros([self.batch_size, self.units]), trainable=False)
        output = tf.squeeze(initial_y)
        output_states = []
        i = 0
        while(i != 30):  # token !=STOP probably

            attention = Attention(prev_state_shape=initial_y.shape,
                                  enc_hidden_shape=enc_hidden_states.shape, units=self.attn_units)
            context = attention(prev_state=output,
                                enc_hidden=enc_hidden_states)
            context = tf.squeeze(context)

            forget_gate = tf.matmul(output, self.forget_input_weight) + \
                tf.matmul(hidden, self.forget_hidden_weight) + \
                tf.matmul(context, self.forget_context_weight) + \
                tf.tile(self.forget_bias, [self.batch_size, 1])
            forget_gate = tf.math.sigmoid(forget_gate)

            extinput_gate = tf.matmul(output, self.extinput_input_weight) + \
                tf.matmul(hidden, self.extinput_hidden_weight) + \
                tf.matmul(context, self.extinput_context_weight) + \
                tf.tile(self.extinput_bias, [self.batch_size, 1])
            extinput_gate = tf.math.sigmoid(extinput_gate)

            output_gate = tf.matmul(output, self.output_input_weight) + \
                tf.matmul(hidden, self.output_hidden_weight) + \
                tf.matmul(context, self.output_context_weight) + \
                tf.tile(self.output_bias, [self.batch_size, 1])
            output_gate = tf.math.sigmoid(output_gate)

            imm_state_gate = tf.matmul(output, self.imm_state_input_weight) + \
                tf.matmul(hidden, self.imm_state_hidden_weight) + \
                tf.matmul(context, self.forget_context_weight) + \
                tf.tile(self.imm_state_bias, [self.batch_size, 1])
            imm_state_gate = tf.math.sigmoid(imm_state_gate)

            state = tf.multiply(forget_gate, state) + \
                tf.multiply(extinput_gate, imm_state_gate)

            hidden = tf.multiply(tf.math.tanh(state), output_gate)

            output = tf.matmul(hidden, self.pred_weight) + self.pred_bias
            output = tf.nn.softmax(output)
            # i = i+1 # For testing
            # tf.print(output) # For testing
            output_states.append(output)

        return tf.stack(output_states, axis=1)


class BiLSTMModel(tf.keras.Model):

    def __init__(self, batch_size, encoder_units, decoder_units, attn_units, output_dim, **kwargs):
        super(BiLSTMModel, self).__init__()
        self.batch_size = batch_size
        self.encoder_units = encoder_units
        self.decoder_units = decoder_units
        self.output_dim = output_dim
        self.attn_units = attn_units

        self.forward_encoder = EncoderLSTM(
            batch_size=self.batch_size, units=self.encoder_units)
        self.backward_encoder = EncoderLSTM(
            batch_size=self.batch_size, units=self.encoder_units)
        self.decoder = DecoderLSTM(
            batch_size=self.batch_size, units=self.decoder_units, output_dim=self.output_dim, encoder_units=2*self.encoder_units, attn_units=self.attn_units)

    def call(self, X, initial_y):
        forward_hidden_states = self.forward_encoder(X)
        backward_hidden_states = self.backward_encoder(tf.reverse(X, axis=[1]))

        enc_hidden_states = tf.concat(
            [forward_hidden_states, backward_hidden_states], -1)

        output_states = self.decoder(initial_y, enc_hidden_states)

        return output_states
