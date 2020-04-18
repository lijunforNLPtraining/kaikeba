import tensorflow as tf
from encoders import encoder
from decoders import decoder
from utils.data_utils import load_word2vec


class SequenceToSequence(tf.keras.Model):
    def __init__(self, params):
        super(SequenceToSequence, self).__init__()
        self.embedding_matrix = load_word2vec(params)
        self.params = params
        self.encoder = encoder.Encoder(params["vocab_size"],
                                           params["embed_size"],
                                           params["enc_units"],
                                           params["batch_size"],
                                           self.embedding_matrix)
        self.attention = decoder.BahdanauAttention(params["attn_units"])
        self.decoder = decoder.Decoder(params["vocab_size"],
                                           params["embed_size"],
                                           params["dec_units"],
                                           params["batch_size"],
                                           self.embedding_matrix)

    def call_encoder(self, enc_inp):
        enc_hidden = self.encoder.initialize_hidden_state()
        # [batch_sz, max_train_x, enc_units], [batch_sz, enc_units]
        enc_output, enc_hidden = self.encoder.call(enc_inp, enc_hidden)
        return enc_output, enc_hidden

    def call(self, enc_output, dec_hidden, enc_inp, dec_inp):
        if self.params["mode"] == "train":
            outputs = self._decode_target(enc_output, dec_hidden, dec_inp)
            return outputs

    def _decode_target(self, enc_output, dec_hidden, dec_inp):
        predictions = []
        attentions = []
        context_vector, attn_dist = self.attention.call(dec_hidden,  # shape=(16, 256)
                                                   enc_output)  # shape=(16, 200, 256)
        for t in range(dec_inp.shape[1]):
            dec_x, pred, dec_hidden = self.decoder.call(tf.expand_dims(dec_inp[:, t], 1),
                                                   dec_hidden,
                                                   enc_output,
                                                   context_vector)
            context_vector, attn_dist = self.attention.call(dec_hidden, enc_output)
            predictions.append(pred)
            attentions.append(attn_dist)
        outputs = dict(logits=tf.stack(predictions, 1), dec_hidden=dec_hidden, attentions=attentions)
        return outputs

