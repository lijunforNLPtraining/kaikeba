import tensorflow as tf
import time
from models.losses import loss_function


def train_model(model, dataset, params, ckpt, ckpt_manager):
    optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=params["learning_rate"])
    # @tf.function()
    def train_step(enc_inp, enc_extended_inp, dec_inp, dec_tar, batch_oov_len, enc_padding_mask, padding_mask):
        with tf.GradientTape() as tape:
            print('enc_inp shape is final for model :', enc_inp.get_shape())
            enc_output, enc_hidden = model.call_encoder(enc_inp)
            dec_hidden = enc_hidden
            if params["model"] == "SequenceToSequence":
                print('enc_output shape is :',enc_output.get_shape())
                print('dec_hidden shape is :', dec_hidden.get_shape())
                print('enc_inp shape is :', enc_inp.get_shape())
                print('dec_inp shape is :', dec_inp.get_shape())
                outputs = model(enc_output,  # shape=(3, 200, 256)
                                dec_hidden,  # shape=(3, 256)
                                enc_inp,  # shape=(3, 200)
                                dec_inp)  # shape=(3, 50)
            loss = loss_function(dec_tar,
                                 outputs,
                                 padding_mask,
                                 0.5,
                                 False)

        variables = model.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return loss

    for epoch in range(params['epochs']):
        t0 = time.time()
        step = 0
        for step, batch in enumerate(dataset.take(params['steps_per_epoch'])):
            # for batch in dataset:
            loss = train_step(batch[0]["enc_input"],  # shape=(16, 200)
                              batch[0]["extended_enc_input"],  # shape=(16, 200)
                              batch[1]["dec_input"],  # shape=(16, 50)
                              batch[1]["dec_target"],  # shape=(16, 50)
                              batch[0]["max_oov_len"],  # ()
                              batch[0]["sample_encoder_pad_mask"],  # shape=(16, 200)
                              batch[1]["sample_decoder_pad_mask"])  # shape=(16, 50)

            step += 1
            if step % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, step, loss.numpy()))

        if epoch % 1 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {} ,best loss {}'.format(epoch + 1, ckpt_save_path, loss))
            print('Epoch {} Loss {:.4f}'.format(epoch + 1, loss))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - t0))