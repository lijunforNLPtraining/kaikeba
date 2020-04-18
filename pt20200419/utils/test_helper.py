import tensorflow as tf
import numpy as np
from utils.batcher_utils import output_to_words

def batch_greedy_decode(model, enc_data, vocab, params):
    # 判断输入长度
    # print(enc_data)
    global outputs
    batch_data = enc_data[0]["enc_input"]
    batch_size = enc_data[0]["enc_input"].shape[0]
    # 开辟结果存储list
    predicts = [''] * batch_size
    inputs = batch_data
    # print(batch_size, batch_data.shape)
    print('inputs shape is :', inputs.get_shape())
    enc_output, enc_hidden = model.call_encoder(inputs)
    dec_hidden = enc_hidden
    # dec_input = tf.expand_dims([vocab.word_to_id(vocab.START_DECODING)] * batch_size, 1)
    dec_input = tf.constant([vocab.word_to_id('[START]')] * batch_size)
    print('dec_input shape is :', dec_input.get_shape())
    dec_input = tf.expand_dims(dec_input, axis=1)
    print('enc_output shape is :',enc_output.get_shape())
    print('dec_hidden shape is :', dec_hidden.get_shape())
    print('inputs shape is :', inputs.get_shape())
    print('dec_input shape is :', dec_input.get_shape())
    # context_vector, _ = model.BahdanauAttention(dec_hidden, enc_output)

    for t in range(params['max_dec_len']):
        # 单步预测
        # final_dist (batch_size, 1, vocab_size+batch_oov_len)
        if params["model"] == "SequenceToSequence":
            outputs = model(enc_output,  # shape=(3, 200, 256)
                            dec_hidden,  # shape=(3, 256)
                            inputs,  # shape=(3, 200)
                            dec_input)  # shape=(3, 50)

        # id转换
        final_dist = tf.squeeze(outputs["logits"], axis=1)
        # print(final_dist)
        predicted_ids = tf.argmax(final_dist, axis=1)
        # print(predicted_ids)

        for index, predicted_id in enumerate(predicted_ids.numpy()):
            predicts[index] += vocab.id_to_word(predicted_id) + ' '
    print(predicts)
    results = []
    for predict in predicts:
        # 去掉句子前后空格
        predict = predict.strip()
        # 句子小于max len就结束了 截断vocab.word_to_id('[STOP]')
        if '[STOP]' in predict:
            # 截断stop
            predict = predict[:predict.index('[STOP]')]
        # 保存结果
        results.append(predict)
    print(results)
    return results


