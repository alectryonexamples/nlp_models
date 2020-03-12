import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm
import argparse
import os

from nlp_dataset import *

# gpu setup copied from https://www.tensorflow.org/guide/gpu
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

class SimpleCharacterLSTM(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim=256, rnn_units=1024):
        super().__init__()

        self.vocab_size = vocab_size

        self.encoding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(
            units=rnn_units,
            return_sequences=True,
            return_state=True)
        self.decoding = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, initial_state=None):
        embedding = self.encoding(inputs)
        output, hidden_state, model_state = self.lstm(embedding, initial_state=initial_state)
        decoding = self.decoding(output)
        return decoding, [hidden_state, model_state]

    def sample(self, starting_char, num_chars, temperature=1.0):

        curr_char = tf.reshape(starting_char, (1, 1))
        curr_state = None

        # (hidden state, cell state)
        char_list = [starting_char]
        for _ in range(num_chars):
            output, curr_state = self.call(curr_char, initial_state=curr_state)
            char_log_probs = np.reshape(output, (-1))
            char_log_probs = char_log_probs / temperature

            dist = tfp.distributions.Categorical(logits=char_log_probs, dtype=tf.int32)
            char_token = tfp.distributions.Sample(dist, sample_shape=(1,)).sample()
            curr_char = tf.reshape(char_token, (1, 1))
            char_list.append(int(char_token))
        return char_list

def character_prediction_loss(y, y_hat):
    loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat), axis=1)
    return loss

@tf.function
def train(x, y, model, optimizer):
    with tf.GradientTape() as tape:
        y_hat, _ = model(x)
        loss = character_prediction_loss(y, y_hat)
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss

def main(modeldir, logdir, epochs):
    train_logdir = os.path.join(logdir, "train")
    train_summary_writer = tf.summary.create_file_writer(train_logdir)

    text = load_tiny_shakespeare()
    vocab = sorted(set(text))
    text_processor = TextProcessor(vocab)
    idx_list = text_processor.convert_to_idx(text)
    ds = create_char_pred_ds(idx_list)
    vocab_size = len(vocab)

    model = SimpleCharacterLSTM(vocab_size=vocab_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    train_loss = tf.keras.metrics.Mean(name="train_loss")

    for i in range(epochs):
        train_loss.reset_states()
        ds = ds.shuffle(10000)
        for (x, y) in tqdm(ds):

            x = tf.convert_to_tensor(x)
            y = tf.convert_to_tensor(y)
            loss = train(x, y, model, optimizer)
            train_loss(loss)
        print("Epoch {} loss: {}".format(i, train_loss.result()))
        model_path = os.path.join(modeldir, "model_" + str(i) + ".ckpt")
        model.save_weights(model_path)
        with train_summary_writer.as_default():
            tf.summary.scalar("train_loss", train_loss.result(), i)

    model_path = os.path.join(modeldir, "model.ckpt")
    model.save_weights(model_path)

def generate_text(model, num_chars, starting_char, temperature=1.0):
    idx_list = model.sample(
        starting_char=text_processor.convert_to_idx(starting_char)[0], 
        num_chars=num_chars,
        temperature=1.0)
    sampled_text = text_processor.convert_to_tokens(idx_list)
    sampled_text = ''.join(sampled_text)
    return sampled_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Character LSTM model")
    parser.add_argument("--modeldir", 
        dest="modeldir", 
        type=str, 
        default="/tmp/models", 
        help="Directory to save trained models in.")
    parser.add_argument("--logdir",
        dest="logdir",
        type=str,
        default="/tmp/logs",
        help="Directory to save logs in.")
    parser.add_argument("--epochs",
        dest="epochs",
        type=int,
        default=10,
        help="Number of epochs to train for")
    args = parser.parse_args()

    main(args.modeldir, args.logdir, args.epochs)
