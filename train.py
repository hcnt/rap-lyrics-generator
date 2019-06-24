import tensorflow as tf

tf.compat.v1.enable_eager_execution()
import numpy as np
import os

text = open("teksty").read()
# print(len(text))

vocab = sorted(set(text))
# print(vocab)

char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = [char2idx[c] for c in text]

seq_len = 100
examples_per_epoch = len(text) // seq_len

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_len + 1, drop_remainder=True)


def split_imput_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


dataset = sequences.map(split_imput_target)

BATCH_SIZE = 64
steps_per_epoch = examples_per_epoch // BATCH_SIZE
BUFFER_SIZE = 10000

dataset = dataset.shuffle(1000).batch(BATCH_SIZE, drop_remainder=True)

# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            recurrent_activation='sigmoid',
                            return_sequences=True,
                            recurrent_initializer='glorot_uniform',
                            stateful=True),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


model = build_model(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


# for input_example_batch, target_example_batch in dataset.take(1):
#     example_batch_predictions = model(input_example_batch)
#     sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
#     sampled_indices = tf.squeeze(sampled_indices, axis=-1)
#     string = ""
#     for i in sampled_indices.numpy():
#         string += idxtochar[i]
#     print(string)
#     example_batch_loss = loss(target_example_batch, example_batch_predictions)
#     print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
#     print("scalar_loss:      ", example_batch_loss.numpy().mean())

# model.summary()

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
#model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss=loss)


checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS = 30

history = model.fit(dataset.repeat(), epochs=EPOCHS, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback])

## PREDICTION

def generate_text(model, start_string):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 500

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 0.8

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a multinomial distribution to predict the word returned by the model
        predictions = predictions / temperature
        predicted_id = tf.multinomial(predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)



#model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
#model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
#model.build(tf.TensorShape([1, None]))
#print(generate_text(model,"chuj"))