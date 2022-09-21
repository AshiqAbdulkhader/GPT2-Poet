# finetune gpt2 on poems
import tensorflow as tf
from transformers import GPT2Config, TFGPT2LMHeadModel, GPT2Tokenizer
from transformers import WEIGHTS_NAME, CONFIG_NAME
import os

# load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# load model
configuration = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id
)
model = TFGPT2LMHeadModel(configuration)

# load dataset
textfile = open("data\poetry.txt", "r", encoding='utf-8')
text = textfile.read()
textfile.close()

# tokenize dataset
string_tokenized = tokenizer.encode(text)
print("Done tokenizing")

# create dataset
examples = []
block_size = 100
BATCH_SIZE = 12
BUFFER_SIZE = 1000
for i in range(0, len(string_tokenized) - block_size + 1, block_size):
    examples.append(string_tokenized[i:i + block_size])
inputs, labels = [], []

for ex in examples:
    inputs.append(ex[:-1])
    labels.append(ex[1:])
dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print("Done creating dataset")

# create model
optimizer = tf.keras.optimizers.Adam(
    learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# train model
model.compile(optimizer=optimizer, loss=loss)
model.fit(dataset, epochs=1)

# save model
save_location = "models"
if not os.path.exists(save_location):
    os.makedirs(save_location)
model.save_pretrained(save_location)
tokenizer.save_pretrained(save_location)
