
import re
import pandas as pd
import numpy as np
import tensorflow as tf

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from collections import Counter

def review_to_wordlist(review, remove_stopwords=False):
  review_text = BeautifulSoup(review, "html5lib").get_text()
  review_text = re.sub("[^a-zA-Z]"," ", review_text)
  words = review_text.lower().split()
  if remove_stopwords:
    stops = set(stopwords.words("english"))
    words = [w for w in words if not w in stops]
  return(words)

def review_to_sentences(review, tokenizer, remove_stopwords=False):
  raw_sentences = tokenizer.tokenize(review.decode('utf8').strip())
  sentences = []
  for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
      sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
  return sentences

def build_vocab(sentences, max_words=None):
  word_count = Counter()
  for sentence in sentences:
    for word in sentence:
      word_count[word] += 1

  print("The dataset has %d different words totally" % len(word_count))
  if not max_words:
    max_words = len(word_count)
  else:
    filter_out_words = len(word_count) - max_words

  word_dict = word_count.most_common(max_words)
  return {word[0]: index + 1 for (index, word) in enumerate(word_dict)}

def vectorize(data, word_dict, verbose=True):
  reviews = []
  for idx, line in enumerate(data):
    seq_line = [word_dict[w] if w in word_dict else 0 for w in line]
    reviews.append(seq_line)

    if verbose and (idx % 5000 == 0):
      print("Vectorization: processed {}".format(idx))
  return reviews

def padding_data_for_rnn(sentences):
  lengths = [len(s) for s in sentences]
  n_samples = len(sentences)
  max_len = np.max(lengths)
  pdata = np.zeros((n_samples, max_len)).astype('int32')
  for idx, seq in enumerate(sentences):
    pdata[idx, :lengths[idx]] = seq
  return pdata, lengths 

def get_batchidx(n_data, batch_size, shuffle=True):
  """
    batch all data index into a list
  """
  idx_list = np.arange(n_data)
  if shuffle:
    np.random.shuffle(idx_list)
  batch_index = []
  num_batches = int(np.ceil(float(n_data) / batch_size))
  for idx in range(num_batches):
    start_idx = idx * batch_size
    batch_index.append(idx_list[start_idx: min(start_idx + batch_size, n_data)])
  return batch_index

def get_batches(sentences, labels, batch_size, max_len=400, type="cnn"):
  """
    read all data into ram once
  """
  minibatches = get_batchidx(len(sentences), batch_size)
  all_batches = []
  for minibatch in minibatches:
    seq_batch = [sentences[t] for t in minibatch]
    lab_batch = [labels[t] for t in minibatch]
    if type == "cnn":
      seq = tf.keras.preprocessing.sequence.pad_sequences(seq_batch, max_len)
      seq_len = [max_len] * batch_size
    elif type == "rnn":
      seq, seq_len = padding_data_for_rnn(seq_batch)
    all_batches.append((seq, seq_len, lab_batch))
  return all_batches

def get_config_proto(log_device_placement=False, allow_soft_placement=True):
  config_proto = tf.ConfigProto(
      log_device_placement=log_device_placement,
      allow_soft_placement=allow_soft_placement)
  config_proto.gpu_options.allow_growth = True
  return config_proto

def one_hot(labels, nb_classes=None):
  labels = np.array(labels).astype("int32")
  if not nb_classes:
    nb_classes = np.max(labels) + 1
  onehot_labels = np.zeros((len(labels), nb_classes)).astype("float32")
  for i in range(len(labels)):
    onehot_labels[i, labels[i]] = 1.
  return onehot_labels