
import os
import utils
import time
import config
import pandas as pd
from model import TextCNN, TextRNN
import tensorflow as tf

def main(args):
  print "loadding reviews and labels from dataset"
  data = pd.read_csv('data/labeledTrainData.tsv.zip', compression='zip', delimiter='\t', 
                      header=0, quoting=3)
  reviews = data["review"]
  labels = list(data['sentiment'])
  sentences = []
  for review in reviews:
    if len(review) > 0:
      sentences.append(utils.review_to_wordlist(review.decode('utf8').strip(), remove_stopwords=True))
  print "loaded %d reviews from dataset" % len(sentences)

  word_dict = utils.build_vocab(sentences, max_words=10000)
  vec_reviews = utils.vectorize(sentences, word_dict, verbose=True)
  train_x = vec_reviews[0: 20000]
  train_y = labels[0:20000]
  train_y = utils.one_hot(train_y, args.nb_classes)
  test_x = vec_reviews[20000:]
  test_y = labels[20000:]
  test_y = utils.one_hot(test_y, args.nb_classes)

  save_dir = args.save_dir
  log_dir = args.log_dir
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)

  with tf.Graph().as_default():
    config_proto = utils.get_config_proto()
    sess = tf.Session(config=config_proto)
    if args.model_type == "cnn":
      model = TextCNN(args, "TextCNN")
      test_batch = utils.get_batches(test_x, test_y, args.max_size)
    elif args.model_type == "rnn":
      model = TextRNN(args, "TextRNN")
      test_batch = utils.get_batches(test_x, test_y, args.max_size, type="rnn")

    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

    for epoch in range(1, args.nb_epochs + 1):
      print "epoch %d start" % epoch
      print "- " * 50

      loss = 0.
      total_reviews = 0
      accuracy = 0.
      if args.model_type == "cnn":
        train_batch = utils.get_batches(train_x, train_y, args.batch_size)
      elif args.model_type == "rnn":
        train_batch = utils.get_batches(train_x, train_y, args.batch_size, type="rnn")
      epoch_start_time = time.time()
      step_start_time = epoch_start_time
      for idx, batch in enumerate(train_batch):
        reviews, reviews_length, labels = batch
        _, loss_t, accuracy_t, global_step, batch_size, summaries = model.train(sess, 
                                  reviews, reviews_length, labels, args.keep_prob)

        loss += loss_t * batch_size
        total_reviews += batch_size
        accuracy += accuracy_t * batch_size
        summary_writer.add_summary(summaries, global_step)

        if global_step % 50 == 0:
          print "epoch %d, step %d, loss %f, accuracy %.4f, time %.2fs" % \
            (epoch, global_step, loss_t, accuracy_t, time.time() - step_start_time)
          step_start_time = time.time()

      epoch_time = time.time() - epoch_start_time
      print "%.2f seconds in this epoch" % (epoch_time)
      print "train loss %f, train accuracy %.4f" % (loss / total_reviews, accuracy / total_reviews)

      total_reviews = 0
      accuracy = 0.
      for batch in test_batch:
        reviews, reviews_length, labels = batch
        accuracy_t, batch_size = model.test(sess, reviews, reviews_length, labels, 1.0)
        total_reviews += batch_size
        accuracy += accuracy_t * batch_size
      print "accuracy %.4f in %d test reviews" % (accuracy / total_reviews, total_reviews)

if __name__ == '__main__':
  args = config.get_args()
  main(args)