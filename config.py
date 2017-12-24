
import argparse

def get_args():

  parser = argparse.ArgumentParser()

  parser.add_argument('--train_dir', type=str, default='data/labeledTrainData.tsv.zip')
  parser.add_argument('--log_dir', type=str, default='save/logs')
  parser.add_argument('--save_dir', type=str, default='save/saves')
  parser.add_argument('--nb_classes', type=int, default=2)

  parser.add_argument('--sentence_length', type=int, default=400, help='The length of input x')
  parser.add_argument('--vocab_size', type=int, default=10000, help='data vocab size')
  parser.add_argument('--embed_size', type=int, default=50, help='dims of word embedding')
  parser.add_argument('--filter_sizes', type=list, default=[3, 4, 5], help='')
  parser.add_argument('--num_filters', type=int, default=65, help='num of filters')
  parser.add_argument('--keep_prob', type=float, default=0.5, help='keep prob in dropout')

  parser.add_argument('--batch_size', type=int, default=32, help='Example numbers every batch')
  parser.add_argument('--max_size', type=int, default=1000, help='max numbers every batch')
  parser.add_argument('--nb_epochs', type=int, default=20, help='Number of epoch')
  parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate')
  parser.add_argument('--max_grad_norm', type=float, default=10.0, help='Max norm of gradient')

  return parser.parse_args()