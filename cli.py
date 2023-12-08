from argparse import ArgumentParser

from data_alg import ALL_OPERATIONS
from train_grokking import main

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--operation", type=str, choices=ALL_OPERATIONS.keys(), default="x/y")
    parser.add_argument("--training_fraction", type=float, default=0.5)
    parser.add_argument("--prime", type=int, default=97)
    parser.add_argument("--vocab_size", type=int, default=99)
    parser.add_argument("--sequence_length", type=int, default=5)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bias", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=512) # 512
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1)
    parser.add_argument("--num_steps", type=int, default=1e4)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    main(args)
