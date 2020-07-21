import argparse

class BaseParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument("--mode", default="train", choices=["train", "test"])
        self.parser.add_argument("--checkpoint", help="path to checkpoint to restore", default='./checkpoint')
        self.parser.add_argument("--train_bsize", type=int, default=4)
        self.parser.add_argument("--test_bsize", type=int, default=8)
        self.parser.add_argument("--epochs", type=int, default=5)
        self.parser.add_argument("--steps", type=int, default=10)

    def parse(self):
        return self.parser.parse_args()
    