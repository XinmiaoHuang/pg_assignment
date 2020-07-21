from base.baseparser import BaseParser


class CustomParser(BaseParser):
    def __init__(self):
        super(CustomParser, self).__init__()
        self.parser.add_argument("--base_dir", type=str, default='/media/homee/Data/Dataset/deepfashion')
        self.parser.add_argument("--index_dir", type=str, default='/media/homee/Data/Dataset/deepfashion/index.p')
        self.parser.add_argument("--map_dir", type=str, default='/media/homee/Data/Dataset/deepmap_test')
        self.parser.add_argument("--save_dir", type=str, default='./checkpoint')
        self.parser.add_argument("--load_dir", type=str, default='./checkpoint')
        self.parser.add_argument("--bilinear", type=bool, default=True)
        self.parser.add_argument("--lambda_perceptual", type=float, default=10)
        self.parser.add_argument("--use_lsgan", type=bool, default=True)
        self.parser.add_argument("--use_sigmoid", type=bool, default=True)
        self.parser.add_argument("--img_size", type=tuple, default=(256, 256, 3))
        self.parser.add_argument("--inputc", type=int, default=21)
        self.parser.add_argument("--posec", type=int, default=18)
        self.parser.add_argument("--d_inputc", type=int, default=6)
        self.parser.add_argument("--lr", type=float, default=0.001)
        self.parser.add_argument("--save_per_epoch", type=int, default=600)
        self.parser.add_argument("--show_per_epoch", type=int, default=200)
        self.parser.add_argument("--test_per_epoch", type=int, default=200)

    def parse(self):
        return self.parser.parse_args()
