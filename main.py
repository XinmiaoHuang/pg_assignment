import argparse
from torch.utils.data import DataLoader
from Dataset import DeepfashionPoseDataset
from options import CustomParser
from model import pix2pixHDModel


def train(opt):
    # load training set
    deepfashion = DeepfashionPoseDataset(opt.img_size, opt.base_dir, opt.index_dir, opt.map_dir, training=True)
    deepfashionTrainLoader = DataLoader(deepfashion, batch_size=opt.train_bsize,
                                shuffle=True, drop_last=True)
    # load testset
    deepfashion_test = DeepfashionPoseDataset(opt.img_size, opt.base_dir, opt.index_dir, opt.map_dir, training=True)
    deepfashionTestLoader = DataLoader(deepfashion_test, batch_size=opt.test_bsize,
                                shuffle=True, drop_last=True)
    # initialize
    model = pix2pixHDModel(opt, deepfashionTrainLoader, deepfashionTestLoader)
    model.init()

    if opt.checkpoint is not None:
        # model.load_model(load_g=True, load_d=True)
        print(f'Checkpoint loaded from {opt.checkpoint}')
    model.train()


# def test(opt):
#     # load testset
#     deepfashion_test = DeepfashionPoseDataset(opt.img_size, opt.base_dir, opt.index_dir, opt.map_dir, training=False)
#     deepfashionTestLoader = DataLoader(deepfashion_test, batch_size=opt.test_bsize,
#                                 shuffle=True, drop_last=True)
#     # initialize
#     model = pix2pixHDModel(opt, None, deepfashionTestLoader)
#     model.init('xavier')
    
#     if opt.checkpoint is not None:
#         model.load_model(opt.checkpoint, load_g=True, load_d=False)
#         print(f'Checkpoint loaded from {opt.checkpoint}')
#     model.test()
    
if __name__ == '__main__':
    parser = CustomParser()
    opt = parser.parse()
    if opt.mode == "train":
        train(opt)
    else:
        test(opt)