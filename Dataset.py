import torch
import os
from torch.utils.data import Dataset, DataLoader
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from utils import make_parts_shape, make_shape_parts
from skimage.draw import circle, line_aa, polygon
import torchvision.utils as vutils
"""
Reference:
https://github.com/CompVis/vunet.git
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""

MISSING_VALUE = -1
LIMB_SEQ = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9],
           [9,10], [1,11], [11,12], [12,13], [1,0], [0,14], [14,16],
           [0,15], [15,17], [2,16], [5,17]]

COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


class DeepfashionPoseDataset(Dataset):
    """Deepfashion dataset with pose. """
    def __init__(self, img_shape, base_dir, index_dir, map_dir, transform=None, training=True):
        self.base_dir = base_dir
        self.index_dir = index_dir
        self.map_dir = map_dir
        self.transform = transform
        self.training = training
        with open(index_dir, 'rb') as f:
            self.index = pickle.load(f)
        self.train = []
        self.test = []
        self.train_joint = []
        self.test_joint = []
        for idx, path in enumerate(self.index['imgs']):
            if path.startswith('train'):
                self.train.append(path)
                self.train_joint.append(self.index["joints"][idx])
            else:
                self.test.append(path)
                self.test_joint.append(self.index["joints"][idx])
        self.joint_order = self.index['joint_order']
        self.img_shape = img_shape
        self.rescale = 256 // self.img_shape[0]
        # scale the joints coordinate
        h, w = self.img_shape[:2]
        wh = np.array([[[w, h]]])
        self.train_joint = self.train_joint * wh
        self.test_joint = self.test_joint * wh

    def __len__(self):
        if self.training:
            return len(self.train)
        else:
            return len(self.test)

    def __getitem__(self, idx):
        if self.training:
            current_set = self.train
            current_joint = self.train_joint
        else:
            current_set = self.test
            current_joint = self.test_joint
        if torch.is_tensor(idx):
            idx = idx.to_list()
        img_dir = os.path.join(self.base_dir, current_set[idx])
        image = cv2.cvtColor(cv2.imread(img_dir), cv2.COLOR_BGR2RGB)

        # to_edge = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # edges = cv2.Canny(to_edge, 60, 150)
        # plt.imshow(edges)
        # plt.show()

        image = cv2.resize(image, self.img_shape[:2])

        s_dir = os.path.join(self.map_dir, current_set[idx][:-4] + '.png')
        s_map = cv2.cvtColor(cv2.imread(s_dir), cv2.COLOR_BGR2RGB)
        s_map = cv2.resize(s_map, self.img_shape[:2])


        # pose_img = self.make_joint_img(self.img_shape, self.joint_order, self.index["joints"][idx])
        pose = self.make_joint_map(current_joint[idx], k_size=9)
        color, mask = self.draw_pose_from_cords(np.floor(current_joint[idx]).astype(int), self.img_shape[:2])
        
        color = np.transpose(color, (2, 0, 1)) * 1./255
        # mask = np.transpose(np.expand_dims(mask, axis=-1)+0, (2, 0, 1)).astype(np.float)
        pose = np.concatenate((pose, color), axis=0)

        pose_img = self.make_joint_img(self.img_shape, self.joint_order, current_joint[idx])
        n_img, n_joint = self.normalize(image, current_joint[idx], pose_img, self.joint_order, 2)
        h, w = n_img.shape[0], n_img.shape[1]
        n_img = np.concatenate((n_img, cv2.resize(image, (h, w))), axis=-1)

        n_img = n_img * 1. / 255

        image = image * 1. / 255
        s_map = s_map * 1. / 255

        image = image.transpose((2, 0, 1))
        s_map = s_map.transpose((2, 0, 1))
        n_img = n_img.transpose((2, 0, 1))

        input_dir = current_set[idx]
        target_set = [id for id, img in enumerate(current_set[idx - 10:idx + 10]) if
                      img.startswith(input_dir[:11])]
        if 10 in target_set:
            target_set.remove(10)
        if len(target_set) == 0:
            target_set.append(10)
        t_idx = random.choice(target_set) + idx - 10
        target_dir = os.path.join(self.base_dir, current_set[t_idx])
        target = cv2.cvtColor(cv2.imread(target_dir), cv2.COLOR_BGR2RGB)
        target = cv2.resize(target, self.img_shape[:2])

        t_dir = os.path.join(self.map_dir, current_set[t_idx][:-4] + '.png')
        t_map = cv2.cvtColor(cv2.imread(t_dir), cv2.COLOR_BGR2RGB)
        t_map = cv2.resize(t_map, self.img_shape[:2])


        t_pose_img = self.make_joint_img(self.img_shape, self.joint_order, current_joint[t_idx])
        t_n_img, t_n_joint = self.normalize(target, current_joint[t_idx], t_pose_img, self.joint_order, 2)
        h, w = t_n_img.shape[0], t_n_img.shape[1]
        t_n_img = np.concatenate((t_n_img, cv2.resize(target, (h, w))), axis=-1)

        t_map = t_map * 1. / 255
        target = target * 1. / 255
        t_n_img = t_n_img * 1. / 255

        # plt.imshow(target)
        # plt.show()
        # plt.close()
        # for i in range(3):
        #     for j in range(3):
        #         # if i == 2 and j == 2:
        #         #     continue
        #         plt.subplot(3, 3, 3 * i + j + 1)
        #         plt.imshow(t_n_img[:, :, 3*(3*i+j): 3*(3*i+j+1)])
        # plt.show()
        # plt.close()

        t_map = t_map.transpose((2, 0, 1))
        target = target.transpose((2, 0, 1))
        t_n_img = t_n_img.transpose((2, 0, 1))

        t_pose = self.make_joint_map(current_joint[t_idx], k_size=9)
        t_color, t_mask = self.draw_pose_from_cords(np.floor(current_joint[t_idx]).astype(int), self.img_shape[:2])
        t_color = np.transpose(t_color, (2, 0, 1)) * 1./255
        # t_mask = np.transpose(np.expand_dims(t_mask, axis=-1)+0, (2, 0, 1)).astype(np.float)
        t_pose = np.concatenate((t_pose, t_color), axis=0)

        sample = {'image': image, 'pose': pose, 'target': target, 't_pose': t_pose, 's_map': s_map,
                  't_map': t_map, 'color': color, 't_color': t_color, 'n_img': n_img, 'tn_img': t_n_img}
        for key in sample.keys():
            sample[key] = torch.from_numpy(sample[key])
            if self.transform:
                sample[key] = self.transform(sample[key])
        sample = {'input': sample}
        return sample


    def draw_pose_from_cords(self, pose_joints, img_size, radius=2, draw_joints=True):
        colors = np.zeros(shape=img_size + (3, ), dtype=np.uint8)
        mask = np.zeros(shape=img_size, dtype=bool)

        if draw_joints:
            for f, t in LIMB_SEQ:
                from_missing = pose_joints[f][0] < 0 or pose_joints[f][1] < 0
                to_missing = pose_joints[t][0] < 0 or pose_joints[t][1] < 0
                if from_missing or to_missing:
                    continue
                yy, xx, val = line_aa(pose_joints[f][0], pose_joints[f][1], pose_joints[t][0], pose_joints[t][1])
                colors[xx, yy] = np.expand_dims(val, 1) * 255
                mask[xx, yy] = True

        for i, joint in enumerate(pose_joints):
            if pose_joints[i][0] < 0 or pose_joints[i][1] < 0:
                continue
            yy, xx = circle(joint[0], joint[1], radius=radius, shape=img_size)
            colors[xx, yy] = COLORS[i]
            mask[xx, yy] = True

        return colors, mask

    def cords_to_map(self, cords, img_size, sigma=6):
        result = np.zeros(img_size + cords.shape[0:1], dtype='float32')
        for i, point in enumerate(cords):
            if point[0] < 0 or point[1] < 0:
                continue
            xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
            result[..., i] = np.exp(-((yy - point[0]) ** 2 + (xx - point[1]) ** 2) / (2 * sigma ** 2))
        return result

    def make_joint_map(self, joints, k_size=9):
        joint_map = np.zeros((18, )+self.img_shape[:-1], dtype=np.float)
        gaussian_vec = cv2.getGaussianKernel(k_size, 0)
        gaussian_matrix = np.dot(gaussian_vec, gaussian_vec.T)
        _range = np.max(gaussian_matrix) - np.min(gaussian_matrix)
        gaussian_matrix = (gaussian_matrix - np.min(gaussian_matrix)) / _range
        for idx, joint in enumerate(joints):
            y = int(joint[0])
            x = int(joint[1])
            if x >= 0 and y >= 0:
                init_x = x - k_size//2
                init_y = y - k_size//2
                for i in range(k_size):
                    for j in range(k_size):
                        new_x = init_x + i
                        new_y = init_y + j
                        if 0 <= new_x < self.img_shape[0] and 0 <= new_y < self.img_shape[1]:
                            joint_map[idx][new_x][new_y] += gaussian_matrix[i][j]
        return joint_map

    @staticmethod
    def valid_joints(*joints):
        j = np.stack(joints)
        return (j >= 0).all()

    def get_crop(self, bpart, joints, jo, wh, o_w, o_h, ar=1.0):
        bpart_indices = [jo.index(b) for b in bpart]
        part_src = np.float32(joints[bpart_indices])

        # fall backs
        if not self.valid_joints(part_src):
            if bpart[0] == "lhip" and bpart[1] == "lknee":
                bpart = ["lhip"]
                bpart_indices = [jo.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices])
            elif bpart[0] == "rhip" and bpart[1] == "rknee":
                bpart = ["rhip"]
                bpart_indices = [jo.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices])
            elif bpart[0] == "lshoulder" and bpart[1] == "rshoulder" and bpart[2] == "cnose":
                bpart = ["lshoulder", "rshoulder", "rshoulder"]
                bpart_indices = [jo.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices])

        if not self.valid_joints(part_src):
            return None

        if part_src.shape[0] == 1:
            # leg fallback
            a = part_src[0]
            b = np.float32([a[0], o_h - 1])
            part_src = np.float32([a, b])

        if part_src.shape[0] == 4:
            pass
        elif part_src.shape[0] == 3:
            # lshoulder, rshoulder, cnose
            if bpart == ["lshoulder", "rshoulder", "rshoulder"]:
                segment = part_src[1] - part_src[0]
                normal = np.array([-segment[1], segment[0]])
                if normal[1] > 0.0:
                    normal = -normal

                a = part_src[0] + normal
                b = part_src[0]
                c = part_src[1]
                d = part_src[1] + normal
                part_src = np.float32([a, b, c, d])
            else:
                assert bpart == ["lshoulder", "rshoulder", "cnose"]
                neck = 0.5 * (part_src[0] + part_src[1])
                neck_to_nose = part_src[2] - neck
                part_src = np.float32([neck + 2 * neck_to_nose, neck])

                # segment box
                segment = part_src[1] - part_src[0]
                normal = np.array([-segment[1], segment[0]])
                alpha = 1.0 / 2.0
                a = part_src[0] + alpha * normal
                b = part_src[0] - alpha * normal
                c = part_src[1] - alpha * normal
                d = part_src[1] + alpha * normal
                # part_src = np.float32([a,b,c,d])
                part_src = np.float32([b, c, d, a])
        else:
            assert part_src.shape[0] == 2

            segment = part_src[1] - part_src[0]
            normal = np.array([-segment[1], segment[0]])
            alpha = ar / 2.0
            a = part_src[0] + alpha * normal
            b = part_src[0] - alpha * normal
            c = part_src[1] - alpha * normal
            d = part_src[1] + alpha * normal
            part_src = np.float32([a, b, c, d])

        dst = np.float32([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
        part_dst = np.float32(wh * dst)

        M = cv2.getPerspectiveTransform(part_src, part_dst)
        return M

    def normalize(self, imgs, coords, stickmen, jo, box_factor):
        img = imgs
        joints = coords
        stickman = stickmen

        h, w = img.shape[:2]
        o_h = h
        o_w = w
        h = h // 2 ** box_factor
        w = w // 2 ** box_factor
        wh = np.array([w, h])
        wh = np.expand_dims(wh, 0)

        bparts = [
            ["lshoulder", "lhip", "rhip", "rshoulder"],
            ["lshoulder", "rshoulder", "cnose"],
            ["lshoulder", "lelbow"],
            ["lelbow", "lwrist"],
            ["rshoulder", "relbow"],
            ["relbow", "rwrist"],
            ["lhip", "lknee"],
            ["rhip", "rknee"]]
        ar = 0.5

        part_imgs = list()
        part_stickmen = list()
        for bpart in bparts:
            part_img = np.zeros((h, w, 3))
            part_stickman = np.zeros((h, w, 3))
            M = self.get_crop(bpart, joints, jo, wh, o_w, o_h, ar)

            if M is not None:
                part_img = cv2.warpPerspective(img, M, (h, w), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
                part_stickman = cv2.warpPerspective(stickman, M, (h, w), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

            part_imgs.append(part_img)
            part_stickmen.append(part_stickman)
        img = np.concatenate(part_imgs, axis=2)
        stickman = np.concatenate(part_stickmen, axis=2)

        return img, stickman

    def make_joint_img(self, img_shape, jo, joints):
        # three channels: left, right, center
        scale_factor = img_shape[1] / 128
        thickness = int(3 * scale_factor)
        imgs = list()
        for i in range(3):
            imgs.append(np.zeros(img_shape[:2], dtype="uint8"))

        body = ["lhip", "lshoulder", "rshoulder", "rhip"]
        body_pts = np.array([[joints[jo.index(part), :] for part in body]])
        if np.min(body_pts) >= 0:
            body_pts = np.int_(body_pts)
            cv2.fillPoly(imgs[2], body_pts, 255)

        right_lines = [
            ("rankle", "rknee"),
            ("rknee", "rhip"),
            ("rhip", "rshoulder"),
            ("rshoulder", "relbow"),
            ("relbow", "rwrist")]
        for line in right_lines:
            l = [jo.index(line[0]), jo.index(line[1])]
            if np.min(joints[l]) >= 0:
                a = tuple(np.int_(joints[l[0]]))
                b = tuple(np.int_(joints[l[1]]))
                cv2.line(imgs[0], a, b, color=255, thickness=thickness)

        left_lines = [
            ("lankle", "lknee"),
            ("lknee", "lhip"),
            ("lhip", "lshoulder"),
            ("lshoulder", "lelbow"),
            ("lelbow", "lwrist")]
        for line in left_lines:
            l = [jo.index(line[0]), jo.index(line[1])]
            if np.min(joints[l]) >= 0:
                a = tuple(np.int_(joints[l[0]]))
                b = tuple(np.int_(joints[l[1]]))
                cv2.line(imgs[1], a, b, color=255, thickness=thickness)

        rs = joints[jo.index("rshoulder")]
        ls = joints[jo.index("lshoulder")]
        cn = joints[jo.index("cnose")]
        neck = 0.5 * (rs + ls)
        a = tuple(np.int_(neck))
        b = tuple(np.int_(cn))
        if np.min(a) >= 0 and np.min(b) >= 0:
            cv2.line(imgs[0], a, b, color=127, thickness=thickness)
            cv2.line(imgs[1], a, b, color=127, thickness=thickness)

        cn = tuple(np.int_(cn))
        leye = tuple(np.int_(joints[jo.index("leye")]))
        reye = tuple(np.int_(joints[jo.index("reye")]))
        if np.min(reye) >= 0 and np.min(leye) >= 0 and np.min(cn) >= 0:
            cv2.line(imgs[0], cn, reye, color=255, thickness=thickness)
            cv2.line(imgs[1], cn, leye, color=255, thickness=thickness)

        img = np.stack(imgs, axis=-1)
        if img_shape[-1] == 1:
            img = np.mean(img, axis=-1)[:, :, None]
        img = img.astype(np.float) * 1./255
        return np.transpose(img, (2, 0, 1))

    def make_pose(self, training=True):
        base_target_img_dir = '../PoseGuide/VUNet/splited_dataset/imgs/'          # 这里改为分割数据集存放路径
        base_target_pose_dir = '../PoseGuide/VUNet/splited_dataset/poses/'        # 这里改为分割数据集pose存放路径
        import shutil
        current_set = self.train if training else self.test
        current_joint = self.train_joint if training else self.test_joint
        for i in range(len(current_set)):
            img_dir = os.path.join(self.base_dir, current_set[i])
            shutil.copyfile(img_dir, os.path.join(base_target_img_dir, current_set[i]))
            pose = self.make_joint_img(self.img_shape, self.joint_order, current_joint[i])
            current_set[i] = current_set[i][:-3] + 'png'
            cv2.imwrite(os.path.join(base_target_pose_dir, current_set[i]), pose)


if __name__ == '__main__':
    base_dir = '/media/homee/Data/Dataset/deepfashion'
    index_dir = '/media/homee/Data/Dataset/deepfashion/index.p'
    map_dir = '/media/homee/Data/Dataset/deepmap_test'
    dataset = DeepfashionPoseDataset((256, 256, 3), base_dir, index_dir, map_dir, training=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    for index, sample in enumerate(dataloader):
        imgs = sample['input']['image'].numpy()
        poses = sample['input']['pose'].numpy()
        tposes = sample['input']['t_pose'].numpy()
        target = sample['input']['target'].numpy()
        color = sample['input']['color'].numpy()
        t_color = sample['input']['t_color'].numpy()
        # # plt.imsave("./small_data/imgs/{}.jpg".format(str(index)), np.transpose(imgs[0], (1, 2, 0)))
        # # plt.imsave("./small_data/poses/{}.jpg".format(str(index)), np.transpose(poses[0], (1, 2, 0)))
        # # plt.imsave("./small_data/target/{}.jpg".format(str(index)), np.transpose(target[0], (1, 2, 0)))
        # # if index > 1200:
        # #     break

        plt.subplot(1, 4, 1)
        plt.imshow(np.transpose(imgs[0], (1, 2, 0)))
        plt.subplot(1, 4, 2)
        plt.imshow(np.sum(np.transpose(poses[0], (1, 2, 0)), axis=2))
        plt.subplot(1, 4, 3)
        plt.imshow(np.transpose(target[0], (1, 2, 0)))
        plt.subplot(1, 4, 4)
        plt.imshow(np.sum(np.transpose(tposes[0], (1, 2, 0)), axis=2))
        plt.show()
    print("process ended.")
