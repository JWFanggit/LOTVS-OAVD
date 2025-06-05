import os
import random
import json
from typing import Any

import numpy as np
import cv2
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

import glob
from PIL import Image
# import decord
# decord.bridge.set_bridge('torch')

from torch.utils.data import Dataset
from einops import rearrange


class DADA2KS3(Dataset):
    def __init__(self, root_path, interval, phase,
                 data_aug=False):
        self.root_path = root_path
        self.interval = interval
        # self.transforms = transforms
        self.data_aug = data_aug
        self.fps = 30
        self.phase = phase
        self.data_list, self.tar, self.tai, self.tco, self.NC_text, self.R_text, self.P_text, self.C_text = self.get_data_list()
        self.bbx_path = r""


    def get_data_list(self):
        if self.phase == "train":
            list_file = os.path.join(self.root_path + "/" + 'OOD_train.txt')
            # ff=open(os.path.join(self.root_path, self.phase + '\word.txt'),encoding='utf-8')
            assert os.path.exists(list_file), "File does not exist! %s" % (list_file)
            fileIDs, tar, tai, tco, NC_text, R_text, P_text, C_text = [], [], [], [], [], [], [], []
            # samples_visited, visit_rows = [], []
            with open(list_file, 'r', encoding='utf-8') as f:
                # for ids, line in enumerate(f.readlines()):
                for ids, line in enumerate(f.readlines()):
                    # print(line)
                    parts = line.strip().split('，')
                    if len(parts) == 2:
                        ID, tr, ta, tc = parts[0].split(' ')
                        fileIDs.append(ID)
                        tar.append(tr)
                        tai.append(ta)
                        tco.append(tc)
                        subparts = parts[1].split('//')
                        if len(subparts) == 4:
                            NC_text.append(subparts[0])
                            R_text.append(subparts[1])
                            P_text.append(subparts[2])
                            C_text.append(subparts[3])
            return fileIDs, tar, tai, tco, NC_text, R_text, P_text, C_text
        if self.phase == "val":
            # list_file = os.path.join(self.root_path + "/" + 'OOD_test.txt')
            list_file = os.path.join(self.root_path + "/" + 'OOD_test.txt')
            assert os.path.exists(list_file), "File does not exist! %s" % (list_file)
            fileIDs, tar, tai, tco, NC_text, R_text, P_text, C_text = [], [], [], [], [], [], [], []
            # samples_visited, visit_rows = [], []
            with open(list_file, 'r', encoding='utf-8') as f:
                # for ids, line in enumerate(f.readlines()):
                for ids, line in enumerate(f.readlines()):
                    # print(line)
                    parts = line.strip().split('，')
                    if len(parts) == 2:
                        ID, tr, ta, tc = parts[0].split(' ')
                        fileIDs.append(ID)
                        tar.append(tr)
                        tai.append(ta)
                        tco.append(tc)
                        subparts = parts[1].split('//')
                        if len(subparts) == 4:
                            NC_text.append(subparts[0])
                            R_text.append(subparts[1])
                            P_text.append(subparts[2])
                            C_text.append(subparts[3])
            return fileIDs, tar, tai, tco, NC_text, R_text, P_text, C_text


    def __len__(self):
        return len(self.data_list)

    def pross_video_data(self, video):
        video_datas = []
        for fid in range(len(video)):
            video_data = video[fid]
            video_data = Image.open(video_data)
            video_data = video_data.resize((224, 224))
            video_data = np.asarray(video_data, np.float32)
            if len(video_data.shape) <3:
                video_data = np.stack((video_data,video_data,video_data),-1)
            video_datas.append(video_data)

        video_data = np.array(video_datas, dtype=np.float32)  # 4D tensor
        video_data = rearrange(video_data, 'f w h c -> f c w h')
        return video_data

    def read_nomarl_rgbvideo(self, video_file):
        """Read video frames
        """
        # assert os.path.exists(video_file), "Path does not exist: %s" % (video_file)
        # get the video data

        video_data = self.pross_video_data(video_file)

        return video_data

    def pdbbx(self, bbx, max_N):
        N = bbx.shape[0]
        if N < max_N:
            pad_objects = torch.zeros(max_N - N, 4)
            bbx = torch.cat([bbx, pad_objects], dim=0)
        elif N > max_N:
            bbx = bbx[:max_N, :]

        return bbx
    #

    def to_valid(self,x0, y0, x1, y1, image_size, min_box_size):
        valid = True

        if x0 > image_size or y0 > image_size or x1 < 0 or y1 < 0:
            valid = False  # no way to make this box vide, it is completely cropped out
            return valid, (None, None, None, None)

        x0 = max(x0, 0)
        y0 = max(y0, 0)
        x1 = min(x1, image_size)
        y1 = min(y1, image_size)

        if (x1 - x0) * (y1 - y0) / (image_size * image_size) < min_box_size:
            valid = False
            return valid, (None, None, None, None)

        return valid, (x0, y0, x1, y1)


    def recalculate_box_and_verify_if_valid(self, normalized_bbx, original_image_size, target_image_size,image_size):
        # normalized_bbx = (x1, y1, x2, y2)
        x1_orig, y1_orig, x2_orig, y2_orig = normalized_bbx
        # Scale coordinates from original image size to target image size
        x1_target = x1_orig * (target_image_size[0] / original_image_size[0])
        y1_target = y1_orig * (target_image_size[1] / original_image_size[1])
        x2_target = x2_orig * (target_image_size[0] / original_image_size[0])
        y2_target = y2_orig * (target_image_size[1] / original_image_size[1])
        valid, (x0, y0, x1, y1) = self.to_valid(x1_target,y1_target,x2_target,y2_target,image_size, min_box_size=0.01)

        # if valid:
        #     # we also perform random flip.
        #     # Here boxes are valid, and are based on image_size
        #     # if trans_info["performed_flip"]:
        #         x0, x1 = image_size - x1, image_size - x0

        return valid, (x0, y0, x1, y1)


    def mapping_caption(self, category_number):
        category_mapping = {'{}': 0, 'motorcycle': 1, 'truck': 2, 'bus': 3, 'traffic light': 4,
                            'person': 5, 'bicycle': 6, 'car': 7}
        if category_number in category_mapping.values():
            caption = next(key for key, value in category_mapping.items() if value == category_number)
            # print(caption)
            return caption

    def extract_masks(self, video_frames, bounding_boxes):
        masks = []
        for i in range(video_frames.shape[0]):
            frame = video_frames[i]
            frame_masks = np.zeros_like(frame)  # Create a blank mask with the same dimensions as the frame

            for j in range(bounding_boxes.shape[1]):
                bbx = bounding_boxes[i, j].int()  # Convert the bounding box coordinates to integers
                x1, y1, x2, y2 = bbx

                # 保留 bounding box 区域内的像素
                frame_masks[..., y1:y2, x1:x2] = frame[..., y1:y2, x1:x2]  # 复制 bounding box 区域内的像素到 mask

            masks.append(frame_masks)
        return np.array(masks)


    def recalculate_bbx(self, normalized_bbx, original_image_size, target_image_size):
        # normalized_bbx = (x1, y1, x2, y2)
        x1_orig, y1_orig, x2_orig, y2_orig = normalized_bbx

        # Scale coordinates from original image size to target image size
        x1_target = x1_orig * (target_image_size[0] / original_image_size[0])
        y1_target = y1_orig * (target_image_size[1] / original_image_size[1])
        x2_target = x2_orig * (target_image_size[0] / original_image_size[0])
        y2_target = y2_orig * (target_image_size[1] / original_image_size[1])

        return torch.tensor([x1_target, y1_target, x2_target, y2_target])

    def bbx_caption_process(self, bbx_info, original_image_size, target_image_size, max_N):
        caption = bbx_info[:, 0]  # Add a new dimension
        caption_text = [self.mapping_caption(category_number.item()) for category_number in caption]
        # Extract the second to fifth elements of each row and keep them as (N, 4) tensor
        bbx = bbx_info[:, 1:]
        new_bbx = torch.zeros_like(bbx)
        # 遍历每一行的坐标并应用处理函数
        for i in range(bbx.shape[0]):
            row_bbx = bbx[i]
            processed_row_bbx =self.recalculate_bbx(row_bbx, original_image_size, target_image_size)
            processed_row_bbx[processed_row_bbx < 0] = 0
            new_bbx[i] = processed_row_bbx

        new_bbx = self.pdbbx(new_bbx, max_N)
        return caption_text, new_bbx

    def gather_info(self, index):
        # accident_id = int(self.data_list[index].split('/')[0])
        accident_id = self.data_list[index]
        video_id = self.data_list[index].split('/')[1]
        N_T = self.NC_text[index]
        R_T = self.R_text[index]
        P_T = self.P_text[index]
        C_T = self.C_text[index]
        return accident_id, video_id, N_T, R_T, P_T, C_T


    def __getitem__(self, index):
        def __getitem__(self, index):
            # read RGB video (trimmed)
            tar = int(self.tar[index])
            tai = int(self.tai[index])
            tco = int(self.tco[index])
            video_path = os.path.join(self.root_path + "/", self.data_list[index] + "/" + "images")
            bbx_path = os.path.join(self.bbx_path + "/", self.data_list[index])
            # video_path=glob.glob(video_path+'/'+"*.[jp][pn]g")
            # video_path= sorted(video_path, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))
            v_o = [video_path + "/" + f'{i:06d}' + ".jpg" for i in range(1, 17)]
            bbx_o_path = [bbx_path + "/" + f'{i:04d}' + ".txt" for i in range(1, 17)]
            selected_r = sorted(random.sample(range(tar, tai + 1), 16))
            selected_r_reverse = selected_r[::-1]
            v_r = [video_path + "/" + f'{i:06d}' + ".jpg" for i in selected_r]
            bbx_r_path = [bbx_path + "/" + f'{i:04d}' + ".txt" for i in selected_r]
            v_p = [video_path + "/" + f'{i:06d}' + ".jpg" for i in selected_r_reverse]
            bbx_p_path = [bbx_path + "/" + f'{i:04d}' + ".txt" for i in selected_r_reverse]
            v_a = [video_path + "/" + f'{i:06d}' + ".jpg" for i in range(tco - 16, tco)]
            bbx_a_path = [bbx_path + "/" + f'{i:04d}' + ".txt" for i in range(tco - 16, tco)]

            accident_id, video_id, N_T, R_T, P_T, C_T = self.gather_info(index)

            vo = self.read_nomarl_rgbvideo(v_o)
            vr = self.read_nomarl_rgbvideo(v_r)
            vp = self.read_nomarl_rgbvideo(v_p)
            va = self.read_nomarl_rgbvideo(v_a)
            V = (vo, vr, vp, va)
            T = (N_T, R_T, P_T, C_T)
            Box = (bbx_o_path, bbx_r_path, bbx_p_path, bbx_a_path )
            random_index = random.randint(0, 3)
            train_vdata = V[random_index]
            prompt_ids = T[random_index]
            boxes = Box[random_index]

            bbx_info_list = []
            caption_list = []

            for bbx_file in boxes:
                with open(bbx_file, 'r') as file:
                    lines = file.readlines()
                    # for key, value in  lines.items():
                    # print(f"Key: {key}, Value: {value}")
                    # with open(bbx_file,"r") as file:
                    if not lines or len(lines) == 0 or all(line.isspace() for line in lines):

                        filtered_datas = torch.zeros(1, 5, dtype=torch.float32)
                    else:
                        # Process valid data
                        # bbx_info = torch.stack(
                        #     [torch.tensor(list(map(line.split())), dtype=torch.float32) for line in lines["bboxes"]])

                        # bbx_info=torch.tensor(lines["bboxes"])
                        bbx_info = lines[:4]
                        scores = lines[4]
                        label = lines[5]
                        # filtered_bbx_info = [info for info, s in zip(bbx_info,scores) if s > 0.3]
                        filtered_data = [[lbl, *info] for info, scr, lbl in zip(bbx_info, scores, label) if scr > 0.3]
                        if not filtered_data or len(filtered_data) == 0:
                            filtered_datas = torch.stack(
                                [torch.zeros(1, 5, dtype=torch.float32) for _ in range(16)]).squeeze(1)
                        else:
                            filtered_datas = torch.stack(
                                [torch.tensor(list(map(float, line)), dtype=torch.float32) for line in filtered_data])


                        # label_info=torch.stack(
                        #     [torch.tensor(list(map(float,filtered_data.split(","))), dtype=torch.float32) for line in lines["labels"]])
                    caption, bbx = self.bbx_caption_process(filtered_datas, original_image_size=(720, 1280),
                                                            target_image_size=(224, 224), max_N=4)
                    merged_caption = ", ".join(caption)
                    bbx_info_list.append(bbx)
                    caption_list.append(merged_caption)

            bboxes = torch.stack(bbx_info_list)
            mask = self.extract_masks(train_vdata, bboxes)
            example = {
                "pixel_values": train_vdata / 127.5 - 1.0,
                "prompt_ids": prompt_ids,
                "bbx": bboxes,
                "mask": mask / 127.5 - 1.0,
                "accident_id": accident_id,
                "video_id": video_id}
            return example



if __name__=="__main__":
    train_dataset = DADA2KS3(root_path=r"/media/work/My Passport/CAPDATA", interval=1,phase="train")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=False,
        pin_memory=True,num_workers=4, drop_last=True)

    for id, (data) in enumerate(train_dataloader):
        # print(data["answer_id"])
        # print(data["question"].shape)
        # print(data["option"].shape)
        # print(data["mask"].shape)
        # print(data['captions'])
        # print(data['non_option_token'].shape)
        print(data['pixel_values'].shape)













