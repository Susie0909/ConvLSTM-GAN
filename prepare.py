import os
import re
import random
import torch
from collections import deque
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm


def extract_number(filename: str) -> int:
    # 使用正则表达式提取文件名中的数字部分
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    else:
        return -1
def random_crop_3d(tensor: torch.tensor, crop_size: tuple) -> torch.tensor:
    assert len(tensor.shape) == 3, "输入张量必须是三维的"
    assert all(crop_size[i] <= tensor.shape[i] for i in range(3)), "裁剪尺寸大于输入张量的尺寸"

    d, h, w = tensor.shape
    td, th, tw = crop_size

    h0 = random.randint(0, h - th)
    w0 = random.randint(0, w - tw)
    d0 = random.randint(0, d - td)

    return tensor[d0:d0+td, h0:h0+th, w0:w0+tw]


# 现在 image_list 中包含了按照文件名顺序读取的所有图片

def split_images(input_dir: str, output_dir: str, seq_len: int, sub_image_size: tuple, num_sub_images: int) -> None:
    """
    input_dir:包含所有序列图像的文件夹路径，图像文件名需包含数字、且该数字大小与顺序先后一致
    output_dir:输出tensor目录

    seq_len:序列长度
    sub_image_size: tuple, [seq_len, h, w]
    num_sub_images:每个序列组内采样的子序列图像个数
    
    """
    assert sub_image_size[0] == seq_len, "输出子序列图像的序列长度参数不一致"
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # 获取文件夹中的所有文件并按文件名排序
    file_names = sorted(os.listdir(input_dir), key=extract_number)
    tensor_list=deque()
    for end_index in tqdm(range(len(file_names))):
        image = Image.open(os.path.join(input_dir, file_names[end_index]))
        tensor_list.append(ToTensor()(image).squeeze(0))
        image.close()
        if end_index>=seq_len-1:
            tensor = torch.stack(tuple(tensor_list), dim=0)
            for i in range(num_sub_images):
                # 随机截取一个子张量
                cropped_tensor =random_crop_3d(tensor,sub_image_size)
                torch.save(cropped_tensor,os.path.join(output_dir, f'{end_index+1}_{i+1}.pt'))
            tensor_list.popleft()



if __name__ == '__main__':

    # 设置输入参数
    input_directory = "./input"
    output_directory = "/media/cw/0EC03BB6C03BA33F/dataset/rock600_6"
    seq_len=6
    sub_image_size =(seq_len,600,600)
    num_sub_images = 5

    # 进行图像截取和保存
    split_images(input_directory, output_directory,seq_len, sub_image_size, num_sub_images)