import os
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_size=(64, 64)):
        self.data_dir = data_dir
        self.image_paths = self._get_image_paths()
        self.transform = transform
        self.target_size = target_size

    def _get_image_paths(self):
        image_paths = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        if self.target_size is not None:
            image = image.resize(self.target_size, Image.ANTIALIAS)

        if self.transform:
            image = self.transform(image)

        return image
# 使用示例
if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])  # 可以添加其他图像变换

    dataset = ImageDataset("./datasets/pokemon/images", transform=transform)

    data_tf = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])

    batch_size = 8
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    examples = enumerate(test_loader)
    batch_idx, imgs= next(examples)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    imgs = imgs.numpy()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(imgs[i].transpose(1, 2, 0))  # 调整通道顺序
        plt.axis('off')

    plt.show()