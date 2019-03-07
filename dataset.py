import torch
import os
from torch.utils import data
import util

class PosDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

###############

root_dir = "/home/m17/hruska/PycharmProjects/pytorch-postagger"

emb_file = "embeddings/glove.txt" #pre-trained embeddings
data_dir = "english"
tag_file = "UDtags.txt"

emb_file = os.path.join(root_dir, emb_file)
tag_file = os.path.join(root_dir, tag_file)
train_file = os.path.join(root_dir, data_dir, "train.txt")

word2i, _, embeddings = util.load_embeddings(emb_file)
print("emb loaded")

tag2i, _ = util.load_postags(tag_file)
print("tags loaded")

train_x, train_y, train_y_list = util.prepare_data(train_file, word2i, tag2i, 50)


###############


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

#trainloader = data.DataLoader(train_x, train_y, batch_size=32, shuffle=False, num_workers=8)

for ind, dat in enumerate(trainloader):
    print(ind, dat)