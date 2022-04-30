import torch
import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self) -> None:
        super(VGG16, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.MaxPool2d((2, 2)),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        self.full = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.Softmax(),
        )

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size()[0], -1)
        return self.full(x)


if __name__ == '__main__':
    import numpy as np

    # test_img = np.zeros((1, 3, 224, 224), dtype=np.uint8)
    # test_tensor = torch.LongTensor(test_img)

    # aa = VGG16()
    # out = aa(test_tensor)
    # print("done")

    # class testDataset(Dataset):
    #     def __init__(self, X, y=None):
    #         self.data = torch.tensor(X)

    #     def __getitem__(self, idx):
    #         return self.data[idx]

    #     def __len__(self):
    #         return len(self.data)

    # test_dataset = testDataset(test_img)
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     # collate_fn=collate_batch,
    # )

    # aa = VGG16().to('cuda:0')

    # for data in test_loader:
    #     data = data.to('cuda:0')
    #     out = aa.forward(data)

    # test_tensor = torch.randn(1, 3, 224, 224)
    # np.zeros(( 224, 224, 3), dtype=np.uint8)

    # [[[2, 4, 5][5, 6, 7]]][[10, 11, 12][13, 14, 15]]]]  numpy
    # [[[2, 5]][[4, 6]][[5, 7]][[10, 13]][[11, 14]][[12, 15]]] torch
    test_img = np.zeros((1, 3, 224, 224), dtype=np.uint8)
    test_tensor = torch.from_numpy(test_img).type(torch.float32)

    aa = VGG16()
    out = aa(test_tensor)
