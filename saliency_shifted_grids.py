import torch
import torch.nn as nn
import torch.optim as optim
from video_to_data import *
import matplotlib.pyplot as plt



class SaliencyShiftedGridsNet(nn.Module):

    def __init__(self, N):
        super(SaliencyShiftedGridsNet, self).__init__()

        # First 5 layers of AlexNet
        self.alexnet = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Additional convolutional layer to turn into saliency map
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # 5 shifted grids
        # self.shifted_grids = []
        # for i in range(5):
        #     shifted_grid = nn.Sequential(
        #         nn.Flatten(),
        #         nn.Linear(13 * 13, N * N),
        #         nn.ReLU(inplace=True)
        #     ).to(device)
        #     self.shifted_grids.append(shifted_grid)
        self.shifted_grid_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(13 * 13, N * N),
            nn.ReLU(inplace=True)
        )
        self.shifted_grid_2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(13 * 13, N * N),
            nn.ReLU(inplace=True)
        )
        self.shifted_grid_3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(13 * 13, N * N),
            nn.ReLU(inplace=True)
        )
        self.shifted_grid_4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(13 * 13, N * N),
            nn.ReLU(inplace=True)
        )
        self.shifted_grid_5 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(13 * 13, N * N),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.alexnet(x)
        x = self.classifier(x)
        # for shifted_grid in self.shifted_grids:
        #     out = shifted_grid(x)
        #     outs.append(shifted_grid(x))
        outs = [
            self.shifted_grid_1(x),
            self.shifted_grid_2(x),
            self.shifted_grid_3(x),
            self.shifted_grid_4(x),
            self.shifted_grid_5(x),
        ]
        return outs

if __name__ == '__main__':
    # print("Dummy test...")
    # net = SaliencyNet()
    # # print(net)

    # optimizer = optim.SGD(net.parameters(), lr=0.01)
    # optimizer.zero_grad()

    # img = torch.randn(1, 3, 227, 227)
    # # print(img)
    # out = net(img)
    # print(list(out.size()))

    # net.zero_grad()
    # out.backward(torch.randn(1, 1, 13, 13), retain_graph=True)

    # target = torch.randn((1, 1, 13, 13))
    # criterion = nn.MSELoss()

    # loss = criterion(out, target)
    # print(loss)

    # net.zero_grad()
    # print('before')
    # print(net.alexnet[0].bias.grad)
    # loss.backward()
    # print('after')
    # print(net.alexnet[0].bias.grad.size())
    # print(loss.item())

    # optimizer.step()

    ########## INTEGRATION ##########

    # print("Integration test...")

    N = 5

    # data
    size_transform = SetSizeShiftedGrids((227,227), N)
    tensor_transform = ToTensorShiftedGrids()
    transform = transforms.Compose([size_transform, tensor_transform])
    data_path = "./data/"
    videos_list = ["2020-03-15_19-27-56-f2472745", "2020-06-22_11-14-22-319eaf00", 
                   "2020-06-25_17-25-16_alexl_everyday-tyingshoelaces-189703d3"]
    dataset = GazeFrameDataset(data_path, videos_list, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    print("dataloader done")

    # # net
    # net = SaliencyShiftedGridsNet(N)

    # # loss functions
    # criterions = []
    # for i in range(5):
    #     criterions.append(nn.CrossEntropyLoss())

    # # optimizer
    # optimizer = optim.SGD(net.parameters(), lr=0.01)
    # print("net setup")

    # for epoch in range(2):
    #     running_loss = 0.0
    #     for i, data in enumerate(dataloader, 0):
    #         inputs, labels = data
    #         optimizer.zero_grad()
    #         outs = net(inputs)
    #         total_loss = None
    #         for j in range(len(outs)):   
    #             criterion = criterions[j]
    #             loss = criterion(outs[j], labels[j])
    #             if total_loss:
    #                 total_loss = torch.add(total_loss, loss)
    #             else:
    #                 total_loss = loss
    #         total_loss.backward()
    #         optimizer.step()

    #         running_loss += total_loss.item()
    #         if i%100 == 99:
    #             print('[%d, %5d] loss: %.10f' %
    #                 (epoch + 1, i + 1, running_loss / 100))
    #             running_loss = 0.0

    # torch.save({
    #     'model_state_dict': net.state_dict(), 
    #     'optimizer_state_dict': optimizer.state_dict()
    #     }, 'model_shifted_grids.tar')

    ########## Load saved model ##########

    model = torch.load('model.tar')
    net = SaliencyShiftedGridsNet(N)
    net.load_state_dict(model['model_state_dict'])
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    optimizer.load_state_dict(model['optimizer_state_dict'])
    net.eval()
    # print("Model's state_dict:")
    # for param_tensor in net.state_dict():
    #     print(param_tensor, "\t", net.state_dict()[param_tensor])
    # print("Optimizer's state_dict:")
    # for var_name in optimizer.state_dict():
    #     print(var_name, "\t", optimizer.state_dict()[var_name])

    for i, data in enumerate(dataloader, 0):
        if i % 1000 == 0:
            inputs, labels = data
            outputs = net(inputs)
            x = inputs
            ys = labels

            for i in range(len(outputs)):
                out = outputs[i]
                out = out.detach().numpy()
                yout = ys[i]
                yout = yout.detach().numpy()
                
                for j in range(len(out)):
                    o = out[j]
                    o = o.reshape((N, N))
                    plt.imshow(o, cmap='gray')
                    plt.colorbar()
                    plt.show()

                    yo = yout[j]
                    y = np.zeros(N*N)
                    y[yo] = 1
                    y = y.reshape((N, N))
                    plt.imshow(y, cmap='gray')
                    plt.colorbar()
                    plt.show()


    print("Finished")
