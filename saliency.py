import torch
import torch.nn as nn
import torch.optim as optim
from video_to_data import *
import matplotlib.pyplot as plt



class SaliencyNet(nn.Module):

    def __init__(self):
        super(SaliencyNet, self).__init__()

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

    def forward(self, x):
        x = self.alexnet(x)
        x = self.classifier(x)
        return x

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

    print("Integration test...")

    # data
    size_transform = SetSize((227,227), (13,13), (3,3))
    tensor_transform = ToTensor()
    transform = transforms.Compose([size_transform, tensor_transform])
    data_path = "./data/"
    videos_list = ["2020-03-15_19-27-56-f2472745", "2020-06-22_11-14-22-319eaf00", 
                   "2020-06-25_17-25-16_alexl_everyday-tyingshoelaces-189703d3"]
    dataset = GazeFrameDataset(data_path, videos_list, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset)
    print("dataloader done")

    # # # net
    # net = SaliencyNet()

    # # loss function
    # criterion = nn.MSELoss()

    # # optimizer
    # optimizer = optim.SGD(net.parameters(), lr=0.01)
    # print("net setup")

    # for epoch in range(2):
    #     running_loss = 0.0
    #     for i, data in enumerate(dataloader, 0):
    #         inputs, labels = data
    #         # print(inputs.size())
    #         # print(labels.size())
    #         optimizer.zero_grad()
    #         out = net(inputs)
    #         loss = criterion(out, labels)
    #         loss.backward()
    #         optimizer.step()

    #         running_loss += loss.item()
    #         if i%100 == 99:
    #             print('[%d, %5d] loss: %.3f' %
    #                 (epoch + 1, i + 1, running_loss / 100))
    #             running_loss = 0.0
    #         if i > 3900:
    #             break

    # torch.save({
    #     'model_state_dict': net.state_dict(), 
    #     'optimizer_state_dict': optimizer.state_dict()
    #     }, 'model.tar')

    ########## Load saved model ##########

    model = torch.load('model.tar')
    net = SaliencyNet()
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
        if i % 1000 == 999:
            inputs, labels = data
            output = net(inputs)
            x = inputs
            y = labels

            out = np.squeeze(output.detach().numpy())
            plt.imshow(out, cmap='gray')
            plt.colorbar()
            plt.show()

            yout = np.squeeze(y.detach().numpy())
            plt.imshow(yout, cmap='gray')
            plt.colorbar()
            plt.show()


    print("Finished")
