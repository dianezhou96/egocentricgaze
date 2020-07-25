from saliency import *
from saliency_shifted_grids import *
import torch

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    # data
    size_transform = SetSize((227,227), (13,13), (3,3))
    tensor_transform = ToTensor()
    transform = transforms.Compose([size_transform, tensor_transform])
    data_path = "./data/"
    videos_list = ["2020-03-15_19-27-56-f2472745", "2020-06-22_11-14-22-319eaf00", 
                   "2020-06-25_17-25-16_alexl_everyday-tyingshoelaces-189703d3"]

    # net
    net = SaliencyNet()
    net.to(device)

    # loss function
    criterion = nn.MSELoss()

    # optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    print("Net setup")

    for epoch in range(10):
        print("Epoch", epoch + 1)
        running_loss = 0.0
        dataset = GazeFrameDataset(data_path, videos_list, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset)
        print("Dataloader done")
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            out = net(inputs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i%1000 == 999:
                print('[%d, %5d] loss: %.10f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    # torch.save({
    #     'model_state_dict': net.state_dict(), 
    #     'optimizer_state_dict': optimizer.state_dict()
    #     }, 'model.tar')

    print("Finished")
