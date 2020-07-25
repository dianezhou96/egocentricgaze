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

    N = 5 # shifted grid is size N x N

    # data
    size_transform = SetSizeShiftedGrids((227,227), N)
    tensor_transform = ToTensorShiftedGrids()
    transform = transforms.Compose([size_transform, tensor_transform])
    data_path = "./data/"
    videos_list = ["2020-03-15_19-27-56-f2472745", "2020-06-22_11-14-22-319eaf00", 
                   "2020-06-25_17-25-16_alexl_everyday-tyingshoelaces-189703d3"]

    # net
    net = SaliencyShiftedGridsNet(N)
    net.to(device)

    # loss functions
    criterions = []
    for i in range(5):
        criterions.append(nn.CrossEntropyLoss())

    # optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    print("net setup")

    for epoch in range(2):
        print("Epoch", epoch + 1)
        running_loss = 0.0
        dataset = GazeFrameDataset(data_path, videos_list, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset)
        print("dataloader done")
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data[0].to(device), data[1]
            optimizer.zero_grad()
            outs = net(inputs)
            total_loss = None
            for j in range(len(outs)):   
                criterion = criterions[j]
                label = labels[j].to(device)
                loss = criterion(outs[j], labels[j])
                if total_loss:
                    total_loss = torch.add(total_loss, loss)
                else:
                    total_loss = loss
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            if i%1000 == 999:
                print('[%d, %5d] loss: %.10f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print("Finished")
