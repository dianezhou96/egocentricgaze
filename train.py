from parser import parser
from saliency import *
from saliency_shifted_grids import *
import torch



def train(device, gaussian_blur_size=(3,3), learning_rate=0.01, num_epochs=10, data_path="./data/"):

    # net
    net = SaliencyNet()
    net.to(device)

    # loss function
    criterion = nn.MSELoss()

    # optimizer
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    print("Net setup")

    # data transform
    transform = make_transform(gaussian_blur_size=gaussian_blur_size)

    for epoch in range(num_epochs):
        print("Epoch", epoch + 1)
        running_loss = 0.0
        dataset = GazeFrameDataset(data_path, videos_list, transform=transform, shuffle=True)
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

    return net, optimizer

def train_shifted_grids(device, N=5, learning_rate=0.01, num_epochs=10, data_path="./data/"):

    # net
    net = SaliencyShiftedGridsNet(N)
    net.to(device)

    # loss functions
    criterions = []
    for i in range(5):
        criterions.append(nn.CrossEntropyLoss())

    # optimizer
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    # a = net.parameters()
    # for x in a:
    #     print(x)
    print("net setup")

    # data transform
    transform = make_transform_shifted_grids(N=N)

    for epoch in range(num_epochs):
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
                out = outs[j]
                label = labels[j].to(device)
                criterion = criterions[j]
                loss = criterion(out, label)
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

    return net, optimizer

def save_model(net, optimizer):
    torch.save({
        'model_state_dict': net.state_dict(), 
        'optimizer_state_dict': optimizer.state_dict()
        }, 'model.tar')



if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    args = parser.parse_args()

    videos_list = get_videos_list_from_file(args.train_file)

    blur = (args.gaussian_blur_size, args.gaussian_blur_size)
    net, optimizer = train(device, 
        gaussian_blur_size=blur, 
        learning_rate=args.learning_rate, 
        num_epochs=args.num_epochs
    )

    # net, optimizer = train_shifted_grids(device)

    print("Finished")
