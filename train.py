from datetime import datetime
from parser import parser
from saliency import *
from saliency_shifted_grids import *
import torch



def train(device, videos_list, gaussian_blur_size=(3,3), learning_rate=0.01, num_epochs=10, 
    batch_size=32, saved_model=None, data_path="./data/"):

    if not saved_model:
        # net
        net = SaliencyNet()
        net.to(device)

        # optimizer
        optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    else:
        # Load net and optimizer
        model = torch.load(saved_model, map_location=device)
        net = SaliencyNet()
        net.load_state_dict(model['model_state_dict'])
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=learning_rate)
        optimizer.load_state_dict(model['optimizer_state_dict'])

    # loss function
    criterion = nn.MSELoss()
    print("Net setup")

    # data transform
    transform = make_transform(gaussian_blur_size=gaussian_blur_size)

    for epoch in range(num_epochs):
        print("Epoch", epoch + 1)
        running_loss = 0.0
        dataset = GazeFrameDataset(data_path, videos_list, transform=transform, shuffle=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        print("Dataloader done")
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            out = net(inputs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i%10 == 9:
                print('[%d, %5d] loss: %.10f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        save_model(net, optimizer, "model_" + str(epoch+1) + '.tar')

    return net, optimizer

def train_shifted_grids(device, videos_list, N=5, learning_rate=0.01, num_epochs=10, 
    batch_size=32, saved_model=None, data_path="./data/"):

    if not saved_model:
        # net
        net = SaliencyNet()
        net.to(device)

        # optimizer
        optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    else:
        # Load net and optimizer
        model = torch.load(saved_model, map_location=device)
        net = SaliencyNet()
        net.load_state_dict(model['model_state_dict'])
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=learning_rate)
        optimizer.load_state_dict(model['optimizer_state_dict'])

    # loss function
    criterion = nn.MSELoss()
    print("net setup")

    # data transform
    transform = make_transform_shifted_grids(N=N)

    for epoch in range(num_epochs):
        print("Epoch", epoch + 1)
        running_loss = 0.0
        dataset = GazeFrameDataset(data_path, videos_list, transform=transform, shuffle=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
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
            if i%10 == 9:
                print('[%d, %5d] loss: %.10f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        save_model(net, optimizer, "model_shifted_" + str(epoch+1) + '.tar')

    return net, optimizer

def save_model(net, optimizer, save_as):
    torch.save({
        'model_state_dict': net.state_dict(), 
        'optimizer_state_dict': optimizer.state_dict()
        }, save_as)



if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    args = parser.parse_args()

    videos_list = get_videos_list_from_file(args.train_file)

    current_time = datetime.now()
    print("Starting at", current_time.strftime("%m/%d/%Y %H:%M:%S"))
    if not args.shifted_grids:
        blur = (args.gaussian_blur_size, args.gaussian_blur_size)
        net, optimizer = train(device, videos_list, 
            gaussian_blur_size=blur, 
            learning_rate=args.learning_rate, 
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            saved_model=args.model
        )
        save_as = "model_blur_" + str(args.gaussian_blur_size)
    else:
        net, optimizer = train_shifted_grids(device, videos_list, 
             N=args.N,
             learning_rate=args.learning_rate,
             num_epochs=args.num_epochs,
             batch_size=args.batch_size,
             saved_model=args.model
        )
        save_as = "model_shifted_grids_N_" + str(args.N)

    save_as += "_lr_" + str(args.learning_rate) + \
        "_epochs_" + str(args.num_epochs) + \
        "_batch_" + str(args.batch_size) + \
        "_" + current_time.strftime("%m-%d-%Y_%H-%M") + ".tar"

    current_time = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    print("Finished at", current_time)

    save_model(net, optimizer, save_as)
