from datetime import datetime
from parser import parser
from saliency import *
from saliency_shifted_grids import *
from sklearn.metrics import roc_auc_score
import torch



def mse_loss(device, videos_list, model_file, gaussian_blur_size, data_path="./data/"):

    # Load net
    model = torch.load(model_file, map_location=device)
    net = SaliencyNet()
    net.to(device)
    net.load_state_dict(model['model_state_dict'])
    net.eval()

    # loss function
    criterion = nn.MSELoss()
    print("Net setup")

    # data transform
    transform = make_transform(gaussian_blur_size=gaussian_blur_size)

    dataset = GazeFrameDataset(data_path, videos_list, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
    print("dataloader done")

    running_loss = 0
    total = None
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        out = net(inputs)
        loss = criterion(out, labels)
        running_loss += loss.item()
        if i%10 == 9:
            print(i+1)
        total = i
    return running_loss / (total+1)

def ce_loss(device, videos_list, model_file, N, data_path="./data/"):

    # net
    net = SaliencyShiftedGridsNet(N)
    net.to(device)
    model = torch.load(model_file, map_location=device)
    net.load_state_dict(model['model_state_dict'])
    net.eval()

    # loss functions
    criterions = []
    for i in range(5):
        criterions.append(nn.CrossEntropyLoss())
    print("net setup")

    # data transform
    transform = make_transform_shifted_grids(N=N)

    dataset = GazeFrameDataset(data_path, videos_list, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
    print("dataloader done")

    running_loss = 0
    total = None
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data[0].to(device), data[1]
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

        running_loss += total_loss.item()
        if i%10 == 9:
            print(i+1)
        total = i

    return running_loss / (total+1)

def auroc_helper(y_true, y_score, resize):
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    scores = []
    weights = []
    for i_class in range(resize[0] * resize[1]):
        y_class_true = (y_true == i_class).reshape(-1, 1)
        y_class_score = y_score[:, i_class]
        if np.count_nonzero(y_class_true) > 0:
            scores.append(roc_auc_score(y_class_true, y_class_score))
            weights.append(np.sum(y_class_true))
        else:
            scores.append(0)
            weights.append(0)
    return np.array(scores), np.array(weights)

def auroc(videos_list, model_file, resize=(5,5), data_path="./data/"):

    device = torch.device('cpu')

    # Load net and optimizer
    model = torch.load(model_file, map_location=device)
    net = SaliencyNet()
    net.load_state_dict(model['model_state_dict'])
    net.eval()
    print("Net setup")

    # data transform
    transform = make_transform(gaussian_blur_size=None, class_size=resize)

    dataset = GazeFrameDataset(data_path, videos_list, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset)
    print("dataloader done")

    y_true = []
    y_score = []
    for i, data in enumerate(dataloader, 0):
        if i%100 == 99:
            print(i+1)
        im, target = data[0].to(device), data[1].to(device)
        target = np.squeeze(target.detach().numpy())
        output = net(im)
        output = np.squeeze(output.detach().numpy())
        output = cv2.resize(output, dsize=resize).flatten() # resize
        output = output / np.sum(output) # normalize array to sum to 1
       # output = np.zeros(resize[0] * resize[1])
       # output[resize[0] * resize[1] // 2] = 1
        y_true.append(target)
        y_score.append(output)

    return auroc_helper(y_true, y_score, resize)

def auroc_shifted_grids(videos_list, model_file, N=5, resize=(5,5), data_path = "./data/"):

    device = torch.device('cpu')

    # Load net and optimizer
    model = torch.load(model_file, map_location=device)
    net = SaliencyShiftedGridsNet(N)
    net.to(device)
    net.load_state_dict(model['model_state_dict'])
    net.eval()
    print("Net setup")

    # data transform
    transform = make_transform_shifted_grids(N=N, class_size=resize)

    dataset = GazeFrameDataset(data_path, videos_list, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset)
    print("dataloader done")

    y_true = []
    y_score = []
    for i, data in enumerate(dataloader, 0):
        if i%100 == 99:
            print(i+1)
        im, targets = data[0].to(device), data[1]
        outputs = net(im)
        final_output = np.zeros(resize[0] * resize[1])
        final_target = np.zeros(resize[0] * resize[1])
        for j in range(len(outputs)):
            output = outputs[j]
            output = np.squeeze(output.detach().numpy())
            output = np.reshape(output, (N, N))
            output = cv2.resize(output, dsize=resize).flatten() # resize
            final_output += output
            target = targets[j].to(device)
            target = np.squeeze(target.detach().numpy())
            final_target[target] += 1
        y_true.append(np.argmax(final_target))
        y_score.append(final_output)

    return auroc_helper(y_true, y_score, resize)

def compute_distance(videos_list, model_file, data_path = "./data/"):

    device = torch.device('cpu')

    # Load net and optimizer
    model = torch.load(model_file, map_location=device)
    net = SaliencyNet()
    net.load_state_dict(model['model_state_dict'])
    net.eval()
    print("Net setup")

    # data transform
    transform = make_transform(gaussian_blur_size=None, class_size=None)

    dataset = GazeFrameDataset(data_path, videos_list, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset)
    print("dataloader done")

    distances = []
    for i, data in enumerate(dataloader, 0):
        if i%100 == 99:
            print(i+1)
        im, target = data[0].to(device), data[1].to(device)
        target = np.squeeze(target.detach().numpy())
        output = net(im)
        output = np.squeeze(output.detach().numpy())
        # get grid coordinates
        H, W = output.shape
        idx = np.argmax(output) 
        y, x = np.unravel_index(idx, (H, W))
        # get normalized coordinates
        y = y / H + 1 / (2 * H)
        x = x / W + 1 / (2 * W)
        output = (y, x)
        #output = (0.5, 0.5) 
        distance = np.linalg.norm(output - target)
        distances.append(distance)

    return np.mean(distances)

def compute_distance_shifted_grids(videos_list, model_file, N, data_path = "./data/"):

    device = torch.device('cpu')

    # Load net and optimizer
    model = torch.load(model_file, map_location=device)
    net = SaliencyShiftedGridsNet(N)
    net.to(device)
    net.load_state_dict(model['model_state_dict'])
    net.eval()
    print("Net setup")

    # data transform
    transform = make_transform_shifted_grids(N=None)

    dataset = GazeFrameDataset(data_path, videos_list, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset)
    print("dataloader done")

    distances = []
    for i, data in enumerate(dataloader, 0):
        if i%100 == 99:
            print(i+1)
        im, target = data[0].to(device), data[1]
        target = np.array(target)
        outputs = net(im)
        final_output = np.zeros(N*N)
        for j in range(len(outputs)):
            output = outputs[j]
            output = np.squeeze(output.detach().numpy())
            final_output += output
        idx = np.argmax(final_output)
        y, x = np.unravel_index(idx, (N, N))
        # get normalized coordinates
        y = y / N + 1 / (2 * N)
        x = x / N + 1 / (2 * N)
        output = (y, x) 
        distance = np.linalg.norm(output - target)
        distances.append(distance)

    return np.mean(distances)


# def visualize(device, videos_list, model_file, gaussian_blur_size, data_path = "./data/"):

#     # Load net and optimizer
#     model = torch.load(model_file, map_location=device)
#     net = SaliencyNet()
#     net.load_state_dict(model['model_state_dict'])
#     net.eval()
#     print("Net setup")

#     # data transform
#     transform = make_transform(gaussian_blur_size=gaussian_blur_size)

#     dataset = GazeFrameDataset(data_path, videos_list, transform=transform)
#     dataloader = torch.utils.data.DataLoader(dataset)
#     print("dataloader done")

#     for i, data in enumerate(dataloader, 0):

#         # Plotting visualization
#         if i % 1000 == 999:
#             inputs, labels = data[0].to(device), data[1].to(device)
#             output = net(inputs)
#             x = inputs
#             y = labels

#             out = np.squeeze(output.detach().numpy())
#             plt.imshow(out, cmap='gray')
#             plt.colorbar()
#             plt.show()

#             yout = np.squeeze(y.detach().numpy())
#             plt.imshow(yout, cmap='gray')
#             plt.colorbar()
#             plt.show()



if __name__ == '__main__':

    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    videos_list = get_videos_list_from_file(args.train_file)
    if not args.shifted_grids:
        blur = (args.gaussian_blur_size, args.gaussian_blur_size)
        print(mse_loss(device, videos_list, args.model, blur))
    else:
        print(ce_loss(device, videos_list, args.model, args.N))

