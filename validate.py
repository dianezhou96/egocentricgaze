from datetime import datetime
from parser import parser
from saliency import *
from saliency_shifted_grids import *
from sklearn.metrics import roc_auc_score
import torch



def validate(device, videos_list, model_file, resize=(5,5), data_path = "./data/"):

    # Load net and optimizer
    model = torch.load(model_file, map_location=device)
    net = SaliencyNet()
    net.load_state_dict(model['model_state_dict'])
    net.eval()
    print("Net setup")

    # data transform
    transform = make_transform(gaussian_blur_size=0, class_size=resize)

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
        y_true.append(target)
        y_score.append(output)

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
    return scores, weights

def validate_shifted_grids(device, videos_list, model_file, N=5, resize=(5,5), data_path = "./data/"):

    # Load net and optimizer
    model = torch.load(model_file, map_location=device)
    net = SaliencyShiftedGridsNet(N)
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
        final_output 
        for j in range(len(outputs)):
            output = outputs[j]
            target = targets[j].to(device)
            target = np.squeeze(target.detach().numpy())
            output = np.squeeze(output.detach().numpy())
        output = cv2.resize(output, dsize=resize).flatten() # resize
        output = output / np.sum(output) # normalize array to sum to 1
        y_true.append(target)
        y_score.append(output)

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
    return scores, weights

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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    args = parser.parse_args()

    videos_list = get_videos_list_from_file(args.validation_file)

    current_time = datetime.now()
    print("Starting at", current_time.strftime("%m/%d/%Y %H:%M:%S"))
    if not args.shifted_grids:
        scores, weights = validate(device, videos_list, args.model)
    #     save_as = "model_blur_" + str(args.gaussian_blur_size)
    # else:
    #     net, optimizer = train_shifted_grids(device,
    #          N=args.N,
    #          learning_rate=args.learning_rate,
    #          num_epochs=args.num_epochs,
    #          batch_size=args.batch_size
    #     )
    #     save_as = "model_shifted_grids_N_" + str(args.N)

    # save_as += "_lr_" + str(args.learning_rate) + \
    #     "_epochs_" + str(args.num_epochs) + \
    #     "_batch_" + str(args.batch_size) + \
    #     "_" + current_time.strftime("%m-%d-%Y_%H-%M") + ".tar"

    weighted_score = np.sum(np.array(scores) * np.array(weights) / np.sum(weights))
    print(weighted_score)
    current_time = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    print("Finished at", current_time)
