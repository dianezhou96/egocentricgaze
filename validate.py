from datetime import datetime
from parser import parser
from evaluate import *



if __name__ == '__main__':

    args = parser.parse_args()

    videos_list = get_videos_list_from_file(args.validation_file)

    current_time = datetime.now()
    print("Starting at", current_time.strftime("%m/%d/%Y %H:%M:%S"))
    if not args.shifted_grids:
        distance = compute_distance(videos_list, args.model)
        scores, weights = auroc(videos_list, args.model)
    else:
        distance = compute_distance_shifted_grids(videos_list, args.model, args.N)
        scores, weights = auroc_shifted_grids(videos_list, args.model, args.N)

    print("Average distance:", distance)

    print("Scores:", scores)
    print("Weights:", weights)

    nonzero = weights > 0
    weighted_score = np.sum(np.array(scores[nonzero]) * np.array(weights[nonzero]) / np.sum(weights[nonzero]))
    print("Weighted average AUROC:", weighted_score)

    current_time = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    print("Finished at", current_time)
