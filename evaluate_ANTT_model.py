# --------------------------------------------------------------------------------------------------
#  Copyright (c) 2021 Microsoft Corporation
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
#  associated documentation files (the "Software"), to deal in the Software without restriction,
#  including without limitation the rights to use, copy, modify, merge, publish, distribute,
#  sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or
#  substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
#  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# --------------------------------------------------------------------------------------------------

# Script to evaluate trained ANTT model on video pairs used in HNTT behavioural study
# Generates data for reproducing Table 2 in the appendix

import argparse
import os
import torch
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr

parser = argparse.ArgumentParser(
    description='Script to evaluate a trained ANTT model',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--path-to-eval-data',
    type=str,
    default='./data/ICML2021-eval-data',
    metavar='STR',
    help='Path to folder of trajectories in format required by ANTT models')
parser.add_argument(
    '--path-to-models',
    type=str,
    default='./data/ICML2021-trained-models/SYM-FF',
    metavar='STR',
    help='Path to folder of trained models (.pt) or saved model outputs (.pkl)')
parser.add_argument(
    '--model-type',
    choices=[
        'BARCODE',
        'CNN',
        'SYMBOLIC',
        'TOPDOWN'],
    default='SYMBOLIC',
    help='Type of model to be evaluated')
parser.add_argument(
    '--subsequence-length',
    type=int,
    default=1,
    metavar='INT',
    help='length of subsequence input to recurrent CNN or SYMBOLIC models')
parser.add_argument('--load-model-output', action='store_true', default=False,
                    help='Load saved model output')
args = parser.parse_args()

if args.model_type == "BARCODE":
    from barcodes.barcodes_classifier import BarcodesClassifier as modelClass
    from PIL import Image
elif args.model_type == "CNN":
    from visuals.visuals_classifier import VisualsClassifier as modelClass
    import base64
    import json
    import io
    import itertools
    from PIL import Image
elif args.model_type == "SYMBOLIC":
    from symbolic.symbolic_classifier import SymbolicClassifier as modelClass
    from symbolic.symbolic_dataset import read_trajectories
elif args.model_type == "TOPDOWN":
    from topdown.topdown_classifier import TopdownClassifier as modelClass
    import torchvision

# Each sublist is a pair of trajectories shown to study participants
user_study_1_human_hybrid = [["___ReplayDebug-Map_Rooftops_Seeds_Main-2021.01.15-12.17.36",
                              "___ReplayDebug-Map_Rooftops_Seeds_Main-2020.12.16-16.08.06"],
                             ["___ReplayDebug-Map_Rooftops_Seeds_Main-2020.12.15-18.25.22",
                              "___ReplayDebug-Map_Rooftops_Seeds_Main-2021.01.15-12.23.26"],
                             ["___ReplayDebug-Map_Rooftops_Seeds_Main-2021.01.15-15.30.37",
                              "___ReplayDebug-Map_Rooftops_Seeds_Main-2020.12.17-11.40.11"],
                             ["___ReplayDebug-Map_Rooftops_Seeds_Main-2020.12.15-18.14.12",
                              "___ReplayDebug-Map_Rooftops_Seeds_Main-2021.01.15-15.35.22"],
                             ["___ReplayDebug-Map_Rooftops_Seeds_Main-2020.12.16-15.57.17",
                              "___ReplayDebug-Map_Rooftops_Seeds_Main-2021.01.15-14.37.26"],
                             ["___ReplayDebug-Map_Rooftops_Seeds_Main-2021.01.15-15.24.55",
                              "___ReplayDebug-Map_Rooftops_Seeds_Main-2020.12.16-16.03.38"]]

# Labels 1 if 2nd video is human, 0 if human is the 1st video
user_study_1_human_hybrid_labels = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0])

# Each sublist is a pair of trajectories shown to study participants
user_study_1_symbolic_hybrid = [["___ReplayDebug-Map_Rooftops_Seeds_Main-2021.01.11-18.11.48",
                                 "___ReplayDebug-Map_Rooftops_Seeds_Main-2021.01.15-15.22.10"],
                                ["___ReplayDebug-Map_Rooftops_Seeds_Main-2021.01.11-18.16.45",
                                 "___ReplayDebug-Map_Rooftops_Seeds_Main-2021.01.15-15.23.34"],
                                ["___ReplayDebug-Map_Rooftops_Seeds_Main-2021.01.11-18.14.45",
                                 "___ReplayDebug-Map_Rooftops_Seeds_Main-2021.01.15-15.23.57"],
                                ["___ReplayDebug-Map_Rooftops_Seeds_Main-2021.01.15-14.34.52",
                                 "___ReplayDebug-Map_Rooftops_Seeds_Main-2021.01.11-18.09.52"]]

# Each sublist is a pair of trajectories shown to study participants
user_study_2_human_symbolic = [["___ReplayDebug-Map_Rooftops_Seeds_Main-2021.01.11-18.13.07",
                                "___ReplayDebug-Map_Rooftops_Seeds_Main-2020.12.15-18.23.57"],
                               ["___ReplayDebug-Map_Rooftops_Seeds_Main-2020.12.17-11.33.59",
                                "___ReplayDebug-Map_Rooftops_Seeds_Main-2021.01.11-18.26.15"],
                               ["___ReplayDebug-Map_Rooftops_Seeds_Main-2021.01.11-18.25.18",
                                "___ReplayDebug-Map_Rooftops_Seeds_Main-2020.12.15-18.21.30"],
                               ["___ReplayDebug-Map_Rooftops_Seeds_Main-2020.12.17-11.41.46",
                                "___ReplayDebug-Map_Rooftops_Seeds_Main-2021.01.11-18.16.22"],
                               ["___ReplayDebug-Map_Rooftops_Seeds_Main-2020.12.16-16.10.12",
                                "___ReplayDebug-Map_Rooftops_Seeds_Main-2021.01.11-16.54.44"],
                               ["___ReplayDebug-Map_Rooftops_Seeds_Main-2021.01.11-18.25.42",
                                "___ReplayDebug-Map_Rooftops_Seeds_Main-2020.12.17-11.38.34"]]

# Labels 1 if 2nd video is human, 0 if human is the 1st video
user_study_2_human_symbolic_labels = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0])

# Each sublist is a pair of trajectories shown to study participants
user_study_2_symbolic_hybrid = [["___ReplayDebug-Map_Rooftops_Seeds_Main-2021.01.11-18.23.05",
                                 "___ReplayDebug-Map_Rooftops_Seeds_Main-2021.01.15-15.34.59"],
                                ["___ReplayDebug-Map_Rooftops_Seeds_Main-2021.01.11-18.12.27",
                                 "___ReplayDebug-Map_Rooftops_Seeds_Main-2021.01.15-13.34.24"],
                                ["___ReplayDebug-Map_Rooftops_Seeds_Main-2021.01.11-18.17.19",
                                 "___ReplayDebug-Map_Rooftops_Seeds_Main-2021.01.15-13.34.55"],
                                ["___ReplayDebug-Map_Rooftops_Seeds_Main-2021.01.15-15.34.05",
                                 "___ReplayDebug-Map_Rooftops_Seeds_Main-2021.01.11-18.17.50"]]

# Labels 1 if 2nd video is human for all human vs agent comparisons ([0:6] & [10:16])
# Or if 2nd video is hybrid for hybrid vs symbolic comparisons ([6:10] & [16:20])
all_study_labels = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0,
                             1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loadHumanResponses():
    max_vote_user_response = np.array([])
    percentage_user_response = np.array([])

    df = pd.read_csv('./data/HNTT_data.csv', header=0)
    for study in [1, 2]:
        study_df = df[df.studyno == study]
        for question in range(1, 11):
            question_df = study_df[study_df.question_id == question]

            response_counts = {}
            value_counts = question_df['subj_resp'].value_counts()
            response_counts["A"] = value_counts.get("A", 0)
            response_counts["B"] = value_counts.get("B", 0)

            # User study participants vote in favour of left video
            if response_counts["A"] > response_counts["B"]:
                max_vote_user_response = np.append(max_vote_user_response, 0.0)
            elif response_counts["A"] == response_counts["B"]:
                # Break ties randomly
                max_vote_user_response = np.append(
                    max_vote_user_response, np.random.randint(2))
            else:
                max_vote_user_response = np.append(max_vote_user_response, 1.0)

            # Human or hybrid is on right so get percentage that agree with
            # this
            if all_study_labels[question - 1] == 1.0:
                percentage_user_response = np.append(
                    percentage_user_response,
                    response_counts["B"] / (response_counts["A"] + response_counts["B"]))
            else:  # Human or hybrid is on left so get percentage that agree with this
                percentage_user_response = np.append(
                    percentage_user_response,
                    response_counts["A"] / (response_counts["A"] + response_counts["B"]))

    print("Max Vote User Responses: {}".format(max_vote_user_response))
    print("Percentage User Responses: {}".format(percentage_user_response))

    print("Max Vote Human Response Accuracy On All: {}".format(
        accuracy_score(max_vote_user_response, all_study_labels)))
    print("Max Vote Human Response Accuracy In Human-Agent Q's: {}".format(
        accuracy_score(np.append(max_vote_user_response[0:6], max_vote_user_response[10:16]),
                       np.append(all_study_labels[0:6], all_study_labels[10:16]))))
    print("Max Vote Human Response Accuracy Picking Hybrid Agent In Hybrid-Symbolic Agent Q's: {}".format(
        accuracy_score(np.append(max_vote_user_response[6:10], max_vote_user_response[16:20]),
                       np.append(all_study_labels[6:10], all_study_labels[16:20]))))
    print("------------------------------------------------------------")

    return max_vote_user_response, percentage_user_response


if __name__ == "__main__":
    print("LOADING HUMAN USER STUDY RESPONSES TO COMPARE MODEL OUTPUT AGAINST")
    max_vote_user_response, percentage_user_response = loadHumanResponses()

    # Initialise lists to store stats for every model in directory
    ground_truth_accuracy_list = []
    human_agent_userlabel_accuracy_list = []
    hybrid_symbolic_userlabel_accuracy_list = []
    spearman_rank_human_agent = []
    spearman_rank_hybrid_symbolic = []

    # Loop over all trained models in directory
    for filename in os.listdir(args.path_to_models):
        if not filename.endswith(".pt"):
            continue
        PATH_TO_MODEL = os.path.join(args.path_to_models, filename)
        PATH_TO_MODEL_OUTPUT = os.path.join(args.path_to_models, filename[:-3] + "-model_output.pkl")

        if args.load_model_output:
            print("LOADING SAVED OUTPUT FOR MODEL: {}".format(PATH_TO_MODEL))
            print("FROM: {}".format(PATH_TO_MODEL_OUTPUT))
            model_output_dict = pickle.load(open(PATH_TO_MODEL_OUTPUT, "rb"))
        else:
            print("LOADING TRAINED MODEL: {}".format(PATH_TO_MODEL))
            model = modelClass(device).to(device)
            model.load_state_dict(torch.load(PATH_TO_MODEL,
                                             map_location=device))
            model.eval()  # Do not update params of model
            # Create empty dictionary to fill then save
            model_output_dict = {}

        # For every pair of trajectories shown to human participants predict most human-like trajectory
        # For models that classify only one trajectory, classify both trajectories separately
        # then pick the one given highest probability of being human
        model_predictions = np.array([])
        percentage_model = np.array([])
        for j, traj_pair in enumerate(user_study_1_human_hybrid +
                                      user_study_1_symbolic_hybrid +
                                      user_study_2_human_symbolic +
                                      user_study_2_symbolic_hybrid):
            percentage_humanlike = []
            for traj in traj_pair:
                if not args.load_model_output:
                    model_output_dict[traj] = []
                if args.model_type == "BARCODE":
                    # load the barcode corresponding to this trajectory
                    in_barcode = os.path.join(
                        args.path_to_eval_data, "barcodes", traj + 'Trajectories.png')

                    img = Image.open(in_barcode)
                    img = np.array(img) / 255
                    img = np.transpose(img, (2, 0, 1))
                    print("barcode trajectory shape:", img.shape)

                    with torch.no_grad():
                        human_count = 0
                        agent_count = 0

                        # sample four random 320x200 windows from the barcode
                        for i in range(0, 4):
                            if img.shape[1] - 200 < 0:
                                start_y = 0
                            else:
                                start_y = np.random.randint(
                                    0, img.shape[1] - 200)
                            cut_barcode = img[:, start_y:start_y + 200, :]
                            y_shape = cut_barcode.shape[1]
                            if y_shape < 200:
                                cut_barcode = np.pad(
                                    cut_barcode, ((0, 0), (0, 200 - y_shape), (0, 0)), mode='edge')

                            input_bc = torch.Tensor(cut_barcode)
                            input_bc = torch.unsqueeze(input_bc, 0).to(device)

                            if args.load_model_output:
                                model_output = model_output_dict[traj][0]
                            else:
                                model_output = model(input_bc)
                                model_output_dict[traj].append(model_output)

                            _, prediction = torch.max(model_output.data, 1)

                            if prediction == 1:
                                human_count += 1
                            else:
                                agent_count += 1

                        percentage_humanlike.append(
                            human_count / (human_count + agent_count))

                elif args.model_type == "TOPDOWN":
                    transform = torchvision.transforms.Compose(
                        [torchvision.transforms.Resize((512, 512)),
                         torchvision.transforms.ToTensor(),
                         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

                    PATH_TO_IMAGE_FOLDER = os.path.join(
                        args.path_to_eval_data, 'topdown_320x200', traj + 'Trajectories.json/')
                    data = torchvision.datasets.ImageFolder(
                        root=PATH_TO_IMAGE_FOLDER, transform=transform)
                    if args.load_model_output:
                        model_output = model_output_dict[traj][0]
                    else:
                        model_output = model(torch.unsqueeze(data[0][0], 0))
                        model_output_dict[traj].append(model_output)

                    # model_output is in the range [-1,1], so normalise to
                    # [0,1] like all other models
                    normalised_model_output = (
                        model_output.data[0][1].item() + 1) / 2
                    percentage_humanlike.append(normalised_model_output)

                elif args.model_type == "CNN":
                    PATH_TO_TRAJECTORY = os.path.join(
                        args.path_to_eval_data,
                        'study_videos_cut_jpg',
                        traj + 'Trajectories.json')
                    with open(PATH_TO_TRAJECTORY) as main_file:
                        video = []
                        for line in itertools.islice(main_file, 0, None, 10):
                            step = json.loads(line)
                            key = list(step.keys())[0]

                            encoded_img = step[key]["Observations"]["Players"][0]["Image"]["ImageBytes"]
                            decoded_image_data = base64.decodebytes(
                                encoded_img.encode('utf-8'))
                            image = Image.open(io.BytesIO(decoded_image_data))
                            img = np.array(image)
                            video.append(img)

                        videodata = np.array(video) / 255
                        videodata = np.transpose(videodata, (0, 3, 1, 2))
                        print("video trajectory shape:", videodata.shape)

                        with torch.no_grad():
                            human_count = 0
                            agent_count = 0
                            number_sequences = len(
                                video) // args.subsequence_length
                            for i in range(number_sequences):
                                sequence_start_idx = i * args.subsequence_length

                                input_seq = torch.Tensor(
                                    videodata[sequence_start_idx:sequence_start_idx + args.subsequence_length, :])
                                input_seq = torch.unsqueeze(
                                    input_seq, 0).to(device)

                                if args.load_model_output:
                                    model_output = model_output_dict[traj][i]
                                else:
                                    model_output = model(input_seq)
                                    model_output_dict[traj].append(
                                        model_output)

                                _, prediction = torch.max(model_output.data, 1)

                                if prediction == 1:
                                    human_count += 1
                                else:
                                    agent_count += 1

                        percentage_humanlike.append(
                            human_count / (human_count + agent_count))

                elif args.model_type == "SYMBOLIC":
                    PATH_TO_TRAJECTORY = os.path.join(
                        args.path_to_eval_data,
                        'study_videos_cut_jpg',
                        traj + 'Trajectories.json')
                    traj_data = read_trajectories(PATH_TO_TRAJECTORY, -1)[0][0]

                    with torch.no_grad():
                        human_count = 0
                        agent_count = 0
                        number_sequences = len(
                            traj_data["obs"]) // args.subsequence_length
                        for i in range(number_sequences):
                            sequence_start_idx = i * args.subsequence_length
                            sample_trajectory = traj_data["obs"][sequence_start_idx:
                                                                 sequence_start_idx + args.subsequence_length]

                            if args.load_model_output:
                                model_output = model_output_dict[traj][i]
                            else:
                                model_output = model(
                                    torch.tensor([sample_trajectory]))
                                model_output_dict[traj].append(model_output)

                            if round(model_output.item()) == 1:
                                human_count += 1
                            else:
                                agent_count += 1

                        percentage_humanlike.append(
                            human_count / (human_count + agent_count))
                else:
                    raise NotImplementedError(
                        "Model type " + args.model_type + " evaluation not implemented")

            # Model votes left video is more humanlike
            if percentage_humanlike[0] > percentage_humanlike[1]:
                model_predictions = np.append(model_predictions, 0.0)
            elif percentage_humanlike[0] == percentage_humanlike[1]:
                # Break ties randomly
                model_predictions = np.append(
                    model_predictions, np.random.randint(2))
            else:  # Model votes right video is more humanlike
                model_predictions = np.append(model_predictions, 1.0)

            # Human or hybrid is on right so get percentage that agree with
            # this
            if all_study_labels[j] == 1.0:
                percentage_model = np.append(
                    percentage_model, percentage_humanlike[1])
            else:  # Human or hybrid is on left so get percentage that agree with this
                percentage_model = np.append(
                    percentage_model, percentage_humanlike[0])

        # Save model output to enable faster stats re-running
        pickle.dump(model_output_dict, open(PATH_TO_MODEL_OUTPUT, "wb"))
        print("Ground Truth Labels: {}".format(all_study_labels))
        print("Model Predictions: {}".format(model_predictions))

        # Calculate model accuracy on held-out test dataset compared to ground
        # truth label (only on human vs agent examples)
        ground_truth_accuracy = accuracy_score(np.append(user_study_1_human_hybrid_labels, user_study_2_human_symbolic_labels), np.append(
            model_predictions[0:6], model_predictions[10:16]))  # 1st 6 questions in both studies are human vs agent
        print('Per Trajectory Model Accuracy With Ground Truth Labels: {:.4f}'.format(
            ground_truth_accuracy))
        ground_truth_accuracy_list.append(ground_truth_accuracy)

        model_accuracy_userlabels_human_agent = accuracy_score(np.append(
            max_vote_user_response[0:6], max_vote_user_response[10:16]), np.append(model_predictions[0:6], model_predictions[10:16]))
        print('Model Accuracy on Human-Agent Comparisons With Max Vote User Study Responses As Labels: {:.4f}'.format(
            model_accuracy_userlabels_human_agent))
        human_agent_userlabel_accuracy_list.append(
            model_accuracy_userlabels_human_agent)

        # Spearman rank correlation of model predictions to percentage user
        # ranking
        print(percentage_user_response[0:6])
        print(percentage_user_response[10:16])
        coef, p = spearmanr(np.append(percentage_user_response[0:6], percentage_user_response[10:16]),
                            np.append(percentage_model[0:6], percentage_model[10:16]))
        print(
            'Spearmans correlation coefficient of all human vs agent comparisons: {} (p={})'.format(
                coef,
                p))
        if not np.isnan(coef):
            spearman_rank_human_agent.append(coef)

        model_accuracy_userlabels_hybrid_symbolic = accuracy_score(np.append(
            max_vote_user_response[6:10], max_vote_user_response[16:20]), np.append(model_predictions[6:10], model_predictions[16:20]))
        print('Model Accuracy on Hybrid-Symbolic Agent Comparisons With Max Vote User Study Responses As Labels: {:.4f}'.format(
            model_accuracy_userlabels_hybrid_symbolic))
        hybrid_symbolic_userlabel_accuracy_list.append(
            model_accuracy_userlabels_hybrid_symbolic)

        coef, p = spearmanr(np.append(percentage_user_response[6:10], percentage_user_response[16:20]),
                            np.append(percentage_model[6:10], percentage_model[16:20]))
        print(
            'Spearmans correlation coefficient of all hybrid vs symbolic agent comparisons: {} (p={})'.format(
                coef,
                p))
        if not np.isnan(coef):
            spearman_rank_hybrid_symbolic.append(coef)
        print("------------------------------------------------------------")

    print("Results Summary From All Models in: {}".format(args.path_to_models))
    print(
        "Model Ground Truth Accuracy: Mean {} - STD {}".format(
            np.array(ground_truth_accuracy_list).mean(),
            np.array(ground_truth_accuracy_list).std()))

    print("Model Accuracy on Human-Agent Comparisons With Max Vote User Study Responses As Labels: Mean {} - STD {}".format(
        np.array(human_agent_userlabel_accuracy_list).mean(), np.array(human_agent_userlabel_accuracy_list).std()))

    print("Spearman Rank Correlation Coefficient on Human vs Agent Rankings: Mean {} - STD {}".format(
        np.array(spearman_rank_human_agent).mean(), np.array(spearman_rank_human_agent).std()))

    print("Model Accuracy on Hybrid-Symbolic Agent Comparisons With Max Vote User Study Responses As Labels: Mean {} - STD {}".format(
        np.array(hybrid_symbolic_userlabel_accuracy_list).mean(), np.array(hybrid_symbolic_userlabel_accuracy_list).std()))

    print("Spearman Rank Correlation Coefficient on Hybrid vs Symbolic Agent Rankings: Mean {} - STD {}".format(
        np.array(spearman_rank_hybrid_symbolic).mean(), np.array(spearman_rank_hybrid_symbolic).std()))
