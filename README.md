# Navigation Turing Test (NTT): Learning to Evaluate Human-Like Navigation
All code and data to reproduce results in Section 5 of [Navigation Turing Test (NTT): Learning to Evaluate Human-Like Navigation [ICML 2021]](https://arxiv.org/abs/2105.09637)

# Getting Started

All code developed in Ubuntu 18.04.2 with WSL and Python 3.6.7.

Optional, but recommended. Setup a virtual environment (e.g. [VirtualEnv](https://virtualenv.pypa.io/).)

Install Dependencies:

    pip install -r requirements.txt

# Human Navigation Turing Test (HNTT)
To reproduce our analysis of responses to the HNTT included in Section 5.1 of the ICML paper:

Download HNTT_data.csv from [this link](https://papersdata.blob.core.windows.net/icml21/ICML2021-hntt-data/HNTT_data.zip) (30KiB)

Step through HNTT_data_analysis.ipynb notebook

## HNTT Survey Templates and Videos
The HNTT survey templates, with the answer key embedded, can be downloaded from [this link](https://papersdata.blob.core.windows.net/icml21/ICML2021-hntt-survey-templates/icml2021-hntt-survey-templates.zip) (301KiB) 

The corresponding HNTT videos can be downloaded from [this link](https://papersdata.blob.core.windows.net/icml21/ICML2021-hntt-videos/icml2021-hntt-videos.zip) (134MiB)

# Automated Navigation Turing Test (ANTT)
## Training ANTT Models (Section 3.3)
To train ANTT models, download the training dataset from [this link](https://papersdata.blob.core.windows.net/icml21/ICML2021-train-data/ICML2021-train-data.zip) (1.95 GiB) then run:

    python train.py --model-type ['visuals', 'symbolic', 'topdown', 'barcode'] --human-train <path> --human-test <path> --agent-train <path> --agent-test <path>

Alternatively, to run a hyperparameter sweep with 5-fold cross validation, first update hyperparameters.json, then run:
    
    python cross_validation.py --model-type ['visuals', 'symbolic', 'topdown', 'barcode'] --human-dirs <path(s)> --agent-dirs <path(s)>

To see all the parameters along with their default values, run  `python cross_validation.py --help`.

To monitor training runs:

    tensorboard --logdir ./logs/

To plot learning curves with variance (e.g. to reproduce figure 2 in the paper):

    python plot_ANTT_training.py

[Optional] To reproduce the barcode or topdown data from the raw trajectories, run:
    
    python topdown/create_topdown_img.py --folders <path(s)> --outdir <path>
    python barcodes/create_barcodes.py --indir <path> --outdir <path>

## Evaluation (Section 5.2)

To reproduce ANTT analysis included in Section 5.2 of the ICML paper:

Download HNTT_data.csv from [this link](https://papersdata.blob.core.windows.net/icml21/ICML2021-hntt-data/HNTT_data.zip) (30KiB)

Download evaluation dataset from [this link](https://papersdata.blob.core.windows.net/icml21/ICML2021-eval-data/ICML2021-eval-data.zip) (264MiB)

Then either:
 + Download trained models (.pt files) and saved model output (.pkl) from [this link](https://papersdata.blob.core.windows.net/icml21/ICML-trained-models/ICML2021-trained-models.zip) (9GiB)
 + Train your own ANTT models as described above

To evaluate a trained model:

    python evaluate_ANTT_model.py --path-to-models PATH --model-type ['BARCODE', 'CNN', 'SYMBOLIC', 'TOPDOWN']

If model is a recurrent CNN or SYMBOLIC model, also pass --subsequence-length N

If model has been previously evaluated, its output for each question in the behavioural study will be saved in a .pkl file. For faster re-evaluation (without classifying replays again) add --load-model-output

To reproduce Figures 9 and 10, plot the evaluation of all ANTT models by:

    python plot_ANTT_evaluation.py
    
# Passing the Navigation Turing Test

In later work on ["How Humans Perceive Human-like Behavior in Video Game Navigation"](https://www.microsoft.com/en-us/research/publication/how-humans-perceive-human-like-behavior-in-video-game-navigation/) published at CHI 2022 we presented the first agent to pass the Navigation Turing Test. [Videos of the CHI 2022 agent are available here.](https://papersdata.blob.core.windows.net/rewardshapingagent2022/videos-new-agent.zip)

# License
Code is licensed under MIT, data and all other content is licensed under Microsoft Research License Agreement (MSR-LA). See LICENSE folder.
