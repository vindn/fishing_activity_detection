# Fishing Activity Detection in Vessels Trajectories

This is a comparison model that detects fishing vessel activity in an AIS dataset. Two approaches were used: the first uses LR, DT, SVM, and NN with trajectory-based data from the trajectories built by moving pandas. The second employs raw data in traditional CNN and RNN.

## Prerequisites

- Python 3.6+
- Tensorflow
- Keras
- Geo Pandas
- Moving Pandas

## Training

1. Fisrtval, uncompress dataset_fishing_train.csv.zip 
2. Starting training with: python fishing_activity_detection_exec.py
3. After the first execution, you built the trajectories and saved them in the objects folder. You can load trajectories from files rather than rebuilding them all by setting load_trajectories_collection_from_file=True.
3. You can do the same for load_images_dataset=True if you don't wish to build all images for CNN again.

## Dataset

After building the vessel's trajectories:
- Train: 5868 trajectories by class
- Test: 1468 trajectories by class

## Accuracy

| Model  | Accuracy |
| ------------- | :---: |
| LR  | 0.85  |
| DT  | 0.87  |
| SVM  | 0.83  |
| RF  | 0.88  |
| NN  | 0.91  |
| CNN  | 0.78  |
| RNN  | 1.00  |

