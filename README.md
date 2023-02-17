# Fishing Activity Detection in Vessels Trajectories

This is a comparison model that detects fishing vessel activity in an AIS dataset. Two approaches were used: the first uses LR, DT, SVM, and NN with trajectory-based data from the trajectories built by moving pandas. The second employs raw data in traditional CNN and RNN.

## Prerequisites

- Python 3.6+
- Tensorflow
- Keras
- Geo Pandas
- Moving Pandas

## Training

1. Fisrtval, uncompress dataset_fishing_train.7z. 
2. Starting training with: python fishing_activity_detection_exec.py
3. After the first execution, you built the trajectories and saved them in the objects folder. You can load trajectories from files rather than rebuilding them all by setting load_trajectories_collection_from_file=True.
3. You can do the same for load_images_dataset=True if you don't wish to build all images for CNN again.

## Accuracy

| Model  | Accuracy |
| ------------- | :---: |
| LR  | 0.83  |
| DT  | 0.88  |
| SVM  | 0.81  |
| NN  | 0.89  |
| CNN  | 0.72  |
| RNN  | 0.99  |
# fishing_activity_detection
