# M5 Forecasting Competition - Accuracy and Uncertainty 
Code for the [accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy) and 
[uncertainty](https://www.kaggle.com/c/m5-forecasting-uncertainty) streams of the M5 Forecasting competition.

This repository includes:
- Code for training (using PyTorch) and generating submissions for both the streams (using k-fold or single split validation)
- Code for WRMSSE and SPL loss functions as given in the competition guidelines
- 2 main model architectures - Dilated Sequence2Sequence model and Sequence2Sequence with attention 
(applied from all encoder outputs to hidden state of each step in the decoder)
- Options for sliding window training, rolling and lag features, and addition of random noise while training

Please refer to [my Kaggle post](https://www.kaggle.com/c/m5-forecasting-uncertainty/discussion/163389) for additional information.

## Steps for training the model and generating submissions

Steps are identical for both the streams. They have been given below:

1. Create a folder named `data` according to the structure below and place all the
 data files in it. Download links - [accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy/data) and
  [uncertainty](https://www.kaggle.com/c/m5-forecasting-uncertainty/data).
    ```bash
    ├── accuracy_stream
    ├── uncertainty_stream
    ├── data
    ```
2. Enter the corresponding directory for the stream you want to train.
3. If required, modify the `config.py` file to change model architecture, loss function, number of epochs, training and validation periods, etc.
3. From the `data` directory, run:
    ```bash
    python prepare_data.py
    ```
    This will prepare the data for training and evaluation by merging different data files and creating the required features.
4. Start model training by running the below command from the root directory of the stream under consideration:
    ```bash
    python train.py
    ```
5. To generate submission files, just run:
    ```bash
    python generate_submission.py
    ```
   Separate submission files will be generated for each fold. Ensembles have to be created manually.

## Results

Accuracy Stream

| Model Architecture        | Details | Private Leaderboard Score | Private Leaderboard Rank
|:-------------------------:|:-------:|:-------------------------:|:-----------------------:|
| seq2seq_w_attn_on_hid | Sliding window training (window length = 28*13), 3-fold validation on last 3 28-day periods | 0.68081 | 482 (top 9%)


\
Uncertainty Stream

| Model Architecture        | Details | Private Leaderboard Score | Private Leaderboard Rank
|:-------------------------:|:-------:|:-------------------------:|:-----------------------:|
| seq2seq_w_attn_on_hid                                 | Sliding window training (window length = 28*13), 3-fold validation on last 3 28-day periods | 0.18317 | NA
| dilated_seq2seq                                       | Sliding window training (window length = 28*13), 3-fold validation on last 3 28-day periods | 0.18068 | NA
| Ensemble of seq2seq_w_attn_on_hid and dilated_seq2seq | Sliding window training (window length = 28*13), 3-fold validation on last 3 28-day periods | 0.17850 | 48 (top 6%)

## Citations
- [PyTorch implementation of Dilated Recurrent Neural Networks (DilatedRNN)](https://github.com/zalandoresearch/pytorch-dilated-rnn)
