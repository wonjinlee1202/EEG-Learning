# 🧠 EEG Motor Movement Classification with Deep Learning

This project implements deep learning models to classify different **motor movements** based on **EEG brain signal data**. We explore multiple neural network architectures, including **Convolutional Neural Networks (CNNs)** and **Recurrent Neural Networks (RNNs)**, to effectively model the temporal and spatial characteristics of EEG signals.

Achieved over **80% testing accuracy** after careful data preprocessing and hyperparameter tuning.

---

## Dependencies

Install the following Python packages before running the code:

```
pip install numpy matplotlib scikit-learn tensorflow
```

---

## Data Preprocessing

- Trimmed EEG data to first 800 time bins to reduce noise
- Applied maxpooling, averaging, and subsampling (4-bin intervals)
- Increased dataset size to boost generalization and reduce overfitting

---

## Model Architectures

We developed and evaluated four deep learning models for EEG data classification: a CNN, a GRU-based RNN, and two hybrid CNN-RNN models (CNN-LSTM and CNN-GRU):
- **CNN**: A 1D CNN was designed using 10×1 filters to focus on temporal features. The final model used two convolutional layers, Dropout (p=0.5), and L2 regularization (λ=0.03) to mitigate overfitting.
- **GRU**: To leverage the sequential nature of EEG signals, we implemented a lightweight GRU-based RNN, chosen for its efficiency over LSTMs given our limited computational resources.
- **CNN-LSTM**: This hybrid model combined convolutional layers for spatial feature extraction with LSTM layers for temporal modeling. We fine-tuned hyperparameters for improved performance.
- **CNN-GRU**: Our most successful model appended a GRU layer to the CNN architecture, achieving test accuracy over 70%. Early stopping was applied to enhance generalization and reduce training time.


Find the full report [here](ECE_C147_Report.pdf)
