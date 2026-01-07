# ASSIGNMENT 1: Deeper Regression, Smarter Features (PyTorch Assignment)
The goal of this Assignment is to predict delivery time (in minutes) using multiple input features, and improve model performance through feature engineering+a multi-layer neural network.
Problem Overview:
Unlike the earlier labs where delivery time depended only on distance, this assignment uses a richer dataset (100 deliveries) where delivery time is influenced by:
distance_miles: delivery distance in miles
time_of_day_hours: dispatch time (24h format, e.g. 16.07 ≈ 4 PM)
is_weekend: 1 weekend, 0 weekday
delivery_time_minutes (target): actual delivery time in minutes
Business constraints:
deliveries happen between 8.0 and 20.0
maximum distance is 20 miles
What I did was
1) Data Loading
Loaded data_with_features.csv using Pandas
Verified dataset shape (100 rows × 4 columns)
Used helper plots to visualize patterns

2) Feature Engineering (Rush Hour)
I created a new binary feature to capture traffic patterns:
rush_hour = 1 if:
it is a weekday (is_weekend == 0)
and the delivery time is in:
morning rush: 8.0 ≤ time < 10.0
evening rush: 16.0 ≤ time < 19.0
Otherwise rush_hour = 0.
This makes “traffic effect” explicit instead of hoping the model learns it indirectly.

3) Data Preparation Pipeline
Implemented prepare_data(df) to:
convert DataFrame to a PyTorch tensor
slice raw columns (distance, hour, weekend, target)
generate rush hour feature
reshape features using .unsqueeze(1)
normalize continuous features (distance + hour)
concatenate final feature tensor
Final model input features become 4 columns:
normalized distance
normalized time_of_day
is_weekend
is_rush_hour
Targets are reshaped into a (N, 1) tensor.

4) Neural Network Model  
Built a deeper regression network:
Linear(4 → 64) + ReLU
Linear(64 → 32) + ReLU
Linear(32 → 1)
Optimizer: SGD with learning rate 0.01
Loss: MSELoss (mean squared error)

5) Training Loop
Implemented train_model(features, targets, epochs) that:
initializes model using init_model()
runs forward pass → loss → zero grad → backward → step
logs loss every 5000 epochs
Trained for 30,000 epochs, and the loss consistently decreased.

6) Evaluation + Prediction
evaluated performance using an Actual vs Predicted plot
used the trained model to predict delivery time for a new unseen order through the provided helper function
Files
Deeper_Regression_Smarter_Features.ipynb (main notebook solution)
data_with_features.csv (dataset)
helper_utils.py (provided utilities for plotting/prediction)
unittests.py (provided unit tests)
README.md (this file)
How to Run (Google Colab)
Upload the notebook + data_with_features.csv and helper files  into Colab.
Run the notebook from top to bottom.
Make sure these imports exist at the top:
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
Run unit tests to confirm correctness:
unittests.exercise_1(...)
unittests.exercise_2(...)
unittests.exercise_3(...)
unittests.exercise_4(...)

Notes:
Tensors shape matters a lot: using .unsqueeze(1) avoids silent shape bugs.
Example Prediction
After training, the model can estimate delivery time for a new order:
distance: 20.0 miles

time: 10.0 (10 AM)

weekend: 1

The notebook uses helper_utils.prediction() to normalize the new input and output the predicted delivery time.

#  C1 M1 Lab 1: Building a Simple Neural Network (Delivery Time Predictor)
This lab builds a very simple neural network in PyTorch: a single linear neuron that learns a relationship between delivery distance (miles) and delivery time (minutes).
We follow a compact Machine Learning Pipeline:
Data ingestion & preparation (already cleaned for this lab)
Model building (a single nn.Linear)
Training (optimize weight & bias using MSE + SGD)
Evaluation (predict time for a new distance, e.g., 7 miles)
Debugging/inspection (print learned weight & bias)
Generalization test (show failure on mixed bike+car non-linear data)
Problem Statement
You are a delivery rider with a 7-mile delivery. The company expects delivery in under 30 minutes.
We train a model using historical delivery records to predict whether a new delivery is likely to be late.

Requirements
Python 3.x (Colab already provides this)
PyTorch (already installed in Colab)
Google Drive mounted if you store files in Drive:
from google.colab import drive
drive.mount('/content/drive')
How to Run in Google Colab
Upload the lab folder into Google Drive (or open from your Drive).
Mount Drive:
from google.colab import drive
drive.mount('/content/drive')
Change directory to your project folder:
import os
os.chdir("/content/drive/MyDrive/<YOUR_PATH>/C1_M1_Lab_1_simple_nn")
Confirm files exist:
!ls
Run cells top-to-bottom.
Dataset Used (Bike-only)
We train on simple delivery data:
Distance (miles)	Time (minutes)
1.0	6.96
2.0	12.11
3.0	16.77
4.0	22.21
In PyTorch tensor format (2D shape: [N, 1]):
distances = torch.tensor([[1.0],[2.0],[3.0],[4.0]], dtype=torch.float32)
times     = torch.tensor([[6.96],[12.11],[16.77],[22.21]], dtype=torch.float32)
assert distances.shape == times.shape, "Each distance must have a corresponding time"
Model
A single neuron (linear regression in neural-network form):
Time=W×Distance+B
PyTorch implementation:
model = nn.Sequential(nn.Linear(1, 1))
Training Setup
Loss: Mean Squared Error (MSE)
Optimizer: Stochastic Gradient Descent (SGD)
Learning rate: 0.01
Epochs: 500
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
Training loop:
for epoch in range(500):
    optimizer.zero_grad()
    outputs = model(distances)
    loss = loss_function(outputs, times)
    loss.backward()
    optimizer.step()
Results
After training, the model learns a straight line that fits the bike-only data reasonably well.
You can visualize:
helper_utils.plot_results(model, distances, times)
Prediction (Evaluation)
Example: Predict delivery time for 7 miles:
distance_to_predict = 7.0
with torch.no_grad():
    new_distance = torch.tensor([[distance_to_predict]], dtype=torch.float32)
    predicted_time = model(new_distance).item()
Inspect Learned Parameters (Weight & Bias)
layer = model[0]
weights = layer.weight.data.numpy()
bias = layer.bias.data.numpy()
print(f"Weight: {weights}")
print(f"Bias: {bias}")
Interpretation:
Weight (W): how many minutes increase per extra mile
Bias (B): base time even when distance is 0 (pickup/setup time)
Bonus: Why the Linear Model Fails on Mixed Bike + Car Data
When we include longer deliveries done by car, the relationship becomes non-linear (traffic, highways, etc.).
A single linear neuron can only draw a straight line, so loss becomes huge:
with torch.no_grad():
    predictions = model(new_distances)
new_loss = loss_function(predictions, new_times)
print(f"Loss on new, combined data: {new_loss.item():.2f}")
Visualization:
helper_utils.plot_nonlinear_comparison(model, new_distances, new_times)
This motivates the next lesson: activation functions and deeper networks to learn curves.
Common Issues & Fixes
1) No such file or directory when using os.chdir or !ls
You must mount Drive first:
from google.colab import drive
drive.mount('/content/drive')
2) Images not showing
3) Shape mismatch

# C1 M1 Lab 2: Modeling Non-Linear Patterns with Activation Functions (ReLU)
This lab upgrades the previous linear delivery-time model into a non-linear neural network by adding an activation function (ReLU) and training on mixed bike + car delivery data.
Problem Statement
In Lab 1, a straight-line model fit bike-only deliveries well, but failed once car deliveries were added because the real relationship between distance and time becomes curved (traffic → highway → different speeds).
Goal: train a model that can learn this non-linear pattern and make practical predictions (e.g., decide if delivery can be promised under 45 minutes, and whether to use bike or car).
Requirements
Google Colab (recommended)
PyTorch (pre-installed in Colab)
Drive mounting if using Google Drive storage:
from google.colab import drive
drive.mount("/content/drive")
Running in Google Colab
Mount Drive:
from google.colab import drive
drive.mount("/content/drive")
Confirm files exist:
!ls 

Common Drive/Path Issue:
FileNotFoundError: No such file or directory: '/content/drive/MyDrive/...'
It usually means the folder name/path is different in your Drive.
Fix it by locating your real path:
!ls /content/drive/MyDrive
Then drill down:
!find /content/drive/MyDrive -maxdepth 4 -type d | head -n 50
Copy the correct folder path and use it in os.chdir(...).
Dataset (Bike + Car)
The lab uses a combined dataset with distances from 1 to 20 miles and corresponding times that form a curve:
distances = torch.tensor([[1.0],[1.5],...,[20.0]], dtype=torch.float32)
times     = torch.tensor([[6.96],[9.67],...,[92.98]], dtype=torch.float32)
Visualization:
helper_utils.plot_data(distances, times)
Data Preparation: Standardization (Z-Score Normalization)
Because neural nets are sensitive to scale, we standardize both inputs and targets:
distances_mean, distances_std = distances.mean(), distances.std()
times_mean, times_std= times.mean(), times.std()
distances_norm = (distances - distances_mean) / distances_std
times_norm= (times - times_mean) / times_std
Plot normalized data:
helper_utils.plot_data(distances_norm, times_norm, normalize=True)
Model Architecture (Non-Linear Network)\
This network adds ReLU between two linear layers:
torch.manual_seed(27)
model = nn.Sequential(
    nn.Linear(1, 3),  # hidden layer: 3 neurons
    nn.ReLU(),        # non-linearity
    nn.Linear(3, 1)   # output layer
)
Why this works
nn.Linear alone can only make straight lines.
ReLU creates “bends” by zeroing negative values.
Multiple hidden neurons → multiple bends → curve approximation.
Training
Loss: Mean Squared Error
Optimizer: SGD (lr=0.01)
Epochs: 3000
Final fit visualization:
helper_utils.plot_final_fit(model, distances, times, distances_norm, times_std, times_mean)
Inference (Prediction) — Normalize Input, De-Normalize Output
Because the model was trained on normalized values:
Normalize new distance
Predict normalized time
Convert back to minutes
distance_to_predict = 5.1
Decision Example (45-minute promise + choose vehicle)
if predicted_time_actual.item() > 45:
    print("Decision: Do NOT promise delivery under 45 minutes.")
else:
    if distance_to_predict <= 3:
        print("Decision: Delivery possible. Use a bike.")
    else:
        print("Decision: Delivery possible. Use a car.")
what i learned in this lab was:
More linear neurons without activation still produce a linear model.
ReLU makes the network capable of learning non-linear patterns.
Standardization helps training stay stable and converge better.

# C1 M1 Lab 3:Tensors: The Core of PyTorch
This lab introduces PyTorch tensors, the core data structure used in deep learning.
Before building models or training neural networks, it is essential to understand how data is represented, reshaped, combined, and manipulated using tensors.
This lab focuses on practical tensor skills that are required to avoid and debug the most common PyTorch errors, especially shape and data type issues.
Why Tensors Matter
Tensors are not just containers for data:
They are optimized for fast numerical computation
They support automatic differentiation (used during training)
They enable vectorized operations over entire batches of data
Many PyTorch runtime errors happen because of:
Shape mismatches
Missing batch dimensions
Incorrect data types (int vs float)
Ensure required files exist (for example data.csv):
!ls
Run the notebook cells sequentially from top to bottom.
If image paths fail to load, verify your working directory using:
!pwd
what i understand from this lab was:
1. Tensor Creation
Tensors can be created from:
Python lists (torch.tensor)
NumPy arrays (torch.from_numpy)
pandas DataFrames (torch.tensor(df.values))
2. Tensor Shapes & Dimensions
A tensor with shape:
[batch_size, features]: First dimension → number of samples, Second dimension → number of features per sample
3. Indexing & Slicing
Standard indexing (x[1], x[1, 2])
Slicing (x[:, 2], x[0:2])
Combining indexing and slicing
Extracting Python values using .item()
4. Advanced Indexing
Powerful data selection techniques:
Boolean masking (x[x > 6])
5. Mathematical & Logical Operations
Element-wise arithmetic (+, *)
Dot products (torch.matmul)
Broadcasting (automatic dimension expansion)
Comparison operators (>, <, ==)
Logical operators (&, |)
Statistical functions (mean, std)
Broadcasting was highlighted as a core mechanism behind how neural networks efficiently apply weights and biases.
6. Data Types (dtype)
How PyTorch handles type promotion automatically
what i learned from this lab:
Tensor shape mistakes are the #1 cause of PyTorch errors
Always print tensor.shape when debugging
Models expect a batch dimension
Broadcasting eliminates the need for loops
Correct data types (float32) are essential for training
Tensor operations form the foundation of every neural network

# ASSIGNMENT 2:  EMNIST Letter Detective

In this assignment, we build and train a neural network using PyTorch to recognize handwritten letters from the EMNIST Letters dataset.

This project extends the classic MNIST digit classification task and introduces additional challenges:
	•	26 classes (letters a–z) instead of 10 digits
	•	More variation and noise in handwriting styles
	•	Increased model complexity and evaluation needs
At the end of the assignment, the trained model is used to decode a secret handwritten message from Andrew Ng.
 what i learned in this assignment
	•	Load and explore the EMNIST Letters dataset
	•	Apply preprocessing techniques such as normalization and tensor conversion
	•	Build a multi-layer neural network using nn.Sequential
	•	Train and evaluate a model using PyTorch
	•	Analyze model performance per class
	•	Use the trained model to decode handwritten text
 Dataset
	•	Dataset: EMNIST Letters
	•	Image Size: 28 × 28 grayscale
	•	Training Samples: 124,800
	•	Test Samples: 20,800
	•	Classes: 26 lowercase letters (a–z)
	•	Labels: Originally 1–26 (shifted to 0–25 during training)
 Data Preprocessing
The following preprocessing steps are applied:
	•	Conversion from PIL images to PyTorch tensors
	•	Normalization using precomputed mean and standard deviation
	•	Batching and shuffling using DataLoader
 Image orientation correction is used only for visualization, not for training.
 Model Architecture
The neural network is implemented using nn.Sequential and follows these constraints:
	•	Input layer: Flatten
	•	Hidden layers: Linear + ReLU
	•	Maximum hidden units: ≤ 256
	•	Maximum total layers: ≤ 7
	•	Output layer: Linear with 26 outputs
Loss Function: CrossEntropyLoss
Optimizer: Adam (learning rate = 0.001)
 Training & Evaluation
	•	Training is performed for up to 15 epochs
	•	Accuracy and loss are tracked per epoch
	•	Evaluation is done on unseen test data
	•	Additional evaluation is performed per letter class
Example performance:
	•	Test Accuracy: ~88–90%
	•	Some letters (e.g., I, Q, G) are more challenging due to visual similarity
Requirements
	•	Python 3.x
	•	PyTorch
	•	torchvision
	•	Google Colab (recommended)
 Conclusion
This assignment covers the complete deep learning workflow:
	•	Data loading and preprocessing
	•	Model design and training
	•	Evaluation and error analysis
	•	Real-world application (handwritten message decoding)


# C1 M2 Lab1: Building Your First Image Classifier (MNIST) with PyTorch
This lab builds a complete end-to-end image classification pipeline using PyTorch.
The goal is to train a neural network that recognizes handwritten digits (0–9) from the MNIST dataset.

By the end of the lab:
Load and inspect MNIST data
Apply essential transformations (tensor conversion + normalization)
Build a custom neural network using nn.Module
Train the model using a full training loop
Evaluate accuracy on unseen test data
Visualize predictions and training metrics (loss & accuracy)
Dataset
MNIST is a classic benchmark dataset:
60,000 training images
10,000 test images
Each image is 28×28, grayscale (1 channel)
Labels are integers from 0 to 9
The dataset is downloaded automatically using:
torchvision.datasets.MNIST(download=True)
Main libraries:
torch
torchvision
numpy
matplotlib
If running locally, install dependencies with:
pip install torch torchvision numpy matplotlib
Running in Google Colab
Open the notebook in Colab.
Run all cells from top to bottom.
MNIST will be downloaded automatically into:
./data
Data Pipeline (Transforms + DataLoaders)
Transformations
Two key transforms are applied:
ToTensor()
Converts PIL image → PyTorch tensor
Scales pixel values from 0–255 → 0–1
Normalize((0.1307,), (0.3081,))
Standardizes pixels using MNIST mean/std
Helps training converge faster and more reliably
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
DataLoaders
Training loader:
batch_size=64
shuffle=True (important for learning well)
Test loader:
batch_size=1000
shuffle=False
Model Architecture
The model is a simple fully-connected neural network:
Input shape
Images arrive as:
Single image: [1, 28, 28]
Batch of 64: [64, 1, 28, 28]
Layers
Flatten() converts [1, 28, 28] → [784]
Linear(784 → 128) + ReLU
Linear(128 → 10) outputs logits for 10 classes
Input (1×28×28)
 → Flatten (784)
 → Linear(784, 128)
 → ReLU
 → Linear(128, 10)
Output: logits for digits 0–9
Training Setup
Loss Function
CrossEntropyLoss()
Standard for multi-class classification
Works directly with logits (no softmax needed manually)
Optimizer
Adam(lr=0.001)
Fast and stable optimizer for neural networks
Training Process
Training is done for 5 epochs, and each epoch includes:
Training over all batches (train_epoch)
Evaluation on test set (evaluate)
Tracking:
train_loss
test_acc
Results (Example Output)
Typical results after 5 epochs (CPU in Colab):
Test accuracy around 97%
Training loss decreases steadily
Accuracy curve rises then stabilizes
Example:
Epoch 1 Test Accuracy: ~96%
Epoch 5 Test Accuracy: ~97%
Evaluation & Visualization
After training:
Visualize random predictions from the test set:
helper_utils.display_predictions(trained_model, test_loader, device)
Plot learning curves:
helper_utils.plot_metrics(train_loss, test_acc)
These plots help verify:
Loss goes down → model learns training data
Test accuracy increases → model generalizes to unseen data
what i learned from this lab was
Data normalization is essential for stable training.
Flattening is required before feeding images into linear layers.
Training loop always follows the same pattern:
zero gradients → forward → loss → backward → step
