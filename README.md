# 1 Deeper Regression, Smarter Features (PyTorch Assignment)
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
