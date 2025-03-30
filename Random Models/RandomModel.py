import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Random Model Architecture with even more randomness
class RandomModel(nn.Module):
    def __init__(self, input_size=20, output_size=3):
        super(RandomModel, self).__init__()
        self.layers = nn.ModuleList()
        num_layers = torch.randint(2, 6, (1,)).item()  # Random number of layers between 2 and 5
        
        for _ in range(num_layers):
            output_size_layer = torch.randint(32, 128, (1,)).item()  # Random number of neurons per layer
            self.layers.append(nn.Linear(input_size, output_size_layer))
            input_size = output_size_layer
            
            # Random dropout layers with random probabilities
            if torch.rand(1).item() > 0.5:  # 50% chance to add dropout layer
                dropout_prob = torch.rand(1).item()  # Random dropout probability
                self.layers.append(nn.Dropout(dropout_prob))

            # Random batch normalization layer
            if torch.rand(1).item() > 0.5:  # 50% chance to add batch normalization
                self.layers.append(nn.BatchNorm1d(output_size_layer))

            # Random weight scaling at initialization
            weight_scale = random.uniform(0.5, 2.0)
            for param in self.layers[-1].parameters():
                param.data *= weight_scale

            # Random choice of activation function
            activation_fn = torch.randint(0, 3, (1,)).item()
            if activation_fn == 0:
                self.layers.append(nn.ReLU())  # Use nn.ReLU module
            elif activation_fn == 1:
                self.layers.append(nn.Sigmoid())  # Use nn.Sigmoid module
            else:
                self.layers.append(nn.Tanh())  # Use nn.Tanh module

        self.output_layer = nn.Linear(input_size, output_size)  # Output layer

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)  # Each layer (Linear, Dropout, BatchNorm, Activation) is applied
        x = self.output_layer(x)
        return x

# Random Data Generation
def generate_random_data(n_samples=1000, n_features=20, n_classes=3):
    # Ensure parameters satisfy the condition: n_classes * n_clusters_per_class <= 2**n_informative
    n_clusters_per_class = 1  # Set to 1 to avoid exceeding the limit
    n_informative = max(2, n_classes)  # Ensure at least enough informative features for the classes
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters_per_class,
        n_informative=n_informative,
        random_state=random.randint(0, 1000)
    )
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random.randint(0, 1000))
    return torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long), torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)

# Create Random Data
X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = generate_random_data()

# Random Optimizer and Loss Function
optimizer_fn = random.choice([optim.Adam, optim.SGD, optim.RMSprop])
loss_fn = random.choice([nn.CrossEntropyLoss(), nn.MSELoss(), nn.BCEWithLogitsLoss()])

# Random Batch Size
batch_size = random.randint(16, 128)  # Random batch size between 16 and 128

# Initialize Model and Optimizer
model = RandomModel(input_size=20, output_size=3)  # Random 20 input features, 3 classes
optimizer = optimizer_fn(model.parameters(), lr=random.uniform(0.0001, 0.01))  # Random learning rate

# Monitoring: Store Losses for Plotting
train_losses = []

# Training Loop with Real-Time Monitoring and Random Epochs
def train(model, num_epochs=5):
    # Random number of epochs
    num_epochs = random.randint(5, 500)  # Random epochs between 5 and 500
    
    # Setup real-time plotting
    plt.ion()  # Turn on interactive mode for real-time updates
    fig, ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Real-Time Training Loss')
    line, = ax.plot([], [], label='Training Loss')

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Random batch splitting to add more randomness
        permutation = torch.randperm(X_train_tensor.size(0))
        for i in range(0, X_train_tensor.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]
            outputs = model(batch_x)
            
            # Handle one-hot encoding for MSELoss
            if isinstance(loss_fn, nn.MSELoss):
                batch_y_one_hot = torch.nn.functional.one_hot(batch_y, num_classes=3).float()
                loss = loss_fn(outputs, batch_y_one_hot)
            else:
                loss = loss_fn(outputs, batch_y)
            
            loss.backward()
        
        optimizer.step()

        # Store the loss for plotting
        train_losses.append(loss.item())

        # Update the plot in real-time
        line.set_xdata(range(len(train_losses)))
        line.set_ydata(train_losses)
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.1)  # Pause to allow the plot to update

        # Print epoch and loss
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    # Finalize plot after training
    plt.ioff()  # Turn off interactive mode
    plt.show()

train(model)
