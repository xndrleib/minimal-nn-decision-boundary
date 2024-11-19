from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim

np.random.seed(42)


def generate_custom_data(cluster_params):
    """
    Generate a dataset with customizable clusters.

    Parameters:
    - cluster_params: A list of dictionaries, each containing parameters for a cluster.
        Each dictionary should have:
            - 'center': Coordinates of the cluster center (list or tuple of length 2).
            - 'std': Standard deviation of the cluster (float).
            - 'n_samples': Number of samples in the cluster (int).
            - 'label': Class label for the cluster (int).

    Returns:
    - X: Generated samples (numpy array).
    - y: Integer labels for class membership of each sample (numpy array).
    """
    X_list = []
    y_list = []

    for params in cluster_params:
        X_cluster, _ = make_blobs(
            n_samples=params['n_samples'],
            centers=[params['center']],
            cluster_std=params['std']
        )
        X_list.append(X_cluster)
        y_list.extend([params['label']] * params['n_samples'])

    X = np.vstack(X_list)
    y = np.array(y_list)

    return X, y


class MinimalNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=1):
        """
        Initialize the minimal neural network.

        Parameters:
        - input_size: Number of input features.
        - hidden_size: Number of neurons in the hidden layer.
        """
        super(MinimalNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        - x: Input tensor.

        Returns:
        - out: Output tensor after passing through the network.
        """
        out = self.fc1(x)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


def train_model(model, criterion, optimizer, dataloader, num_epochs=100):
    losses = []
    # Add a learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    for epoch in range(num_epochs):
        epoch_loss = 0
        for inputs, labels in dataloader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

        # Adjust the learning rate at the end of each epoch
        scheduler.step()

        # Record average epoch loss
        losses.append(epoch_loss / len(dataloader))

        if (epoch + 1) % 10 == 0:
            print(
                f'Epoch [{epoch+1}/{num_epochs}], Loss: {losses[-1]:.4f}, '
                f'LR: {scheduler.get_last_lr()[0]:.6f}'
            )

    return model, losses


def visualize_results(model, X, y, scaler):
    """
    Visualize the data and decision boundary.

    Parameters:
    - model: Trained neural network model.
    - X: Original (unnormalized) data features.
    - y: Data labels.
    - scaler: Fitted scaler used for normalization.
    """
    # Create a grid of points
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.05  # step size in the mesh
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    # Flatten and scale the grid
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler.transform(grid)

    # Predict classes for each point in the grid
    with torch.no_grad():
        model.eval()
        inputs = torch.from_numpy(grid_scaled).float()
        outputs = model(inputs)
        Z = outputs.reshape(xx.shape)
        Z = Z.detach().numpy()
        Z = (Z > 0.5).astype(int)

    # Plot the contour and training examples
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()


def get_decision_boundary_lines(model, scaler):
    """
    Extract the formulas of lines for each boundary, rescaled to the original feature space.

    Parameters:
    - model: Trained neural network model.
    - scaler: MinMaxScaler instance used to normalize the data.

    Returns:
    - lines: List of line equations as strings in the original feature space.
    """
    lines = []
    # Get weights and biases from the first linear layer
    weights = model.fc1.weight.detach().numpy()
    biases = model.fc1.bias.detach().numpy()

    # Scaling parameters
    data_min = scaler.data_min_
    data_max = scaler.data_max_
    scale = data_max - data_min

    for idx in range(weights.shape[0]):
        w1, w2 = weights[idx]
        b = biases[idx]

        # Rescale weights to the original feature space
        w1_orig = w1 / scale[0]
        w2_orig = w2 / scale[1]

        # Adjust the bias
        b_orig = (
            b - (w1 * data_min[0] / scale[0]) - (w2 * data_min[1] / scale[1])
        )

        # Equation of the line in the original space: w1_orig * x1 + w2_orig * x2 + b_orig = 0
        # Solving for x2: x2 = (-w1_orig * x1 - b_orig) / w2_orig
        if abs(w2_orig) > 1e-6:
            slope = -w1_orig / w2_orig
            intercept = -b_orig / w2_orig
            line_eq = f"y = {slope:.4f} * x + {intercept:.4f}"
            lines.append(line_eq)
        else:
            # Vertical line: x = -b_orig / w1_orig
            x_intercept = -b_orig / w1_orig
            line_eq = f"x = {x_intercept:.4f}"
            lines.append(line_eq)

    return lines


def visualize_initialized_lines(model):
    """
    Visualize the lines represented by the neurons before training.
    """
    weights = model.fc1.weight.detach().numpy()
    biases = model.fc1.bias.detach().numpy()

    plt.figure(figsize=(10, 6))
    x_vals = np.linspace(0, 1, 100)

    for idx in range(weights.shape[0]):
        w1, w2 = weights[idx]
        b = biases[idx]

        # Equation of the line: w1 * x + w2 * y + b = 0
        # Solve for y: y = (-w1 * x - b) / w2
        if abs(w2) > 1e-6:
            y_vals = (-w1 * x_vals - b) / w2
            plt.plot(
                x_vals,
                y_vals,
                label=(
                    f'Neuron {idx+1}: {w1:.2f} * x + {w2:.2f} * y '
                    f'+ {b:.2f} = 0'
                )
            )
        else:
            # Vertical line at x = -b / w1
            x_intercept = -b / w1
            plt.axvline(
                x=x_intercept,
                label=(
                    f'Neuron {idx+1}: x = -{b:.2f} / {w1:.2f}'
                )
            )

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Feature 1 (scaled)')
    plt.ylabel('Feature 2 (scaled)')
    plt.title('Initialized Decision Boundaries')
    plt.legend()
    plt.show()


def main():
    # User-specified dataset parameters
    # General Lines
    cluster_params = [
        {'center': [0, -2], 'std': 0.2, 'n_samples': 2500, 'label': 0},
        {'center': [2, 0], 'std': 0.2, 'n_samples': 2500, 'label': 1},
        {'center': [4, 2], 'std': 0.2, 'n_samples': 2500, 'label': 0},
        {'center': [6, 4], 'std': 0.2, 'n_samples': 2500, 'label': 1}
    ]

    # Horizontal Lines
    # cluster_params = [
    #     {'center': [0, -2], 'std': 0.2, 'n_samples': 2500, 'label': 0},
    #     {'center': [0, 0], 'std': 0.2, 'n_samples': 2500, 'label': 1},
    #     {'center': [0, 2], 'std': 0.2, 'n_samples': 2500, 'label': 0},
    #     {'center': [0, 4], 'std': 0.2, 'n_samples': 2500, 'label': 1}
    # ]

    # Vertical Lines
    # cluster_params = [
    #     {'center': [0, 0], 'std': 0.2, 'n_samples': 2500, 'label': 0},
    #     {'center': [2, 0], 'std': 0.2, 'n_samples': 2500, 'label': 1},
    #     {'center': [4, 0], 'std': 0.2, 'n_samples': 2500, 'label': 0},
    #     {'center': [6, 0], 'std': 0.2, 'n_samples': 2500, 'label': 1}
    # ]

    # Step 1: Generate data
    X, y = generate_custom_data(cluster_params)

    # Scale the data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert data to PyTorch tensors
    inputs = torch.from_numpy(X_scaled).float()
    labels = torch.from_numpy(y).float().unsqueeze(1)

    # Create DataLoader
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Step 2: Determine minimal required architecture
    hidden_size = 3

    # Step 3: Define the neural network architecture
    input_size = 2
    model = MinimalNN(input_size=input_size, hidden_size=hidden_size)

    def initialize_weights(m):
        """
        Initialize weights to ensure diverse decision boundaries covering
        all possible slopes and intercepts.
        """
        if isinstance(m, nn.Linear):
            num_neurons = m.weight.size(0)
            input_dim = m.weight.size(1)

            # Only apply to the first layer with 2D inputs
            if input_dim == 2:
                # w1*x1 + w2*x2 + b = 0

                # Sample theta uniformly from [0, pi)
                theta = torch.rand(num_neurons) * np.pi
                w1 = torch.cos(theta)
                w2 = torch.sin(theta)
                m.weight.data = torch.stack((w1, w2), dim=1)

                m.bias.data = -torch.rand(num_neurons) * w2
            else:
                # For other layers, initialize normally
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                nn.init.constant_(m.bias, 0.0)

    model.apply(initialize_weights)
    visualize_initialized_lines(model)

    # Step 4: Train the neural network
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-4)
    model, losses = train_model(
        model, criterion, optimizer, dataloader, num_epochs=100
    )

    # Step 5: Visualize the result and return metrics
    visualize_results(model, X, y, scaler)

    # Predict on training data
    with torch.no_grad():
        model.eval()
        predictions = (model(inputs) > 0.5).float().squeeze().numpy()

    # Classification metrics
    print("Classification Report:")
    print(classification_report(y, predictions.astype(int)))

    # Step 6: Decision Boundaries
    lines = get_decision_boundary_lines(model, scaler)
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
