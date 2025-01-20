import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import time

np.random.seed(int(time.time()))

# Step 1: Generate a dataset
def generate_data(n=100, d=2):
    #x = np.random.uniform(-1,1,(n,d))  # Random points in R^d generated with uniform distribution
    #W_bar = np.random.uniform(-1,1,d) # True weights generated with uniform distribution
    x = np.random.randn(n, d)  # Random points in R^d generated with std normal distribution
    W_bar = np.random.randn(d)  # True weights generated with std normal distribution
    y = np.sign(x.dot(W_bar))  # Generate labels/classes based on the sign of w^Tx
    return x, y, W_bar

# Step 2: Implement the SGD
def stochastic_gradient_descent(x, y, max_epochs=100, tol=1e-3):
    n, d = x.shape
    w = np.zeros(d)  # W^0
    #eps=np.array([1/(i+1) for i in range(max_epochs)]) # decreasing seq to 0
    eps=0.001
    for epoch in range(max_epochs):
        prev_risk = np.inf
        for i in range(n):
            xi, yi = x[i], y[i]
            gradient_w = -2 * xi * (yi - w.dot(xi))
            w -= eps * gradient_w
        # Calculate empirical risk to check for convergence
        risk = np.mean((y - x.dot(w)) ** 2)
        if abs(prev_risk - risk) < tol:
            break
        prev_risk = risk
    print("\n the risk= ", risk)
    return w

# Step 3: Plot decision boundaries for comparison in subplots
def plot_decision_boundaries(x, y, W_bar, w_estimated, x_noisy, y_noisy, w_estimated_noisy):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Define function to plot points and boundaries
    def plot_boundary(ax, x, y, W_bar, w_estimated, title):
        # Separate points by class for clearer plotting
        x_pos = x[y == 1]
        x_neg = x[y == -1]
        
        # Scatter plot with different colors for classes
        ax.scatter(x_pos[:, 0], x_pos[:, 1], color='blue', label='$y = 1$', alpha=0.7)
        ax.scatter(x_neg[:, 0], x_neg[:, 1], color='red', label='$y = -1$', alpha=0.7)
        
        # Define limits for plot
        x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
        xx = np.linspace(x_min, x_max, 100)
        
        # True hyperplane
        yy_true = (-W_bar[0] / W_bar[1]) * xx
        ax.plot(xx, yy_true, 'g--', label='True hyperplane')
        
        # Estimated hyperplane from SGD
        yy_estimated = (-w_estimated[0] / w_estimated[1]) * xx
        ax.plot(xx, yy_estimated, 'b-', label='SGD estimated hyperplane')
        
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend()
        ax.set_title(title)
    
    # Plot without noise
    plot_boundary(axes[0], x, y, W_bar, w_estimated, "SGD without noise")
    
    # Plot with noise
    plot_boundary(axes[1], x_noisy, y_noisy, W_bar, w_estimated_noisy, "SGD with added Gaussian noise")
    
    plt.tight_layout()
    plt.show()

# Generate data and compute SGD without noise
x, y, W_bar = generate_data()
w_estimated = stochastic_gradient_descent(x, y)

# Add noise to the generated data and compute SGD
x_noisy = x + 0.5*np.random.randn(x.shape[0],x.shape[1])
w_estimated_noisy = stochastic_gradient_descent(x_noisy,y)

print("w_bar = ",W_bar,"  w_estimated= ",w_estimated," and  w_estimated_noisy= ",w_estimated_noisy )

# Plot both results in subplots
plot_decision_boundaries(x, y, W_bar, w_estimated, x_noisy, y, w_estimated_noisy)


#################
#Last question
#################


 

# Load the Breast Cancer dataset
data = fetch_ucirepo(id=17)
X = data.data.features
y = data.data.targets


# Map labels from {'M', 'B'} to {-1, 1} ('M' for malignant = 1, 'B' for benign = -1)
y = np.where(y == 'M', 1, -1)
y=y.ravel()


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Apply SGD on training data
w_estimated = stochastic_gradient_descent(X_train, y_train)


# Predict on training and testing sets
y_pred_train = np.sign(X_train.dot(w_estimated))
y_pred_test = np.sign(X_test.dot(w_estimated))

# Convert predictions back to {0, 1} format for comparison
y_pred_train = np.where(y_pred_train == -1, 0, 1)
y_pred_test = np.where(y_pred_test == -1, 0, 1)
y_train_bin = np.where(y_train == -1, 0, 1)
y_test_bin = np.where(y_test == -1, 0, 1)

# Calculate metrics
train_accuracy = accuracy_score(y_train_bin, y_pred_train)
test_accuracy = accuracy_score(y_test_bin, y_pred_test)
precision = precision_score(y_test_bin, y_pred_test)
recall = recall_score(y_test_bin, y_pred_test)

print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Testing Accuracy: {test_accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")


