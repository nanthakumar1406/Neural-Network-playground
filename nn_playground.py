import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.datasets import make_circles, make_moons, make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Page configuration
st.set_page_config(page_title="Neural Network Playground", layout="wide")
st.title("Neural Network Playground")
st.subheader("Experiment with different neural network parameters")

# Initialize session state for animation control
if 'play' not in st.session_state:
    st.session_state.play = False
if 'frame' not in st.session_state:
    st.session_state.frame = 0
if 'loss_history' not in st.session_state:
    st.session_state.loss_history = []
if 'accuracy_history' not in st.session_state:
    st.session_state.accuracy_history = []

# Animation controls
col1, col2, col3 = st.columns([1, 1, 6])
with col1:
    if st.button("⏮️ Replay"):
        st.session_state.frame = 0
        st.session_state.play = True
with col2:
    if st.button("⏭️ Skip Next" if st.session_state.play else "▶️ Play"):
        st.session_state.play = not st.session_state.play

# Sidebar controls
with st.sidebar:
    st.header("Network Parameters")
    
    # Network architecture
    input_size = st.slider("Input layer size", 2, 10, 2)
    hidden_layers = st.text_input("Hidden layers (comma separated)", "4, 2")
    output_size = st.slider("Output layer size", 1, 3, 1)
    
    # Learning parameters
    learning_rate = st.selectbox("Learning rate", [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0], index=3)
    activation = st.selectbox("Activation", ["tanh", "relu", "logistic", "identity"], index=0)
    regularization = st.selectbox("Regularization", ["None", "L1", "L2"], index=0)
    reg_rate = st.selectbox("Regularization rate", [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3], index=0)
    problem_type = st.selectbox("Problem type", ["Classification", "Regression"], index=0)
    
    # Data parameters
    dataset = st.selectbox(
        "Which dataset do you want to use",
        ["Circle", "Moon", "Gaussian", "Spiral", "XOR"],
        index=0
    )
    noise_level = st.slider("Noise", 0, 100, 0)
    train_ratio = st.slider("Ratio of training to test data", 10, 90, 50)
    batch_size = st.selectbox("Batch size", [1, 5, 10, 20, 50, 100], index=2)
    animation_speed = st.slider("Animation Speed", 1, 10, 3)
    
    # Features
    features = st.multiselect(
        "Which properties do you want to feed in",
        ["X1", "X2", "X1*X2", "X1^2", "X2^2", "sin(X1)", "sin(X2)"],
        default=["X1", "X2"]
    )

# Generate dataset based on selection
def generate_data(dataset_name, noise):
    np.random.seed(42)
    noise = noise / 100.0  # Convert from percentage to decimal
    
    if dataset_name == "Circle":
        X, y = make_circles(n_samples=1000, noise=noise, factor=0.5)
    elif dataset_name == "Moon":
        X, y = make_moons(n_samples=1000, noise=noise)
    elif dataset_name == "Gaussian":
        X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                                  n_informative=2, n_clusters_per_class=1, flip_y=noise)
    elif dataset_name == "XOR":
        X = np.random.randn(1000, 2)
        y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
        X += noise * np.random.randn(1000, 2)
    else:  # Spiral
        n = 1000
        t = np.linspace(0, 4 * np.pi, n // 2)
        X1 = np.zeros((n, 2))
        X1[:n//2, 0] = t * np.cos(t)
        X1[:n//2, 1] = t * np.sin(t)
        X1[n//2:, 0] = t * np.cos(t + np.pi)
        X1[n//2:, 1] = t * np.sin(t + np.pi)
        X = X1 + noise * np.random.randn(n, 2)
        y = np.hstack([np.zeros(n//2), np.ones(n//2)])
    
    return X, y

# Process features
def process_features(X, features):
    X1 = X[:, 0]
    X2 = X[:, 1]
    processed = []
    
    for feat in features:
        if feat == "X1":
            processed.append(X1)
        elif feat == "X2":
            processed.append(X2)
        elif feat == "X1*X2":
            processed.append(X1 * X2)
        elif feat == "X1^2":
            processed.append(X1**2)
        elif feat == "X2^2":
            processed.append(X2**2)
        elif feat == "sin(X1)":
            processed.append(np.sin(X1))
        elif feat == "sin(X2)":
            processed.append(np.sin(X2))
    
    return np.column_stack(processed)

# Function to draw neural network
def draw_neural_network(ax, layer_sizes, frame=0):
    left = 0.1
    right = 0.9
    bottom = 0.1
    top = 0.9
    
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    
    # Nodes
    for i, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for j in range(layer_size):
            circle = Circle((i*h_spacing + left, layer_top - j*v_spacing), 
                          v_spacing/4., 
                          ec='k', 
                          fc='white',
                          zorder=4)
            ax.add_patch(circle)
            
            # Animate activation (growing circle)
            if frame > 0 and i > 0:
                activation_circle = Circle((i*h_spacing + left, layer_top - j*v_spacing), 
                                         v_spacing/4. * min(1, frame/10), 
                                         fc='orange',
                                         alpha=0.6,
                                         zorder=5)
                ax.add_patch(activation_circle)
    
    # Edges
    for i, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for j in range(layer_size_a):
            for k in range(layer_size_b):
                line = plt.Line2D([i*h_spacing + left, (i + 1)*h_spacing + left],
                                 [layer_top_a - j*v_spacing, layer_top_b - k*v_spacing], 
                                 c='k',
                                 alpha=0.5 if frame == 0 else min(1, max(0, (frame - i*3)/10)),
                                 zorder=3)
                ax.add_line(line)

# Function to show activations
def show_activations(model, sample):
    activations = [sample]
    for i in range(len(model.coefs_)):
        activations.append(np.maximum(0, np.dot(activations[-1], model.coefs_[i]) + model.intercepts_[i]))
    
    act_fig, axes = plt.subplots(1, len(activations), figsize=(16, 4))
    for i, (ax, act) in enumerate(zip(axes, activations)):
        ax.imshow(act.reshape(-1, 1), cmap='viridis', aspect='auto')
        ax.set_title(f"Layer {i} Activation")
        ax.axis('off')
    return act_fig

# Main app
def main():
    # Generate and split data
    X, y = generate_data(dataset, noise_level)
    X_processed = process_features(X, features)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, train_size=train_ratio/100, random_state=42
    )
    
    # Parse hidden layers
    try:
        hidden_layer_sizes = [int(x.strip()) for x in hidden_layers.split(",") if x.strip()]
    except:
        st.error("Invalid hidden layers format. Please use comma separated integers (e.g., '4, 2')")
        hidden_layer_sizes = [4, 2]
    
    # Train model
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        learning_rate_init=learning_rate,
        batch_size=batch_size,
        max_iter=1,  # We'll control iterations manually
        random_state=42,
        warm_start=True,
        early_stopping=True,
        validation_fraction=0.2
    )
    
    if regularization == "L1":
        model.alpha = reg_rate
        model.solver = "sgd"
    elif regularization == "L2":
        model.alpha = reg_rate
    else:
        model.alpha = 0.0
    
    # Initialize model
    model.fit(X_train, y_train)
    
    # Update frame if playing
    if st.session_state.play:
        st.session_state.frame += animation_speed
        if st.session_state.frame > 30:  # Reset animation
            st.session_state.frame = 0
            # Perform one training iteration
            model.max_iter += 1
            model.fit(X_train, y_train)
            st.session_state.loss_history.append(model.loss_)
            st.session_state.accuracy_history.append(model.score(X_train, y_train))
    
    # Calculate accuracy
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    # Main visualization
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2])
    
    # Data plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, edgecolors='k', label='Train')
    ax1.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Paired, edgecolors='k', marker='x', label='Test')
    ax1.set_title("Data Distribution")
    ax1.legend()
    
    # Network plot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    full_network = [input_size] + hidden_layer_sizes + [output_size]
    draw_neural_network(ax2, full_network, st.session_state.frame)
    ax2.set_title("Neural Network Architecture")
    
    st.pyplot(fig)
    
    # Decision Boundary Visualization
    if st.session_state.frame > 0:
        fig2, ax3 = plt.subplots(figsize=(8, 6))
        h = 0.02
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        mesh_processed = process_features(mesh_points, features)
        Z = model.predict(mesh_processed)
        Z = Z.reshape(xx.shape)
        ax3.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
        ax3.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
        ax3.set_title(f"Decision Boundary (Iteration: {model.n_iter_})")
        st.pyplot(fig2)
    
    # Display model information
    st.subheader("Model Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Training Accuracy", f"{train_acc:.2%}")
    with col2:
        st.metric("Test Accuracy", f"{test_acc:.2%}")
    with col3:
        st.metric("Hidden Layers", str(hidden_layer_sizes))
    
    st.write(f"**Layers:** Input({input_size}) → " + 
             " → ".join([f"Hidden({size})" for size in hidden_layer_sizes]) + 
             f" → Output({output_size})")
    st.write(f"**Activation:** {activation} | **Learning Rate:** {learning_rate} | "
             f"**Regularization:** {regularization} ({reg_rate}) | "
             f"**Batch Size:** {batch_size}")
    st.write(f"**Iterations:** {model.n_iter_} | **Loss:** {model.loss_:.4f}")
    
    # Training progress
    if len(st.session_state.loss_history) > 1:
        progress_fig, (ax4, ax5) = plt.subplots(1, 2, figsize=(16, 5))
        ax4.plot(st.session_state.loss_history, 'r-')
        ax4.set_title("Training Loss")
        ax5.plot(st.session_state.accuracy_history, 'b-')
        ax5.set_title("Training Accuracy")
        st.pyplot(progress_fig)
    
    # Layer activations
    if st.session_state.frame > 0:
        st.subheader("Layer Activations")
        sample_idx = np.random.randint(0, len(X_train))
        st.pyplot(show_activations(model, X_train[sample_idx:sample_idx+1]))

if __name__ == "__main__":
    main()