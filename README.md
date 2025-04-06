# **Neural Network Playground - Interactive Visualization Tool**

## **🧠 Project Overview**
This interactive Streamlit application lets you experiment with neural networks through visualizations of:
- Network architectures
- Decision boundaries
- Training progress
- Layer activations

## **✨ Key Features**

### **Interactive Controls**
✅ **Network Architecture**:
- Adjust input/hidden/output layer sizes
- Choose activation functions (tanh, relu, etc.)
- Set learning rate and regularization

✅ **Data Customization**:
- Multiple synthetic datasets (Circles, Moons, XOR, etc.)
- Adjustable noise levels
- Feature engineering options

✅ **Visualizations**:
- Animated neural network diagram
- Evolving decision boundary
- Loss/accuracy curves
- Layer activation heatmaps

## **⚙️ Installation & Setup**

1. Install requirements:
```bash
pip install streamlit numpy matplotlib scikit-learn
```

2. Run the app:
```bash
streamlit run neural_playground.py
```

## **🚀 Quick Start Guide**

1. **Select a dataset** from the sidebar (Circles, Moons, etc.)
2. **Adjust network parameters**:
   - Layer sizes (e.g., "4, 2" for hidden layers)
   - Learning rate (start with 0.03)
   - Activation function (tanh works well)
3. **Click "Play"** to start training animation
4. **Explore visualizations**:
   - Network architecture diagram
   - Decision boundary evolution
   - Training metrics

## **📊 Visualization Features**

| Visualization | Description |
|--------------|-------------|
| **Network Diagram** | Interactive animation showing signal flow |
| **Decision Boundary** | How the network separates classes |
| **Training Curves** | Loss and accuracy over time |
| **Layer Activations** | Heatmaps of neuron activations |

## **🧠 Educational Value**
- Perfect for understanding:
  - How neural networks learn
  - Effect of different architectures
  - Impact of hyperparameters
  - Decision boundary formation

## **📜 License**
MIT 
https://neural-network-playground.streamlit.app/

---

### **Pro Tip**
Try the "XOR" dataset with:
- Hidden layers: "4, 2"
- Activation: "tanh"
- Learning rate: 0.1
to see how neural networks solve non-linear problems!
