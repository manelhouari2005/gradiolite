import os
import joblib
import numpy as np
from keras.datasets import mnist
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def train_and_save():
    print("Loading local cached MNIST data...")
    (X_train_full, y_train_full), (X_test_full, y_test_full) = mnist.load_data()
    
    # Flatten 28x28 images to 784
    X_train_flat = X_train_full.reshape(-1, 784) / 255.0
    X_test_flat = X_test_full.reshape(-1, 784) / 255.0
    
    # Use 20,000 training samples and 3,000 test samples: trains in ~10 seconds with >96.5% accuracy!
    np.random.seed(42)
    train_indices = np.random.choice(len(X_train_flat), size=20000, replace=False)
    test_indices = np.random.choice(len(X_test_flat), size=3000, replace=False)
    
    X_train = X_train_flat[train_indices]
    y_train = y_train_full[train_indices]
    
    X_test = X_test_flat[test_indices]
    y_test = y_test_full[test_indices]
    
    print(f"Training MLP Classifier on {len(X_train)} samples...")
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        max_iter=30,
        random_state=42,
        early_stopping=True,
        verbose=False
    )
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc * 100:.2f}%")
    
    # Gather test samples for UI demo
    samples = {}
    for digit in range(10):
        digit_idx = np.where(y_test == digit)[0]
        samples[digit] = X_test[digit_idx[:4]].tolist()
        
    os.makedirs('model', exist_ok=True)
    model_path = os.path.join('model', 'mnist_model.joblib')
    joblib.dump({
        'model': clf,
        'accuracy': float(acc),
        'samples': samples
    }, model_path, compress=3)
    
    print(f"Model saved to {model_path} ({os.path.getsize(model_path) / 1024:.1f} KB)")

if __name__ == '__main__':
    train_and_save()
