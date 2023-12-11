import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats


def load_face_model():
    # Assuming you have trained a face recognition model using TensorFlow
    # and saved it as 'face_model.h5'
    face_model = tf.keras.models.load_model('face_model.h5')
    return face_model

using SciPy
def load_regression_model(X_train, y_train):
    
    model, _ = stats.linregress(X_train, y_train)
    return model

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x)

def load_neural_network(X_train, y_train):
    input_size = X_train.shape[1]
    model = NeuralNetwork(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1000):
        inputs = torch.tensor(X_train.values, dtype=torch.float32)
        targets = torch.tensor(y_train.values, dtype=torch.float32)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    return model

nsorFlow
def recognize_face(img, face_model):
   
    img_array = cv2.resize(img, (224, 224))
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    
    predictions = face_model.predict(img_array)
    return predictions.flatten()


def calculate_regression_prediction_scipy(model, face_encoding):
    return model.slope * face_encoding + model.intercept


def calculate_regression_prediction_torch(model, face_encoding):
    X_regression_nn = pd.DataFrame([face_encoding])
    return model(torch.tensor(X_regression_nn.values, dtype=torch.float32)).detach().numpy()[0][0]


def mark_attendance(name):
    # Assuming you have a function to mark attendance
    pass

# Function to create GUI for attendance marking
def create_gui(name, window):
    def on_mark_attendance():
     
        mark_attendance(name)
        messagebox.showinfo("Attendance Marked", f"Attendance for {name} marked successfully!")

    mark_button = ttk.Button(window, text="Mark Attendance", command=on_mark_attendance)
    mark_button.pack()

 purposes)
data = pd.read_csv("your_regression_data.csv")
y_regression = data.pop("target_column")
X_regression_train, _, y_regression_train, _ = train_test_split(data, y_regression, test_size=0.2, random_state=42)


face_model = load_face_model()
scipy_model = load_regression_model(X_regression_train, y_regression_train)
torch_model = load_neural_network(X_regression_train, y_regression_train)


cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    # Recognize face using TensorFlow
    face_encoding = recognize_face(img, face_model)

    # Calculate regression prediction using SciPy
    regression_prediction_scipy = calculate_regression_prediction_scipy(scipy_model, face_encoding)

    # Calculate regression prediction using PyTorch
    regression_prediction_torch = calculate_regression_prediction_torch(torch_model, face_encoding)

    
    name = "Dummy Name"  # Replace with your logic to get the actual name
    root = tk.Tk()
    root.title("Attendance Marking")

    ttk.Label(root, text=f"SciPy Regression Prediction: {regression_prediction_scipy}").pack()
    ttk.Label(root, text=f"PyTorch Regression Prediction: {regression_prediction_torch}").pack()

    create_gui(name, root)

    root.mainloop()

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
