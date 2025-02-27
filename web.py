from flask import Flask, render_template, request
import torch
import torch.nn as nn

app = Flask(__name__)

#-----------------------------------------------NN model-----------------------------------------------#
# Define Net-class
class Net(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        self.extract = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ELU(),
            nn.Linear(128, 1024),

            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout1d(p = 0.1),

            nn.Linear(1024, 8192),
            nn.ELU(),
            nn.Linear(8192, 1024),

            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout1d(p = 0.01),

            nn.Linear(1024, 128),
            nn.ELU()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x = self.extract(x)
        x = self.classifier(x)
        return x

#---------------------------------------------Load model---------------------------------------------#
# Load model and set to eval()
model = Net(n_features=8)
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()

#----------------------------------------------Website----------------------------------------------#

# Home page
@app.route("/", methods = ["GET", "POST"])
def home():
    prob = None
    prediction = None
    if request.method == "POST":
        try:
            # Get variables for inputs to predict outcome
            input_data = [
                float(request.form.get("Pregnancies")),
                float(request.form.get("Glucose")),
                float(request.form.get("BloodPressure")),
                float(request.form.get("SkinThickness")),
                float(request.form.get("Insulin")),
                float(request.form.get("BMI")),
                float(request.form.get("DiabetesPedigreeFunction")),
                float(request.form.get("Age"))
            ]

            # Create tensor from variables
            input_tensor = torch.tensor([input_data], dtype=torch.float32)

            # Prediction
            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.sigmoid(output.squeeze(0)).item()
            
            # Convert prediction to binary classification
            prediction = 1 if prob >= 0.5 else 0
        except Exception as e:
            print(f'Error: {e}')
            prediction = None

    # Return prediction
    return render_template("index.html", prediction = prediction, prob = prob)

if __name__ == "__main__":
    app.run(port = 3000, debug = True)