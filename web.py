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
@app.route("/", methods = ["GET"])
def home():
    return render_template("index.html")

# User input
@app.route("/", methods = ['POST'])
def predicter():
    pregnancies = request.files["Pregnancies"]

if __name__ == "__main__":
    app.run(port = 3000, debug = True)