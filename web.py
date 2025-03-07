from flask import Flask, render_template, request
import torch
import torch.nn as nn
import joblib
import pandas as pd

# Load Scaler
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)

#-----------------------------------------------NN model-----------------------------------------------#
# Define Net-class (neural network)
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

#---------------------------------------Main links for website---------------------------------------#
# Home 
@app.route("/", methods = ["GET", "POST"])
def home():
    return render_template("index.html")

# Projects
@app.route("/projects", methods = ["GET", "POST"])
def projects():
    return render_template("projects.html")

# About
@app.route("/about", methods = ["GET", "POST"])
def about():
    return render_template("about.html")

# Contact
@app.route("/contact", methods = ["GET", "POST"])
def contact():
    return render_template("contact.html")

#----------------------------------------------Blood cell----------------------------------------------#
# Blood cell cancer detection
@app.route("/projects/blood_cell_cancer_detection", methods = ["GET", "POST"])
def blood_cell():
    return render_template("/projects/blood_cell_cancer/blood_cell.html")

#---------------------------------------Brain tumor classification---------------------------------------#
# Brain tumor classification
@app.route("/projects/brain_tumor_detection", methods = ["GET", "POST"])
def brain_tumor():
    return render_template("/projects/brain_tumor/brain_tumor.html")

#-----------------------------------------Diabetes classifier-----------------------------------------#
# Diabetes classifier
@app.route("/projects/diabetes_classifier", methods = ["GET", "POST"])
def diabetes_classifier():
    return render_template("projects/diabetes_classification/diabetes_classifier.html")

# Predict page for diabetes classifier
@app.route("/projects/diabetes_classifier/predict", methods = ["GET", "POST"])
def predict():
    prob = None
    prediction = None
    if request.method == "POST":
        try:
            # Debug 1: Print raw input
            print("Raw input data:")
            print(request.form)

            # Feauture columns
            columns = [
                'Pregnancies', 
                'Glucose', 
                'BloodPressure', 
                'SkinThickness', 
                'Insulin', 
                'BMI', 
                'DiabetesPedigreeFunction', 
                'Age'
            ]

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

            # Input data DataFrame
            input_data_df = pd.DataFrame([input_data], columns = columns)

            # Debug 2: Print input data
            print("Input data as floats:", input_data)

            # Scaled data
            input_scaled = scaler.transform(input_data_df)

            # Debug 3: Print scaled data
            print("Scaled input data:", input_scaled)

            # Create tensor from variables
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

            # Prediction
            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.sigmoid(output.squeeze(0)).item()
            
            # Debug 4: Print model output
            print("Model output (probability):", prob)
            
            # Convert prediction to binary classification
            prediction = 1 if prob >= 0.5 else 0
        except Exception as e:
            print(f'Error: {e}')
            prediction = None

    # Return prediction
    return render_template("projects/diabetes_classification/predict.html", prediction = prediction, prob = prob)

#-----------------------------------------PCA breast cancer-----------------------------------------#
# PCA breast cancer
@app.route("/projects/pca_gene_expression", methods = ["GET", "POST"])
def pca_gene():
    return render_template("/projects/pca_gene/pca_gene.html")

#-----------------------------------------Penguin clustering-----------------------------------------#
# Penguin clustering
@app.route("/projects/clustering_of_penguins", methods = ["GET", "POST"])
def penguin_clustering():
    return render_template("/projects/penguin_cluster/penguin.html")

#----------------------------------------------Utility----------------------------------------------#
if __name__ == "__main__":
    app.run(port = 3000, debug = True)