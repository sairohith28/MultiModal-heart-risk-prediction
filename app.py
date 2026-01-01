import os
import numpy as np
import pandas as pd
import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import cv2
from PIL import Image

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the trained model
print("Loading model...")
model = load_model('models/multimodal_health_risk_model.h5')
print("Model loaded successfully!")

# Initialize tokenizer
max_sequence_length = 100
max_num_words = 1000
tokenizer = Tokenizer(num_words=max_num_words)

# Load sample data to fit tokenizer
print("Initializing tokenizer...")
master_df = pd.read_csv('data/master_dataset.csv')
all_notes = []
for patient_id in master_df['patient_id'][:100]:  # Use first 100 for fitting
    note_path = f'data/text/{patient_id}_note.txt'
    if os.path.exists(note_path):
        with open(note_path, 'r') as f:
            all_notes.append(f.read())
tokenizer.fit_on_texts(all_notes)

def preprocess_image(image):
    """Preprocess the uploaded X-ray image"""
    if image is None:
        return np.zeros((112, 112, 3))
    
    # Convert PIL Image to numpy array
    img = np.array(image)
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Resize to model input size
    img = cv2.resize(img, (112, 112))
    
    # Convert to 3 channels
    img = np.stack([img, img, img], axis=-1)
    
    # Normalize
    img = img / 255.0
    
    return img.astype(np.float32)

def preprocess_text(clinical_note):
    """Preprocess the clinical note text"""
    if not clinical_note or clinical_note.strip() == "":
        clinical_note = "No clinical notes available."
    
    # Tokenize and pad
    text_seq = tokenizer.texts_to_sequences([clinical_note])[0]
    text_seq = pad_sequences([text_seq], maxlen=max_sequence_length)[0]
    
    return text_seq.astype(np.int32)

def predict_mortality_risk(age, gender, weight, height, glucose, hemoglobin, 
                          white_blood_cells, platelets, creatinine,
                          heart_rate, systolic_bp, diastolic_bp, 
                          temperature, respiratory_rate, oxygen_saturation,
                          xray_image, clinical_note):
    """
    Make a prediction using the multimodal model
    """
    try:
        # Calculate BMI
        bmi = weight / ((height/100)**2)
        
        # Prepare tabular features
        gender_numeric = 1 if gender == "Male" else 0
        tabular_features = np.array([
            age, gender_numeric, bmi, glucose, hemoglobin, 
            white_blood_cells, platelets, creatinine,
            heart_rate, systolic_bp, diastolic_bp, 
            temperature, respiratory_rate, oxygen_saturation
        ], dtype=np.float32).reshape(1, -1)
        
        # Preprocess image
        image_features = preprocess_image(xray_image).reshape(1, 112, 112, 3)
        
        # Preprocess text
        text_features = preprocess_text(clinical_note).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict([tabular_features, image_features, text_features], verbose=0)[0][0]
        
        # Create result message
        risk_percentage = prediction * 100
        
        if risk_percentage < 30:
            risk_level = "LOW"
            color = "🟢"
            recommendation = "Patient appears to be at low risk. Continue standard monitoring."
        elif risk_percentage < 70:
            risk_level = "MODERATE"
            color = "🟡"
            recommendation = "Patient shows moderate risk factors. Consider enhanced monitoring and preventive measures."
        else:
            risk_level = "HIGH"
            color = "🔴"
            recommendation = "Patient is at high risk. Immediate medical attention and intervention may be required."
        
        result = f"""
## {color} Mortality Risk Assessment
        
**Risk Level:** {risk_level}  
**Risk Score:** {risk_percentage:.1f}%

### Clinical Interpretation:
{recommendation}

### Model Confidence:
- Prediction: {prediction:.4f}
- This prediction is based on multimodal analysis of:
  - Patient demographics and vital signs
  - Laboratory test results
  - Chest X-ray imaging
  - Clinical notes

### Next Steps:
1. Review all clinical findings with the patient
2. Discuss risk factors and potential interventions
3. Schedule appropriate follow-up appointments
4. Consider additional diagnostic tests if indicated

⚠️ **Note:** This is an AI-assisted prediction tool. All decisions should be made by qualified healthcare professionals with consideration of the complete clinical picture.
        """
        
        return result
        
    except Exception as e:
        return f"Error making prediction: {str(e)}"

def load_sample_patient(patient_num):
    """Load data for a sample patient"""
    try:
        patient_num = int(patient_num)
        master_df = pd.read_csv('data/master_dataset.csv')
        demographics_df = pd.read_csv('data/tabular/demographics.csv')
        labs_df = pd.read_csv('data/tabular/labs.csv')
        vitals_df = pd.read_csv('data/tabular/vitals.csv')
        
        if patient_num < 0 or patient_num >= len(master_df):
            return "Invalid patient number"
        
        patient_id = master_df.iloc[patient_num]['patient_id']
        demo = demographics_df[demographics_df['patient_id'] == patient_id].iloc[0]
        labs = labs_df[labs_df['patient_id'] == patient_id].iloc[0]
        vitals_avg = vitals_df[vitals_df['patient_id'] == patient_id].drop(['patient_id', 'time_point'], axis=1).mean()
        
        # Load X-ray image
        img_path = f'data/images/{patient_id}_xray.png'
        img = Image.open(img_path) if os.path.exists(img_path) else None
        
        # Load clinical note
        note_path = f'data/text/{patient_id}_note.txt'
        note = ""
        if os.path.exists(note_path):
            with open(note_path, 'r') as f:
                note = f.read()
        
        return (
            int(demo['age']),
            "Male" if demo['gender'] == 'M' else "Female",
            float(demo['weight']),
            float(demo['height']),
            float(labs['glucose']),
            float(labs['hemoglobin']),
            float(labs['white_blood_cells']),
            float(labs['platelets']),
            float(labs['creatinine']),
            float(vitals_avg['heart_rate']),
            float(vitals_avg['systolic_bp']),
            float(vitals_avg['diastolic_bp']),
            float(vitals_avg['temperature']),
            float(vitals_avg['respiratory_rate']),
            float(vitals_avg['oxygen_saturation']),
            img,
            note
        )
    except Exception as e:
        print(f"Error loading patient: {e}")
        return None

# Create Gradio interface
with gr.Blocks(title="Multimodal Health Risk Assessment", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏥 Multimodal Health Risk Assessment System
    
    This AI-powered system analyzes patient data from multiple sources to assess mortality risk:
    - 📊 **Tabular Data**: Demographics, vital signs, and lab results
    - 🔬 **Medical Imaging**: Chest X-ray analysis
    - 📝 **Clinical Notes**: Natural language processing of medical records
    
    Enter patient information below or load a sample patient to get started.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 👤 Patient Demographics & Vitals")
            
            with gr.Group():
                age = gr.Slider(18, 100, value=65, label="Age (years)", step=1)
                gender = gr.Radio(["Male", "Female"], label="Gender", value="Male")
                weight = gr.Slider(40, 150, value=70, label="Weight (kg)", step=0.1)
                height = gr.Slider(140, 200, value=170, label="Height (cm)", step=0.1)
            
            gr.Markdown("### 🔬 Laboratory Results")
            with gr.Group():
                glucose = gr.Slider(50, 200, value=90, label="Glucose (mg/dL)", step=1)
                hemoglobin = gr.Slider(8, 20, value=14, label="Hemoglobin (g/dL)", step=0.1)
                white_blood_cells = gr.Slider(2, 20, value=7, label="White Blood Cells (×10⁹/L)", step=0.1)
                platelets = gr.Slider(50, 600, value=250, label="Platelets (×10⁹/L)", step=1)
                creatinine = gr.Slider(0.3, 3, value=1, label="Creatinine (mg/dL)", step=0.1)
            
            gr.Markdown("### 💓 Vital Signs")
            with gr.Group():
                heart_rate = gr.Slider(40, 150, value=75, label="Heart Rate (bpm)", step=1)
                systolic_bp = gr.Slider(80, 200, value=120, label="Systolic BP (mmHg)", step=1)
                diastolic_bp = gr.Slider(40, 120, value=80, label="Diastolic BP (mmHg)", step=1)
                temperature = gr.Slider(35, 40, value=36.8, label="Temperature (°C)", step=0.1)
                respiratory_rate = gr.Slider(8, 30, value=16, label="Respiratory Rate (breaths/min)", step=1)
                oxygen_saturation = gr.Slider(80, 100, value=98, label="Oxygen Saturation (%)", step=1)
        
        with gr.Column(scale=1):
            gr.Markdown("### 📸 Chest X-ray Image")
            xray_image = gr.Image(type="pil", label="Upload Chest X-ray", height=300)
            
            gr.Markdown("### 📝 Clinical Notes")
            clinical_note = gr.Textbox(
                label="Clinical Notes",
                placeholder="Enter clinical notes here...",
                lines=10
            )
            
            with gr.Row():
                sample_patient_num = gr.Number(label="Load Sample Patient (0-999)", value=0, precision=0)
                load_sample_btn = gr.Button("Load Sample Patient", variant="secondary")
            
            predict_btn = gr.Button("🔍 Assess Mortality Risk", variant="primary", size="lg")
    
    gr.Markdown("### 📊 Risk Assessment Results")
    output = gr.Markdown()
    
    # Set up event handlers
    predict_btn.click(
        fn=predict_mortality_risk,
        inputs=[age, gender, weight, height, glucose, hemoglobin, 
                white_blood_cells, platelets, creatinine,
                heart_rate, systolic_bp, diastolic_bp, 
                temperature, respiratory_rate, oxygen_saturation,
                xray_image, clinical_note],
        outputs=output
    )
    
    load_sample_btn.click(
        fn=load_sample_patient,
        inputs=[sample_patient_num],
        outputs=[age, gender, weight, height, glucose, hemoglobin, 
                white_blood_cells, platelets, creatinine,
                heart_rate, systolic_bp, diastolic_bp, 
                temperature, respiratory_rate, oxygen_saturation,
                xray_image, clinical_note]
    )
    
    gr.Markdown("""
    ---
    ⚠️ **Disclaimer**: This tool is for demonstration purposes only. It should not be used for actual medical diagnosis or treatment decisions. 
    Always consult qualified healthcare professionals for medical advice.
    """)

# Launch the app
if __name__ == "__main__":
    print("Starting Gradio interface...")
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
