import streamlit as st
import os
from PIL import Image
import io
import numpy as np
import textwrap
from langchain_community.llms import Ollama
import google.generativeai as genai
from IPython.display import Markdown
import ollama
from concurrent.futures import ThreadPoolExecutor

# Set Google API Key for generative AI
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
genai.configure(api_key=GOOGLE_API_KEY)

# Set up the language models
text_model = Ollama(model="llama3.2")
visual_model = genai.GenerativeModel('gemini-1.5-flash-latest')

# Utility function for markdown formatting
def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# Define the multimodal radiology pipeline
class MultimodalRadiologyPipeline:
    def __init__(self, visual_model, text_model):
        self.visual_model = visual_model
        self.text_model = text_model

    def generate_txt_from_image(self, image):
        # Get description or features from the visual model
        image_description = self.visual_model.generate_content([image, "describe"])
        return image_description.text
    
    def generate_report(self,text):
        output = ollama.generate(
                        model='mistral:latest', 
                        prompt=text)
        return output['response']

    def generate_text(self, text):
        return self.text_model(text)
    
    def generate_image_from_text(self, prompt):
        try:
            image_bytes = self.visual_model.generate(prompt=prompt)
            image = Image.open(io.BytesIO(image_bytes))
            return image
        except Exception as e:
            print(f"Error generating image: {e}")
            return Image.new("RGB", (512, 512), color="white")

    def generate_findings(self, user_data, imaging_Modality_Data):
        prompt = f"""
        The following patient has undergone an imaging study with the provided modalities. The patient's details and history are as follows:
    
        Patient ID: {user_data['patient_id']}
        Name: {user_data['patient_name']}
        Date of Birth: {user_data['dob']}
        Gender: {user_data['gender']}
        Exam Date: {user_data['exam_date']}
        Chief Complaint: {user_data['chief_complaint']}
        Clinical History: {user_data['clinical_history']}
        Imaging Modality Data: {imaging_Modality_Data}
    
        As a radiologist, please provide :
        Findings: Describe the observed imaging findings based on the study.
        """
        # Use the text model to generate the output
        response = self.generate_text(prompt)
        return response

    def generate_impressions(self, user_data, imaging_Modality_Data):
        prompt = f"""
        The following patient has undergone an imaging study with the provided modalities. The patient's details and history are as follows:
    
        Patient ID: {user_data['patient_id']}
        Name: {user_data['patient_name']}
        Date of Birth: {user_data['dob']}
        Gender: {user_data['gender']}
        Exam Date: {user_data['exam_date']}
        Chief Complaint: {user_data['chief_complaint']}
        Clinical History: {user_data['clinical_history']}
        Imaging Modality Data: {imaging_Modality_Data}
    
        As a radiologist, please provide :
        Impression: Provide a brief diagnosis or interpretation of the findings.
        """
        # Use the text model to generate the output
        response = self.generate_text(prompt)
        return response

    def generate_recommendations(self, user_data, imaging_Modality_Data):
        prompt = f"""
        The following patient has undergone an imaging study with the provided modalities. The patient's details and history are as follows:
    
        Patient ID: {user_data['patient_id']}
        Name: {user_data['patient_name']}
        Date of Birth: {user_data['dob']}
        Gender: {user_data['gender']}
        Exam Date: {user_data['exam_date']}
        Chief Complaint: {user_data['chief_complaint']}
        Clinical History: {user_data['clinical_history']}
        Imaging Modality Data: {imaging_Modality_Data}
    
        As a radiologist, please provide :
        Recommendations: Provide recommendations for the next steps (e.g., additional imaging, follow-up).
        """
        # Use the text model to generate the output
        response = self.generate_text(prompt)
        return response
    
    def generate_radiology_report(self, user_data, knowledge_base):
        print("*******************************")
        imaging_modality_image = Image.open(user_data['imaging_modality'])
        imaging_Modality_Data = self.generate_txt_from_image(imaging_modality_image)
        
        # findings = self.generate_findings(user_data, imaging_Modality_Data)
        # impression = self.generate_impressions(user_data, imaging_Modality_Data)
        # recommendations = self.generate_recommendations(user_data, imaging_Modality_Data)

        with ThreadPoolExecutor() as executor:
        # Submit the tasks for findings, impressions, and recommendations to be executed in parallel
            findings_future = executor.submit(self.generate_findings, user_data, imaging_Modality_Data)
            impression_future = executor.submit(self.generate_impressions, user_data, imaging_Modality_Data)
            recommendations_future = executor.submit(self.generate_recommendations, user_data, imaging_Modality_Data)

            # Wait for all tasks to complete and get the results
            findings = findings_future.result()
            impression = impression_future.result()
            recommendations = recommendations_future.result()
    
        # Format the input data into the report template
        prompt = f"""
        You are a Radiologist writing a report based on an imaging study. Please follow this standardized RSNA Radiology Report template 
        and fill in the relevant information based on the following patient details and imaging study findings. Use {knowledge_base}.
    
        ### RSNA Radiology Report Template:
        1. **Patient Information**: 
           - Patient ID: {user_data['patient_id']}
           - Name: {user_data['patient_name']}
           - Date of Birth: {user_data['dob']}
           - Gender: {user_data['gender']}
           - Exam Date: {user_data['exam_date']}
    
        2. **Clinical History**:
           - Chief Complaint: {user_data['chief_complaint']}
           - Relevant Clinical History: {user_data['clinical_history']}
           
        3. **Imaging Modality**: {user_data['imaging_modality']}
    
        4. **Findings**:
             - {findings}
    
        5. **Impression**:
             - {impression}
    
        6. **Recommendations**:
             - {recommendations}
        """
    
        # Call the text model to generate the final radiology report
        #response = self.generate_text(prompt)
        response = self.generate_report(prompt)
        return response

# Streamlit app
def main():
    st.title("Radiology Report Generation")
    
    # Create the pipeline object
    pipeline = MultimodalRadiologyPipeline(visual_model=visual_model, text_model=text_model)
    
    # Input form for patient details
    st.header("Enter Patient Details")
    with st.form(key="patient_form"):
        patient_id = st.text_input("Patient ID")
        patient_name = st.text_input("Patient Name")
        dob = st.date_input("Date of Birth")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        exam_date = st.date_input("Exam Date")
        
        # Clinical History
        chief_complaint = st.text_area("Chief Complaint")
        clinical_history = st.text_area("Relevant Clinical History")
        
        # Imaging Modality upload
        imaging_modality = st.file_uploader("Upload Imaging Modality Image", type=["jpg", "jpeg", "png", "dicom"])
        
        # Submit button
        submit_button = st.form_submit_button("Generate Report")
        
        if submit_button:
            if imaging_modality is None:
                st.error("Please upload an imaging modality image.")
                return
            
            # Check if the directory exists, and create it if not
            upload_dir = "C:/temp"
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)
            
            # Save the uploaded file to the specified directory
            file_path = os.path.join(upload_dir, imaging_modality.name)
            with open(file_path, "wb") as f:
                f.write(imaging_modality.getbuffer())
            
            # Pass the new file path to user_data
            user_data = {
                "patient_id": patient_id,
                "patient_name": patient_name,
                "dob": str(dob),
                "gender": gender,
                "exam_date": str(exam_date),
                "chief_complaint": chief_complaint,
                "clinical_history": clinical_history,
                "imaging_modality": file_path  # Use the new path
            }
            
            # Generate the report
            knowledge_base = "I am an expert radiologist."
            report = pipeline.generate_radiology_report(user_data, knowledge_base)
            
            # Display the generated report
            st.subheader("Generated Radiology Report")
            st.write(report)
            
            # Display the uploaded image
            st.image(file_path, caption="Uploaded Imaging Modality", use_container_width=True)

if __name__ == "__main__":
    main()
