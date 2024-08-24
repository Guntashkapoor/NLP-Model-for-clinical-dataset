import streamlit as st
import os
from fpdf import FPDF
import base64
import spacy
import en_core_med7_lg
from spacy import displacy
from PIL import Image 
from pytesseract import pytesseract 
from PyPDF2 import PdfReader
import numpy as np
import fitz
import time
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import nltk
from nltk.tag.stanford import StanfordNERTagger
import re
import phonenumbers

nltk.download()

os.environ['JAVA_HOME'] = r'C:\Program Files\Java\jdk-22'  
STANFORD_MODELS=r'C:\Users\HP\Documents\Guntash\RPL FLASH\NLP Model\stanford-ner-2013-11-12\stanford-ner-2013-11-12\classifiers\english.all.3class.distsim.crf.ser.gz'
CLASSPATH=r'C:\Users\HP\Documents\Guntash\RPL FLASH\NLP Model\stanford-ner-2013-11-12\stanford-ner-2013-11-12\stanford-ner.jar'

stanf=StanfordNERTagger(STANFORD_MODELS,CLASSPATH)

#----------------------------------------- FRONTEND UI DESIGN ------------------------------------------------------------

# Set page configuration
st.set_page_config(page_title="Clinical Information Extraction", page_icon="💬", layout="centered")

# Function to encode the image as a base64 string
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Path to your local image
image_path = r"static\images\Data-extraction-rafiki.png"  # Update this to the relative path of your image
#image_path=r"static\images\chatbots-in-healthcare-its-benefits-use-cases-and-challenges-768x389.jpg"

# Encode the image
bg_image = get_base64_image(image_path)

# Custom CSS for background image and chatbot UI
st.markdown(
    f"""
    <style>    
    .stApp {{
        /*background-image: url("data:image/jpg;base64,{bg_image}");*/
        background: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), url("http://www.dataentryhelp.com/assets/img/dataentryservices/data-extraction.png");,
        /*background-image: url("https://tse2.mm.bing.net/th?id=OIP.rgKTqZkbC72ABLIIPK3e5wHaF2&pid=Api&P=0&h=220");*/
        /*background-image: url("http://www.dataentryhelp.com/assets/img/dataentryservices/data-extraction.png"); */               
        background-size: cover;
        background-position: center;
        color: white;
        
    }}
    .chat-bubble {{
        background-color: rgba(0, 123, 255, 0.8);
        padding: 10px;
        border-radius: 10px;
        max-width: 60%;
        margin-bottom: 10px;
    }}
    .user-bubble {{
        background-color: rgba(255, 255, 255, 0.8);
        padding: 10px;
        border-radius: 10px;
        max-width: 60%;
        margin-bottom: 10px;
        margin-left: auto;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Clinical Information Extraction Chatbot 💬")

# Display user input
#if user_input:
 #   st.markdown(f'<div class="user-bubble">{user_input}</div>', unsafe_allow_html=True)

    # Extract information
  #  extracted_info = handle_file_upload(user_input)

    # Display extracted information
   # st.markdown('<div class="chat-bubble">Here is the extracted information:</div>', unsafe_allow_html=True)
    
   # for key, value in extracted_info.items():
    #    if value:
     #       st.markdown(f'<div class="chat-bubble">{key}: {", ".join(value)}</div>', unsafe_allow_html=True)
      #  else:
       #     st.markdown(f'<div class="chat-bubble">{key}: None found</div>', unsafe_allow_html=True)




#--------------------------------------------------- EXTRACTION LOGIC----------------------------------------


# Initialize session state
if 'file_uploaded' not in st.session_state:
    st.session_state['file_uploaded'] = False
if 'folder_path' not in st.session_state:
    st.session_state['folder_path'] = ""
if 'file_name' not in st.session_state:
    st.session_state['file_name'] = ""
if 'text_area_visible' not in st.session_state:
    st.session_state['text_area_visible'] = False
if 'submit_disabled' not in st.session_state:
    st.session_state['submit_disabled'] = False


def ExtractTextFromImage(path=None):
    model="en_core_med7_lg"
    nlp = spacy.load(model)
    img=Image.open(path)    
    img1 = np.array(img)
    text=pytesseract.image_to_string(img1)
    PiiData=extract_personal_demographies(text) #extracting personal data    
    doc = nlp(text)
    entities = []
    annotations=[]
    for ent in doc.ents:
        entities.append((ent.start_char, ent.end_char, ent.label_))
    annotations.append((text, {'entities': entities}))    
    return annotations

def ExtractTextFromPdf(path=None):
    doc = fitz.open(path)
    pdf_text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        pdf_text += page.get_text()
        
    return pdf_text

#extract Pii from the text
def extract_personal_demographies(text):
    # Initialize a dictionary to store PII
    pii_data = {
        "FullNames": [],
        "Names": [],
        "Emails": [],
        "Phone Numbers": [],
        "Credit Card Numbers": [],
        "Genders": [],
        "Ages": [],
        "Addresses": [],
        "Heights": [],
        "Weights": []
    }
    
    personal_nlp = spacy.load("en_core_web_sm")
    # Process the text using spaCy
    doc = personal_nlp(text)
    
    # Extract Names (PERSON entities)
    for sent in nltk.sent_tokenize(text):
        tokens = nltk.tokenize.word_tokenize(sent)
        tags = stanf.tag(tokens)
        for tag in tags:
            if tag[1]=='PERSON': pii_data["Names"].append(tag)
            
    # Extract FullNames (PERSON entities)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            pii_data["FullNames"].append(ent.text)
            
    # Extract Email addresses using regex
    email_regex = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_regex, text)
    pii_data["Emails"].extend(emails)
    
    # Extract Phone Numbers using the phonenumbers library
    for match in phonenumbers.PhoneNumberMatcher(text, "US"):
        pii_data["Phone Numbers"].append(phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.INTERNATIONAL))
    
    # Extract Credit Card numbers using regex
    credit_card_regex = r'\b(?:\d[ -]*?){13,16}\b'
    credit_cards = re.findall(credit_card_regex, text)
    pii_data["Credit Card Numbers"].extend(credit_cards)
    
    # Extract Gender based on keywords
    gender_keywords = {"male", "female", "non-binary", "man", "woman", "transgender", "cisgender"}
    words = text.lower().split()
    genders = set(gender_keywords).intersection(words)
    pii_data["Genders"].extend(genders)
    
    # Extract Age using regex (e.g., "25 years old" or "age 25" or"AGE:45")
    age_regex = r'\b(?:\d{1,3})\s*(?:years old|year old|yo|age|AGE:\d{1,3}|Age: \d{1,3})\b|\bAGE:\d{1,3}\b|\bAge: \d{1,3}\b'
    ages = re.findall(age_regex, text)
    pii_data["Ages"].extend(ages)
    
    # Extract Addresses (IDENTIFIED AS "GPE" entities)
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC", "FAC", "ORG"]:
            pii_data["Addresses"].append(ent.text)
    
    # Extract Heights using regex (e.g., "6 feet", "170 cm")
    height_regex = r'\b\d{1,3}\s*(?:cm|inches|inch|ft|feet|foot)\b'
    heights = re.findall(height_regex, text)
    pii_data["Heights"].extend(heights)
    
    # Extract Weights using regex (e.g., "70 kg", "150 lbs")
    weight_regex = r'\b\d{2,3}\s*(?:kg|kilograms|lbs|pounds)\b'
    weights = re.findall(weight_regex, text)
    pii_data["Weights"].extend(weights)
    
    return pii_data


def generate_medical_annotation(model,document):
    nlp = spacy.load(model)        
    annotations = []
    for text in document:
        doc = nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append((ent.start_char, ent.end_char, ent.label_))
        annotations.append((text, {'entities': entities}))    
   
    return annotations

def visualize_annotations(transcription):
        nlp = spacy.load("en_core_med7_lg")
        # Create distict colours for labels
        col_dict = {}
        s_colours = ['#e6194B', '#3cb44b', '#ffe119', '#ffd8b1', '#f58231', '#f032e6', '#42d4f4']
        for label, colour in zip(nlp.pipe_labels['ner'], s_colours):
            col_dict[label] = colour
        options = {'ents': nlp.pipe_labels['ner'], 'colors':col_dict}
        text = """
                Patient Discharge Summary:

                Patient Name: John Doe
                Diagnosis: Hypertension, Diabetes
                Medications:
                - Lisinopril 10 mg daily
                - Metformin 500 mg twice daily

                Instructions:
                - Continue medications as prescribed.
                - Follow up with cardiology in 3 months.
                - Monitor blood pressure regularly.
                """
        # doc = nlp(text)
        doc=nlp(transcription)
        print(transcription)
        # spacy.displacy.render(doc, style = 'ent', jupyter = True, options = options)
        st.write()
        prescriptions = []
        print('doc ets',doc.ents)
        annos=[]
        # for i in prescriptions:
        #     annos.append(':'.join(i))
        annots=[(ent.text, ent.label_) for ent in doc.ents]
        for i in annots:
            annos.append(':'.join(i))
        print("annots",annots)
        print('annos','\n'.join(annos))
        return '\n'.join(annos)
    

# Function to handle file upload
def handle_file_upload():
    uploaded_file = st.session_state['uploaded_file']
    allowed_types = ["pdf", "txt", "png", "jpeg", "jpg"]
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension in allowed_types:
        uploads_dir = "uploads"
    
        # Check if the uploads directory exists; if not, create it
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
        
        # Save the uploaded file to the uploads directory
        file_path = os.path.join(uploads_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    if file_extension=='pdf':
        output1=ExtractTextFromPdf(file_path)
        PiiData=extract_personal_demographies(output1)
        print("Personal demographies", PiiData)
        annotations1=visualize_annotations(output1)
        print('number of lines',annotations1.count('\n')+1)
        # annotations1='\n'.join(annotations)
        output_file=file_path+'_Solution.txt'
        with open(output_file,'w')as f:
            f.write(annotations1)
        with st.spinner('Processing...'):
            time.sleep(20)
        st.text_area("Prescription-1",value=annotations1)
        
        output_file1=file_path+'_Solution.pdf'
        c = canvas.Canvas(output_file1, pagesize=letter)
        width, height = letter  # Get the dimensions of the page
        data=open(output_file,'r')
        data1=data.readlines()
        # Add UTF-8 content to the PDF
        text_object = c.beginText(40, height - 40)  # Starting position of text
        text_object.setFont("Helvetica", 12)
        # Add the text content, handling line breaks
        for line in data1:
            lines=line.replace('\n','')
            text_object.textLine(lines)
        c.drawText(text_object)
        c.showPage()
        c.save()
        pdf_output_path=file_path+'_Solution.pdf'
        with open(pdf_output_path,'rb') as filed:
            st.download_button(
                        label="Download as PDF",
                        data=filed,
                        file_name='Prescription.pdf',
                        mime="application/pdf"
                    )
        st.markdown("\n\n\n")

# Function to create a PDF and return its binary content
def create_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Hello, this is a test PDF file.", ln=True, align='C')
    pdf_output_path = os.path.join(st.session_state['folder_path'], 'prescription.pdf')
    pdf.output(pdf_output_path)
    
    # Read the PDF file as binary
    with open(pdf_output_path, 'rb') as f:
        pdf_data = f.read()
    
    return pdf_data

# Function to generate download link
def generate_download_link(pdf_data):
    b64 = base64.b64encode(pdf_data).decode() # B64 encode
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="prescription.pdf">Download PDF</a>'
    return href

def file_upload():
    # File uploader with on_change callback
    
    #st.image("static\images\Data-extraction-rafiki.png")
    st.file_uploader("Upload a file", type=["pdf", "png", "jpeg", "jpg", "txt"], key='uploaded_file', on_change=handle_file_upload)

file_upload()

