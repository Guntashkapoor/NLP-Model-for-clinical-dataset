import spacy
import en_core_med7_lg
from spacy import displacy
from PIL import Image 
from pytesseract import pytesseract 
from PyPDF2 import PdfReader
import numpy as np
import fitz



# Defining paths to tesseract.exe 

image_path = r"\images\DischargeSummary1.jpg"
pdf_path=r"C:\Users\HP\Documents\Guntash\RPL FLASH\NLP Model\static\PDF\HospitalDischargeSummary.pdf"


def ExtractTextFromImage(path=image_path):
    img=Image.open(path)    
    img1 = np.array(img)
    text=pytesseract.image_to_string(img1)
    
    return text;
  
def ExtractTextFromPdf(path=pdf_path):
    doc = fitz.open(pdf_path)
    pdf_text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        pdf_text += page.get_text()
        
    return pdf_text

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

            doc = nlp(transcription)

            spacy.displacy.render(doc, style = 'ent', jupyter = True, options = options)

            [(ent.text, ent.label_) for ent in doc.ents]
    



#output=ExtractTextFromImage(image_path)
#print("Text obtained from image")
#print(output)

output1=ExtractTextFromPdf(pdf_path)
annotations = generate_medical_annotation("en_core_med7_lg",output1)
print(annotations)
 # visualize annotation
visualize_annotations(output1)

    
#def ExtractTextFromPdf():
    
