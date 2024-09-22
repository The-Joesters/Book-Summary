from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import re
import pdfplumber
import arabic_reshaper
from bidi.algorithm import get_display
import stanza
from langdetect import detect
import os
import tempfile
import torch

app = Flask(__name__)

# Determine the device to run models on (GPU if available, else CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load models
print("Loading models...")
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
summarizer_en = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if device == 'cuda' else -1)
summarizer_ar = pipeline('summarization', model='csebuetnlp/mT5_multilingual_XLSum', device=0 if device == 'cuda' else -1)

# Download stanza resources if not already downloaded
stanza_dir = os.path.expanduser('~/stanza_resources')
if not os.path.exists(os.path.join(stanza_dir, 'ar')):
    stanza.download('ar')

nlp_ar = stanza.Pipeline('ar', processors='tokenize', use_gpu=(device == 'cuda'))

print("Models loaded.")

    
# Extract text from the PDF file using pdfplumber
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''.join(page.extract_text() for page in pdf.pages)
    return text

# Function: Arabic Text Reshaping and Bidi Fix
def fix_arabic_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)


# Clean the text by removing URLs, numbers, and extra spaces
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'\b[A-Za-z]\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

"""## Step 5: Set Up the Sentence-BERT Model and Summarizer Models

"""

# Load pre-trained Sentence-BERT model for semantic embeddings (ensure GPU usage)
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
summarizer_en = pipeline("summarization", model="facebook/bart-large-cnn", device=0)
summarizer_ar = pipeline('summarization', model='csebuetnlp/mT5_multilingual_XLSum', device=0)
nlp_ar = stanza.Pipeline('ar', processors='tokenize')

"""## Step 6: Define the Function for Semantic Chunking in English

"""

def divide_by_semantics_with_length(text, threshold=0.6, max_words=800, min_words=400):
    sentences = text.split('. ')
    embeddings = model.encode(sentences, convert_to_tensor=True)
    chunks = []
    current_chunk = sentences[0]

    for i in range(1, len(sentences)):
        similarity = util.pytorch_cos_sim(embeddings[i], embeddings[i - 1])
        current_word_count = len(current_chunk.split())

        if similarity < threshold or current_word_count + len(sentences[i].split()) > max_words:
            if current_word_count >= min_words:
                chunks.append(current_chunk.strip())
                current_chunk = sentences[i]
            else:
                current_chunk += '. ' + sentences[i]
        else:
            current_chunk += '. ' + sentences[i]

    if len(current_chunk.split()) >= min_words:
        chunks.append(current_chunk.strip())

    return chunks

"""## Step 7: Define the Function for Semantic Chunking in Arabic"""

# Function: Semantic Chunking (Arabic)
def chunk_arabic_text(text, min_words=300, max_words=500):
    """Break the Arabic text into semantically meaningful chunks."""
    doc = nlp_ar(text)
    chunks = []
    current_chunk = []
    current_chunk_word_count = 0

    for sentence in doc.sentences:
        sentence_text = sentence.text
        sentence_word_count = len(sentence_text.split())

        # If the sentence is too long, split it into smaller sentences
        if sentence_word_count > max_words:
            split_sentences = split_long_sentence(sentence_text, max_words)
        else:
            split_sentences = [sentence_text]

        # Add the split sentences to the current chunk
        for split_sentence in split_sentences:
            split_sentence_word_count = len(split_sentence.split())
            if current_chunk_word_count + split_sentence_word_count > max_words and current_chunk_word_count >= min_words:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_chunk_word_count = 0

            current_chunk.append(split_sentence)
            current_chunk_word_count += split_sentence_word_count

    # Add the last chunk if it meets the minimum word requirement
    if current_chunk_word_count >= min_words:
        chunks.append(' '.join(current_chunk))

    return chunks

# Helper function to split long Arabic sentences
def split_long_sentence(sentence_text, max_words):
    words = sentence_text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

"""## Step 8: Define the Summarization and Text Generation Functions

"""

# Summarize text chunks
def summarize_chunks(chunks, summarizer, min_chunk_length=50, max_summary_length=300, min_summary_length=80):
    summaries = []
    for chunk in chunks:
        if len(chunk.split()) > min_chunk_length:
            try:
                summary = summarizer(chunk, max_length=max_summary_length, min_length=min_summary_length, do_sample=False)[0]['summary_text']
                summaries.append(summary)
            except Exception as e:
                print(f"Error summarizing chunk: {e}")
                summaries.append(chunk)
        else:
            summaries.append(chunk)
    return summaries


# Process the title based on the language
def get_title(language):
    if language == 'ar':
        title = "ملخص الكتاب"
    else:
        title = 'Book Summary'
    return title


# Generate the text file
def generate_txt(summary_text, txt_output_path, language='en'):
    # Process the title
    title = get_title(language)

    # Process the body text
    if language == 'ar':
        reshaped_text = arabic_reshaper.reshape(summary_text)
        body_text = get_display(reshaped_text)
    else:
        body_text = summary_text

    # Define A4 page parameters
    characters_per_line = 80  # تقديريًا لعرض السطر في A4
    effective_line_width = characters_per_line

    # Adjust alignment based on language
    if language == 'ar':
        # For Arabic, define a function to right-align text
        def align_line(line):
            return line.rjust(effective_line_width)
    else:
        # For English, define a function to left-align text
        def align_line(line):
            return line.ljust(effective_line_width)

    # Center the title considering alignment
    centered_title = title.center(effective_line_width)

    # Format the body text with alignment
    formatted_body = ''
    for paragraph in body_text.split('\n'):
        words = paragraph.split()
        line = ''
        for word in words:
            if len(line) + len(word) + 1 <= effective_line_width:
                line += word + ' '
            else:
                # Strip extra space and align the line
                line = line.strip()
                formatted_line = align_line(line)
                formatted_body += formatted_line + '\n'
                line = word + ' '
        if line:
            line = line.strip()
            formatted_line = align_line(line)
            formatted_body += formatted_line + '\n'
        formatted_body += '\n'  # إضافة سطر فارغ بين الفقرات

    # Write the title and body to a text file
    with open(txt_output_path, 'w', encoding='utf-8') as f:
        f.write(centered_title + '\n\n')
        f.write(formatted_body)

"""## Step 9: Define the Summarization Pipelines for English and Arabic

### **English Summarization Pipeline**
"""

def summarize_english(book_text, text_output_path="english_summary.txt"):
    # Step 1: Divide text into semantic chunks
    semantic_chunks = divide_by_semantics_with_length(book_text)

    # Step 2: Clean the chunks
    cleaned_chunks = [clean_text(chunk) for chunk in semantic_chunks]

    # Step 3: Summarize the chunks
    summarized_chunks = summarize_chunks(cleaned_chunks, summarizer_en)

    # Step 4: Generate PDF
    final_summary = '\n\n'.join(summarized_chunks)
    generate_txt(final_summary, text_output_path, language='en')

    print(f"Summarization completed!, saved to {text_output_path}")

    return final_summary

"""### **Arabic Summarization Pipeline**

"""

def summarize_arabic(pdf_path, text_output_path="arabic_summary.txt"):
    # Step 1: Extract text from PDF and fix Arabic text direction
    text = extract_text_from_pdf(pdf_path)
    fixed_text = fix_arabic_text(text)  # Fixing the text direction

    # Step 2: Chunk the text semantically
    chunks = chunk_arabic_text(fixed_text)  # Now the chunking function is defined

    # Step 3: Summarize the chunks
    summarized_chunks = summarize_chunks(chunks, summarizer_ar)

    # Step 4: Clean and generate the final summary
    cleaned_summaries = [clean_text(chunk) for chunk in summarized_chunks]
    final_summary = '\n\n'.join(cleaned_summaries)

    # Step 5: Generate txt
    final_summary_arabic = fix_arabic_text(final_summary)
    generate_txt(final_summary_arabic, text_output_path, language='ar')

    # Notify the user that the txt has been created
    print(f"Summarization completed!, saved to {text_output_path}")

    return final_summary

"""## Step 10: Language Detection and Pipeline Execution

"""

def detect_language_and_summarize(pdf_path, text_output_path_ar="arabic_summary.txt", text_output_path_en="english_summary.txt"):
    text = extract_text_from_pdf(pdf_path)
    language = detect(text)

    if language == 'ar':
        print("Detected Arabic. Running Arabic summarization pipeline...")
        return summarize_arabic(pdf_path, text_output_path=text_output_path_ar)
    else:
        print("Detected English. Running English summarization pipeline...")
        return summarize_english(text, text_output_path=text_output_path_en)
    

@app.route('/summarize', methods=['POST'])
def summarize():
    # Get the file from the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
        file.save(temp_pdf.name)
        pdf_path = temp_pdf.name

    try:
        # Run the summarization pipeline
        final_summary = detect_language_and_summarize(pdf_path)
    except Exception as e:
        print(f"Error during summarization: {e}")
        return jsonify({'error': 'Summarization failed'}), 500
    finally:
        # Clean up the temp file
        os.remove(pdf_path)

    # Return the summary
    return jsonify({'summary': final_summary})

if __name__ == '__main__':
    app.run(debug=True)
