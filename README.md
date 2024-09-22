# PDF Summarization API

This project provides a Flask-based API for summarizing PDF documents in both English and Arabic languages. It leverages advanced natural language processing models to extract text, perform semantic chunking, and generate concise summaries.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the API](#running-the-api)
  - [Making a Request](#making-a-request)
- [API Endpoint](#api-endpoint)
- [Project Structure](#project-structure)
- [Models and Libraries Used](#models-and-libraries-used)
- [Notes](#notes)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- Multilingual Support Summarizes both English and Arabic PDF documents.
- Semantic Chunking Divides text into semantically meaningful chunks for better summarization.
- Advanced NLP Models Utilizes state-of-the-art models like SentenceTransformer, BART, and mT5.
- Flask API Provides a RESTful API endpoint for easy integration.
- GPU Acceleration Supports CUDA-enabled GPUs for faster processing.

## Prerequisites

- Python 3.7 or higher
- CUDA-compatible GPU (optional but recommended for improved performance)
- Internet Connection Required for downloading pre-trained models on the first run.

## Installation

1. Clone the Repository

   ```bash
   git clone https://github.com/The-Joesters/Book-Summary-AR-EN.git
   cd Book-Summary-AR-EN
   ```

2. Create a Virtual Environment

   ```bash
   python -m venv venv
   source venvbinactivate  # On Windows use `venvScriptsactivate`
   ```

3. Install Dependencies

   ```bash
   pip install -r requirements.txt
   ```

   ```bash
   pip install flask sentence-transformers transformers pdfplumber arabic-reshaper python-bidi stanza langdetect torch
   ```

4. Download Stanza Resources

   Open a Python shell and run

   ```python
   import stanza
   stanza.download('ar')
   ```

   Alternatively, you can include this in your code or run it as a script.

## Usage

### Running the API

```bash
python app.py
```

By default, the Flask app will run on `http127.0.0.15000`.

### Making a Request

You can use `curl`, `Postman`, or any HTTP client to send a POST request.

Using `curl`

```bash
curl -X POST -F 'file=@pathtoyourdocument.pdf' http127.0.0.15000summarize
```

Response

```json
{
  summary Your summarized text here...
}
```

## API Endpoint

### `POST summarize`

- Description Accepts a PDF file and returns a summarized version of its content.
- Parameters
  - `file` The PDF file to be summarized (multipartform-data).
- Responses
  - `200 OK` Returns the summary in JSON format.
  - `400 Bad Request` Missing file or invalid request.
  - `500 Internal Server Error` Summarization failed due to server error.

## Project Structure

- `app.py` Main Flask application containing the API endpoint and summarization logic.
- `requirements.txt` List of required Python packages.
- `README.md` Project documentation.

## Models and Libraries Used

- SentenceTransformer (`all-MiniLM-L6-v2`) For generating semantic embeddings.
- Transformers Pipeline
  - BART (`facebookbart-large-cnn`) For English summarization.
  - mT5 (`csebuetnlpmT5_multilingual_XLSum`) For Arabic summarization.
- Stanza For Arabic NLP tasks like tokenization.
- pdfplumber For extracting text from PDF files.
- arabic-reshaper and python-bidi For correct display and handling of Arabic text.
- langdetect For automatic language detection.

## Notes

- First-Time Model Downloads The first run will download the necessary NLP models, which may take some time depending on your internet speed.
- GPU Acceleration If a CUDA-compatible GPU is available, the models will utilize it for faster processing.
- Text Alignment The API formats the summary to align text from the right for Arabic and from the left for English.
- File Cleanup Temporary files are automatically deleted after processing to conserve disk space.

## Troubleshooting

- Model Loading Issues Ensure that your internet connection is stable during the first run to download all necessary models.
- Stanza Download Errors If you encounter issues with Stanza resources, manually download them using the provided commands in the installation section.
- CUDA Errors If you have a GPU but encounter CUDA errors, ensure that the correct CUDA drivers and PyTorch version are installed.
- Language Detection The `langdetect` library may not always accurately detect the language if the text is too short or ambiguous.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face Transformers](httpshuggingface.cotransformers) For the pre-trained models and pipelines.
- [SentenceTransformers](httpswww.sbert.net) For semantic similarity and embedding models.
- [Stanza](httpsstanfordnlp.github.iostanza) For NLP tasks in multiple languages.
- [pdfplumber](httpsgithub.comjsvinepdfplumber) For PDF text extraction.
- [arabic-reshaper](httpspypi.orgprojectarabic-reshaper) and [python-bidi](httpspypi.orgprojectpython-bidi) For proper handling of Arabic text.
- [langdetect](httpspypi.orgprojectlangdetect) For automatic language detection.

---

Feel free to contribute to this project by submitting issues or pull requests.
