import streamlit as st
from PIL import Image
import io
import os
from openai import OpenAI
from dotenv import load_dotenv 
import base64
import requests

load_dotenv()
languages = [
    "Spanish",
    "English",
    "Arabic",
    "Bengali",
    "Chinese (Cantonese)",
    "Chinese (Mandarin)",
    "French",
    "German",
    "Haitian Creole",
    "Hindi",
    "Italian",
    "Polish",
    "Japanese",
    "Khmer",
    "Korean",
    "Persian",
    "Portuguese",
    "Russian",
    "Somali",
    "Tagalog",
    "Urdu",
    "Vietnamese",
]
# Load OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OpenAI API key not found in environment variable 'OPENAI_API_KEY'.")
    st.stop()

# Initialize the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def analyze_document(file_bytes, mime_type, filename="uploaded_document", language="english"):
    """
    Uses the OpenAI API to analyze the uploaded document.
    The prompt is simplified: it asks whether the document is administrative or criminal
    and what actions the recipient should take.
    """
    prompt = (
        f"""
        
Using the language as specified in {language},
You are a careful legal-information explainer. Your job is to read an official immigration or government document or notice and produce a plain-language explanation for a non-lawyer.

Objectives:

You must clearly explain:

What this document is

Why the person received it

What it means for their case or situation

What actions, if any, they must take

Any deadlines, requirements, or consequences

Where they can get help or more information

Important Rules:

Do not provide legal advice or predict outcomes.

Write at a 6th–8th grade reading level in a concise, neutral, and helpful tone.

Use “you” when addressing the reader.

Do not invent facts. If the input text does not include certain information, state that it is not provided.

If the notice contains warnings or deadlines, clearly highlight them.

If the document indicates approval, denial, transfer, missing evidence, or an appointment, explain the practical meaning.

Use the Markdown format in the structure below.

Required Output Format (Markdown only):

[Document Type] (Explained)

Overview
2–3 sentences summarizing the document’s purpose and who it applies to.

Key Information

Date issued:

Deadline or appointment date (if any):

Location (if any):

Case number (if provided):

Other important facts or identifiers:

What This Means for You
Explain the purpose of the notice in practical, plain terms.

Next Steps

List required actions as bullet points in order.

Include instructions for what to do if the person cannot comply with the requirement (e.g., reschedule or request more time).

If You Do Not Respond (include only if relevant)
Explain consequences stated in the notice.

Where to Get Help
Include only phone numbers, links, or resources that were provided in the input or resource list.
        """
    )

    try:
        # First upload the file to OpenAI's Files API to obtain a file_id.
        # Use a direct multipart/form-data POST to avoid depending on the
        # specific shape of the Python client in this environment.
        upload_url = "https://api.openai.com/v1/files"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        files = {"file": (filename, io.BytesIO(file_bytes), mime_type)}
        # Choose an allowed purpose for the Files API upload.
        # Use 'vision' for image uploads (helps vision-related models),
        # otherwise use 'user_data' which is a generic allowed purpose.
        if mime_type and mime_type.startswith("image/"):
            purpose = "vision"
        else:
            purpose = "user_data"
        data = {"purpose": purpose}

        upload_resp = requests.post(upload_url, headers=headers, files=files, data=data)
        if upload_resp.status_code not in (200, 201):
            return f"Error uploading file: {upload_resp.status_code} - {upload_resp.text}"

        upload_json = upload_resp.json()
        # The Files API returns the file id under the 'id' key.
        file_id = upload_json.get("id")
        if not file_id:
            return f"Error uploading file: no file id returned: {upload_json}"

        # Now call the chat completions endpoint, referencing the uploaded file by id.
        # Build a minimal file reference payload. The chat API requires
        # a previously uploaded file's id; it generally does not accept
        # extra arbitrary fields like 'mime_type' in the file object.
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": "You are a helpful legal analysis assistant."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "file", "file": {
                        "file_id": file_id
                    }}
                ]}
            ]
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"Error during analysis: {e}"

def process_uploaded_file(uploaded_file):
    """
    Processes the uploaded file:
    - For image files, returns a PIL Image for preview.
    - For PDFs, no preview is shown.
    Returns (file_bytes, mime_type, preview_image).
    """
    file_bytes = uploaded_file.getvalue()
    mime_type = uploaded_file.type
    # Try to get the original filename if available (UploadedFile has .name).
    filename = getattr(uploaded_file, "name", "uploaded_document")

    preview_image = None
    if mime_type != "application/pdf":
        uploaded_file.seek(0)
        try:
            preview_image = Image.open(uploaded_file)
        except Exception:
            st.warning("Could not open the image for preview.")
    return file_bytes, mime_type, preview_image, filename

def main():
    st.title("Document analysis tool")
    st.write("This tool analyzes documents using OpenAI's GPT-5-Nano model.")
    st.write("Upload an image (JPEG/PNG) or PDF of the document, or capture one using your camera.")

    input_method = st.radio("Choose input method:", ("Upload File", "Capture Image"))
    selected_language = st.selectbox("Select target language", languages)

    uploaded_file = None
    if input_method == "Upload File":
        uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png", "pdf"])
    else:
        uploaded_file = st.camera_input("Capture an image")

    if uploaded_file is not None:
        file_bytes, mime_type, preview_image, filename = process_uploaded_file(uploaded_file)

        if mime_type != "application/pdf" and preview_image:
            st.image(preview_image, caption="Uploaded/Captured Image", use_column_width=True)
        else:
            st.write("PDF file uploaded.")

        st.write("Analyzing the document, please wait...")
        analysis_result = analyze_document(file_bytes, mime_type, filename=filename, language=selected_language)
        st.success("Analysis Complete")
        st.write(analysis_result)

if __name__ == "__main__":
    main()
