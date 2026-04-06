import os
import re
import json
import time
import shutil
import requests
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from database import SessionLocal
from models import DocumentAnalysis

from typing import List
from fastapi import UploadFile, File

from dateutil import parser
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# PDF Loader
from langchain_community.document_loaders import PyPDFLoader

# DB (Assuming already configured)
from database import SessionLocal
from models import DocumentAnalysis

# ===============================
# APP INIT
# ===============================
app = FastAPI(title="OCR Extraction API", docs_url="/api/docs", redoc_url="/api/redoc")

UPLOAD_FOLDER = "/tmp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OCR_API_KEY = os.getenv("OCR_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===============================
# OCR FUNCTIONS
# ===============================
def extract_text_from_image(file_obj):
    filename = "upload.jpg"

    response = requests.post(
        "https://api.ocr.space/parse/image",
        files={"file": (filename, file_obj)},
        data={
            "apikey": OCR_API_KEY,
            "language": "eng",
        },
    )

    result = response.json()

    if result.get("IsErroredOnProcessing"):
        return ""

    try:
        return result["ParsedResults"][0]["ParsedText"]
    except:
        return ""


def extract_text_from_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    return "\n".join([doc.page_content for doc in documents])


# ===============================
# LLM FUNCTION (UPDATED)
# ===============================
def extract_final_data(text):
    current_year = datetime.now().year

    prompt = f"""
            You are an extraction system.

            Rules:
            - Extract name of person → if missing return "unknown employee"
            - Extract date → if missing return "date not present"
            - If date belongs to current year ({current_year}) → is_traveled = true
            - else false
            - Also want confidence score of he/she is traveled or not (0.0 to 1.0), if is_traveled is true then confidence score should be greater than 0.9 else less than 0.3
            - Flight name if is present in the text → if not present return "flight name not present"
            - Seat number if is present in the text → if not present return "seat number not present"
            - From location if is present in the text → if not present return "from location not present"
            - To location if is present in the text → if not present return "to location not present"

            Return ONLY JSON:

            {{
            "name": "string",
            "date": "string",
            "is_traveled": true,
            "confidence_score": "string",
            "flight_name": "string",
            "seat_number": "string",
            "from_location": "string",
            "to_location": "string"
            }}

            Text:
            {text}
            """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Strict JSON extractor"},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        return json.loads(response.choices[0].message.content)

    except Exception as e:
        return {
            "name": "unknown employee",
            "date": "date not present",
            "is_traveled": False,
            "error": str(e)
        }

@app.get("/")
def home():
    return {"message": "FastAPI Backend Running 🚀"}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    filename = f"{int(time.time())}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    # ===============================
    # SAVE FILE
    # ===============================
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ===============================
    # EXTRACT TEXT
    # ===============================
    if filename.lower().endswith(".pdf"):
        extracted_text = extract_text_from_pdf(file_path)
    else:
        with open(file_path, "rb") as f:
            extracted_text = extract_text_from_image(f)

    result = extract_final_data(extracted_text)
    status = "approved" if result.get("is_traveled") else "rejected"

    db = SessionLocal()

    try:
        record = DocumentAnalysis(
            name=result.get("name"),
            date=result.get("date"),
            is_traveled=str(result.get("is_traveled")),

            confidence_score=str(result.get("confidence_score")),
            flight_name=result.get("flight_name"),
            seat_number=result.get("seat_number"),
            from_location=result.get("from_location"),
            to_location=result.get("to_location"),

            status=status,
            file_path=file_path
        )

        db.add(record)
        db.commit()

    except Exception as e:
        db.rollback()

        record = DocumentAnalysis(
            status="error",
            error_message=str(e),
            file_path=file_path
        )
        db.add(record)
        db.commit()

        return {"error": str(e)}

    finally:
        db.close()

    # ===============================
    # DELETE FILE AFTER SAVE ✅
    # ===============================
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print("File deleted:", file_path)
    except Exception as e:
        print("File delete error:", e)

    result["status"] = status
    return result



@app.get("/api/documents")
def dashboard():

    db = SessionLocal()

    try:
        records = db.query(DocumentAnalysis)\
                    .order_by(DocumentAnalysis.created_at.desc())\
                    .all()

        total = len(records)
        approved = sum(1 for r in records if r.status == "approved")
        rejected = sum(1 for r in records if r.status == "rejected")
        error = sum(1 for r in records if r.status == "error")

        data = []

        for r in records:
            data.append({
                "id": r.id,
                "name": r.name,
                "date": r.date,
                "is_traveled": r.is_traveled,
                "confidence_score": r.confidence_score,
                "flight_name": r.flight_name,
                "seat_number": r.seat_number,
                "from_location": r.from_location,
                "to_location": r.to_location,
                "status": r.status,
                "file_path": r.file_path,
                "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S")
            })

        return {
            "summary": {
                "total": total,
                "approved": approved,
                "rejected": rejected,
                "error": error
            },
            "records": data
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        db.close()


@app.get("/api/document/{doc_id}")
def get_document_by_id(doc_id: int):

    db = SessionLocal()

    try:
        record = db.query(DocumentAnalysis)\
                   .filter(DocumentAnalysis.id == doc_id)\
                   .first()

        # ===============================
        # NOT FOUND
        # ===============================
        if not record:
            raise HTTPException(status_code=404, detail="Record not found")

        # ===============================
        # RESPONSE
        # ===============================
        return {
            "id": record.id,
            "name": record.name,
            "date": record.date,
            "is_traveled": record.is_traveled,
            "confidence_score": record.confidence_score,
            "flight_name": record.flight_name,
            "seat_number": record.seat_number,
            "from_location": record.from_location,
            "to_location": record.to_location,
            "status": record.status,
            "file_path": record.file_path,
            "created_at": record.created_at.strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        db.close()
@app.delete("/api/document/{doc_id}")
def delete_document_by_id(doc_id: int):

    db = SessionLocal()

    try:
        record = db.query(DocumentAnalysis)\
                   .filter(DocumentAnalysis.id == doc_id)\
                   .first()

        # ===============================
        # NOT FOUND
        # ===============================
        if not record:
            raise HTTPException(status_code=404, detail="Record not found")

        # ===============================
        # DELETE
        # ===============================
        db.delete(record)
        db.commit()

        return {"message": f"Document ID {doc_id} deleted successfully"}

    except Exception as e:
        db.rollback()
        return {"error": str(e)}

    finally:
        db.close()



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

