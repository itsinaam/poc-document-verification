from langchain_community.document_loaders import PyPDFLoader
import os,json,time,shutil,base64 
import pandas as pd
import numpy as np
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from database import SessionLocal
from models import DocumentAnalysis, Product, Order
from fastapi import UploadFile, File
from openai import OpenAI
from dotenv import load_dotenv
from models import Base
from database import engine
from pydantic import BaseModel
from typing import Optional
from decimal import Decimal
from chatbot_routes import router


load_dotenv()

app = FastAPI(title="POC-AI-Agent-Empro", docs_url="/api/docs", redoc_url="/api/redoc")
app.include_router(router)

UPLOAD_FOLDER = "/tmp"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


#----------------- Schema ----------------

class ProductCreate(BaseModel):
    name: str
    description: Optional[str] = None
    price: Optional[str] = None
    quantity: Optional[int] = 0
    ingredients: Optional[str] = None
    usage_instructions: Optional[str] = None
    suitable_age_range: Optional[str] = None
    image_url: Optional[str] = None


def extract_text_from_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return "\n".join([doc.page_content for doc in documents])

def extract_travel_data(text=None, base64_image=None, mime_type=None):
    current_year = datetime.now().year

    prompt = f"""
    You are an extraction system.

    Rules:
    - Extract name of person → if missing return "unknown employee"
    - Extract date → if missing return "date not present"
    - If date belongs to 2025 or 2026 → is_traveled = true
    - If date is 2027 or above → is_traveled = false
    - For any other years → is_traveled = false
    - Also want confidence score of he/she is traveled or not (0.0 to 1.0), 
      if is_traveled is true then confidence score should be 1.0 else less than 0.3
    - Flight name if present → else "flight name not present"
    - Seat number if present → else "seat number not present"
    - From location if present → else "from location not present"
    - To location if present → else "to location not present"

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
    """

    try:
        # 🔹 Case 1: Image input
        if base64_image and mime_type:
            messages = [
                {"role": "system", "content": "Strict JSON extractor"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]

        # 🔹 Case 2: Text input
        elif text:
            messages = [
                {"role": "system", "content": "Strict JSON extractor"},
                {"role": "user", "content": f"{prompt}\n\nText:\n{text}"}
            ]

        else:
            raise ValueError("Either text or image must be provided")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
        )

        content = response.choices[0].message.content.strip()

        # 🔹 Clean markdown JSON if exists
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()

        return json.loads(content)

    except Exception as e:
        return {
            "name": "unknown employee",
            "date": "date not present",
            "is_traveled": False,
            "confidence_score": "0.0",
            "flight_name": "flight name not present",
            "seat_number": "seat number not present",
            "from_location": "from location not present",
            "to_location": "to location not present",
            "error": str(e)
        }
    
def save_analysis_to_db(db, result, file_path, status):
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
        db.refresh(record)

        return record

    except Exception as e:
        db.rollback()

        error_record = DocumentAnalysis(
            status="error",
            error_message=str(e),
            file_path=file_path
        )
        db.add(error_record)
        db.commit()

        raise e

def get_all_documents(db):
    return db.query(DocumentAnalysis) \
             .order_by(DocumentAnalysis.created_at.desc()) \
             .all()

def calculate_summary(records):
    return {
        "total": len(records),
        "approved": sum(1 for r in records if r.status == "approved"),
        "rejected": sum(1 for r in records if r.status == "rejected"),
        "error": sum(1 for r in records if r.status == "error"),
    }

def format_documents(records):
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

    return data

def create_product(db, product_data: ProductCreate):
    try:
        product = Product(
            name=product_data.name,
            description=product_data.description,
            price=product_data.price,
            quantity=product_data.quantity,
            ingredients=product_data.ingredients,
            usage_instructions=product_data.usage_instructions,
            suitable_age_range=product_data.suitable_age_range,
            image_url=product_data.image_url
        )

        db.add(product)
        db.commit()
        db.refresh(product)

        return product

    except Exception as e:
        db.rollback()
        raise e

def get_all_orders(db):
    return db.query(Order) \
             .order_by(Order.created_at.desc()) \
             .all()

def calculate_order_summary(orders):
    return {
        "total_orders": len(orders),
        "pending": sum(1 for o in orders if o.status == "pending"),
        "completed": sum(1 for o in orders if o.status == "completed"),
        "cancelled": sum(1 for o in orders if o.status == "cancelled"),
    }

def format_orders(orders):
    data = []

    for o in orders:
        data.append({
            "id": o.id,
            "order_source": o.order_source,
            "type_of_order": o.type_of_order,
            "language": o.language,
            "location": o.location,
            "currency": o.currency,
            "customer_name": o.customer_name,
            "order_items": o.order_items,
            "total_amount": str(o.total_amount) if o.total_amount else None,
            "status": o.status,
            "order_date": o.order_date.strftime("%Y-%m-%d %H:%M:%S") if o.order_date else None,
            "created_at": o.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": o.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
        })

    return data


@app.get("/api/health")
def health_check():
    return {"message": "FastAPI Backend Running 🚀"}

@app.post("/api/init-db")
def init_db():
    Base.metadata.create_all(bind=engine)
    return {"message": "Tables are created successfully!"}

# -------------- DOCUMENT ANALYSIS --------------

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    filename = f"{int(time.time())}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        if filename.lower().endswith(".pdf"):
            extracted_text = extract_text_from_pdf(file_path)
            result = extract_travel_data(text=extracted_text)

        else:
            with open(file_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")

            ext = filename.split('.')[-1].lower()
            mime_type = f"image/{ext if ext != 'jpg' else 'jpeg'}"

            result = extract_travel_data(
                base64_image=base64_image,
                mime_type=mime_type
            )

        status = "approved" if result.get("is_traveled") else "rejected"

        db = SessionLocal()
        try:
            save_analysis_to_db(db, result, file_path, status)
        finally:
            db.close()

        result["status"] = status
        return result

    except Exception as e:
        return {"error": str(e)}

    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print("File deleted:", file_path)
        except Exception as e:
            print("File delete error:", e)

@app.get("/api/documents")
def dashboard():

    db = SessionLocal()

    try:
        records = get_all_documents(db)
        summary = calculate_summary(records)
        formatted_records = format_documents(records)

        return {
            "summary": summary,
            "records": formatted_records
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

        if not record:
            raise HTTPException(status_code=404, detail="Record not found")

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

        if not record:
            raise HTTPException(status_code=404, detail="Record not found")

        db.delete(record)
        db.commit()

        return {"message": f"Document ID {doc_id} deleted successfully"}

    except Exception as e:
        db.rollback()
        return {"error": str(e)}

    finally:
        db.close()


#----------------- DAVID - GIS--------------------

@app.get("/api/parcel-data")
def get_parcel_data():
    """
    Returns all data from the parcel_data.csv file
    """
    try:
        # Read the CSV file
        csv_path = os.path.join(os.path.dirname(__file__), "parcel_data.csv")
        print(f"Looking for CSV at: {csv_path}")
        
        if not os.path.exists(csv_path):
            raise HTTPException(status_code=404, detail="parcel_data.csv file not found")
        
        # Read CSV into DataFrame
        df = pd.read_csv(csv_path)
        
        # Convert all numpy types to native Python types
        df = df.astype(object).where(pd.notna(df), None)
        
        # Custom function to convert numpy types
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            elif isinstance(obj, (pd.Series, np.ndarray)):
                return [convert_types(v) for v in obj]
            elif hasattr(obj, 'item'):
                # Convert numpy types to native Python types
                return obj.item()
            else:
                return obj
        
        # Convert DataFrame to list of dictionaries
        data = df.to_dict(orient='records')
        data = convert_types(data)
        
        return {
            "status": "success",
            "total_records": len(data),
            "data": data
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading parcel data: {str(e)}")

@app.get("/api/zoning-filter")
def get_filtered_zoning_data(
    zones: str = "R-4,R-6,R-10"
):
    """
    Returns filtered zoning data from zoning_data.csv
    Filters by the provided zoning values.
    """
    try:
        zoning_path = os.path.join(os.path.dirname(__file__), "zoning_data.csv")
        if not os.path.exists(zoning_path):
            raise HTTPException(status_code=404, detail="zoning_data.csv not found")

        zoning_df = pd.read_csv(zoning_path)
        zoning_list = [z.strip() for z in zones.split(",") if z.strip()]

        if not zoning_list:
            raise HTTPException(status_code=400, detail="Please provide at least one zoning value")

        filtered_df = zoning_df[zoning_df['ZONE_TYPE'].isin(zoning_list)].copy()

        map_columns = [
            'geometry', 'GLOBALID', 'OBJECTID', 'ZONE_TYPE', 'ZONE_TYPE_DECODE',
            'HEIGHT', 'FRONTAGE', 'CONDITIONAL', 'ZONING', 'ZN_CASE_NUM',
            'EFF_DATE', 'ORDINANCE', 'PLAN_NAME', 'INTO_UDO', 'COND_LINK'
        ]

        existing_cols = [col for col in map_columns if col in filtered_df.columns]
        result_df = filtered_df[existing_cols]
        result_df = result_df.astype(object).where(pd.notna(result_df), None)

        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            elif hasattr(obj, 'item'):
                return obj.item()
            else:
                return obj

        data = result_df.to_dict(orient='records')
        data = convert_types(data)

        return {
            "status": "success",
            "filters_applied": {
                "zones": zoning_list
            },
            "total_records": len(data),
            "data": data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error filtering zoning data: {str(e)}")

#------------------- Products ----------------------


@app.post("/api/products")
def add_product(product: ProductCreate):

    db = SessionLocal()

    try:
        new_product = create_product(db, product)

        return {
            "message": "Product created successfully",
            "product": {
                "id": new_product.id,
                "name": new_product.name,
                "description": new_product.description,
                "price": str(new_product.price),
                "quantity": new_product.quantity,
                "suitable_age_range": new_product.suitable_age_range,
                "ingredients": new_product.ingredients,
                "usage_instructions": new_product.usage_instructions,
                "image_url": new_product.image_url,
                "created_at": new_product.created_at.strftime("%Y-%m-%d %H:%M:%S")
            }
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        db.close()


#------------------ Get Orders--------------------

@app.get("/api/orders")
def get_orders():

    db = SessionLocal()

    try:
        # 🔹 Fetch
        orders = get_all_orders(db)

        # 🔹 Process
        summary = calculate_order_summary(orders)
        formatted_orders = format_orders(orders)

        return {
            "summary": summary,
            "orders": formatted_orders
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        db.close()