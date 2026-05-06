from fastapi import APIRouter, Depends
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from models import History
from database import SessionLocal
from sqlalchemy.orm import Session
from langgraph_chatbot_b2b import chatbot, retrieve_all_threads
from langgraph_chatbot_b2c import chatbot as chatbot_b2c , retrieve_all_threads as retrieve_all_threads_b2c

router = APIRouter()


# Request Schema
class ChatRequest(BaseModel):
    message: str
    thread_id: str 

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

#-------------- B2B Endpoint -------------------------

@router.post("/api/chat/b2b")
def chat_with_bot(request: ChatRequest, db: Session = Depends(get_db)):
    try:
        input_state = {
            "messages": [HumanMessage(content=request.message)]
        }

        config = {
            "configurable": {
                "thread_id": request.thread_id
            }
        }

        # Run chatbot
        result = chatbot.invoke(input_state, config=config)

        messages = result["messages"]
        last_message = messages[-1]
        bot_response = last_message.content

        # ✅ SAVE INTO DB
        history = History(
            thread_id=request.thread_id,
            message=request.message,
            response=bot_response
        )

        db.add(history)
        db.commit()
        db.refresh(history)

        # Try to parse JSON response; if it's an order result, return only the message
        import json, re
        parsed_response = bot_response
        try:
            clean = bot_response.strip()
            # Strip markdown code fences if present
            if clean.startswith("```"):
                clean = re.sub(r'^```[a-z]*\n?', '', clean).rstrip('`').strip()
            parsed_response = json.loads(clean)
        except Exception:
            pass

        return {
            "response": parsed_response,
            "thread_id": request.thread_id
        }

    except Exception as e:
        return {
            "error": str(e)
        }

#-------------- B2C Endpoint -------------------------
@router.post("/api/chat/b2c")
def chat_with_bot(request: ChatRequest, db: Session = Depends(get_db)):
    try:
        input_state = {
            "messages": [HumanMessage(content=request.message)]
        }

        config = {
            "configurable": {
                "thread_id": request.thread_id
            }
        }

        # Run chatbot
        result = chatbot_b2c.invoke(input_state, config=config)

        messages = result["messages"]
        last_message = messages[-1]
        bot_response = last_message.content

        # ✅ SAVE INTO DB
        history = History(
            thread_id=request.thread_id,
            message=request.message,
            response=bot_response
        )

        db.add(history)
        db.commit()
        db.refresh(history)

        return {
            "response": bot_response,
            "thread_id": request.thread_id
        }

    except Exception as e:
        return {
            "error": str(e)
        }

#list threads
@router.get("/api/threads")
def get_threads(db: Session = Depends(get_db)):
    try:
        # Get all unique thread IDs from both B2B and B2C chatbots
        thread_ids_b2b = retrieve_all_threads()
        thread_ids_b2c = retrieve_all_threads_b2c()
        
        # Combine and remove duplicates
        thread_ids = list(set(thread_ids_b2b + thread_ids_b2c))
        
        threads_with_messages = []
        
        for thread_id in thread_ids:
            # Get the latest message for this thread
            latest_record = (
                db.query(History)
                .filter(History.thread_id == thread_id)
                .order_by(History.timestamp.desc())
                .first()
            )
            
            if latest_record:
                # Use the user's message as the thread preview
                threads_with_messages.append({
                    "thread_id": thread_id,
                    "message": latest_record.message
                })
            else:
                # If no history found, use a default message
                threads_with_messages.append({
                    "thread_id": thread_id,
                    "message": "No messages yet"
                })
        
        return {
            "threads": threads_with_messages
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "threads": []
        }

@router.get("/api/chat/history/{thread_id}")
def get_chat_history(thread_id: str, db: Session = Depends(get_db)):
    try:
        records = (
            db.query(History)
            .filter(History.thread_id == thread_id)
            .order_by(History.timestamp.asc())
            .all()
        )

        if not records:
            return {
                "thread_id": thread_id,
                "messages": []
            }

        # Format response
        chat_history = []
        for item in records:
            chat_history.append({
                "user_message": item.message,
                "ai_response": item.response,
                "timestamp": item.timestamp
            })

        return {
            "thread_id": thread_id,
            "messages": chat_history
        }

    except Exception as e:
        return {
            "error": str(e)
        }

# Delete Thread Endpoint
@router.delete("/api/threads/{thread_id}")
def delete_thread(thread_id: str):
    """Delete a thread and all its associated checkpoints"""
    import psycopg
    from psycopg.rows import dict_row
    from langgraph.checkpoint.postgres import PostgresSaver
    import os
    from dotenv import load_dotenv
    
    try:
        load_dotenv()
        DATABASE_URL = os.getenv("PSYCOPG_DATABASE_URL")
        
        if not DATABASE_URL:
            return {
                "error": "Database connection not configured",
                "thread_id": thread_id
            }
        
        # Create a fresh connection for this operation
        with psycopg.connect(
            DATABASE_URL,
            autocommit=True,
            prepare_threshold=None,
            row_factory=dict_row
        ) as conn:
            # Create a new checkpointer with fresh connection
            temp_checkpointer = PostgresSaver(conn=conn)
            
            # Delete the thread using PostgresSaver's built-in method
            temp_checkpointer.delete_thread(thread_id)
            
        return {
            "message": f"Thread '{thread_id}' deleted successfully",
            "thread_id": thread_id
        }
    
    except Exception as e:
        return {
            "error": f"Failed to delete thread: {str(e)}",
            "thread_id": thread_id
        }