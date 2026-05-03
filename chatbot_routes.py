from fastapi import APIRouter, Depends
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from models import History
from database import SessionLocal
from sqlalchemy.orm import Session
from langgraph_chatbot import chatbot, retrieve_all_threads

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

@router.post("/api/chat")
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
def get_threads():
    return {
        "threads": retrieve_all_threads()
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