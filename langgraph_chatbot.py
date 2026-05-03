from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from dotenv import load_dotenv
import os
import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from langchain_core.tools import tool
from database import SessionLocal
from models import Product, Order


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


llm = ChatOpenAI()


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
            "customer_name": o.customer_name,
            "order_items": o.order_items,
            "total_amount": str(o.total_amount) if o.total_amount else None,
            "status": o.status,
            "order_date": o.order_date.strftime("%Y-%m-%d %H:%M:%S") if o.order_date else None,
            "created_at": o.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": o.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
        })

    return data


# Tools
@tool
def get_products() -> dict:
    """Get all products from database"""
    db = SessionLocal()
    try:
        products = db.query(Product).all()
        data = [
            {
                "id": p.id,
                "name": p.name,
                "price": float(p.price),
                "description": p.description,
                "category": p.category,
                "quantity": p.quantity
            }
            for p in products
        ]
        return {"status": "success", "data": data}
    finally:
        db.close()

@tool
def add_product(name: str, price: float, description: str, category: str = None, quantity: int = 0) -> dict:
    """Add new product"""
    db = SessionLocal()
    try:
        product = Product(
            name=name,
            price=price,
            description=description,
            category=category,
            quantity=quantity
        )
        db.add(product)
        db.commit()
        db.refresh(product)

        return {"status": "success", "data": {"id": product.id}, "message": "Product added"}
    finally:
        db.close()

@tool
def create_order(product_id: int, quantity: int, customer_name: str) -> dict:
    """Create order"""
    db = SessionLocal()
    try:
        product = db.query(Product).filter(Product.id == product_id).first()

        if not product:
            return {"status": "error", "error": "Product not found"}

        total_amount = float(product.price) * quantity

        order = Order(
            customer_name=customer_name,
            order_items=[{
                "product_id": product_id,
                "quantity": quantity,
                "product_name": product.name
            }],
            total_amount=total_amount,
            status="pending"
        )

        db.add(order)
        db.commit()
        db.refresh(order)

        return {"status": "success", "data": {"order_id": order.id}}
    finally:
        db.close()

@tool
def get_orders(customer_name: str = None) -> dict:
    """Get orders"""
    db = SessionLocal()
    try:
        query = db.query(Order)

        if customer_name:
            query = query.filter(Order.customer_name == customer_name)

        orders = query.order_by(Order.created_at.desc()).all()

        return {
            "status": "success",
            "data": format_orders(orders),
            "summary": calculate_order_summary(orders)
        }
    finally:
        db.close()


tools = [get_products, add_product, create_order, get_orders]

# Bind all tools to LLM
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------

SYSTEM_PROMPT = """
You are an AI-powered sales assistant for Empro, specializing in Cosmetics and Skincare products.

Your role is to handle both B2B (business clients, bulk orders, resellers) and B2C (individual customers) interactions professionally and efficiently.

Guidelines:
- Maintain a polite, professional, and helpful tone at all times
- Keep responses concise and to the point (avoid long explanations)
- Understand user intent clearly and guide them through product selection and ordering
- For B2C users: assist with product details, pricing, and simple orders
- For B2B users: handle bulk inquiries, large quantities, and business-oriented requests professionally
- Always prefer using available tools (get_products, add_product, create_order, get_orders) for accurate data instead of guessing
- Never make up product or order data

Order Handling Rules (VERY IMPORTANT):
- If a user wants to place an order, ensure the following details are collected:
  1. Customer Name
  2. Product Name or Product ID
  3. Quantity

- If any of these details are missing:
  → Ask for ONLY the missing information (do not repeat everything)

- If customer name is missing:
  → Politely ask for the customer's name

- If product is not specified:
  → Ask which product they want (you may suggest available products using tools)

- If quantity is missing:
  → Ask how many units the user wants

- Do NOT proceed to create an order until all required details are available

Behavior:
- Act as a 24/7 global sales assistant capable of handling multiple customers
- Be efficient, structured, and business-oriented in responses
- Ask short, clear follow-up questions when needed
- Do not ask multiple unnecessary questions at once

Goal:
Help users quickly discover products, collect required order details, and complete orders smoothly with minimal friction.
"""

def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# -------------------
# 5. Checkpointer
# -------------------


# Create a connection pool for better handling in async environments
pool = ConnectionPool(
    DATABASE_URL,
    min_size=1,
    max_size=10,
    kwargs={
        "autocommit": True,
        "prepare_threshold": None  # Disable prepared statements
    }
)

def get_checkpointer():
    # Get a connection from the pool
    with pool.connection() as conn:
        checkpointer = PostgresSaver(conn=conn)
        # Setup will create tables if they don't exist
        try:
            checkpointer.setup()
        except Exception as e:
            # Tables might already exist, which is fine
            print(f"Setup note: {e}")
    
    # Create a new connection for the checkpointer that won't be closed
    conn = psycopg.connect(
        DATABASE_URL,
        autocommit=True,
        prepare_threshold=None
    )
    return PostgresSaver(conn=conn)

# Initialize checkpointer
checkpointer = get_checkpointer()


# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Helper
# -------------------
def retrieve_all_threads():
    """Retrieve all unique thread IDs from checkpoints"""
    threads = set()
    try:
        # List all checkpoints without filtering by thread_id
        # Pass None as config to get all checkpoints
        for cp in checkpointer.list(None):
            config = cp.config
            if config and "configurable" in config:
                thread_id = config["configurable"].get("thread_id")
                if thread_id:
                    threads.add(thread_id)
    except Exception as e:
        print(f"Error retrieving threads: {e}")
        # If listing fails, return empty list
        return []
    return list(threads)