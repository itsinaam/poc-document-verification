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

DATABASE_URL = os.getenv("PSYCOPG_DATABASE_URL")


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
            "order_source": o.order_source,
            "customer_name": o.customer_name,
            "order_items": o.order_items,
            "total_amount": str(o.total_amount) if o.total_amount else None,
            "type_of_order": o.type_of_order,
            "language": o.language,
            "status": o.status,
            "order_date": o.order_date.strftime("%Y-%m-%d %H:%M:%S") if o.order_date else None,
            "created_at": o.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": o.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
        })

    return data

def parse_price(price_string: str) -> float:
    """Parse price string and extract numeric value, handling currency symbols and formatting"""
    import re
    
    if not price_string:
        return 0.0
        
    # Remove currency symbols and extra spaces
    # This regex finds all numbers (including decimals) in the string
    price_matches = re.findall(r'\d+(?:\.\d+)?', str(price_string))
    
    if price_matches:
        # Take the first numeric value found
        return float(price_matches[0])
    
    # If no number found, try to convert as is (fallback)
    try:
        return float(price_string)
    except ValueError:
        return 0.0

def detect_customer_type(customer_name: str, quantity: int) -> str:
    """Automatically detect if customer is B2B or B2C based on name and quantity"""
    
    # Business indicators in customer name
    business_keywords = [
        'llc', 'inc', 'corp', 'ltd', 'company', 'co.', 'enterprise', 'enterprises',
        'group', 'corporation', 'incorporated', 'limited', 'business', 'store',
        'shop', 'retail', 'wholesale', 'distribution', 'distributor', 'trading',
        'international', 'global', 'industries', 'solutions', 'services', 'agency',
        'firm', 'associates', 'partners', 'partnership'
    ]
    
    customer_lower = customer_name.lower()
    
    # Check for business keywords in name
    has_business_indicators = any(keyword in customer_lower for keyword in business_keywords)
    
    # Business detection logic - prioritize business name indicators
    if has_business_indicators:
        return "B2B"  # Business customer regardless of quantity
    
    # For individual names, classify based on quantity
    if quantity >= 10:
        return "B2B"  # Large quantities suggest business use
    else:
        return "B2C"  # Small quantities for individual customers
        
    # Additional patterns that suggest B2B:
    # - Multiple words in business format (e.g., "ABC Beauty Supply")
    # - All caps names (common in business names)
    # - Names ending with numbers (branch locations)
    words = customer_name.split()
    if len(words) >= 3 and any(word.isupper() for word in words):
        return "B2B"
    
    return "B2C"  # Default to B2C

def detect_language(text: str) -> str:
    """Auto-detect language from customer name or text"""
    
    # Simple language detection based on common patterns
    text_lower = text.lower()
    
    # Check for Japanese characters or names
    japanese_indicators = ['san', 'kun', 'chan', 'sama', 'tokyo', 'osaka', 'kyoto', 'honda', 'yamaha', 'suzuki']
    if any(indicator in text_lower for indicator in japanese_indicators):
        return "ja"
    
    # Check for Chinese indicators
    chinese_indicators = ['li', 'wang', 'zhang', 'liu', 'chen', 'yang', 'beijing', 'shanghai']
    if any(indicator in text_lower for indicator in chinese_indicators):
        return "zh"
    
    # Check for Arabic indicators
    arabic_indicators = ['mohammed', 'ahmad', 'hassan', 'ali', 'omar', 'fatima', 'aisha']
    if any(indicator in text_lower for indicator in arabic_indicators):
        return "ar"
    
    # Check for Spanish indicators
    spanish_indicators = ['carlos', 'maria', 'jose', 'luis', 'ana', 'juan', 'pedro']
    if any(indicator in text_lower for indicator in spanish_indicators):
        return "es"
    
    # Check for French indicators
    french_indicators = ['jean', 'marie', 'pierre', 'jacques', 'françois', 'michel']
    if any(indicator in text_lower for indicator in french_indicators):
        return "fr"
    
    # Default to English
    return "en"


# Tools
@tool
def get_products(
    fields_only: list = None,
    include_all: bool = False,
    product_ids: list = None,
    names_only: bool = False
) -> dict:
    """Get products from database with flexible field selection based on user request.
    
    Parameters:
    - fields_only: List of specific fields to return (e.g., ['name', 'price'])
    - include_all: If True, return all available fields
    - product_ids: List of specific product IDs to filter
    - names_only: If True, return only product names
    
    Usage examples:
    - get_products(names_only=True) -> Only product names
    - get_products(fields_only=['name', 'price']) -> Name and price only
    - get_products(include_all=True) -> All product details
    - get_products(product_ids=[1, 2, 3]) -> Specific products with all details
    """
    db = SessionLocal()
    try:
        query = db.query(Product)
        
        # Filter by specific product IDs if provided
        if product_ids:
            query = query.filter(Product.id.in_(product_ids))
            
        products = query.all()
        
        # Return only names if requested
        if names_only:
            data = [{"id": p.id, "name": p.name} for p in products]
            return {"status": "success", "data": data, "display_type": "names_only"}
        
        # Return specific fields only
        if fields_only and not include_all:
            data = []
            for p in products:
                item = {"id": p.id}  # Always include ID
                for field in fields_only:
                    if hasattr(p, field):
                        item[field] = getattr(p, field)
                data.append(item)
            return {"status": "success", "data": data, "display_type": "selective_fields", "fields": fields_only}
        
        # Return all details (default or when include_all=True)
        data = []
        for p in products:
            item = {
                "id": p.id,
                "name": p.name,
                "price": p.price,
                "description": p.description,
                "quantity": p.quantity,
                "ingredients": p.ingredients,
                "usage_instructions": p.usage_instructions,
                "suitable_age_range": p.suitable_age_range,
                "image_url": p.image_url
            }
            data.append(item)
            
        return {"status": "success", "data": data, "display_type": "full_details"}
    finally:
        db.close()

@tool
def add_product(
    name: str, 
    price: str, 
    description: str = None, 
    quantity: int = 0,
    ingredients: str = None,
    usage_instructions: str = None,
    suitable_age_range: str = None,
    image_url: str = None
) -> dict:
    """Add new product with all available fields"""
    db = SessionLocal()
    try:
        product = Product(
            name=name,
            price=price,
            description=description,
            quantity=quantity,
            ingredients=ingredients,
            usage_instructions=usage_instructions,
            suitable_age_range=suitable_age_range,
            image_url=image_url
        )
        db.add(product)
        db.commit()
        db.refresh(product)

        return {"status": "success", "data": {"id": product.id}, "message": "Product added"}
    finally:
        db.close()

@tool
def create_order(
    product_id: int, 
    quantity: int, 
    customer_name: str,
    order_source: str = "chatbot",
    type_of_order: str = "auto",
    language: str = "auto"
) -> dict:
    """Create order with automatic B2B/B2C detection, language detection, and stock validation.
    
    Stock Validation:
    - Checks if requested quantity is available in stock
    - Returns error message if insufficient stock with current availability
    
    Auto-detects:
    - 'B2C': Individual customers, small quantities (1-9 units)
    - 'B2B': Business customers (company names) or large quantities (10+ units)
    
    Language auto-detection based on customer name patterns.
    
    Always asks user for customer name and quantity before creating order.
    """
    db = SessionLocal()
    try:
        product = db.query(Product).filter(Product.id == product_id).first()

        if not product:
            return {"status": "error", "error": "Product not found"}

        # Check if there's enough quantity in stock
        if product.quantity < quantity:
            return {
                "status": "error", 
                "error": f"Insufficient stock. We currently have {product.quantity} units of '{product.name}' available, but you requested {quantity} units. Please adjust your quantity or contact us for restocking information."
            }

        # Auto-detect B2B vs B2C if not explicitly set
        if type_of_order == "auto":
            type_of_order = detect_customer_type(customer_name, quantity)

        # Auto-detect language if not explicitly set
        if language == "auto":
            language = detect_language(customer_name)

        # Parse price to handle currency formatting
        unit_price = parse_price(product.price)
        total_amount = unit_price * quantity

        # Create proper order_items structure
        order_items_data = [{
            "product_id": product_id,
            "product_name": product.name,
            "quantity": quantity,
            "unit_price": unit_price,
            "subtotal": total_amount
        }]

        order = Order(
            order_source=order_source,
            customer_name=customer_name,
            order_items=order_items_data,
            total_amount=total_amount,
            type_of_order=type_of_order,
            language=language,
            status="pending"
        )

        db.add(order)
        db.commit()
        db.refresh(order)

        return {
            "status": "success", 
            "data": {
                "order_id": order.id,
                "customer_type": type_of_order,
                "detected_language": language,
                "total_amount": total_amount
            }
        }
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

Product Information Available:
- Basic details: name, description, price, quantity
- Detailed information: ingredients, usage_instructions, suitable_age_range, image_url
- Use these details to help customers make informed decisions

Tool Usage for Product Information:
- Use get_products() with flexible parameters based on user request:
  * names_only=True when user asks "show me product names", "what products do you have", "list product names"
  * fields_only=['name', 'price'] when user asks for specific details like "show names and prices"
  * fields_only=['ingredients'] when user asks "what are the ingredients", "show ingredients"
  * product_ids=[1,2,3] when user asks about specific products by ID
  * include_all=True when user asks "show all details", "full information", "everything about products"
  * Default (no parameters) returns full details for general product queries

IMPORTANT: Match the tool parameters to user intent:
- "Show me product names" → names_only=True
- "What's the price of products" → fields_only=['name', 'price'] 
- "Tell me about ingredients" → fields_only=['name', 'ingredients']
- "Show all product information" → include_all=True
- "Details about product 1" → product_ids=[1], include_all=True

Order Processing Enhanced:
- ALWAYS ask for customer name and quantity before creating any order
- Stock validation: Automatically checks if requested quantity is available in stock
  * If insufficient stock, informs customer of current availability
  * Only creates order if sufficient stock exists
- Orders automatically detect customer type (B2B vs B2C) based on:
  * Customer name (business indicators like LLC, Corp, Inc, Company, etc.)
  * Order quantity: 1-9 units = B2C, 10+ units = B2B
  * Business names automatically = B2B regardless of quantity
- Language auto-detection based on customer name patterns (en, ja, zh, ar, es, fr)
- System provides detected customer type and language in response
- This helps with better customer service and business analytics

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