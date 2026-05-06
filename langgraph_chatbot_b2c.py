
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


llm = ChatOpenAI(model="gpt-4o")


def calculate_order_summary(orders):
    return {
        "total_orders": len(orders),
        "pending": sum(1 for o in orders if o.status == "pending"),
        "completed": sum(1 for o in orders if o.status == "completed"),
        "cancelled": sum(1 for o in orders if o.status == "cancelled"),
    }

def format_orders(orders):
    data = []
    db = SessionLocal()
    
    try:
        for o in orders:
            # Enhanced order items with product details
            enhanced_order_items = []
            if o.order_items:
                for item in o.order_items:
                    # Fetch product details for each item
                    product_id = item.get('product_id')
                    if product_id:
                        product = db.query(Product).filter(Product.id == product_id).first()
                        if product:
                            enhanced_item = {
                                **item,  # Keep original item data
                                "product_type": product.product_type,
                                "product_currency": product.currency
                            }
                            enhanced_order_items.append(enhanced_item)
                        else:
                            enhanced_order_items.append(item)
                    else:
                        enhanced_order_items.append(item)

            data.append({
                "id": o.id,
                "order_source": o.order_source,
                "customer_name": o.customer_name,
                "country": o.location,
                "order_items": enhanced_order_items,
                "total_amount": str(o.total_amount) if o.total_amount else None,
                "order_currency": o.currency,  # Show order's currency
                "type_of_order": o.type_of_order,
                "language": o.language,
                "status": o.status,
                "order_date": o.order_date.strftime("%Y-%m-%d %H:%M:%S") if o.order_date else None,
                "created_at": o.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "updated_at": o.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
            })
    finally:
        db.close()

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

def detect_language(text: str) -> str:
    """Auto-detect language from message content using character analysis and keywords"""
    
    import re
    
    # Check for Chinese characters (CJK Unified Ideographs)
    if re.search(r'[\u4e00-\u9fff]', text):
        return "Chinese"
    
    # Check for Japanese characters (Hiragana, Katakana, Kanji)
    if re.search(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf]', text):
        # If has Chinese characters but also Japanese specific characters
        if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
            return "Japanese"
    
    # Check for Arabic characters
    if re.search(r'[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]', text):
        return "Arabic"
    
    # Check for common language keywords and patterns
    text_lower = text.lower()
    
    # Spanish keywords and patterns
    spanish_keywords = ['hola', 'gracias', 'por favor', 'sí', 'no', 'cómo', 'qué', 'donde', 'cuando', 'precio', 'producto']
    if any(keyword in text_lower for keyword in spanish_keywords):
        return "Spanish"
    
    # French keywords and patterns  
    french_keywords = ['bonjour', 'merci', 's\'il vous plaît', 'oui', 'non', 'comment', 'quoi', 'où', 'quand', 'prix', 'produit']
    if any(keyword in text_lower for keyword in french_keywords):
        return "French"
    
    # German keywords
    german_keywords = ['hallo', 'danke', 'bitte', 'ja', 'nein', 'wie', 'was', 'wo', 'wann', 'preis', 'produkt']
    if any(keyword in text_lower for keyword in german_keywords):
        return "German"
    
    # Japanese romanized keywords
    japanese_keywords = ['arigatou', 'sumimasen', 'konnichiwa', 'sayonara', 'ikura', 'nani', 'doko']
    if any(keyword in text_lower for keyword in japanese_keywords):
        return "Japanese"
    
    # Default to English
    return "English"

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
                "currency": p.currency,
                "product_type": p.product_type,
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
def check_product_by_name(product_name: str) -> dict:
    """Check if a specific product exists in the database by name (case-insensitive, partial match).
    
    ALWAYS call this tool first when user asks about a specific product before proceeding with any order.
    
    Returns:
    - If found: product details including id, name, price, quantity, currency, product_type
    - If not found: status='not_found' with list of all available product names so user can choose
    """
    db = SessionLocal()
    try:
        # Case-insensitive partial match
        products = db.query(Product).filter(
            Product.name.ilike(f"%{product_name}%")
        ).all()
        
        if products:
            data = []
            for p in products:
                data.append({
                    "id": p.id,
                    "name": p.name,
                    "price": p.price,
                    "currency": p.currency,
                    "quantity": p.quantity,
                    "product_type": p.product_type,
                    "description": p.description
                })
            return {"status": "found", "data": data}
        else:
            # Product not found - return all available product names
            all_products = db.query(Product).all()
            available = [{"id": p.id, "name": p.name} for p in all_products]
            return {
                "status": "not_found",
                "message": f"Product '{product_name}' is not available in our catalog.",
                "available_products": available
            }
    finally:
        db.close()

@tool
def get_products_by_category(category: str) -> dict:
    """Get products filtered by product_type/category (e.g. skincare, cosmetics, face mask, haircare).
    
    Use this when user asks about a specific category like:
    - "show me skincare products"
    - "I want cosmetics"
    - "do you have face masks?"
    
    Returns product name and image_url for each matching product.
    Case-insensitive partial match on product_type field.
    """
    db = SessionLocal()
    try:
        # Build list of search terms to try (handle plural/singular)
        search_terms = [category]
        cat_lower = category.lower().strip()
        if cat_lower.endswith('ics'):
            search_terms.append(cat_lower[:-1])   # cosmetics -> cosmetic
        elif cat_lower.endswith('s') and len(cat_lower) > 3:
            search_terms.append(cat_lower[:-1])   # masks -> mask, products -> product
        elif not cat_lower.endswith('s'):
            search_terms.append(cat_lower + 's')  # cosmetic -> cosmetics

        products = []
        for term in search_terms:
            results = db.query(Product).filter(
                Product.product_type.ilike(f"%{term}%")
            ).all()
            products.extend(results)

        # Deduplicate by product id
        seen = set()
        unique_products = []
        for p in products:
            if p.id not in seen:
                seen.add(p.id)
                unique_products.append(p)
        products = unique_products

        if products:
            data = [
                {
                    "id": p.id,
                    "name": p.name,
                    "price": p.price,
                    "currency": p.currency,
                    "image_url": p.image_url,
                    "product_type": p.product_type
                }
                for p in products
            ]
            return {"status": "found", "category": category, "data": data}
        else:
            all_types = db.query(Product.product_type).distinct().all()
            types_list = [t[0] for t in all_types if t[0]]
            return {
                "status": "not_found",
                "message": f"No products found for category '{category}'.",
                "available_categories": types_list
            }
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
    customer_name: str,
    country: str,
    quantity: int = 1,
    user_message: str = "",
    order_source: str = "chatbot",
    type_of_order: str = "auto",
    language: str = "auto"
) -> dict:
    """Create a B2C order. Only requires product_id, customer_name, and country.
    Quantity defaults to 1 if not specified by user.
    
    Stock Validation:
    - Checks if requested quantity is available in stock
    
    Language auto-detection based on user's message content.
    
    Collect ONLY customer_name and country before creating order.
    Include user_message parameter to detect language from actual message content.
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

        # Auto-detect language if not explicitly set
        if language == "auto":
            # Use user message for language detection, fallback to customer name if message is empty
            detection_text = user_message if user_message.strip() else customer_name
            language = detect_language(detection_text)

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
            location=country,
            order_items=order_items_data,
            total_amount=total_amount,
            type_of_order="B2C",
            language=language,
            currency=product.currency,
            status="pending"
        )

        db.add(order)
        db.commit()
        db.refresh(order)

        return {
            "status": "success",
            "data": {
                "order_id": order.id,
                "product_name": product.name,
                "image_url": product.image_url,
                "unit_price": f"{unit_price} {product.currency}",
                "total_amount": f"{total_amount} {product.currency}",
                "quantity": quantity,
                "customer_name": customer_name,
                "country": country,
                "detected_language": language
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


tools = [check_product_by_name, get_products_by_category, get_products, add_product, create_order, get_orders]

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
- Basic details: name, description, price, quantity, currency
- Product classification: product_type (categories like skincare, cosmetics, haircare, etc.)
- Detailed information: ingredients, usage_instructions, suitable_age_range, image_url
- Use these details to help customers make informed decisions

Product Type Guidance:
When users ask about product types ("what type of product is this?", "what products do you have for skincare?", "show me cosmetic products"), use the product_type field to provide helpful information:
- Clearly identify the product category (skincare, cosmetics, haircare, etc.)
- Explain what the product type is used for and its benefits
- Suggest suitable use cases based on the product_type
- Example: "This is a skincare product, specifically designed for [purpose]. It's suitable for [use case]."

CATEGORY BROWSING (MANDATORY):
When user asks about a category (e.g., "skincare", "cosmetics", "face mask", "haircare", "I want skincare products", "I want to order face mask"):
1. Call get_products_by_category(category="...") — extract the category keyword from user's message
2. If status='found': display each product in this EXACT format:

   **[Product Name]**
   ![View Image]([image_url])
   Price: [price] [currency]

3. If status='not_found': politely list available_categories and ask user to choose

Image URL Formatting Rule:
- ALWAYS render image URLs as markdown images: ![View Image](url)
- NEVER show raw URLs as plain text
- Apply this to ALL product displays, not just category browsing

PRODUCT VALIDATION FLOW (MANDATORY):
When a user asks about or mentions a specific product by name (e.g., "I want to order X", "do you have X", "tell me about X"):
1. FIRST call check_product_by_name(product_name="X") to verify it exists
2. If status='found': proceed normally — show product details and continue with order flow
3. If status='not_found': respond politely, e.g.:
   "I'm sorry, we don't currently carry '[product name]' in our catalog.
   Here are the products we have available:
   [list available_products from tool response]
   Would you like to order any of these instead?"
- NEVER assume a product exists without calling check_product_by_name first
- NEVER proceed to create_order for a product that was not found

Tool Usage for Product Information:
- Use get_products() with flexible parameters based on user request:
  * names_only=True when user asks "show me product names", "what products do you have", "list product names"
  * fields_only=['name', 'price'] when user asks for specific details like "show names and prices"
  * fields_only=['name', 'product_type'] when user asks "what types of products do you have", "show product categories"
  * fields_only=['ingredients'] when user asks "what are the ingredients", "show ingredients"
  * fields_only=['name', 'product_type', 'usage_instructions'] when user asks about product types and usage
  * product_ids=[1,2,3] when user asks about specific products by ID
  * include_all=True when user asks "show all details", "full information", "everything about products"
  * Default (no parameters) returns full details for general product queries

IMPORTANT: Match the tool parameters to user intent:
- "Show me product names" / "list products" / "what products do you have" → fields_only=['name', 'image_url']
- "What's the price of products" → fields_only=['name', 'price', 'image_url']
- "What types of products do you have" → fields_only=['name', 'product_type', 'image_url']
- "Tell me about product categories" → fields_only=['name', 'product_type', 'usage_instructions', 'image_url']
- "Tell me about ingredients" → fields_only=['name', 'ingredients', 'image_url']
- "Show all product information" → include_all=True
- "Details about product 1" → product_ids=[1], include_all=True

NEVER use names_only=True — always include image_url in every product listing.

Image URL Formatting (MANDATORY):
- ALWAYS render image URLs as markdown images: ![View Image](url)
- NEVER show raw URLs as plain text
- When listing products, display EVERY product in this format:

  **[Product Name]**
  ![View Image]([image_url])

- When price is available, add: Price: [price] [currency]
- Apply this format consistently for ALL product listings

Order Processing Enhanced:
- ALWAYS ask for customer name, quantity, and country before creating any order
- Stock validation: Automatically checks if requested quantity is available in stock
  * If insufficient stock, informs customer of current availability
  * Only creates order if sufficient stock exists
- Orders automatically detect customer type (B2B vs B2C) based on:
  * Customer name (business indicators like LLC, Corp, Inc, Company, etc.)
  * Order quantity: 1-9 units = B2C, 10+ units = B2B
  * Business names automatically = B2B regardless of quantity
- Language auto-detection based on user's message content and character analysis (English, Japanese, Chinese, Arabic, Spanish, French, German)
- Country information is collected for shipping and regional business analytics
- System provides detected customer type, language, and country in response
- This helps with better customer service, shipping logistics, and business analytics

Order Handling Rules (VERY IMPORTANT — B2C):
- If a user wants to place an order, collect these details ONE AT A TIME if missing:
  1. Customer Name
  2. Country/Location
  (Quantity is automatically 1 — do NOT ask for quantity unless user specifies a different amount)

- Ask for ONLY the missing information — never ask all questions at once
- Do NOT proceed to create_order until customer_name and country are available
- When calling create_order, ALWAYS include user_message for language detection

After Order Success — respond with this SHORT format ONLY:
✅ Order Confirmed!
- **Product:** [product_name]
  ![View Image]([image_url])
- **Price:** [unit_price]
- **Customer:** [customer_name] | [country]

Thank you for your order! We'll deliver it to you soon.

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
    
    # Return the list of unique thread IDs
    return list(threads)