
from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from dotenv import load_dotenv
import os , json
import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from langchain_core.tools import tool
from database import SessionLocal
from models import Product, Order


load_dotenv()

DATABASE_URL = os.getenv("PSYCOPG_DATABASE_URL")


llm = ChatOpenAI(model="gpt-4o")


@tool
def create_order(
    customer_name: str,
    product_name : str,
    country: str,
    currency: str,
    quantity: int,
    language: str,
    image_url: str, 
    price: float
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
      

    
        unit_price = price
        total_amount = unit_price * quantity

        # Create proper order_items structure
        order_items_data = [{
            "product_name": product_name,
            "quantity": quantity,
            "unit_price": unit_price,
            "subtotal": total_amount
        }]

        order = Order(
            order_source="chatbot",
            customer_name=customer_name,
            location=country,
            order_items=order_items_data,
            total_amount=total_amount,
            type_of_order="B2C",
            language=language,
            currency=currency,
            status="pending"
        )

        db.add(order)
        db.commit()
        db.refresh(order)

        return {
            "status": "success",
            "data": {
                "order_id": order.id,
                "product_name": product_name,
                "image_url": image_url,
                "unit_price": f"{unit_price} {currency}",
                "total_amount": f"{total_amount} {currency}",
                "quantity": quantity,
                "customer_name": customer_name,
                "country": country,
                "detected_language": language
            }
        }
    finally:
        db.close()



tools = [ create_order]

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
You are a B2C sales assistant chatbot for a global cosmetics and skincare brand. Your main goal is to help users discover products, answer their questions, and assist them in placing orders smoothly.


Order Handling Rules (VERY IMPORTANT — B2C):
- If a user wants to place an order, collect these details ONE AT A TIME if missing:
    1. Customer Name
    2. Country/Location
    (Quantity is automatically 1 — do NOT ask for quantity unless user specifies a different amount)

- Ask for ONLY the missing information — never ask all questions at once
- Do NOT proceed to create_order until customer_name and country are available
- When calling create_order:
    - ALWAYS include the original user_message for language detection.
    - You MUST auto-detect the language from the user's order message using langchain's built-in capabilities or OpenAI's system if available.
    - NEVER set the language to "en", "na", "not available", "puchra", "english", or "en na aya". If you can't detect the language, default to "English".
    - If the user is speaking German, set language: "German". If Urdu, set: "Urdu". For French, set: "French". For Spanish: "Spanish". Etc.
    - The language value must be the full language name in English (e.g., "German", "Spanish", "Urdu", "English").
    - NEVER use short codes like "en"/"ur"/etc. Only proper language names.

PRODUCT LISTING RULE:
Whenever you list available products as options or examples for the user, ALWAYS show each product's name and its image using this format:
**Product:** [product_name]
![View Image]([image_url])

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

def load_products_data():
    """Load products data from products_rows.json file"""
    try:
        with open('products_rows.json', 'r', encoding='utf-8') as f:
            products = json.load(f)
        return products
    except Exception as e:
        print(f"Error loading products data: {e}")
        return []

def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        # Load products data and include it in the system message
        products_data = load_products_data()
        enhanced_system_prompt = SYSTEM_PROMPT + f"""

---

## 📋 AVAILABLE PRODUCTS DATABASE

Here are all the products available in our inventory. Use this data to create accurate orders:

{json.dumps(products_data, indent=2, ensure_ascii=False)}

**IMPORTANT ORDER CREATION RULES:**

1. When user mentions product names, find the matching product from the above list
2. Use the actual product data from above (name, price, currency, product_type, etc.)
3. For product_id in create_multi_product_order tool, create a sequential ID based on the order in the list (starting from 1)
4. Use the actual price from the products data, not dummy price of 50
5. Use the actual product names and details from the above list
6. If user mentions a product not in the list, inform them it's not available

**How to match products (WORKS IN ANY LANGUAGE):**
- User says "face mask" / "मास्क" / "masque" / "máscara" → look for products with product_type "face mask"
- User says "SpaceLift" → look for products containing "SpaceLift" in name (brand names are universal)
- User says "eyebrow pencil" / "आईब्रो पेंसिल" / "crayon sourcils" / "lápiz de cejas" → look for products containing "eyebrow" or "brow" in name
- User says specific product names in ANY language → try to match with English product names in database
- For non-English product requests, translate the request to English keywords and then match

**CRITICAL: ALWAYS try to match products even if user speaks in Hindi, Urdu, French, Spanish, etc.**
- Translate user's product request to English keywords
- Then search in the English product database
- If no match found, suggest available products
- NEVER say "technical issue" - always try to process the order

---

"""
        messages = [SystemMessage(content=enhanced_system_prompt)] + messages
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