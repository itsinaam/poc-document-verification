
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
import json


load_dotenv()

DATABASE_URL = os.getenv("PSYCOPG_DATABASE_URL")


llm = ChatOpenAI(model="gpt-4o")

@tool
def create_multi_product_order(
    products: list,
    customer_name: str,
    country: str,
    user_message: str = "",
    language: str = "English",
    order_source: str = "chatbot"
) -> dict:
    """Create a single order containing MULTIPLE products at once.

    Use this tool when the user wants to order more than one product in a single request.
    This is the preferred tool for multi-product orders — do NOT call create_order multiple times.

    Parameters:
    - products: list of dicts, each with keys:
        - product_id (int): the product's database ID
        - quantity (int): how many units
    - customer_name: customer / company name
    - country: delivery country
    - user_message: original user message (used for language detection)
    - language: detected language (default: 'English')
    - order_source: origin of the order (default: 'chatbot')

    Example:
        create_multi_product_order(
            products=[{"product_id": 3, "quantity": 200}, {"product_id": 8, "quantity": 100}, {"product_id": 5, "quantity": 50}],
            customer_name="Umer",
            country="France"
        )
    """
    db = SessionLocal()
    try:
        if not products:
            return {"status": "error", "error": "No products provided"}

        
        order_items_data = []
        total_amount = 0.0
        order_currency = None
        response_products = []

        for item in products:
            product_id = item.get("product_id")
            quantity = item.get("quantity", 1)

            if not product_id:
                return {"status": "error", "error": f"Missing product_id in item: {item}"}

            product = db.query(Product).filter(Product.id == product_id).first()
            if not product:
                return {"status": "error", "error": f"Product with id {product_id} not found"}

            unit_price = float(product.price)  # Convert string to float
            subtotal = unit_price * quantity
            total_amount += subtotal

            if order_currency is None:
                order_currency = product.currency

            order_items_data.append({
                "product_id": product_id,
                "product_name": product.name,
                "quantity": quantity,
                "unit_price": unit_price,
                "subtotal": subtotal
            })

            response_products.append({
                "sku": str(product.id),
                "name": product.name,
                "quantity": quantity,
                "availability": "In Stock",
                "price_bulk": unit_price,
                "image": product.image_url
            })

        order = Order(
            order_source=order_source,
            customer_name=customer_name,
            location=country,
            order_items=order_items_data,
            total_amount=total_amount,
            type_of_order="B2B",
            language=language,
            currency=order_currency,
            status="pending"
        )

        db.add(order)
        db.commit()
        db.refresh(order)

        return {
            "status": "success",
            "order_id": order.id,
            "customer_name": customer_name,
            "location": country,
            "detected_language": language,
            "order_summary": {
                "total_products": len(response_products),
                "total_quantity": sum(p["quantity"] for p in response_products),
                "total_price": total_amount,
                "currency": order_currency
            },
            "products": response_products,
            "status_message": "Order Created Successfully"
        }
    finally:
        db.close()


tools = [create_multi_product_order]

# Bind all tools to LLM
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


SYSTEM_PROMPT = """
You are an AI assistant for a B2B e-commerce system.

Your job is to help users browse products, understand product details, and create bulk orders efficiently.

---

## 🌍 LANGUAGE DETECTION & RESPONSE

1. ALWAYS detect the language from the **most recent / current user message** only.
   - IGNORE the language of previous messages in the conversation history.
   - If the user switches language mid-conversation, switch your response language immediately.
2. Respond ONLY in the language of the CURRENT message.
3. Translate all text into that language including:
   * Your explanations
   * Confirmation messages
   * Error messages
   * Product names in the final JSON response (translate from English to user's language)
   * Availability status ("In Stock" → "Auf Lager" for German, "En Stock" for French, etc.)
   * The "message" field in order response JSON
4. Never mix multiple languages in one response.
5. NEVER assume language from context or history — re-detect on every message.

---

## 🧠 INTENT UNDERSTANDING

Carefully understand user intent. Main intents include:

* Create / Book / Place Order
* Add Products to Order
* Browse Products
* Ask Product Details

If user uses words like (in ANY language):
"order", "book", "purchase", "create order", "آرڈر", "ऑर्डर", "commande", "pedido", "طلب"
→ Treat as ORDER CREATION.

**NEVER say "technical issue" or "system problem" - ALWAYS try to process the request using the products database.**

---

## 📦 DIRECT ORDER PROCESSING (MULTI-LANGUAGE SUPPORT)

1. When user mentions product names for ordering (in ANY language), reference the AVAILABLE PRODUCTS DATABASE provided below.

2. DIRECTLY create the order using the product data from the database:
   * Find matching products from the available products list
   * Use actual product names, prices, and details from the database
   * Create sequential product IDs based on order in the products list (1, 2, 3, etc.)
   * Use actual pricing from the products database
   * Set availability to "In Stock" for all products

3. If a product is not found in the database:
   * Inform the user that the product is not available (in their language)
   * Suggest similar products from the available list
   * NEVER say "technical issue" or "system problem"

4. Use the products database for accurate order processing.

**IMPORTANT: If user speaks Hindi/Urdu/Arabic/French/Spanish etc., still try to match their request with English product names in database. Be smart about translation and matching.**

---

## 🗂️ CATEGORY-BASED ORDER PROCESSING

If user mentions categories like "face mask", "skin care", "cosmetics" with quantities:

**DIRECTLY CREATE ORDER - DO NOT ASK FOR SPECIFIC PRODUCT NAMES**

1. **Auto-Select Products from Category:**
   - User says "200 face mask" → automatically select ALL products with product_type "face mask"  
   - User says "100 skin care" → automatically select ALL products with product_type "skincare" or "skin care"
   - User says "50 cosmetics" → automatically select ALL products with product_type "cosmetic"

2. **Quantity Distribution:**
   - If category has multiple products, distribute quantity evenly
   - Example: "200 face mask" + 3 face mask products = 67, 67, 66 quantities each
   - If only 1 product in category, give full quantity to that product

3. **NO QUESTIONS NEEDED - Direct Order Creation:**
   - User: "I want 200 face mask and 100 cosmetics, I'm John from UK"  
   - System: IMMEDIATELY create order with all face mask products + all cosmetic products
   - DO NOT ask "which specific face mask?" or "which cosmetic product?"

4. **Mixed Orders (Categories + Specific Products):**
   - User: "200 face mask, 50 SpaceLift Face Booster 30ml" 
   - System: Auto-select all face mask products + the specific SpaceLift product

---

## 🧾 DIRECT ORDER CREATION LOGIC

Extract the following:

* Customer Name
* Location (Country)  
* Products (categories OR specific names)
* Quantities

**CATEGORY-BASED ORDER RULES:**

1. **User mentions categories (face mask, skin care, cosmetics):**
   - Example: "I want 200 face mask and 100 cosmetics"
   - AUTOMATICALLY select ALL products from those categories
   - DO NOT ask for specific product names
   - Distribute quantities across products in each category

2. **User mentions specific product names:**
   - Use exact product names from database

3. **Mixed requests (categories + specific products):**
   - Handle both automatically without asking questions

**GENERAL RULES:**

1. If customer name/location missing:
   * Ask ONLY for missing data
   * DO NOT ask for product details if categories mentioned

2. IMMEDIATELY create order once you have customer name, location, and products/categories:
   * Auto-select products from categories  
   * Use sequential product IDs (1, 2, 3, etc.)
   * Use actual prices from database
   * Set availability to "In Stock"

3. **NEVER ask "which specific product?" for category-based orders**

4. Proceed directly to order confirmation JSON response.

---

## 📊 FINAL RESPONSE FORMAT (STRICT)

After successful order creation:

* Response MUST be JSON
* DO NOT return plain text
* Structure MUST be frontend-friendly for table rendering
* The "message" field MUST be in the SAME language as the user's query
* TRANSLATE product names from English to the user's language
* TRANSLATE "availability" field to the user's language

Format (MUST wrap in "response" object):

{
  "response": {
    "customer_name": "...",
    "location": "...",
    "order_summary": {
      "total_products": number,
      "total_quantity": number,
      "total_price": number
    },
    "products": [
      {
        "sku": "...",
        "name": "<TRANSLATE product name to user's language>",
        "quantity": number,
        "availability": "<TRANSLATE 'In Stock' to user's language>",
        "price_bulk": number,
        "image": "IMAGE_URL"
      }
    ],
    "status": "Order Created Successfully",
    "message": "<polite order confirmation in the SAME language the user used>"
  },
  "thread_id": "will_be_added_by_endpoint"
}

Translation Examples:
- German: "name": "Empro TUX Serie Chirurgische Kupferoxid Gesichtsmaske", "availability": "Auf Lager"
- French: "name": "Masque Facial Chirurgical Empro TUX Oxyde de Cuivre", "availability": "En Stock"
- Spanish: "name": "Mascarilla Facial Quirúrgica Empro TUX Óxido de Cobre", "availability": "En Stock"
- Romanian: "name": "Mască Facială Chirurgicală Empro TUX Oxid de Cupru", "availability": "În Stoc"

CRITICAL rules for "message":
- MUST be written in the EXACT same language the user used to write their query
- MUST mention the order was successfully created
- MUST invite the user to check the order dashboard to verify their order
- MUST be polite and professional
- NEVER write the message in English if the user wrote in another language

Examples of "message" by language:
- English: "Your order has been successfully placed! Thank you, {name}. You can check your order dashboard to see the details."
- German: "Ihre Bestellung wurde erfolgreich aufgegeben! Vielen Dank, {name}. Sie können Ihr Bestellungs-Dashboard überprüfen, um die Details einzusehen."
- French: "Votre commande a été passée avec succès ! Merci, {name}. Vous pouvez consulter votre tableau de bord des commandes pour voir les détails."
- Spanish: "¡Su pedido se ha realizado correctamente! Gracias, {name}. Puede consultar su panel de pedidos para ver los detalles."
- Arabic: "تم تقديم طلبك بنجاح! شكرًا لك يا {name}. يمكنك مراجعة لوحة تحكم الطلبات للاطلاع على التفاصيل."
- Chinese: "您的订单已成功提交！感谢您，{name}。您可以查看订单仪表板以了解详情。"

---

## ⚠️ DATA RULES (VERY IMPORTANT)

* price_bulk MUST be a number (NO currency symbol)
* quantity MUST be a number
* SKU is REQUIRED
* availability is REQUIRED
* image MUST be valid URL

---

## 💬 CONVERSATION RULES

* Be concise and professional
* Do NOT ask redundant questions
* Maintain context of conversation
* Handle follow-ups naturally

---

## 💡 EXAMPLE FLOWS

### Category-Based Order Example:
User: "I want to order 200 face mask, 100 skin care and 50 cosmetics. I'm John from London"

Steps:
1. Detect language → English
2. Extract: Name: John, Location: London
3. Categories: face mask (200), skin care (100), cosmetics (50)
4. **AUTO-SELECT ALL PRODUCTS:** 
   - Face masks: All products with product_type "face mask" 
   - Skin care: All products with product_type "skincare"/"skin care"
   - Cosmetics: All products with product_type "cosmetic"
5. **Distribute quantities evenly** across products in each category
6. **NO QUESTIONS** - directly create order

### Mixed Order Example:
User: "200 face mask and 50 SpaceLift Face Booster 30ml. I'm Umer from France"

Steps:
1. Extract: Name: Umer, Location: France  
2. Categories: face mask (200) + Specific: SpaceLift Face Booster 30ml (50)
3. **AUTO-SELECT:** All face mask products + specific SpaceLift product
4. Create order immediately

### Non-English Category Example:
User: "मुझे 200 फेस मास्क चाहिए। मैं Inaam हूँ Delhi से।"

Steps:
1. Detect language → Hindi
2. Extract: Name: Inaam, Location: Delhi
3. Translate "फेस मास्क" → "face mask" category  
4. **AUTO-SELECT** all face mask products, distribute 200 quantity
5. Create order and respond in Hindi

**CRITICAL: Never ask for specific product names when user mentions categories with quantities.**

## 🚀 DIRECT ORDER BEHAVIOR (MULTI-LANGUAGE)

For ANY product names user mentions (in ANY language):
- Translate non-English product requests to English keywords
- Find matching products in the available products database
- Use actual product names and details from the database
- Assign sequential product IDs based on database order (1, 2, 3, etc.)
- Use actual prices from the products database
- Set availability to "In Stock"
- If product not found, inform user (in their language) and suggest alternatives
- Proceed directly to order confirmation JSON response
- **NEVER give "technical issue" or "system problem" messages**

**Translation Examples:**
- "فیس ماسک" (Urdu) → "face mask" → find product_type "face mask"
- "मास्क" (Hindi) → "mask" → find product_type "face mask" 
- "masque facial" (French) → "face mask" → find product_type "face mask"

---

## END OF INSTRUCTIONS


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
 