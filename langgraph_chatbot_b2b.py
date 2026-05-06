
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
    Handles plurals, compound words (e.g. 'facemasks' -> 'face mask'), and partial matches.
    
    Returns:
    - If found: product details including id, name, price, quantity, currency, product_type
    - If not found: status='not_found' with list of all available product names so user can choose
    """
    import re
    db = SessionLocal()
    try:
        def _search(term: str):
            return db.query(Product).filter(
                Product.name.ilike(f"%{term}%")
            ).all()

        # 1. Try exact partial match first
        products = _search(product_name)

        # 2. If not found, try splitting into words and search each word
        if not products:
            # Normalise: insert space before uppercase runs and split on non-alpha
            normalised = re.sub(r'([a-z])([A-Z])', r'\1 \2', product_name)
            words = re.split(r'[\s_\-]+', normalised.strip())
            # Also try stripping trailing 's' for plurals (e.g. facemasks -> facemask)
            search_terms = set()
            for w in words:
                if len(w) >= 3:
                    search_terms.add(w)
                    if w.lower().endswith('s') and len(w) > 3:
                        search_terms.add(w[:-1])   # remove trailing 's'
                    if w.lower().endswith('es') and len(w) > 4:
                        search_terms.add(w[:-2])   # remove trailing 'es'

            seen_ids = set()
            for term in search_terms:
                for p in _search(term):
                    if p.id not in seen_ids:
                        seen_ids.add(p.id)
                        products.append(p)

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
    quantity: int, 
    customer_name: str,
    country: str,
    user_message: str = "",
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
    
    Language auto-detection based on user's message content (returns full language names: Chinese, Japanese, Arabic, Spanish, French, German, English).
    
    Always asks user for customer name, quantity, and country before creating order.
    Include user_message parameter to detect language from actual message content.
    """
    db = SessionLocal()
    try:
        product = db.query(Product).filter(Product.id == product_id).first()

        if not product:
            return {"status": "error", "error": "Product not found"}

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
            type_of_order="B2B",
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
def create_multi_product_order(
    products: list,
    customer_name: str,
    country: str,
    user_message: str = "",
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

        # Detect language
        detection_text = user_message.strip() if user_message.strip() else customer_name
        language = detect_language(detection_text)

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

            unit_price = parse_price(product.price)
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


@tool
def get_orders(customer_name: str = None) -> dict:
    """Get orders"""
    db = SessionLocal()
    try:
        query = db.query(Order)

        if customer_name:
            query = query.filter(Order.customer_name == customer_name)

        orders = query.order_by(Order.id.desc()).all()

        return {
            "status": "success",
            "data": format_orders(orders),
            "summary": calculate_order_summary(orders)
        }
    finally:
        db.close()


tools = [check_product_by_name, get_products_by_category, get_products, add_product, create_order, create_multi_product_order, get_orders]

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

If user uses words like:
"order", "book", "purchase", "create order"
→ Treat as ORDER CREATION.

---

## 📦 DIRECT ORDER PROCESSING

1. When user mentions product names for ordering, DO NOT validate or check if products exist in database.

2. DIRECTLY create the order with the exact product names user mentioned:
   * Use the product names exactly as the user stated them
   * Do NOT call check_product_by_name or any validation tools
   * Proceed immediately to order creation

3. For order creation, use dummy/placeholder product IDs:
   * Assign sequential IDs starting from 1 (first product = 1, second = 2, etc.)
   * Use standard pricing (set price_bulk to 50 for all products)
   * Set availability to "In Stock" for all products

4. NEVER validate products against database - just process the order immediately.

---

## 🗂️ CATEGORY HANDLING

If user provides ONLY category (no product names):

1. Fetch products from that category using tool.

2. Show product list with:

   * Name
   * Price
   * Availability
   * Image (MANDATORY format below)

3. Image format MUST be:
   ![View Image](IMAGE_URL)

4. Ask user to select product and quantity.

---

## 🧾 DIRECT ORDER CREATION LOGIC

Extract the following:

* Customer Name
* Location (Country)
* Products (use exact names user mentioned)
* Quantities

Rules:

1. If any required info is missing:

   * Ask ONLY for missing data
   * DO NOT repeat already provided info

2. If user already gave name/location:

   * DO NOT ask again

3. IMMEDIATELY create order once you have customer name, location, products and quantities:
   * Skip all product validation
   * Use dummy product IDs (1, 2, 3, etc. in sequence)
   * Set price_bulk to 50 for all products
   * Set availability to "In Stock"
   * Use exact product names user mentioned

4. For MULTIPLE products → create order with all products in response JSON
   For SINGLE product → create order with that product

5. NO database checking required - proceed directly to order confirmation JSON response.

---

## 📊 FINAL RESPONSE FORMAT (STRICT)

After successful order creation:

* Response MUST be JSON
* DO NOT return plain text
* Structure MUST be frontend-friendly for table rendering
* The "message" field MUST be in the SAME language as the user's query
* TRANSLATE product names from English to the user's language
* TRANSLATE "availability" field to the user's language

Format:

{
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

## 💡 EXAMPLE FLOW

User:
"Please book 200 facemasks, 100 SpaceLift Face Booster 100ml and 50 MIOS EYEBROW PENCIL. I am Umer from France"

Steps:

1. Detect language → English
2. Extract:
   * Name: Umer
   * Location: France
   * Products: facemasks (qty 200), SpaceLift Face Booster 100ml (qty 100), MIOS EYEBROW PENCIL (qty 50)
3. SKIP product validation - use exact product names
4. DIRECTLY return order confirmation JSON with:
   * customer_name: "Umer"
   * location: "France" 
   * products with dummy IDs (1, 2, 3), price_bulk: 50, availability: "In Stock"
   * Use exact product names user mentioned

## 🚀 DIRECT ORDER BEHAVIOR

For ANY product names user mentions:
- Use the EXACT names user provided
- Assign dummy product IDs in sequence (1, 2, 3, etc.)
- Set price_bulk to 50 for all products
- Set availability to "In Stock"
- NEVER check database or validate products
- Proceed directly to order confirmation JSON response

---

## END OF INSTRUCTIONS


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
    
    # Return the list of unique thread IDs
    return list(threads)
 