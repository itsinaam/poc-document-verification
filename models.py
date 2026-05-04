from sqlalchemy import create_engine, Column, Integer, String,Text, DateTime, JSON, DECIMAL,LargeBinary
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func, text
import datetime

Base = declarative_base()

class DocumentAnalysis(Base):
    __tablename__ = "travel_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=True)
    date = Column(String(100), nullable=True)
    is_traveled = Column(String(10), nullable=True)

    confidence_score = Column(String(10), nullable=True)
    flight_name = Column(String(50), nullable=True)
    seat_number = Column(String(50), nullable=True)
    from_location = Column(String(255), nullable=True)
    to_location = Column(String(255), nullable=True)
    status = Column(String(30), nullable=False)
    error_message = Column(String(255), nullable=True) 
    file_path = Column(String(255), nullable=True)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class Product(Base):
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    price = Column(String(255), nullable=False)
    quantity = Column(Integer, default=0)
    ingredients = Column(Text)
    usage_instructions = Column(Text)
    suitable_age_range = Column(String(50))
    image_url = Column(String(255))
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class Order(Base):
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True, index=True)
    order_source = Column(String(255))
    customer_name = Column(String(255))
    location = Column(String(255))
    order_items = Column(JSON, nullable=False)
    total_amount = Column(String(255), nullable=False)
    type_of_order = Column(String(50))
    language = Column(String(50))
    status = Column(String(50), default='pending')
    order_date = Column(DateTime, default=func.now())
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class History(Base):
    __tablename__ = "history"
    
    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=func.now())