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
    category = Column(String(100))
    description = Column(Text)
    price = Column(DECIMAL(10, 2), nullable=False)
    quantity = Column(Integer, default=0)
    brand = Column(String(100))
    ingredients = Column(Text)
    usage_instructions = Column(Text)
    skin_type = Column(String(50))
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class Order(Base):
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_name = Column(String(255))
    order_items = Column(JSON, nullable=False)
    total_amount = Column(DECIMAL(10, 2))
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