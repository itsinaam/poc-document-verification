from sqlalchemy import Column, Integer, String, DateTime, Text, create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
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
