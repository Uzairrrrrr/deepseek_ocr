"""
Database utility - Fix for duplicate flyer_id issues
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()


class Flyer(Base):
    """Flyer table"""
    __tablename__ = 'flyers'
    
    id = Column(Integer, primary_key=True)
    flyer_id = Column(String(100), unique=True, nullable=False)
    filename = Column(String(255))
    total_offers = Column(Integer)
    processing_time_ms = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    offers = relationship("Offer", back_populates="flyer", cascade="all, delete-orphan")


class Offer(Base):
    """Offer table"""
    __tablename__ = 'offers'
    
    id = Column(Integer, primary_key=True)
    flyer_id = Column(Integer, ForeignKey('flyers.id'))
    offer_id = Column(String(100), unique=True, nullable=False)
    image_path = Column(String(500))
    
    x = Column(Integer)
    y = Column(Integer)
    width = Column(Integer)
    height = Column(Integer)
    
    full_text = Column(Text)
    product_title = Column(String(500))
    original_price = Column(String(50))
    discounted_price = Column(String(50))
    discount_percentage = Column(String(20))
    promotional_text = Column(Text)
    
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    flyer = relationship("Flyer", back_populates="offers")


DATABASE_URL = "sqlite:///flyer_detection.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)


def save_flyer_results(flyer_data: dict, offers_data: list):
    """
    Save flyer and offers to database with upsert logic
    """
    db = SessionLocal()
    try:
        flyer_id_str = flyer_data['flyer_id']
        
        # Check if flyer already exists
        existing_flyer = db.query(Flyer).filter(Flyer.flyer_id == flyer_id_str).first()
        
        if existing_flyer:
            # UPDATE existing flyer
            logger.info(f"Updating existing flyer: {flyer_id_str}")
            existing_flyer.filename = flyer_data.get('filename')
            existing_flyer.total_offers = flyer_data['total_offers']
            existing_flyer.processing_time_ms = flyer_data['processing_time_ms']
            existing_flyer.updated_at = datetime.utcnow()
            
            # Delete old offers
            db.query(Offer).filter(Offer.flyer_id == existing_flyer.id).delete()
            
            flyer = existing_flyer
        else:
            # INSERT new flyer
            logger.info(f"Creating new flyer: {flyer_id_str}")
            flyer = Flyer(
                flyer_id=flyer_id_str,
                filename=flyer_data.get('filename'),
                total_offers=flyer_data['total_offers'],
                processing_time_ms=flyer_data['processing_time_ms']
            )
            db.add(flyer)
        
        db.flush()
        
        # Insert new offers
        for offer_data in offers_data:
            offer = Offer(
                flyer_id=flyer.id,
                offer_id=offer_data['offer_id'],
                image_path=offer_data['image_path'],
                x=offer_data['position']['x'],
                y=offer_data['position']['y'],
                width=offer_data['position']['width'],
                height=offer_data['position']['height'],
                full_text=offer_data['extracted_text']['full_text'],
                product_title=offer_data['extracted_text'].get('product_title'),
                original_price=offer_data['extracted_text'].get('original_price'),
                discounted_price=offer_data['extracted_text'].get('discounted_price'),
                discount_percentage=offer_data['extracted_text'].get('discount_percentage'),
                promotional_text=offer_data['extracted_text'].get('promotional_text'),
                confidence=offer_data['confidence']
            )
            db.add(offer)
        
        db.commit()
        logger.info(f"✓ Saved flyer {flyer_id_str} with {len(offers_data)} offers")
        return True
        
    except Exception as e:
        db.rollback()
        logger.error(f"Database save error: {e}")
        raise e
    finally:
        db.close()


def get_flyer_by_id(flyer_id: str):
    """Retrieve flyer and its offers"""
    db = SessionLocal()
    try:
        flyer = db.query(Flyer).filter(Flyer.flyer_id == flyer_id).first()
        return flyer
    finally:
        db.close()


def delete_flyer(flyer_id: str):
    """Delete a flyer and all its offers"""
    db = SessionLocal()
    try:
        flyer = db.query(Flyer).filter(Flyer.flyer_id == flyer_id).first()
        if flyer:
            db.delete(flyer)
            db.commit()
            logger.info(f"Deleted flyer: {flyer_id}")
            return True
        return False
    finally:
        db.close()


# Quick fix script
if __name__ == "__main__":
    print("Database Utility")
    print("================")
    print(f"Database: {DATABASE_URL}")
    
    db = SessionLocal()
    
    # Show all flyers
    flyers = db.query(Flyer).all()
    print(f"\nCurrent flyers: {len(flyers)}")
    for f in flyers:
        print(f"  - {f.flyer_id}: {f.total_offers} offers")
    
    # Option to clean
    if input("\nDelete all flyers? (yes/no): ").lower() == 'yes':
        db.query(Flyer).delete()
        db.commit()
        print("✓ All flyers deleted")
    
    db.close()