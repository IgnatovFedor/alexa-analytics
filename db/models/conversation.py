from sqlalchemy import VARCHAR, Column, UnicodeText, Float, Integer, TIMESTAMP
from sqlalchemy.orm import relationship

from db.models.base import BaseModel


class Conversation(BaseModel):
    __tablename__ = 'conversation'
    id = Column(Integer, unique=True, nullable=False, primary_key=True, autoincrement=True)
    alexa_conversation_id = Column(VARCHAR(64), nullable=False)
    started = Column(TIMESTAMP, nullable=False)
    length = Column(Integer, nullable=False)
    feedback = Column(UnicodeText, nullable=True)
    rating = Column(Float, nullable=True)

    utterances = relationship('Utterance', order_by='Utterance.date_time', back_populates='conversation', lazy='dynamic')
