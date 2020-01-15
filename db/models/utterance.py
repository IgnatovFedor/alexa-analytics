from sqlalchemy import Column, ForeignKey, TIMESTAMP, Integer, VARCHAR, UnicodeText, CHAR
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from db.models.base import BaseModel


class Utterance(BaseModel):
    __tablename__ = 'utterance'
    id = Column(Integer, unique=True, nullable=False, primary_key=True, autoincrement=True)

    text = Column(UnicodeText, nullable=False)
    date_time = Column(TIMESTAMP, nullable=False)
    active_skill = Column(VARCHAR(255), nullable=True)
    attributes = Column(JSONB, nullable=True)

    conversation_id = Column(CHAR(24), ForeignKey('conversation.id'), nullable=False)

    conversation = relationship('Conversation', back_populates='utterances')
