from sqlalchemy import Column, ForeignKey, TIMESTAMP, Integer, VARCHAR, UnicodeText
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import relationship

from db.models.base import BaseModel


class Utterance(BaseModel):
    __tablename__ = 'utterance'
    id = Column(Integer, unique=True, nullable=False, primary_key=True, autoincrement=True)
    type = Column(VARCHAR(8), nullable=False)
    text = Column(UnicodeText, nullable=False)
    date_time = Column(TIMESTAMP, nullable=False)

    active_skill = Column(VARCHAR(255), nullable=True)
    attributes = Column(JSONB, nullable=True)

    @declared_attr
    def conversation_id(cls):
        return Column(Integer, ForeignKey('conversation.id'), nullable=False)

    @declared_attr
    def conversation(cls):
        return relationship('Conversation', back_populates='utterances')
