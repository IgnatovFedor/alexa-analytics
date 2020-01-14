from sqlalchemy import Column, Integer, ForeignKey, VARCHAR
from sqlalchemy.dialects.postgresql import JSONB

from db.models.base import BaseModel


class ConversationPeer(BaseModel):
    __tablename__ = 'conversation_peer'

    id = Column(Integer, unique=True, nullable=False, primary_key=True, autoincrement=True)
    persona = Column(JSONB, nullable=False)
    attributes = Column(JSONB, nullable=False)
    type = Column(VARCHAR(8), nullable=False)

    # Human attrs
    user_telegram_id = Column(VARCHAR(512), nullable=True)
    profile = Column(JSONB, nullable=True)

    conversation_id = Column(Integer, ForeignKey('conversation.id'), nullable=False)
