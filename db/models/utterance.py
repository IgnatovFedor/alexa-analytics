from sqlalchemy import Column, ForeignKey, TIMESTAMP, Integer, VARCHAR, UnicodeText, CHAR, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship

from db.models.base import BaseModel
from db.models.conversation import Conversation

class Utterance(BaseModel):
    __tablename__ = 'utterance'
    id = Column(Integer, unique=True, nullable=False, primary_key=True, autoincrement=True)

    text = Column(UnicodeText, nullable=False)
    date_time = Column(TIMESTAMP, nullable=False)
    active_skill = Column(VARCHAR(255), nullable=True)
    attributes = Column(JSONB, nullable=True)

    conversation_id = Column(CHAR(34), ForeignKey('conversation.id'), nullable=False)

    conversation = relationship('Conversation', back_populates='utterances')

    annotations = relationship("Annotation", back_populates="parent_utterance")

    utterance_hypotheses = relationship("UtteranceHypothesis", back_populates="parent_utterance")

    @hybrid_property
    def rating(self):
        return self.conversation.rating

    @rating.expression
    def rating(cls):
        return select([Conversation.rating]).where(cls.conversation_id == Conversation.id)