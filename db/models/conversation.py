from sqlalchemy import Column, UnicodeText, Float, Integer, TIMESTAMP, CHAR, VARCHAR
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship

from db.models.base import BaseModel


class Conversation(BaseModel):
    __tablename__ = 'conversation'
    id = Column(CHAR(34), unique=True, nullable=False, primary_key=True)
    mgid = Column(CHAR(24), nullable=False)

    date_start = Column(TIMESTAMP, nullable=False)
    date_finish = Column(TIMESTAMP, nullable=False)
    human = Column(JSONB, nullable=False)
    bot = Column(JSONB, nullable=False)
    raw_utterances = Column(JSONB, nullable=False)

    length = Column(Integer, nullable=False)
    feedback = Column(UnicodeText, nullable=True)
    rating = Column(Float, nullable=True)

    amazon_conv_id = Column(VARCHAR(64), nullable=True)

    utterances = relationship('Utterance', order_by='Utterance.date_time', back_populates='conversation',
                              lazy='dynamic')

    # TODO: Make tg_id searchable with placing users and bots to different table
    @hybrid_property
    def tg_id(self):
        return self.human['user_telegram_id']
