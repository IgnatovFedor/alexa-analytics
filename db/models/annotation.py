from sqlalchemy import Column, UnicodeText, Float, Integer, TIMESTAMP, CHAR, VARCHAR, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from db.models.base import BaseModel


class Annotation(BaseModel):
    __tablename__ = 'annotation'
    id = Column(Integer, primary_key=True, autoincrement=True)

    parent_utterance_id = Column(Integer, ForeignKey('utterance.id'), nullable=False)

    parent_utterance = relationship('Utterance', order_by='Utterance.date_time',
                                    back_populates='annotations', lazy='dynamic', uselist=True)

    annotation_type = Column(UnicodeText, nullable=False)

    annotation_data = Column(JSONB, nullable=False)

    def __str__(self):
        return "%s: %s" % (self.annotation_type, self.annotation_data)
