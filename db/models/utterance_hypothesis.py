import json
from sqlalchemy import Column, UnicodeText, Float, Integer, TIMESTAMP, CHAR, VARCHAR, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from db.models.base import BaseModel


class UtteranceHypothesis(BaseModel):
    """
    Each skill produces utterances hypotheses one of which is selected by response selector
    on each step
    """
    __tablename__ = 'utterance_hypothesis'
    id = Column(Integer, primary_key=True, autoincrement=True)

    parent_utterance_id = Column(Integer, ForeignKey('utterance.id'), nullable=False)

    parent_utterance = relationship('Utterance', order_by='Utterance.date_time',
                                    back_populates='utterance_hypotheses')

    skill_name = Column(UnicodeText, nullable=False)
    text = Column(UnicodeText, nullable=False)
    confidence = Column(Float, nullable=False)

    # all stuff that the skill has pushed out:
    other_attrs = Column(JSONB, nullable=True)

    def __str__(self):
        return "(%s:%0.2f): %s" % (self.skill_name, self.confidence, self.text)

    @property
    def pretty_attrs(self):
        """
        prints attrs with pretty indentation
        :return:
        """
        return json.dumps(self.other_attrs, ensure_ascii=False, indent=2)