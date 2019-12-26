from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json

from db.models import Base, ConversationPeer, Conversation, Utterance


class DBManager:
    def __init__(self, user: str, password: str, host: str, dbname: str):
        db_uri = f'postgresql://{user}:{password}@{host}/{dbname}'
        engine = create_engine(db_uri)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        self._session = Session()

    def add_hour_logs(self, dblogs: list):
        for conversation in dblogs:
            conv = Conversation(alexa_conversation_id=conversation['utterances'][0]['attributes']['conversation_id'])
            self._session.add(conv)
            self._session.commit()
            for utterance in conversation['utterances']:
                try:
                    timestamp = datetime.strptime(utterance['date_time'], '%Y-%m-%d %H:%M:%S.%f')
                except ValueError:
                    timestamp = datetime.strptime(utterance['date_time'], '%Y-%m-%d %H:%M:%S')
                utt = Utterance(type='bot' if 'active_skill' in utterance else 'human',
                                active_skill=utterance.get('active_skill'),
                                text=utterance['text'],
                                date_time=timestamp,
                                attributes=utterance.get('attributes'),
                                conversation_id=conv.id)
                self._session.add(utt)
            for peer_name in ['bot', 'human']:
                peer = ConversationPeer(
                    type=peer_name,
                    persona=conversation[peer_name]['persona'],
                    attributes=conversation[peer_name]['attributes'],
                    user_telegram_id=conversation[peer_name].get('user_telegram_id'),
                    profile=conversation[peer_name].get('profile'),
                    conversation_id=conv.id
                )
                self._session.add(peer)
            self._session.commit()
