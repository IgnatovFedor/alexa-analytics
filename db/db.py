from datetime import datetime

from pandas import read_csv, DataFrame
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.exc import NoResultFound, MultipleResultsFound

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
            conv_id = conversation['utterances'][0]['attributes'].get('conversation_id')
            if conv_id is None:
                print('Conversation ID is None')
                continue
            try:
                conv = self._session.query(Conversation).filter_by(alexa_conversation_id=conv_id).one()
            except NoResultFound:
                pass
            except MultipleResultsFound:
                print(f"{conv_id} is already in conversations")
                continue
            else:
                # TODO: make proper warning
                print(f"{conv_id} is already in conversations")
                continue
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

    def add_ratings(self, df: DataFrame):
        for _, row in df.iterrows():
            try:
                conversation = self._session.query(Conversation).filter_by(
                    alexa_conversation_id=row['Conversation ID']).one()
                conversation.rating = row['Rating']
            except MultipleResultsFound as e:
                #TODO: make proper error handling
                print(e)
            except NoResultFound:
                pass
        self._session.commit()

    def add_feedbacks(self, df: DataFrame):
        for _, row in df.iterrows():
            try:
                conversation = self._session.query(Conversation).filter_by(
                    alexa_conversation_id=row['conversation_id']).one()
                conversation.rating = conversation.rating or row['rating']
                conversation.feedback = row['feedback']
            except MultipleResultsFound as e:
                # TODO: make proper error handling
                print(e)
            except NoResultFound:
                pass
        self._session.commit()

    def get_last_utterance_time(self):
        utterance = self._session.query(Utterance).order_by(Utterance.date_time.desc()).first()
        if utterance:
            return utterance.date_time
        else:
            return None
