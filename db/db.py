import time
from datetime import datetime, timedelta
from logging import getLogger

from pandas import DataFrame
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.exc import NoResultFound, MultipleResultsFound
from sqlalchemy.orm.session import Session

from db.models import Base, Conversation, Utterance

log = getLogger(__name__)


def get_session(user: str, password: str, host: str, dbname: str) -> Session:
    db_uri = f'postgresql://{user}:{password}@{host}/{dbname}'
    engine = create_engine(db_uri)
    Base.metadata.create_all(engine)
    session_maker = sessionmaker(bind=engine)
    return session_maker()


class DBManager:
    def __init__(self, session: Session):
        self._session = session

    def add_hour_logs(self, dblogs: list, skip_tg: bool) -> None:
        conversations_added = 0
        for conversation in dblogs:
            if skip_tg and conversation['utterances'][0].get('attributes', {}).get('conversation_id') is None:
                continue
            start = self._parse_time(conversation['date_start'])
            finish = self._parse_time(conversation['date_finish'])
            if finish - start > timedelta(hours=6):
                log.info(f'Conversation {conversation["id"]} duration is greater than 6 hours. Countinue...')
                continue
            conv_id = conversation['id'] + str(int(time.mktime(start.timetuple())))
            try:
                conv: Conversation = self._session.query(Conversation).filter_by(id=conv_id).one()
                conversation['utterances'] = conversation['utterances'][conv.utterances.count():]
                conv.length += len(conversation['utterances'])
                conv.date_finish = finish
                conv.raw_utterances = conversation['utterances']
            except NoResultFound:
                conversation_id = None
                for utter in conversation['utterances']:
                    if 'attributes' in utter:
                        attrs = utter['attributes']
                        if 'conversation_id' in attrs:
                            conversation_id = attrs['conversation_id']
                            break
                conv = Conversation(
                    id=conv_id,
                    mgid=conversation['id'],
                    date_start=start,
                    date_finish=finish,
                    human=conversation['human'],
                    bot=conversation['bot'],
                    length=len(conversation['utterances']),
                    amazon_conv_id=conversation_id,
                    raw_utterances=conversation['utterances']
                )

            except MultipleResultsFound:
                log.error(f"{conv_id} is already in conversations multiple times")
                continue

            for utterance in conversation['utterances']:
                utt = Utterance(text=utterance['text'],
                                date_time=self._parse_time(utterance['date_time']),
                                active_skill=utterance.get('active_skill'),
                                attributes=utterance.get('attributes'),
                                conversation_id=conv_id)
                self._session.add(utt)

            self._session.add(conv)
            conversations_added += 1

        self._session.commit()
        log.info(f'Successfully added {conversations_added} conversations')

    @staticmethod
    def _parse_time(time_str: str):
        try:
            time = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            time = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
        return time

    def add_ratings(self, df: DataFrame):
        for i, row in df.iterrows():
            try:
                conversation = self._session.query(Conversation).filter_by(amazon_conv_id=row['Conversation ID']).one()
                conversation.rating = row['Rating'].replace('*', '')
            except MultipleResultsFound:
                #TODO: make proper error handling
                log.error(f'Multiple conversations found for ID: {row["Conversation ID"]}')
            except NoResultFound:
                pass
            if i%100 == 0:
                log.info(f'Ratings: {i}')
                self._session.commit()
        self._session.commit()

    def add_feedbacks(self, df: DataFrame):
        for i, row in df.iterrows():
            try:
                conversation = self._session.query(Conversation).filter_by(amazon_conv_id=row['conversation_id']).one()
                conversation.rating = conversation.rating or row['rating']
                conversation.feedback = row['feedback']
            except MultipleResultsFound:
                # TODO: make proper error handling
                log.error(f'Multiple conversations found for ID: {row["Conversation ID"]}')
            except NoResultFound:
                pass
            if i%100 == 0:
                log.info(f'Feedbacks: {i}')
                self._session.commit()
        self._session.commit()

    def get_last_utterance_time(self):
        utterance = self._session.query(Utterance).order_by(Utterance.date_time.desc()).first()
        if utterance:
            return utterance.date_time
        else:
            return None
