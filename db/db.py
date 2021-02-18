import time
from datetime import datetime, timedelta
from logging import getLogger

from pandas import DataFrame
from sqlalchemy import create_engine, text
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

def drop_all_tables(session):
    engine = session.get_bind()
    Base.metadata.create_all(engine)
    Base.metadata.drop_all(bind=engine)
    # session.commit()
    log.info(f'All tables are deleted from DB!')


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
            conv_id = conversation['id']
            conversation['human']['user_external_id'] = conversation['human'].get('user_telegram_id', '')
            try:
                conv: Conversation = self._session.query(Conversation).filter_by(id=conv_id).one()
                conv.raw_utterances = conversation['utterances']
                conversation['utterances'] = conversation['utterances'][conv.utterances.count():]
                conv.length += len(conversation['utterances'])
                conv.date_finish = finish
                conv.human = conversation['human']
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
                    mgid=conv_id,
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
        log.info(f'Total ratings to add: {len(df.index)}')
        for i, (j, row) in enumerate(df.iterrows()):
            if i % 10 == 0 and i != 0:
                log.info(f'Ratings: {i}')
                self._session.commit()
            try:
                conversation = self._session.query(Utterance).filter(text(
                    f"attributes->>'conversation_id' = '{row['Conversation ID']}'")).one().conversation
                if conversation.rating is not None:
                    continue
                conversation.rating = row['Rating']
            except MultipleResultsFound:
                #TODO: make proper error handling
                log.error(f'Multiple conversations found for ID: {row["Conversation ID"]}')
            except NoResultFound:
                log.info(f'NoResultFound for {row["Conversation ID"]}')
        self._session.commit()

    def add_feedbacks(self, df: DataFrame):
        log.info(f'Total feedbacks to add: {len(df.index)}')
        for i, (j, row) in enumerate(df.iterrows()):
            if i % 10 == 0:
                log.info(f'Feedbacks: {i}')
                self._session.commit()
            try:
                conversation = self._session.query(Utterance).filter(text(
                    f"attributes->>'conversation_id' = '{row['conversation_id']}'")).one().conversation
                if conversation.feedback is not None:
                    continue
                conversation.rating = conversation.rating or row['rating']
                conversation.feedback = row['feedback']
            except MultipleResultsFound:
                # TODO: make proper error handling
                log.error(f'Multiple conversations found for ID: {row["Conversation ID"]}')
            except NoResultFound:
                pass
        self._session.commit()

    def get_last_utterance_time(self):
        utterance = self._session.query(Utterance).order_by(Utterance.date_time.desc()).first()
        if utterance:
            return utterance.date_time
        else:
            return None


