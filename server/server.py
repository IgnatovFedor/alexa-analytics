from datetime import timedelta, datetime
from logging import getLogger
from time import sleep
from typing import List

from pytz import UTC

from db.db import DBManager
from server.s3 import S3Manager

log = getLogger(__name__)


def update_utts(s3: S3Manager, db: DBManager, last_utt_time, skip_tg: bool):
    last_utt_time_with_tz = last_utt_time.replace(tzinfo=UTC) if last_utt_time is not None else None
    for content in s3.get_all_interval_logs():
        if last_utt_time is not None and content['LastModified'] < last_utt_time_with_tz:
            log.debug(f'Passing file {content["Key"]}')
            continue
        log.info(f'Adding dialogs from {content["Key"]}')
        json = s3.get_hour_log_json(content['Key'])
        db.add_hour_logs(json, skip_tg)

    log.info('utterances updated')


def update_info(s3: S3Manager, db: DBManager, last_utt_time):
    ratings = s3.get_ratings()
    log.info('ratings downloaded')
    if last_utt_time is not None:
        last_utt_time = last_utt_time - timedelta(days=2)
        ratings = ratings[ratings['Approximate Start Time'] > last_utt_time]
    db.add_ratings(ratings)
    log.info('ratings updated')

    feedbacks = s3.get_feedback()
    log.info('feedbacks downloaded')
#    if last_utt_time is not None:
#        feedbacks = feedbacks[feedbacks['conversation_start_time'] > last_utt_time]
    db.add_feedbacks(feedbacks)
    log.info('feedbacks updated')


def start_polling(s3s: List[S3Manager], db: DBManager):
    while True:
        last_utt_time = db.get_last_utterance_time()
        for s3 in s3s:
            update_utts(s3, db, last_utt_time, s3.skip_tg)
        update_info(s3s[0], db, last_utt_time)
        now = datetime.now()
        sleep_for = timedelta(hours=1) - (now - now.replace(minute=5, second=0))
        log.info(f'Started sleep for {sleep_for}')
        sleep(sleep_for.seconds)
