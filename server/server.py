from datetime import timedelta
from logging import getLogger
from time import sleep
from typing import List

from pytz import UTC

from db.db import DBManager
from server.s3 import S3Manager

log = getLogger(__name__)


def update_db(s3: S3Manager, db: DBManager, last_utt_time, skip_tg: bool):
    last_utt_time_with_tz = last_utt_time.replace(tzinfo=UTC) if last_utt_time is not None else None
    for content in s3.get_all_interval_logs():
        if last_utt_time is not None and content['LastModified'] < last_utt_time_with_tz:
            log.debug(f'Passing file {content["Key"]}')
            continue
        log.info(f'Adding dialogs from {content["Key"]}')
        json = s3.get_hour_log_json(content['Key'])
        db.add_hour_logs(json, skip_tg)

    log.info('utterances updated')
    ratings = s3.get_ratings()
    if last_utt_time is not None:
        last_utt_time = last_utt_time - timedelta(days=2)
        ratings = ratings[ratings['Approximate Start Time'] > last_utt_time]
    db.add_ratings(ratings)
    log.info('ratings updated')

    feedbacks = s3.get_feedback()
#    if last_utt_time is not None:
#        feedbacks = feedbacks[feedbacks['conversation_start_time'] > last_utt_time]
    db.add_feedbacks(feedbacks)
    log.info('feedbacks updated')


def start_polling(s3s: List[S3Manager], db: DBManager):
    # TODO: make better update timeout (to update exactly after new hour)
    while True:
        last_utt_time = db.get_last_utterance_time()
        for s3 in s3s:
            update_db(s3, db, last_utt_time, s3.skip_tg)
        sleep(3600)
