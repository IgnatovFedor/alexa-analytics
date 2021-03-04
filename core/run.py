import argparse
import json
import os

import pandas as pd

from admin.admin import start_admin
from db.db import DBManager, get_session
from server.s3 import S3Manager
from server.server import start_polling, update_info

parser = argparse.ArgumentParser()

parser.add_argument('mode', help='select a routine', type=str, choices={'server', 'poller', 'dpa_dumper', 'drop_tables',
                                                                        'update_info'})
parser.add_argument('-p', '--port', help='select admin port', type=int, default=5000)
parser.add_argument('-ac', '--amazon-container', help='http of container to get additional info', type=str)


def verify_config(config):
    bad_keys = [k for k, v in config.items() if v == '']
    print(bad_keys)
    if bad_keys:
        raise ValueError(f'Following parameters at config file are empty: {", ".join(bad_keys)}')


def main():
    args = parser.parse_args()
    # TODO: make proper path handling
    with open('core/config.json') as config_file:
        config = json.load(config_file)
    db_config = config['DB']
    db_config['user'] = db_config.get('user') or os.getenv('DB_USER')
    db_config['password'] = db_config.get('password') or os.getenv('DB_PASSWORD')
    db_config['host'] = db_config.get('host') or os.getenv('DB_HOST')
    db_config['dbname'] = db_config.get('dbname') or os.getenv('DB_NAME')
    verify_config(db_config)
    session = get_session(db_config['user'], db_config['password'], db_config['host'], db_config['dbname'])

    if args.mode == 'poller':
        db = DBManager(session)
        s3_configs = config['S3']
        s3s = []
        for s3_config in s3_configs:
            verify_config(s3_config)
            s3s.append(S3Manager(s3_config['key'],
                                 s3_config['secret'],
                                 s3_config['dialogs-bucket'],
                                 s3_config['stats-bucket'],
                                 s3_config['team-id'],
                                 s3_config['skip-tg']))
        start_polling(s3s, db)

    if args.mode == 'server':
        admin = config['admin']
        admin['user'] = admin.get('user') or os.getenv('ADMIN_USER')
        admin['password'] = admin.get('password') or os.getenv('ADMIN_PASSWORD')
        start_admin(session, admin['user'], admin['password'], args.port, args.amazon_container)

    if args.mode == 'dpa_dumper':
        from server.dump_new_dialogs_from_dpagent import dump_new_dialogs
        dump_new_dialogs(session, dpagent_base_url=args.amazon_container)

    if args.mode == 'update_info':
        s3_config = config['S3'][0]
        aws_access_key_id = s3_config['key'] or os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = s3_config['secret'] or os.getenv('AWS_SECRET_ACCESS_KEY')
        dialog_dumps_bucket = s3_config['dialogs-bucket'] or os.getenv('DIALOG_DUMPS_BUCKET')
        stats_bucket = s3_config['stats-bucket'] or os.getenv('STATS_BUCKET')
        team_id = s3_config['team-id'] or os.getenv('TEAM_ID')
        print((aws_access_key_id,
                           aws_secret_access_key,
                           dialog_dumps_bucket,
                           stats_bucket,
                           team_id,
                           True))
        if aws_access_key_id:
            s3 = S3Manager(aws_access_key_id,
                           aws_secret_access_key,
                           dialog_dumps_bucket,
                           stats_bucket,
                           team_id,
                           True)
            db = DBManager(session)
            last_utt_time = db.get_last_utterance_time()
            try:
                update_info(s3, db, last_utt_time)
            except pd.errors.EmptyDataError as err:
                print(repr(err))

    if args.mode == "drop_tables":
        from db.db import drop_all_tables
        drop_all_tables(session)


if __name__ == "__main__":
    main()
