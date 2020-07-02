import argparse
import json

from admin.admin import start_admin
from db.db import DBManager, get_session
from server.s3 import S3Manager
from server.server import start_polling

parser = argparse.ArgumentParser()

parser.add_argument('mode', help='select a mode: server or poller', type=str, choices={'server', 'poller', 'dpa_dumper'})
parser.add_argument('-p', '--port', help='select admin port', type=int, default=5000)
parser.add_argument('-ac', '--amazon-container', help='http of container to get additional info', type=str)


def verify_config(config):
    bad_keys = [k for k, v in config.items() if v == '']
    if bad_keys:
        raise ValueError(f'Following parameters at config file are empty: {", ".join(bad_keys)}')


def main():
    args = parser.parse_args()
    # TODO: make proper path handling
    with open('core/config.json') as config_file:
        config = json.load(config_file)
    db_config = config['DB']
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
        start_admin(session, admin['user'], admin['password'], args.port, args.amazon_container)

    if args.mode == 'dpa_dumper':
        from server.dump_new_dialogs_from_dpagent import dump_new_dialogs
        dump_new_dialogs(session)

if __name__ == "__main__":
    main()
