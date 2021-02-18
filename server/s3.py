import io
import json
import os
from typing import Generator

import boto3
from pandas import DataFrame, read_table, to_datetime, notnull


class S3Manager:
    def __init__(self, access_key: str, secret_access_key: str, dialog_dumps_bucket: str,
                 ratings_bucket: str, team_id: str, skip_tg: bool) -> None:
        self._s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_access_key)
        self._dialog_dumps_bucket = dialog_dumps_bucket
        self._ratings_bucket = ratings_bucket
        self._team_id = team_id
        self.skip_tg = skip_tg

    def get_all_interval_logs(self) -> Generator[dict, None, None]:
        continuation_token = None
        while True:
            kwargs = {'Bucket': self._dialog_dumps_bucket}
            if continuation_token:
                kwargs['ContinuationToken'] = continuation_token
            response = self._s3.list_objects_v2(**kwargs)
            yield from response.get('Contents', [])
            if not response.get('IsTruncated'):  # At the end of the list?
                break
            continuation_token = response.get('NextContinuationToken')

    def download_logs_to_dir(self, dir):
        for file in self.get_all_interval_logs():
            self._s3.download_file(self._dialog_dumps_bucket, file['Key'], os.path.join(dir, file['Key']))

    def get_hour_log_json(self, key):
        filelike = io.BytesIO(b'')
        self._s3.download_fileobj(self._dialog_dumps_bucket, key, filelike)
        return json.loads(filelike.getvalue().decode())

    def _get_results(self, filename) -> DataFrame:
        filelike = io.BytesIO(b'')
        self._s3.download_fileobj(self._ratings_bucket, f'{self._team_id}/{filename}', filelike)
        filelike.seek(0)
        return read_table(filelike, sep=',', index_col=False, encoding='utf-8')

    def get_feedback(self):
        df = self._get_results('conversation_feedback.csv')
        df = df.where((notnull(df)), None)
        # TODO: Make proper time filtering (problem with different formats)
#        df['conversation_start_time'] = to_datetime(df['conversation_start_time'])
        return df

    def get_ratings(self):
        df = self._get_results('Ratings/ratings.csv')
        df['Approximate Start Time'] = to_datetime(df['Approximate Start Time'])
        return df
