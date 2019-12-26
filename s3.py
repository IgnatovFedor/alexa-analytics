import boto3
import os
import io
import json

from typing import Generator


class S3Manager:
    def __init__(self, access_key: str, secret_access_key: str, dialog_dumps_bucket: str) -> None:
        self._s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_access_key)
        self._dialog_dumps_bucket = dialog_dumps_bucket

    def _get_all_interval_logs(self) -> Generator[dict, None, None]:
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
        for file in self._get_all_interval_logs():
            self._s3.download_file(self._dialog_dumps_bucket, file['Key'], os.path.join(dir, file['Key']))

    def get_interval_logs_json(self, key):
        filelike = io.BytesIO(b'')
        self._s3.download_fileobj(self._dialog_dumps_bucket, key, filelike)
        return json.loads(filelike.getvalue().decode())
