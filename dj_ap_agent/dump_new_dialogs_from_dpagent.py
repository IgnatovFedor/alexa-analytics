################# Universal Import ###################################################
import sys
import os
import argparse

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# PREROOT_DIR = os.path.dirname(ROOT_DIR)
print(ROOT_DIR)
sys.path.append(ROOT_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dj_ap_agent.settings")
import django
django.setup()
# #####################################################
from copy import copy
from dialogs.models import Dialog, Utterance, Annotation, UtteranceHypothesis, Author
import urllib.request, json
import datetime as dt
import pytz
import math

# from db.models.conversation import Conversation
import json
import urllib.request, json
# from admin.admin import start_admin
# from db.db import DBManager, get_session
# from sqlalchemy.orm.exc import NoResultFound, MultipleResultsFound
# from db.models.conversation import Conversation
# from db.models.utterance import Utterance
# from db.models.annotation import Annotation
# from db.models.utterance_hypothesis import UtteranceHypothesis
# from db.db import DBManager
import logging
from copy import copy
import datetime as dt
# from server.s3 import S3Manager
# from server.server import start_polling
logger = logging.getLogger(__name__)
number_of_attempts = 0


class DjDPADumper():
    """
    Class for dumping the staf from DP-Agent DB into django analytical tool
    """
    def __init__(self, dpa_base_url="http://0.0.0.0:4242"):
        self.dpa_base_url = dpa_base_url


    def request_api_page_for_dialogs_list(self, offset=0, limit=100, url_suffix=None,
                                          timeout=15):
        """
        Requests list of recent dialogs
        :param url_suffix: string with page suffix. Ex.: "?offset=30&limit=30"
            if None then request is from offset and limit args
        :param offset: int, how many dialogs to skip

        :param timeout: seconds of timeout for waiting

        :return: dict with dialog_list_ids and next page url. Ex.:
        {
            "dialog_ids": ["5edfaf8a4d595032bf2e7aed", "5ee0a8b3931df828d12362eb",
                "5ee0d5bc4ceff4226ebedd63", "5ee1003c287f6c1327d56a07", "5ee6435afa53109f5c6c8c39",
                "5ee73cb12334f458eb1109ea", "5ee746b8d9bbd35cdd614281", "5ee74be8b359fad8dbaac1ff",
                "5ee74c249dd442dfc2fc637d", "5ee77ab88b9fc0d06ac76d50", "5ee77bfc8b9fc0d06ac76d55",
                "5ee78ceb82d52fa04922bbc3", "5ee79326c28d00fa17666953", "5ee797fc14a371f5e3922915",
                "5ee79867d6c588e1566571a5", "5ee8d221d6c588e1566571aa", "5ee8d272d6c588e1566571af",
                "5ee8d80fd6c588e1566571b4", "5ee94b8ad6c588e1566571d7", "5ee94be5d6c588e1566571dc",
                "5ee94d9cd6c588e1566571e6", "5ee9c7e2d6c588e1566571eb", "5ee9cc7cd6c588e156657202",
                "5eea07e6d6c588e156657263", "5eea3e16d6c588e15665726c", "5eea6bffd6c588e156657275",
                "5eea6d6ad6c588e15665727a", "5eea6eefd6c588e156657294", "5eea71b3d6c588e1566572a4",
                "5eea782fd6c588e1566572a9"],
            "next": "?offset=30&limit=30"}

        """

        url_to_dialogs = "%s/api/dialogs/" % (self.dpa_base_url)

        if url_suffix:
            final_url = url_to_dialogs + url_suffix
        else:
            params = {
                "offset": offset,
                "limit": limit,
                # for dumping finished dialogs only:
                # "_active": 0
            }

            url_suffix = urllib.parse.urlencode(params)
            final_url = url_to_dialogs + "?" + url_suffix


        fails_counter = 0
        remote_data = None
        while not remote_data:
            try:
                logger.info(f"requesting DP_Agent API: {final_url}...")

                with urllib.request.urlopen(final_url, timeout=timeout) as url:
                    remote_data = json.loads(url.read().decode())
            except Exception as e:
                print(e)
                fails_counter +=1
                if fails_counter>=5:
                    logger.info(f"Failed to get {final_url} after several attempts")
                    raise e
                import time
                time.sleep(10)

        return remote_data

    def request_api_for_dialog(self, dp_dialog_id, timeout=15, sleep=10, max_retries=10):
        """
        Requests specific dialog details from DP-Agent
        :param dp_dialog_id:
        :param timeout: seconds of timeout for waiting
        :param sleep: allows to retry request after some sleeping
        :param max_retries: number of failed attempts to request
        :return: serialized dict with dp_agent.Dialog representation
        """
        # import ipdb; ipdb.set_trace()

        url_to_dialog = "%s/api/dialogs/%s" % (self.dpa_base_url, dp_dialog_id)


        fails_counter = 0
        remote_data = None
        while not remote_data:
            try:
                logger.info(f"Requesting DP_Agent API: {url_to_dialog}...")
                with urllib.request.urlopen(url_to_dialog, timeout=timeout) as url:
                    remote_data = json.loads(url.read().decode())
                logger.info(f"Collected data for dialog: {url_to_dialog}.")
            except Exception as e:
                print(e)
                fails_counter += 1
                if fails_counter >= 5:
                    logger.info(f"Failed to get {url_to_dialog} after several attempts")
                    raise e
                import time
                time.sleep(10)

        return remote_data


def parse_time(time_str: str):
    try:
        time = dt.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        time = dt.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
    return time

def dump_new_dialogs(dpagent_base_url="http://0.0.0.0:4242"):
    # TODO make dumper to request only new dialogs and stop paging for old dialogs
    # get the latest dialog from local db
    # then iteratively collect dialog ids in the DPAgent dialog_list_ids API until meet existing
    # then iteratively grab each dialog to local db

    # res = session.query(Conversation).order_by(Conversation.date_finish.desc())
    #
    # # get the latest local id
    # recent_conversation = res.first()
    # print(res.first())
    # if not recent_conversation:
    #     # load all
    #     pass

    ######################################################################
    # request dp_agent api for list of dialogs
    dpad = DjDPADumper(dpa_base_url=dpagent_base_url)

    # page_suffix = "?limit=100"
    # page_suffix = "?limit=1000"
    # page_suffix = "?limit=1000&offset=52000"
    page_suffix = "?limit=1000&offset=73000"
    print(f"page_suffix: {page_suffix}")
    # page_suffix = "?limit=5&_active=0"
    # page_suffix = "?limit=5&_active=1"

    while page_suffix is not None:
        # TODO make DP-Agent to return new dialogs first
        results = dpad.request_api_page_for_dialogs_list(url_suffix=page_suffix)
        # parse results:
        if results:
            page_suffix =results['next']
            dialog_ids = results['dialog_ids']
            for each_dialog_id in dialog_ids:

                try:
                    print(f"Searching the dialog {each_dialog_id} in local db...")
                    conv = Dialog.objects.get(dp_id=each_dialog_id)
                    # conv = session.query(Conversation).filter_by(id=each_dialog_id).one()
                    print(f"Dialog {each_dialog_id} is already in local db!" )
                except Exception as e:
                    # print(f"Got exception: {e}")

                    # import ipdb; ipdb.set_trace()
                    # request api for viewing dialogs
                    try:
                        dialog_data = dpad.request_api_for_dialog(each_dialog_id)
                        start_dt = parse_time(dialog_data['date_start'])
                        start = pytz.utc.localize(start_dt)
                        # print(f"start {start}")
                        # print(f"dt.datetime.now(pytz.utc): {dt.datetime.now(pytz.utc)}")
                        if dialog_data["_active"] and start > dt.datetime.now(pytz.utc)-dt.timedelta(minutes=30):
                            print(f"skipping actve and fresh dialog: {dialog_data['dialog_id']}")
                            continue
                        if len(dialog_data['utterances'])<7:
                            print(f"skipping short dialog: {dialog_data['dialog_id']}, Length={len(dialog_data['utterances'])}")
                            continue
                        conv_id = dialog_data['dialog_id']
                        logger.debug("Adding Conversation to DB...")

                        # find amazon conv id:
                        amazon_conv_id=None
                        for utter in dialog_data['utterances']:
                            if 'attributes' in utter:
                                attrs = utter['attributes']
                                # import ipdb; ipdb.set_trace()
                                print(f"attributes: {attrs}")
                                if 'conversation_id' in attrs:
                                    amazon_conv_id = attrs['conversation_id']
                                    break

                        # ##### create ###########################################
                        # get or create authors
                        print("Creating the authors and the dialog obj")
                        # print(dialog_data['human'])
                        # import ipdb; ipdb.set_trace()
                        author_h, _ = Author.objects.get_or_create(
                            user_type='human',
                            dp_id=dialog_data['human']['id']
                        )

                        author_b, _ = Author.objects.get_or_create(
                            user_type='bot')

                        conv = Dialog.objects.create(
                            dp_id=conv_id,
                            # Fix the shit!:
                            conversation_id=amazon_conv_id,
                            start_time=start,
                            # date_finish=finish,
                            human=author_h,
                            bot=author_b,
                            # length=len(dialog_data['utterances']),
                            # raw_utterances=dialog_data['utterances'],
                        )
                        # ##########################
                        # find ratings
                        # get the last rating:
                        if "attributes" in dialog_data:
                            if 'ratings' in dialog_data['attributes']:
                                for each_rat_dict in dialog_data['attributes']['ratings']:
                                    conv.rating = each_rat_dict['rating']

                                    conv.save()
                        # ##########################
                        print("Added Conversation to DB. Adding utterances...")
                        for utt_idx, utterance in enumerate(dialog_data['utterances']):
                            logger.debug(f"Adding utterance {utt_idx}")

                            if 'user' in utterance:
                                if utterance['user']['user_type'] == "bot":
                                    author = author_b
                                    active_skill = utterance['active_skill']
                                    version = None
                                elif utterance['user']['user_type'] == "human":
                                    author = author_h
                                    active_skill = "human"
                                    version = utterance['attributes'].get('version', "UNKNOWN")
                                else:
                                    print("Unknown error with detection of typo of author! Deleting them!")
                                    import ipdb;
                                    ipdb.set_trace()
                                    # self destruct to avoid corruptedf dialogs:
                                    conv.delete()
                                    return
                            print(f"Adding utterance: {utterance['text']}")
                            utt = Utterance.objects.create(text=utterance['text'],
                                            timestamp= pytz.utc.localize(parse_time(utterance['date_time'])),
                                            active_skill=utterance.get('active_skill'),
                                            # attributes=utterance.get('attributes'),
                                            parent_dialog=conv,
                                           author=author
                                                           )

                            # ANNOTATIONS:
                            try:
                                for each_anno_key, each_anno_dict in dialog_data['utterances'][utt_idx][
                                    'annotations'].items():
                                    # print(f"Adding annotation: {each_anno_key}")
                                    anno = Annotation.objects.create(
                                        parent_utterance=utt,
                                        annotation_type=each_anno_key,
                                        annotation_dict=each_anno_dict
                                    )
                                    # session.add(anno)
                                # session.commit()
                            except Exception as e:
                                logger.warning(e)
                                # import ipdb;
                                # ipdb.set_trace()
                                # print("Investigate")

                            # HYPOTHESES:
                            if 'hypotheses' in dialog_data['utterances'][utt_idx]:
                                for each_hypo in dialog_data['utterances'][utt_idx]['hypotheses']:

                                    # lets add dictionary with extra attributes from skills:
                                    other_attrs = copy(each_hypo)
                                    del other_attrs['skill_name']
                                    del other_attrs['text']
                                    del other_attrs['confidence']
                                    try:
                                        # print(f"Adding hypotheses: {each_hypo}")
                                        hypo = UtteranceHypothesis.objects.create(
                                            parent_utterance=utt,
                                            skill_name=each_hypo['skill_name'],
                                            text=each_hypo['text'],
                                            confidence=each_hypo['confidence'],
                                            other_attrs=other_attrs
                                        )
                                    except Exception as e:
                                        logger.warning(e)
                                        # print(e)
                                        # import ipdb;
                                        # ipdb.set_trace()
                                        # print("Investigate")

                        logger.info(f'Successfully added a new conversation {conv_id}/{amazon_conv_id} to local DB.')
                    except Exception as e:
                        logger.warning(f"Some problem occured during importing the dialog {each_dialog_id}")
                        logger.warning(e)
                        logger.info("Skipping")
                        raise e


        else:
            logger.warning("No dialogs in DP-Agent!")
        # print(results)
    print("Fin.")



if __name__=="__main__":
    dump_new_dialogs(dpagent_base_url="http://a0c5f8bbe459c4cf7a1e04d8807bf007-344976058.us-east-1.elb.amazonaws.com:4242")