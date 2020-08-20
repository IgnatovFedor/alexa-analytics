# from db.models.conversation import Conversation
import json
import urllib.request, json
# from admin.admin import start_admin
# from db.db import DBManager, get_session
from sqlalchemy.orm.exc import NoResultFound, MultipleResultsFound
from db.models.conversation import Conversation
from db.models.utterance import Utterance
from db.models.annotation import Annotation
from db.models.utterance_hypothesis import UtteranceHypothesis
from db.db import DBManager
import logging
from copy import copy
import datetime as dt
# from server.s3 import S3Manager
# from server.server import start_polling
logger = logging.getLogger(__name__)
number_of_attempts = 0

class DPADumper():
    """
    Class for dumping the staf from DP-Agent DB into analytical tool
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

        print(f"requesting DP_Agent API: {final_url}...")
        with urllib.request.urlopen(final_url, timeout=timeout) as url:
            remote_data = json.loads(url.read().decode())

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
        print(f"requesting DP_Agent API: {url_to_dialog}...")
        with urllib.request.urlopen(url_to_dialog, timeout=timeout) as url:
            remote_data = json.loads(url.read().decode())

        return remote_data


def dump_new_dialogs(session, dpagent_base_url="http://0.0.0.0:4242"):
    # TODO make dumper to request only new dialogs and stop paging for old dialogs
    # get the latest dialog from local db
    # then iteratively collect dialog ids in the DPAgent dialog_list_ids API until meet existing
    # then iteratively grab each dialog to local db

    res = session.query(Conversation).order_by(Conversation.date_finish.desc())

    # get the latest local id
    recent_conversation = res.first()
    print(res.first())
    if not recent_conversation:
        # load all
        pass

    ######################################################################
    # request dp_agent api for list of dialogs
    dpad = DPADumper(dpa_base_url=dpagent_base_url)

    page_suffix = "?limit=100"
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
                    conv = session.query(Conversation).filter_by(id=each_dialog_id).one()
                except NoResultFound:
                    # request api for viewing dialogs
                    dialog_data = dpad.request_api_for_dialog(each_dialog_id)
                    start = DBManager._parse_time(dialog_data['date_start'])
                    finish = DBManager._parse_time(dialog_data['date_finish'])

                    if dialog_data["_active"] and finish > dt.datetime.now()-dt.timedelta(minutes=10):
                        logger.info(f"skipping actve and fresh dialog: {dialog_data['dialog_id']}")
                        continue
                    conv_id = dialog_data['dialog_id']

                    conv = Conversation(
                        id=conv_id,
                        # Fix the shit!:
                        mgid=conv_id[:24],
                        date_start=start,
                        date_finish=finish,
                        human=dialog_data['human'],
                        bot=dialog_data['bot'],
                        length=len(dialog_data['utterances']),
                        raw_utterances=dialog_data['utterances']
                    )
                    # ##########################
                    # find ratings
                    # get the last rating:
                    if "attributes" in dialog_data:
                        if 'ratings' in dialog_data['attributes']:
                            for each_rat_dict in dialog_data['attributes']['ratings']:
                                conv.rating = each_rat_dict['rating']
                    # ##########################
                    session.add(conv)
                    session.commit()

                    for utt_idx, utterance in enumerate(dialog_data['utterances']):
                        utt = Utterance(text=utterance['text'],
                                        date_time=DBManager._parse_time(utterance['date_time']),
                                        active_skill=utterance.get('active_skill'),
                                        attributes=utterance.get('attributes'),
                                        conversation_id=conv_id)
                        session.add(utt)
                        session.commit()

                        # ANNOTATIONS:
                        try:
                            for each_anno_key, each_anno_dict in dialog_data['utterances'][utt_idx][
                                'annotations'].items():
                                anno = Annotation(
                                    parent_utterance_id=utt.id,
                                    annotation_type=each_anno_key,
                                    annotation_data=each_anno_dict
                                )
                                session.add(anno)
                            session.commit()
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
                                    # anno, _ = UtteranceHypothesis.objects.get_or_create(
                                    hypo = UtteranceHypothesis(
                                        parent_utterance_id=utt.id,
                                        skill_name=each_hypo['skill_name'],
                                        text=each_hypo['text'],
                                        confidence=each_hypo['confidence'],
                                        other_attrs=other_attrs
                                    )
                                    session.add(hypo)
                                    session.commit()
                                except Exception as e:
                                    logger.warning(e)
                                    # print(e)
                                    # import ipdb;
                                    # ipdb.set_trace()
                                    # print("Investigate")

                    logger.info(f'Successfully added a new conversation {conv_id} to local DB.')
        else:
            logger.warning("No dialogs in DP-Agent!")
        # print(results)
    print("Fin.")



