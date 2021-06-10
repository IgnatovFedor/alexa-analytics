"""
Scripts that given a dataset with dialogs uploads them into Django database
It should assure duplicates
"""
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


number_of_attempts = 0
def request_api_for_dialog(dp_dialog_id, timeout=15, sleep=10, max_retries = 10):
    # url_to_dialog = "http://docker-externalloa-lofsuritnple-525614984.us-east-1.elb.amazonaws.com:4242/api/dialogs/%s" % dp_dialog_id
    url_to_dialog = "http://a1b4e1088651f439d9e82fce0c4533b4-501376769.us-east-1.elb.amazonaws.com:4242/api/dialogs/%s" % dp_dialog_id
    global number_of_attempts
    try:
        number_of_attempts +=1
        with urllib.request.urlopen(url_to_dialog, timeout=timeout) as url:
            remote_data = json.loads(url.read().decode())
            # print(remote_data)
            # import ipdb;ipdb.set_trace()
    except Exception as e:
        print(e)
        # import ipdb; ipdb.set_trace()
        # Should we sleep  a little and try again? because usually it is problem of
        # overloading the remote server
        import time
        print("sleeping. and retry...")
        time.sleep(sleep)
        remote_data = request_api_for_dialog(dp_dialog_id)

    number_of_attempts = 0
    return remote_data

def upload_dataset(df):
    """
    Upload from pandas dataframe with dialogs
    :param path_to_pandas_df:
    :return:
    """
    print(len(df))
    for num_idx, (dialog_idx, each_dialog_row) in enumerate(df.iterrows()):
        print("__" *40)
        print(num_idx)
        # if num_idx<9843:
        #     continue
        # ###########################################################
        # retrieve dialog identifier
        dp_dialog_id = each_dialog_row['dialog']['id']

        rating = each_dialog_row['rating_val']
        try:
            if isinstance(rating, str):
                rating = None
            elif math.isnan(rating):
                rating = None
            else:
                assert isinstance(rating, float)
        except Exception as e:
            print(e)
            import ipdb; ipdb.set_trace()
            print(e)
            return
        if each_dialog_row['conversation_id'] is None:
            if not dp_dialog_id:
                import ipdb; ipdb.set_trace()

                raise Exception("no dp_dialog_id!")
            print("filter dialogs by dp_id")
            dialogs = Dialog.objects.filter(
                dp_id=dp_dialog_id)

        else:
            print("filter dialogs by conversation_id: %s" % each_dialog_row['conversation_id'])
            dialogs = Dialog.objects.filter(
                conversation_id=each_dialog_row['conversation_id']
            )

        # ###########################################################
        # get or create dialog
        if len(dialogs)>0:
            # exist
            print("dialog exists! %s" % each_dialog_row['conversation_id'])
            pass
        else:
            print("dialog not exists!")
            # ##### create ###########################################
            # get or create authors
            author_h, _ = Author.objects.get_or_create(
                user_type='human',
                dp_id=each_dialog_row['dialog']['human']['id']
            )

            author_b, _ = Author.objects.get_or_create(
                user_type='bot')
            # import ipdb; ipdb.set_trace()
            parsed_dt = pytz.utc.localize(each_dialog_row['first_utt_time'])


            # now create each utterance
            # dialog_dict = each_dialog_row['dialog']
            ##############################################################################
            # annotations and hypotheses are available only from remote endpoint
            # so we call it with dirty code:
            # import ipdb; ipdb.set_trace()
            #
            # url_to_dialog = "http://docker-externalloa-lofsuritnple-525614984.us-east-1.elb.amazonaws.com:4242/api/dialogs/%s" %dp_dialog_id

            try:
                remote_data = request_api_for_dialog(dp_dialog_id, timeout=15, sleep=10, max_retries=10)
            except Exception as e:
                print(e)
                print("I have tried to request Dialog API for 10 times but it sucks! Exit and try later!")
                print("dp_dialog_id=%s" % dp_dialog_id)
                return

            ##############################################################################
            # Start data creation:
            dialog = Dialog.objects.create(
                conversation_id=each_dialog_row['conversation_id'],
                start_time=parsed_dt,
                dp_id=dp_dialog_id,
                rating=rating,
                human=author_h
            )
            # for utt_idx, each_utterance in enumerate(dialog_dict['utterances']):
            for utt_idx, each_utterance in enumerate(remote_data['utterances']):
                # is_human = utt_idx % 2 == 0

                if 'user' in each_utterance:
                    if each_utterance['user']['user_type'] == "bot":
                        author = author_b
                        active_skill = each_utterance['active_skill']
                        version = None
                    elif each_utterance['user']['user_type'] == "human":
                        author = author_h
                        active_skill = "human"
                        version = each_utterance['attributes']['version']
                    else:
                        print("Unknown error with detection of typo of author!")
                        import ipdb; ipdb.set_trace()
                        # self destruct to avoid corruptedf dialogs:
                        dialog.delete()
                        return
                # try:
                #     if is_human:
                #         author = author_h
                #         active_skill = "human"
                #         version = each_utterance['attributes']['version']
                #     else:
                #         if "active_skill" not in each_utterance:
                #             # Seems order of utterances was mixed!
                #             print("each_utterance")
                #             print(each_utterance)
                #             import ipdb; ipdb.set_trace()
                #         # assert "active_skill" in each_utterance
                #         author = author_b
                #         active_skill = each_utterance['active_skill']
                #         version=None
                # except Exception as e:
                #     print(e)
                #     print("dp_dialog_id:")
                #     print(dp_dialog_id)
                #
                #     print(e)

                # utt, created = Utterance.objects.get_or_create(
                #     text=each_utterance['text'],
                #     parent_dialog=dialog,
                #     author=author
                # )

                # we use create to write duplicated utterance in one dialog separately
                # TODO annotate with timestamp
                # parse datetime... dp_agent has two formats...
                try:
                    parsed_dt = dt.datetime.strptime(each_utterance['date_time'], "%Y-%m-%d %H:%M:%S.%f")
                except Exception as e:
                    parsed_dt = dt.datetime.strptime(each_utterance['date_time'],
                                                     "%Y-%m-%d %H:%M:%S")
                parsed_dt = pytz.utc.localize(parsed_dt)

                utt = Utterance.objects.create(
                    text=each_utterance['text'],
                    parent_dialog=dialog,
                    author=author,
                    timestamp=parsed_dt,
                    active_skill=active_skill,
                    version=version
                )

                # ANNOTATIONS:
                # import ipdb; ipdb.set_trace()
                try:
                    for each_anno_key, each_anno_dict in remote_data['utterances'][utt_idx]['annotations'].items():
                        # anno, _ = Annotation.objects.get_or_create(
                        anno = Annotation.objects.create(
                            parent_utterance=utt,
                            annotation_type=each_anno_key,
                            annotation_dict=each_anno_dict
                        )
                except Exception as e:
                    print(e)
                    import ipdb; ipdb.set_trace()
                    print("Investigate")
                # if 'annotations' in each_utterance:
                #     # push annotations for human created utterances:
                #     for each_anno_key, each_anno_dict in each_utterance['annotations'].items():
                #         anno, _ = Annotation.objects.get_or_create(
                #             parent_utterance=utt,
                #             annotation_type=each_anno_key,
                #             annotation_dict=each_anno_dict
                #         )
                # else:
                #     # print("annotations are not found!")
                #     # find it from remote object:
                #     for each_anno_key, each_anno_dict in remote_data['utterances'][utt_idx]['annotations'].items():
                #         anno, _ = Annotation.objects.get_or_create(
                #             parent_utterance=utt,
                #             annotation_type=each_anno_key,
                #             annotation_dict=each_anno_dict
                #         )


                # HYPOTHESES:
                if 'hypotheses' in remote_data['utterances'][utt_idx]:
                    for each_hypo in remote_data['utterances'][utt_idx]['hypotheses']:

                        # lets add dictionary with extra attributes from skills:
                        other_attrs = copy(each_hypo)
                        del other_attrs['skill_name']
                        del other_attrs['text']
                        del other_attrs['confidence']
                        try:
                            # anno, _ = UtteranceHypothesis.objects.get_or_create(
                            anno = UtteranceHypothesis.objects.create(
                                parent_utterance=utt,
                                skill_name=each_hypo['skill_name'],
                                text=each_hypo['text'],
                                confidence=each_hypo['confidence'],
                                other_attrs=other_attrs
                            )
                        except Exception as e:
                            print(e)
                            import ipdb; ipdb.set_trace()
                            print("Investigate")

                #
                # if 'hypotheses' in each_utterance:
                #     # push hypotheses:
                #     for each_hypo in each_utterance['hypotheses']:
                #         # lets add dictionary with extra attributes from skills:
                #         other_attrs = copy(each_hypo)
                #         del other_attrs['skill_name']
                #         del other_attrs['text']
                #         del other_attrs['confidence']
                #
                #         anno, _ = UtteranceHypothesis.objects.get_or_create(
                #             parent_utterance=utt,
                #             skill_name=each_hypo['skill_name'],
                #             text=each_hypo['text'],
                #             confidence=each_hypo['confidence'],
                #             other_attrs=other_attrs
                #         )
                #
                # else:
                #     print("hypotheses are not found!")
                #     import ipdb; ipdb.set_trace()
                #     print("hypotheses are not found!")

if __name__=="__main__":
    PATH_TO_DIALOGS_DUMP = "/home/alx/Cloud/aiml_related/dp_agent_alexa/data_reports/dialogs_df.pckl"
    import pandas as pd

    df = pd.read_pickle(PATH_TO_DIALOGS_DUMP)
    # import ipdb; ipdb.set_trace()

    upload_dataset(df)

    print("Uploading dataset to local db complete!")
    print("Fin.")
