from django.db import models

# Create your models here.
from django.db import models
from django.contrib.postgres.fields import JSONField

USER_TYPE_CHOICES = (
    ("bot", "bot"),
    ("human", "human"))


class Author(models.Model):
    user_type = models.CharField(choices=USER_TYPE_CHOICES, max_length=128)

    user_telegram_id =models.CharField(max_length=128, null=True, default=None, blank=True)

    profile = JSONField(null=True, default=None, blank=True)

    dp_id = models.CharField(max_length=128, null=True, default=None, blank=True)

    def __str__(self):
        if self.dp_id:
            return "User: %s, %s" % (self.user_type, self.dp_id)

        return "User: %s" % (self.user_type)


class Dialog(models.Model):

    # question_text = models.CharField(max_length=200)
    # pub_date = models.DateTimeField('date published')

    # dp alexa id?:
    conversation_id = models.CharField(max_length=128, null=True, blank=True, db_index=True)

    # internal dp_agent id?
    dp_id = models.CharField(max_length=128, null=True, blank=True)

    start_time = models.DateTimeField(null=True, blank=True)

    rating = models.IntegerField(default=None, null=True, blank=True)

    human = models.ForeignKey(Author, null=True, blank=True, on_delete=models.CASCADE, related_name='human_utterances')
    bot = models.ForeignKey(Author, null=True, blank=True, on_delete=models.CASCADE, related_name='bot_utterances')

    def __str__(self):
        if self.utterance_set.count()>0:
            output = ""
            for each_utt in reversed(self.utterance_set.all()):
                output += "-%s\n" % each_utt
        else:
            output = "Dialog: %s" % self.conversation_id
        return output

    def get_utterances_segment(self, focus_utterance, prev_utterances=1, forward_utterances=1, skip_focus_utterance=False):
        """
        Method is useful when we want to find context utterances surrounding some target utterance.
        Method outputs a list of utterances which surround the focus_utterance.
        :param focus_utterance: is the utterance surrounding of which we'd like to output
        :param prev_utterances:int, how many utterances preceding the focus_utterance to return
        :param forward_utterances: int, how many utterances after focus_utterance to return
        :param skip_focus_utterance: bool, if True, then result will skip the utterance with focus_utterance
        :return: list of Utterance objects which are part of surrounding context
        """
        # not all utteracnes has timestamp due to buggy dump:
        # all_d_utts= self.utterance_set.all().order_by('timestamp')
        all_d_utts= self.utterance_set.all().order_by('timestamp')
        if focus_utterance in all_d_utts:
            # ok
            list_of_d_utts = list(all_d_utts)
            f_idx = list_of_d_utts.index(focus_utterance)
            from_idx = max(f_idx-prev_utterances, 0)
            up_idx = min(f_idx+forward_utterances+1, len(list_of_d_utts))
            if skip_focus_utterance:
                return all_d_utts[from_idx:f_idx]+all_d_utts[f_idx+1:up_idx]
            else:
                return all_d_utts[from_idx : up_idx]

        else:
            print("can not find the utterance in dialog. investigate")

    def get_noun_phrases(self):
        utts = self.utterance_set.all()
        nps = []
        for each_utt in utts:
            utt_np  = each_utt.get_noun_phrases()
            if utt_np:
                nps+=utt_np
        return nps


    @classmethod
    def find_dialogs_by_topic(cls, topic_str):
        annotations_with_np = Annotation.objects.filter(annotation_type="cobot_nounphrases",
                                  annotation_dict__contains=topic_str)
        dialogs= []
        for each_anno in annotations_with_np:
            dialogs.append(each_anno.parent_utterance.parent_dialog)

        count_of_nps = len(dialogs)
        unique_dialogs = set(dialogs)
        count_of_nps_unique_dialogs = len(unique_dialogs)
        return unique_dialogs



class Utterance(models.Model):
    parent_dialog = models.ForeignKey(Dialog, on_delete=models.CASCADE)

    text = models.CharField(max_length=2064)

    author = models.ForeignKey(Author, on_delete=models.CASCADE)

    # date_time:
    timestamp = models.DateTimeField(null=True, blank=True)

    # active skill is filled for Bot Utterances, may be empty for old ones, "human" for Human's utterances
    active_skill = models.CharField(max_length=2064, null=True, blank=True)

    # sys version
    version = models.CharField(max_length=200, null=True, blank=True)


    # annotations is on2Many field
    # question = models.ForeignKey(Question, on_delete=models.CASCADE)

    def __str__(self):
        return "%s" % self.text

    def get_noun_phrases(self):
        """Retrieves cobot noun phrases from the Utterance object"""
        nps = []
        annotations = self.annotation_set.filter(annotation_type="cobot_nounphrases")
        for each_an in annotations:
            if each_an.annotation_dict:
                nps+=each_an.annotation_dict
        return nps

    def next_utterance(self):
        """returns a next utterance in dialog"""
        parent_dialog = self.parent_dialog
        # TODO optimize me
        all_d_utts = parent_dialog.utterance_set.filter(id__gte=self.id).order_by('timestamp')

        if len(all_d_utts)>1:
            # it must be second in filtered list
            return all_d_utts[1]

        else:
            # print("can not find the utterance in dialog. investigate")
            return None

    def next_utterances_in_dialog(self):
        parent_dialog = self.parent_dialog
        # TODO optimize me
        all_d_utts = parent_dialog.utterance_set.filter(id__gte=self.id).order_by('timestamp')

        if len(all_d_utts) > 1:
            # it must be second in filtered list
            return all_d_utts

        else:
            # print("can not find the utterance in dialog. investigate")
            return None

    def get_sentiment(self, with_confidence=False):
        """
        retrieves sentiment of the utterance
        :return: None or string with leading sentiment category: neutral, positive or negative (usually)
            if with_confidence param provided then it returns tuple, where the first element is sentimen category name,
            the second is confidence
        """
        sentiment_annos = self.annotation_set.filter(annotation_type="sentiment_classification")
        for each_sentiment_anno in sentiment_annos:
            list_of_sentim_data = each_sentiment_anno.annotation_dict['text']
            assert len(list_of_sentim_data) == 2
            # zero erlement of the tuple is a name of sentiment class:
            sentiment_category_name = list_of_sentim_data[0]

            if with_confidence:

                sentiment_category_confidence = list_of_sentim_data[1]
                return sentiment_category_name, sentiment_category_confidence
            return sentiment_category_name
        return None

    def get_emotions(self):
        """Returns emotions distribution as dict
        with keys as emotion names and values their scores.
        Each scores is ranged from 0 to 1.
        Scores are not normalized (for example neutral emotion is quite always around 0.9) and biased.

        Ex:
        {
        "joy": 0.55,
        "fear": 0.18,
        "love": 0.29,
        "anger": 0.44
        "neutral": 0.99
        "sadness": 0.31
        "surprise": 0.12
        }
        """
        emotions_annos = self.annotation_set.filter(annotation_type="emotion_classification")
        emotion_dict = {}
        if len(emotions_annos)>1:
            # investigate
            raise Exception("Multiple Emotion annotations for the utterance %s:%s!" % (self.id, self.text))

        for each_emo_anno in emotions_annos:
            emotion_dict = each_emo_anno.annotation_dict['text']

        return emotion_dict


class BotUtterance(Utterance):
    pass


class HumanUtterance(Utterance):
    "'text', 'user', 'annotations', 'hypotheses'"
    pass


class Annotation(models.Model):
    """Annotations are produced by annotators and decorate input Utterances with additional
    information"""

    parent_utterance = models.ForeignKey(Utterance, on_delete=models.CASCADE)

    annotation_type = models.CharField(max_length=256)

    annotation_dict = JSONField(null=True, default=None, blank=True)

    def __str__(self):
        return "%s: %s" % (self.annotation_type, self.annotation_dict)

class UtteranceHypothesis(models.Model):
    """
    Each skill produces utterances hypotheses one of which is selected by response selector
    on each step
    """
    parent_utterance = models.ForeignKey(Utterance, on_delete=models.CASCADE)

    skill_name = models.CharField(max_length=256)
    text = models.CharField(max_length=2064)
    confidence = models.FloatField()

    # all stuff that the skill has pushed out:
    other_attrs = JSONField(null=True, default=None, blank=True)

    def __str__(self):
        return "(%s:%0.2f): %s" % (self.skill_name, self.confidence, self.text)
