# Lets define which skill we analyze:
from dialogs.models import Dialog, Utterance, Annotation, UtteranceHypothesis, Author
from django.db.models import Q
import datetime as dt
from nltk import FreqDist
import pandas as pd
import numpy as np

# here a list of skills you want to analyze:
skills = [
    "alice",
    # "eliza",
    "cobotqa",
    "program_y",
    "personality_catcher",
    "intent_responder",
    "dummy_skill",
    "alexa_handler",
    "dummy_skill_dialog",
    "misheard_asr",
    "program_y_dangerous",
    "movie_skill",
    "emotion_skill",
    "tfidf_retrieval",
    "convert_reddit",
    "reddit_ner_skill",
    "personal_info_skill",
    "book_tfidf_retrieval",
    "entertainment_tfidf_retrieval",
    "fashion_tfidf_retrieval",
    "movie_tfidf_retrieval",
    "music_tfidf_retrieval",
    "politics_tfidf_retrieval",
    "science_technology_tfidf_retrieval",
    "sport_tfidf_retrieval",
    "animals_tfidf_retrieval",
    "topicalchat_convert_retrieval",
    "book_skill",
    "coronavirus_skill",
    "christmas_new_year_skill",
    "superbowl_skill",
    "oscar_skill",
    "valentines_day_skill",
    "weather_skill",
    "meta_script_skill",
    "small_talk_skill",
    "program_y_wide",
    "news_api_skill",
]


# version_filter = "v8.12.1"
def get_skill_sentiment_observations(skill_name, since_dt=None, version_filter=None):
    """The funstion collects a list of sentiment observations for the human answers to the skill
    optionally you may filter data by since_dt
    since_dt: optional datetime - if provided then only utterance after the date will be analyzed
    version_filter: optional string - if provided then only data from specified version will be collected

    return list of strings with names of sentiment classes like ["neutral", "positive", "neutral", "negative", ...]
    """
    if not since_dt:
        # since_dt = dt.datetime.now() - dt.timedelta(days=2)
        since_dt = dt.datetime(2020,5,6) - dt.timedelta(days=2)
    # 1. select utterances of the skill

    s_utts = Utterance.objects.filter(active_skill=skill_name)
    if since_dt:
        # chain filter:
        s_utts = s_utts.filter(timestamp__gte=since_dt)

    # 2. retrieve all next utterances
    # TODO version filtration
    # if utt.version==version_filter
    n_utts = [utt.next_utterance() for utt in s_utts if utt.next_utterance()]
    if version_filter:
        n_utts = [n_utt for n_utt in n_utts if n_utt.version == version_filter]
    # 3. collect annotation stat

    sentiment_observations = []
    for each_n_utt in n_utts:
        sentiment_annos = each_n_utt.annotation_set.filter(annotation_type="sentiment_classification")
        for each_sentiment_anno in sentiment_annos:
            list_of_sentim_data = each_sentiment_anno.annotation_dict['text']
            assert len(list_of_sentim_data) == 2
            # zero erlement of the tuple is a name of sentiment class:
            sentiment_observations.append(list_of_sentim_data[0])

    return sentiment_observations


def analyze_skills_sentiments(since_dt=None, version_filter=None, skills_filter=None):
    """
    Returns count distribution of sentiments per skill
    :param since_dt: datetime with the date since which we need to collect data
    :param version_filter: string of version
    :param skills_filter: list of skill names to use, by default all skills are anlyzed
    :return: Ex.:
        {
            "alice": {
                "positive":20,
                "neutral":20,
                "negative":20,
            },
            "tfidf":{
                "positive":20,
                "neutral":20,
                "negative":20,
            },
        }
    """
    skill_stat_dict = {}
    # fdist['neutral']
    if not skills_filter:
        skills_filter = skills
    for each_skill in skills_filter:

        # collect statistics of sentiment classes:
        sentiment_observations = get_skill_sentiment_observations(each_skill, since_dt=since_dt,
                                                                  version_filter=version_filter)
        print("Support for skill %s: %d" % (each_skill, len(sentiment_observations)))

        # visualize classes as frequency statistics:
        if len(sentiment_observations) > 0:
            fdist = FreqDist(sentiment_observations)

            #         f_senti_classes, f_counts = zip(*fdist.most_common(50))
            #         fig, ax1 = plt.subplots(figsize=(7, 5))
            #         ax1.set_title("Frequent sentiments for skill %s" % each_skill)
            #         ax1.bar(list(reversed(f_senti_classes)), list(reversed(f_counts)))

            skill_stat_dict[each_skill] = {"positive": 0, "neutral": 0, "negative": 0}
            for each_cls, each_count in fdist.most_common(50):
                skill_stat_dict[each_skill][each_cls] = each_count

            # fig, ax1 = plt.subplots(figsize=(7, 5))
            # ax1.set_title("Frequent sentiments for skill %s" % each_skill)
            # ax1.bar(skill_stat_dict[each_skill].keys(), [v for k, v in skill_stat_dict[each_skill].items()])
        else:
            print("No observations for skill %s" % each_skill)

    return skill_stat_dict


def analyze_daily_sentiments_distribution(days=365):
    """Collect statistics of daily sentiment

    """
    since_dt = dt.datetime.now() - dt.timedelta(days=days)
    bot_user = Author.objects.get(user_type="bot")

    # all user utterances for the last week:
    utterances = Utterance.objects.filter(~Q(author=bot_user) & Q(timestamp__gte=since_dt))

    # for each utterance get its sentiment
    sentiment_data = []
    #     print("iterate utts")
    for num, each_utter in enumerate(utterances):
        #         if num%1000==0:
        #             print(num)
        sentiment = each_utter.get_sentiment()
        if sentiment:
            data_el = {
                'timestamp': each_utter.timestamp,
                'sentiment': sentiment}
            sentiment_data.append(data_el)
    sentiments_df = pd.DataFrame(sentiment_data)
    # resample and calc statistics over each daily group
    sentiments_df.set_index("timestamp", inplace=True)

    def func(data):
        #         print("____")

        #         print(data.values)
        fdist = FreqDist(data.values)
        res_dict = {}
        for each_el in fdist.most_common(3):
            res_dict[each_el[0]] = each_el[1]

        return res_dict

    def cat_wrapper(cat_name):
        def func_extract_positive_category(x):
            if cat_name in x:
                return x[cat_name]
            else:
                return 0
        return func_extract_positive_category

    # df = pd.DataFrame(columns=["negative", "neutral", "positive"])
    res = sentiments_df.resample("D").apply(func)
    res['positive'] = res['sentiment'].apply(cat_wrapper('positive'))
    res['neutral'] = res['sentiment'].apply(cat_wrapper('neutral'))
    res['negative'] = res['sentiment'].apply(cat_wrapper('negative'))
    # res['negative'] = res['sentiment'].apply(lambda x: x['negative'])
    summas = res[['positive', 'neutral', 'negative']].sum(axis=1)
    res_df = res[['positive', 'neutral', 'negative']] / [summas, summas, summas]

    return res_df


# #######################################
# Topics Analysis

def get_rating_for_topic(topic_name, since_dt=None):
    """Given a topic name as string the method searches for all utterances with ratings that mention the topic
    And aggregates the statistics.
    """
    kwargs = {
        "annotation_type": "cobot_nounphrases",
        "annotation_dict__contains": topic_name
    }
    if since_dt:
        kwargs['parent_utterance__timestamp__gt'] = since_dt

    annotations_with_np = Annotation.objects.filter(**kwargs)
    dialogs = []
    for each_anno in annotations_with_np:
        if each_anno.parent_utterance.author.user_type == "bot":
            continue

        dialogs.append(each_anno.parent_utterance.parent_dialog)

    count_of_nps = len(dialogs)
    count_of_nps_unique_dialogs = len(set(dialogs))
    unique_dialogs = set(dialogs)
    ratings = []
    for each_d in unique_dialogs:
        if each_d.rating:
            ratings.append(each_d.rating)

    np_ratings = np.array(ratings)
    np_ratings.mean()
    return {"topic": topic_name,
            "avg_rating": np_ratings.mean(),
            "std_rating": np_ratings.std(),
            "support": len(np_ratings)
            }


def get_mentioned_topics(since_dt=None, only_rated=True):
    """
    :param since_dt: since which date topics must be collected
    :return: FreqDist with mentioned topics
    """
    if not since_dt:
        # last month topics
        since_dt = dt.datetime.now() - dt.timedelta(days=361)
    nps = []

    if only_rated:
        dialogs = Dialog.objects.filter(start_time__gt=since_dt, rating__in=[0, 1, 2, 3, 4, 5])
    else:
        dialogs = Dialog.objects.filter(start_time__gt=since_dt)
    for dialog in dialogs:
        nps += dialog.get_noun_phrases()

    fdist = FreqDist(nps_march)
    fdist.pprint(maxlen=100)
    return fdist


def collect_topics_ratings_statistics(fdist):
    """
    Given a topics distribution it collects ratings mean, support and std for all of them
    :param fdist:
    :return:
    """
    rows_march = []
    for each_topic, topic_count in fdist.most_common(500):
        print("%s: %d" % (each_topic, topic_count))
        stat_dict = get_rating_for_topic(each_topic)
        rows_march.append(stat_dict)
        print("__")

    df = pd.DataFrame(rows_march)
    df['rating_delta'] = df['avg_rating'] - df['avg_rating'].mean()
    return df


def collect_topics_ratings_observation_df(since_dt=None, skip_ratingless=True):
    """
    Given a timestamp it collects all topics of utterances and returns dataframe with schema:
    [(topic, rating),...]
    :param since_dt:
    :return:
    """
    kwargs = {
        "annotation_type": "cobot_nounphrases"
    }
    if since_dt:
        kwargs['parent_utterance__timestamp__gt'] = since_dt

    annotations_with_np = Annotation.objects.filter(**kwargs)
    rows = []
    for each_anno in annotations_with_np:
        for each_anno_item in each_anno.annotation_dict:
            rating = each_anno.parent_utterance.parent_dialog.rating
            if not rating and skip_ratingless:
                continue
            anno_dict = {
                'topic': each_anno_item,
                'rating': rating
            }
            rows.append(anno_dict)

    topics_observations_df = pd.DataFrame(rows)
    return topics_observations_df


def collect_topics_statistics(since_dt=None):
    """Constructs noun_phrases statistics over period with information about support, mean rating, std_rating and
    rating diff"""
    if not since_dt:
        since_dt = dt.datetime.now() - dt.timedelta(days=160)
    observations_df = collect_topics_ratings_observation_df(since_dt)
    counts_df = observations_df.groupby("topic").count()
    counts_df['support'] = counts_df['rating']
    counts_df['avg_rating'] = observations_df.groupby("topic").mean()['rating']
    counts_df['std_rating'] = observations_df.groupby("topic").std()['rating']
    del counts_df['rating']
    # to avoid multirow header:
    counts_df.index.name = None
    counts_df['rating_delta'] = counts_df['avg_rating'] - counts_df['avg_rating'].mean()
    # counts_df.sort_values('support', ascending=False)[:60]
    return counts_df.sort_values('support', ascending=False)


# Collection of topic-sentiment relation
def collect_topic_sentiments_observations(since_dt=None):
    """Method for collection of observation of topics - sentiments"""
    if not since_dt:
        since_dt = dt.datetime.now() - dt.timedelta(days=160)
    # collect human utterances
    # retrieve annotations and sentiments
    # put it into ibservation_df
    # utterance, noun_phrases, sentiment_category
    # or
    # utterance, noun_phrase, sentiment_category
    bot_user = Author.objects.get(user_type="bot")

    # all user utterances for the last week:
    utterances = Utterance.objects.filter(~Q(author=bot_user) & Q(timestamp__gte=since_dt))

    datas = []
    for each_utt in utterances:
        # get nounpharse annotation
        # get sentiment annotation
        sentiment_cat = each_utt.get_sentiment()
        noun_phrases = each_utt.get_noun_phrases()
        for each_np in noun_phrases:
            observation_dict = {
                'noun_phrase': each_np,
                'sentiment': sentiment_cat,
                # 'utterance': each_utt.text
            }
            datas.append(observation_dict)

    observations_df = pd.DataFrame(datas)
    return observations_df


def calc_topic_sentiments_statistics(since_dt=None):
    from nltk.probability import ConditionalFreqDist

    if not since_dt:
        # since_dt = dt.datetime.now() - dt.timedelta(days=3)
        since_dt = dt.datetime.now() - dt.timedelta(days=160)

    bot_user = Author.objects.get(user_type="bot")
    utterances = Utterance.objects.filter(~Q(author=bot_user) & Q(timestamp__gte=since_dt))

    # dictionary which stores nps as keys and values as dict with sentiments observations and counts?
    nps_sentiments = {}
    cfdist = ConditionalFreqDist()

    for each_utt in utterances:
        # get nounpharse annotation
        # get sentiment annotation
        sentiment_cat = each_utt.get_sentiment()
        noun_phrases = each_utt.get_noun_phrases()
        for each_np in noun_phrases:
            cfdist[each_np][sentiment_cat] += 1
            if each_np not in nps_sentiments:
                nps_sentiments[each_np] = {
                    "sentiments": [sentiment_cat]
                }
            else:
                nps_sentiments[each_np]['sentiments'].append(sentiment_cat)

    datas = []

    for each_key, each_dict in nps_sentiments.items():
        #     each_dict['support'] = len(each_dict['sentiments'])
        #     each_dict['sentiments']
        support = len(each_dict['sentiments'])
        obs_dict = {
            "noun_phrase": each_key,
            "support": support,
            "positives_count": cfdist[each_key]['positive'],
            "neutrals_count": cfdist[each_key]['neutral'],
            "negatives_count": cfdist[each_key]['negative'],
            "positives_ratio": float(cfdist[each_key]['positive']) / support,
            "neutrals_ratio": float(cfdist[each_key]['neutral']) / support,
            "negatives_ratio": float(cfdist[each_key]['negative']) / support,
        }
        datas.append(obs_dict)

    df = pd.DataFrame(datas)
    df.sort_values("support", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ####################################################################################
# Emotions analytics
# ######################################################
def get_skill_emotion_observations(skill_name, since_dt=None, version_filter=None):
    """The function collects a list of emotion observations for the human answers to the skill
    optionally you may filter data by since_dt
    since_dt: optional datetime - if provided then only utterance after the date will be analyzed
    version_filter: optional string - if provided then only data from specified version will be collected

    return pandas DataFrame with columns as emotions and rows as float estimations of each emotion in particular utterance

    Ex.:
            joy	        fear	    love	    anger	    neutral	    sadness	    surprise
        0	0.392808	0.199807	0.364568	0.315551	0.999187	0.239706	0.184356
        1	0.298255	0.235878	0.326469	0.287945	0.999280	0.273439	0.213362
        2	0.591905	0.184519	0.323955	0.484032	0.997095	0.443082	0.080949
        3	0.551933	0.182773	0.289024	0.444207	0.998263	0.312497	0.126133
        ...
    """

    # 1. select utterances of the skill

    s_utts = Utterance.objects.filter(active_skill=skill_name)
    if since_dt:
        # chain filter:
        s_utts = s_utts.filter(timestamp__gte=since_dt)

    # 2. retrieve all next utterances
    n_utts = [utt.next_utterance() for utt in s_utts if utt.next_utterance()]
    if version_filter:
        n_utts = [n_utt for n_utt in n_utts if n_utt.version == version_filter]
    # 3. collect annotation stat

    emotions_observations = []
    for each_n_utt in n_utts:
        emotion_dict = each_n_utt.get_emotions()
        emotions_observations.append(emotion_dict)

    emotions_df = pd.DataFrame(emotions_observations)
    return emotions_df


def collect_emotions_after_skill_statistics(renormalize=True, since_dt=None, version_filter=None, skills_filter=None):
    """
    Collects statistics of emotions distributions for Human answers for each skill.

    :param since_dt: datetime, date since which you need to collect data
    :param version_filter: string of version
    :param skills_filter: list of skill names to filter, by default all skills
    :param renormalize: renormalizes emotions estimations over skills (without normalization most of skills has
    neutral emotion with 0.99 quite always)

    :return: pd.DataFrame, Ex.:

                joy	        fear	    love	    anger	    neutral	    sadness	    surprise
        alice	0.305607	0.276640	0.301488	0.357288	0.993714	0.292200	0.213112
        eliza	0.281681	0.270212	0.316569	0.342887	0.996687	0.272544	0.224505
        cobotqa	0.319470	0.265375	0.312298	0.348641	0.992599	0.281040	0.217197
    """

    skill_stat_dict = {}
    if not skills_filter:
        skills_filter = skills
    for each_skill in skills_filter:

        # collect statistics of sentiment classes:
        emotions_df = get_skill_emotion_observations(each_skill, since_dt=since_dt, version_filter=version_filter)
        print("Support for skill %s: %d" % (each_skill, len(emotions_df)))

        # visualize classes as frequency statistics:
        if len(emotions_df) > 0:
            averaged_emotions = emotions_df.mean().to_dict()
            skill_stat_dict[each_skill] = averaged_emotions
    # import ipdb;ipdb.set_trace()

    avg_skill_emotions_df = pd.DataFrame(skill_stat_dict)
    avg_skill_emotions_df = avg_skill_emotions_df.T
    if renormalize:
        # this help to avoid biased estimations, when neutral is around 0.99 for every skill,
        # we renormalize it over all skills (for each emotion independently):
        emo_summas = avg_skill_emotions_df.sum(axis=0)
        normalized_skill_emo_df = avg_skill_emotions_df / emo_summas
        return normalized_skill_emo_df
    return avg_skill_emotions_df
