from flask_admin import BaseView, expose
import json
import os
import pandas as pd
import datetime as dt
from dateutil import tz
from plotly.offline import plot
from db.models import Conversation
from admin.admin import cache
from tqdm import tqdm
import datetime as dt

skill_names_map = {
     'meta_script_skill' : 'Activities Discussion',
     'comet_dialog_skill': 'Personal Events Discussion',
     'alice': 'Alice',
     'book_skill' : 'Book skill',
     'convert_reddit_with_personality' : 'ConveRT Reddit with Personality',
     'convert_reddit': 'ConveRT Reddit Retrieval',
     'eliza': 'Eliza',
     'game_cooperative_skill': 'Game Skill',
     'movie_skill': 'Movie Skill',
     'news_api_skill': 'News Skill',
     'program_y': 'AIML DREAM Chit-Chat',
     'program_y_dangerous': 'AIML Dangerous Topics',
     'program_y_wide': 'AIML General Chit-Chat',
     'reddit_ner_skill': 'NER-Skill on Reddit',
     'short_story_skill': 'Short-Story Skill',
     'topicalchat_convert_retrieval': 'TopicalChat ConveRT Retrieval',
    'intent_responder': 'Intent Responder',
    'cobotqa': 'CoBot QA',
    'valentines_day_skill': 'Valentines Day',
    'weather_skill': 'Weather Skill',
    'christmas_new_year_skill': 'Christmas & New Year',
    'coronavirus_skill' : 'Coronavirus Skill',
    'personal_info_skill' : 'Personal Info Skill',
    'emotion_skill' : 'Emotion Skill',
    'dummy_skill_dialog': 'Dummy Skill Dialog',
    'fashion_tfidf_retrieval': 'Fashion TF-IDF Retrieval',
    'sport_tfidf_retrieval': 'Sport TF-IDF Retrieval',
    'movie_tfidf_retrieval': 'Movie TF-IDF Retrieval',
    'music_tfidf_retrieval': 'Music TF-IDF Retrieval',
    'animals_tfidf_retrieval': 'Animals TF-IDF Retrieval',
    'entertainment_tfidf_retrieval': 'Entertainment TF-IDF Retrieval',
    'science_technology_tfidf_retrieval': 'Science TF-IDF Retrieval',
    'book_tfidf_retrieval': 'Book TF-IDF Retrieval',
    'tfidf_retrieval' : 'TF-IDF Retrieval',
    'superbowl_skill' : 'Superbowl',
    'misheard_asr' : 'Misheard ASR Skill',
    'oscar_skill' : 'Oscar',
    'small_talk_skill' : 'Small Talk Skill',
    'dummy_skill' : 'Dummy Skill',
}

def read_releases(path):
    """search file with releases and parses it to provide information about dates and releases.

    https://raw.githubusercontent.com/dilyararimovna/dp-dream-alexa/feat/releases/releases.txt
    """
    releases = pd.read_csv(path, sep=',')
    releases = releases.iloc[::-1]
    releases['date'] = pd.to_datetime(releases['date'], utc=True, format='%d.%m.%Y %H:%M')
    releases['release'] = releases['release'].apply(lambda x: x.replace('A/B:', ''))
    return releases

class OverviewChartsView(BaseView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO fix the shit, fuck the flask:
        from db.db import DBManager, get_session
        with open('core/config.json') as config_file:
            config = json.load(config_file)
        db_config = config['DB']
        db_config['user'] = db_config.get('user') or os.getenv('DB_USER')
        db_config['password'] = db_config.get('password') or os.getenv('DB_PASSWORD')
        db_config['host'] = db_config.get('host') or os.getenv('DB_HOST')
        db_config['dbname'] = db_config.get('dbname') or os.getenv('DB_NAME')
        self.session = get_session(db_config['user'], db_config['password'], db_config['host'],
                              db_config['dbname'])



    @expose('/')
    @cache.cached(timeout=80400)
    def index(self):
        """
        Main page for analytical overview
        :return:
        """
        print("Retrieve releases data...")
        releases = read_releases("releases.txt")
        print("releases")
        print(releases)

        # retrieve all dialogs
        print("retrieve all dialogs...")
        today = dt.date.today()
        weeks_ago = today - dt.timedelta(weeks=2)
        # weeks_ago = current_time - dt.timedelta(weeks=1)
        # weeks_ago = current_time - dt.timedelta(days=1)
        # dialogs = self.session.query(Conversation).order_by(
        #     Conversation.date_finish.desc()).all()
        dialogs = self.session.query(Conversation).filter(Conversation.date_finish > weeks_ago).filter(Conversation.date_finish < today).order_by(
            Conversation.date_finish.desc()).all()
        ############################################################
        print("calculate_skill_weights...")

        dialog_skills_weights_data = self.calculate_skill_weights(dialogs)
        print("dialog_skills_weights_data")
        print(dialog_skills_weights_data)
        # ratings_df = self.prepare_data_for_ratings_plots(dialogs)
        ############################################################
        print("preparing all data for plotting...")
        dialog_durations_df, skills_ratings_df, dialog_finished_df, ratings_df = self.prepare_all_data(dialogs)
        print("skills_ratings_df")
        print(skills_ratings_df)
        print("dialog_durations_df")
        print(dialog_durations_df)
        # retrieve data for skill frequency chart
        # prepare plot for it
        print("ratings_df")
        print(ratings_df)
        print("dialog_finished_df")
        print(dialog_finished_df)
        print("plot_number_of_dialogs_with_ratings_hrly...")
        try:
            hrly_dialogs_ratings_fig = self.plot_number_of_dialogs_with_ratings_hrly(ratings_df)
            hrly_dialogs_ratings_fig_div = plot(hrly_dialogs_ratings_fig, output_type='div', include_plotlyjs=False)
        except Exception as e:
            print("plot_number_of_dialogs_with_ratings_hrly failed to execute:")
            print(e)
            hrly_dialogs_ratings_fig_div = ""

        # Skill Ratings by Releases
        ratings_by_releases_fig = self.plot_ratings_by_releases(skills_ratings_df, releases)
        ratings_by_releases_fig_div = plot(ratings_by_releases_fig, output_type='div', include_plotlyjs=False)
        # retrieve data for Dialog time(sec), Daily chart

        # Skill Ratings by Releases (EMA 0.5)
        ratings_by_releases_ema_05_fig = self.plot_ratings_by_releases_ema_05(skills_ratings_df, releases, dialog_skills_weights_data)
        ratings_by_releases_ema_05_fig_div = plot(ratings_by_releases_ema_05_fig, output_type='div', include_plotlyjs=False)

        fig_versions_ratings_ema_more_fig = self.plot_ratings_by_releases_ema05_gt7(skills_ratings_df, releases, dialog_skills_weights_data)
        versions_ratings_ema_more_fig_div = plot(fig_versions_ratings_ema_more_fig, output_type='div',
                                                  include_plotlyjs=False)

        fig_versions_ratings_ema_less_fig = self.plot_ratings_by_releases_ema05_lt7(skills_ratings_df, releases, dialog_skills_weights_data)
        versions_ratings_ema_less_fig_div = plot(fig_versions_ratings_ema_less_fig, output_type='div',
                                                     include_plotlyjs=False)

        # Ratings by version
        version_ratings_fig = self.plot_ratings_by_version(skills_ratings_df)
        version_ratings_fig_div = plot(version_ratings_fig, output_type='div',
                                                 include_plotlyjs=False)
        # prepare plot of it
        print("plot_skills_durations...")
        dialog_time_fig, shares_n_utt_fig = self.plot_skills_durations(dialog_durations_df, releases)
        dialog_time_fig_div = plot(dialog_time_fig, output_type='div', include_plotlyjs=False)
        shares_n_utt_div = plot(shares_n_utt_fig, output_type='div', include_plotlyjs=False)

        context_dict = {
            # "plot_title": "Duration analysis",
            "dialog_time_figure_div": dialog_time_fig_div,
            "shares_n_utt_div": shares_n_utt_div,
            "hrly_dialogs_ratings_fig_div": hrly_dialogs_ratings_fig_div,
            "ratings_by_releases_fig_div": ratings_by_releases_fig_div,
            "ratings_by_releases_ema_05_fig_div": ratings_by_releases_ema_05_fig_div,
            "versions_ratings_ema_more_fig_div": versions_ratings_ema_more_fig_div,
            "versions_ratings_ema_less_fig_div": versions_ratings_ema_less_fig_div,
            "version_ratings_fig_div": version_ratings_fig_div,
        }

        skill_names = list(set(skills_ratings_df["active_skill"].values))
        ########################
        # Last skill in dialog, all
        print("plot_last_skill_in_dialog...")
        last_skill_fig = self.plot_last_skill_in_dialog(dialog_finished_df, skill_names)
        last_skill_fig_div = plot(last_skill_fig, output_type='div', include_plotlyjs=False)
        # return render_template('overview_charts.html', name=name)
        context_dict["last_skill_fig_div"] = last_skill_fig_div

        # ######################################
        # Ratings, hist
        print("plot_ratings_hists...")
        rating_hists_fig = self.plot_ratings_hists(skills_ratings_df)
        rating_hists_fig_div = plot(rating_hists_fig, output_type='div', include_plotlyjs=False)
        context_dict["rating_hists_fig_div"] = rating_hists_fig_div


        # ##########################################
        # TODO ratings by version?

        # ######################################
        # Rating by n_turns for last 7 days
        print("plot_rating_by_turns...")
        rating_by_n_turns_fig = self.plot_rating_by_turns(skills_ratings_df)
        rating_by_n_turns_fig_div = plot(rating_by_n_turns_fig, output_type='div', include_plotlyjs=False)
        context_dict["rating_by_n_turns_fig_div"] = rating_by_n_turns_fig_div

        # ######################################
        # Skill was selected, relative
        print("plot_skill_was_selected_relative...")
        daily_counts_relative_fig = self.plot_skill_was_selected_relative(skills_ratings_df, skill_names, releases)
        daily_counts_relative_fig_div = plot(daily_counts_relative_fig, output_type='div', include_plotlyjs=False)
        context_dict["daily_counts_relative_fig_div"] = daily_counts_relative_fig_div

        #
        print("plot_skills_ratings_ma_dialogs_with_gt_7_turns...")
        moving_avg_fig = self.plot_skills_ratings_ma_dialogs_with_gt_7_turns(skills_ratings_df, skill_names, releases)
        moving_avg_fig_div = plot(moving_avg_fig, output_type='div', include_plotlyjs=False)
        context_dict["moving_avg_fig_div"] = moving_avg_fig_div

        print("plot_skill_ratings_total_ma_n_turns_gt_7...")
        skill_ratings_total_ma_n_turns_gt_7_fig = self.plot_skill_ratings_total_ma_n_turns_gt_7(skills_ratings_df, releases)
        skill_ratings_total_ma_n_turns_gt_7_fig_div = plot(skill_ratings_total_ma_n_turns_gt_7_fig, output_type='div',
                                                           include_plotlyjs=False)
        context_dict["skill_ratings_total_ma_n_turns_gt_7_fig_div"] = skill_ratings_total_ma_n_turns_gt_7_fig_div

        #
        print("plot_dialog_finished_reason...")
        dialog_finished_reason_fig = self.plot_dialog_finished_reason(dialog_finished_df)
        dialog_finished_reason_fig_div = plot(dialog_finished_reason_fig, output_type='div',
                                              include_plotlyjs=False)
        context_dict["dialog_finished_reason_fig_div"] = dialog_finished_reason_fig_div

        # ####
        print("plot_dialog_finished_reasons_w_ratings...")
        dialog_finished_reason_w_rats_fig = self.plot_dialog_finished_reasons_w_ratings(dialog_finished_df)
        dialog_finished_reason_w_rats_fig_div = plot(dialog_finished_reason_w_rats_fig, output_type='div',
                                                     include_plotlyjs=False)
        context_dict["dialog_finished_reason_w_rats_fig_div"] = dialog_finished_reason_w_rats_fig_div

        #
        print("plot_last_skill_in_dialog_with_rating...")
        dialog_finished_skill_rating_day_fig = self.plot_last_skill_in_dialog_with_rating(dialog_finished_df,
                                                                                          skill_names, releases)
        dialog_finished_skill_rating_day_fig_div = plot(dialog_finished_skill_rating_day_fig, output_type='div',
                                                        include_plotlyjs=False)
        context_dict["dialog_finished_skill_rating_day_fig_div"] = dialog_finished_skill_rating_day_fig_div

        # ##
        print("plot_last_skill_stop_exit...")
        last_skill_stop_exit_info_fig = self.plot_last_skill_stop_exit(dialog_finished_df, skill_names)
        last_skill_stop_exit_info_fig_div = plot(last_skill_stop_exit_info_fig, output_type='div',
                                                 include_plotlyjs=False)
        context_dict["last_skill_stop_exit_info_fig_div"] = last_skill_stop_exit_info_fig_div
        print("render!")
        return self.render('overview_charts.html', **context_dict)

    def prepare_all_data(self, dialogs):
        """
        Uses utterances from attribute of Conversation
        - prepare_data_for_plotting
        -prepare_dialog_finished_df
        :param dialogs:
        :return:
        """
        skills_ratings = []
        dialog_durations = []
        def get_last_skill(dialog, exit_intent=False):
            try:
                if exit_intent and len(dialog.raw_utterances) >= 3 and 'active_skill' in dialog.raw_utterances[-3]:
                    return dialog.raw_utterances[-3]['active_skill']
                return dialog.raw_utterances[-1]['active_skill']
            except Exception as e:
                print(e)
                # print("dialog.raw_utterances:")
                # print(dialog.raw_utterances)
                print("Exception")
                print(e)
                return "UnrecognizedSkill"

        dialog_finished_data = []

        rating_data = []
        for dialog in tqdm(dialogs, desc="Analyzing conversations"):
            if dialog.rating:
                rating = float(dialog.rating)
                has_rating = True
            else:
                rating = 'no_rating'
                has_rating = False
            date = dialog.date_start
            alexa_command = 'no_alexa_command'
            bot_respond_with_goodbye = False
            # n_turns = len(dialog.utterances) // 2
            n_utt = len(dialog.raw_utterances)
            # n_utt = len(list(dialog.utterances))
            n_turns = n_utt // 2
            last_skill = None

            # if 'alexa_commands' in dialog:
            #     alexa_command = dialog['alexa_commands'][0]['text']
            #     last_skill = get_last_skill(dialog)

            # if '#+#exit' in dialog.raw_utterances[-1].text:
            if '#+#exit' in dialog.raw_utterances[-1]['text']:
                bot_respond_with_goodbye = True
                last_skill = get_last_skill(dialog, exit_intent=True)

            if last_skill is None:
                last_skill = get_last_skill(dialog)

            no_command_no_goodbye = (alexa_command == 'no_alexa_command') and not bot_respond_with_goodbye

            conv_id = dialog.id

            dialog_finished_data += [
                [date, alexa_command, bot_respond_with_goodbye, no_command_no_goodbye, rating,
                 has_rating, n_turns, last_skill, conv_id, None]]
                 # has_rating, n_turns, last_skill, conv_id, dialog['version']]]

            if has_rating:
                if dialog.date_start:
                    rating_data.append({
                        'id': dialog.id,
                        'rating': dialog.rating,
                        'start_time': dialog.date_start
                    })
                time = (dialog.date_finish - dialog.date_start).seconds

                dialog_durations += [[date, time, n_utt]]

                for utt in dialog.raw_utterances:
                    # if hasattr(utt, 'active_skill'):
                    if 'active_skill' in utt and utt['active_skill']:
                        #         skills_ratings += [[date, utt.active_skill, rating, conv_id, dialog.version]]
                        skills_ratings += [[date, utt['active_skill'], rating, conv_id, None]]

        ratings_df = pd.DataFrame(rating_data)

        dialog_finished_df = pd.DataFrame(dialog_finished_data,
                                          columns=['date', 'alexa_command', 'bot_goodbye',
                                                   'no_command_no_goodbye', 'rating', 'has_rating',
                                                   'n_turns', 'last_skill', 'conv_id', 'version'])
        dialog_finished_df['date'] = pd.to_datetime(dialog_finished_df['date'], utc=True)

        skills_ratings = pd.DataFrame(skills_ratings,
                                      columns=['date', 'active_skill', 'rating', 'conv_id',
                                               'version'])
        skills_ratings['date'] = pd.to_datetime(skills_ratings['date'], utc=True)
        #
        n_turns = skills_ratings['conv_id'].value_counts().to_dict()
        skills_ratings['n_turns'] = skills_ratings['conv_id'].apply(lambda x: n_turns[x])
        # print("skills_ratings")
        # print(skills_ratings)
        # skills_ratings = skills_ratings[skills_ratings['n_turns'] > 1]

        dialog_durations = pd.DataFrame(dialog_durations, columns=['date', 'time', 'n_utt'])
        dialog_durations['date'] = pd.to_datetime(dialog_durations['date'], utc=True)

        return dialog_durations, skills_ratings, dialog_finished_df, ratings_df

    def calculate_skill_weights(self, dialogs):
        from collections import defaultdict

        def get_skills_weights(dialog, alpha):
            skills_weights = defaultdict(int)

            # for utt in dialog['utterances']:
            for utt in dialog.raw_utterances:
                # print("utt repr:")
                # print(utt)
                # if utt['spk'] == 'Bot':
                if utt['user']['user_type'] == 'human':
                    pass
                else:
                # if utt['user']['user_type'] == 'bot':
                #     print(utt['user']['user_type'] )
                    active_skill = utt['active_skill']
                    for sn in skills_weights:
                        skills_weights[sn] *= (1 - alpha)
                    skills_weights[active_skill] += 1 * alpha
            return skills_weights

        def get_skills_active_n(dialog):
            skills_active_n = defaultdict(int)
            # for utt in dialog['utterances']:
            for utt in dialog.raw_utterances:
                # if utt['spk'] == 'bot':
                if utt['user']['user_type'] == 'human':
                    pass
                else:
                    active_skill = utt['active_skill']
                    skills_active_n[active_skill] += 1
            return skills_active_n

        get_skills_weights(dialogs[0], alpha=0.25), get_skills_active_n(dialogs[0])

        # prepare dataframes with weighted rating
        ema_alphas = [0.5, 0.2]
        dialog_skills_weights_data = []
        for dialog in dialogs:
            r = dialog.rating
            # v = dialog.version
            if r == 'no_rating':
                continue
            conv_id = dialog.amazon_conv_id
            # date = dialog['first_utt_time']
            date = dialog.date_start
            skills_active_n = get_skills_active_n(dialog)
            to_add = {
                'conv_id': conv_id,
                'rating': r,
                # 'version': v,
                'date': date,
                # 'n_turns': len(dialog['utterances']) // 2
                'n_turns': len(dialog.raw_utterances) // 2
            }
            for a in ema_alphas:
                skills_weights = get_skills_weights(dialog, alpha=a)
                for sn in skills_weights:
                    to_add[f'{sn}_{a}_w'] = skills_weights[sn]
            for sn in skills_active_n:
                to_add[f'{sn}_n'] = skills_active_n[sn]
            dialog_skills_weights_data += [to_add]

        dialog_skills_weights_data = pd.DataFrame(dialog_skills_weights_data)
        dialog_skills_weights_data['date'] = pd.to_datetime(dialog_skills_weights_data['date'], utc=True)
        dialog_skills_weights_data = dialog_skills_weights_data.fillna(0)
        return dialog_skills_weights_data

    def plot_skills_durations(self, dialog_durations_df, releases):
        """
        - Dialog time(sec), Daily
        - Avg number of utterances, Daily
        - Number of utterances, distribution, Daily

        :param dialog_durations_df:
        :return:
        """
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        dialog_time_fig = make_subplots(rows=2, cols=1, subplot_titles=(
            'Dialog time(sec), Daily', 'Avg number of utterances, Daily',))

        import datetime as dt
        now = dt.datetime.now(tz=tz.gettz('UTC'))
        # now = dt.datetime.now()
        end = now
        # start = end - dt.timedelta(days=50)
        min_by_dialogs_dt = dialog_durations_df['date'].min().replace(tzinfo=tz.gettz('UTC'))
        start = max(end - dt.timedelta(days=50), min_by_dialogs_dt)


        max_lens = [2, 4, 8, 12, 16, 24, 32, 48, 64]
        time_ = []
        n_utt = []
        x = []
        utt_shares = [[] for len_ in max_lens]
        for dt in pd.date_range(start=start, end=end, freq='1D'):
            daily_times = dialog_durations_df[
                (dialog_durations_df['date'] < dt) & (
                            dialog_durations_df['date'] >= dt - dt.freq * 1)]
            if len(daily_times) > 0:
                time_.append(sum(daily_times['time']) / len(daily_times['time']))
                n_utt.append(sum(daily_times['n_utt']) / len(daily_times['n_utt']))
                for i, len_ in enumerate(max_lens):
                    utt_shares[i].append(
                        (len(daily_times[daily_times['n_utt'] <= len_]) + 0.0) / (len(daily_times)))
                x.append(dt)
        dialog_time_fig.add_trace(go.Scatter(x=x, y=time_,
                                         name='Average dialog time(sec)', line={'dash': 'dot'},
                                         # marker={'size': 8}))
                                         marker={'size': 8}), row=1, col=1)

        dialog_time_fig.add_trace(go.Scatter(x=x, y=n_utt,
                                         name='Average number of utterances', line={'dash': 'dot'},
                                         # marker={'size': 8}))
                                         marker={'size': 8}), row=2, col=1)
        ###################
        for d, r in releases.values:
            if d > start:
                dialog_time_fig.add_shape(
                    dict(type="line", x0=d, y0=0, x1=d, y1=200, line=dict(color="RoyalBlue", width=1)),
                    row=1, col=1)
                dialog_time_fig.add_annotation(x=d, y=200, text=r, textangle=-90, showarrow=True,
                                           font=dict(color="black", size=10), opacity=0.7, row=1, col=1)
                dialog_time_fig.add_shape(
                    dict(type="line", x0=d, y0=10, x1=d, y1=35, line=dict(color="RoyalBlue", width=1)),
                    row=2, col=1)
                dialog_time_fig.add_annotation(x=d, y=35, text=r, textangle=-90, showarrow=True,
                                           font=dict(color="black", size=10), opacity=0.7, row=2, col=1)

        #############
        dialog_time_fig.update_layout(height=500, width=1300, showlegend=True)
        dialog_time_fig['layout']['yaxis1']['range'] = [min(time_)*0.9, max(time_)*1.1]
        dialog_time_fig['layout']['yaxis2']['range'] = [0, max(n_utt)*1.1]
        dialog_time_fig.update_layout(hovermode='x')



        # shares of utterances by lengths
        shares_n_utt_fig = make_subplots(rows=1, cols=1,
                                     subplot_titles=(['Number of utterances, distribution, Daily']))
        # dialog_time.show()
        for i in range(len(utt_shares)):
            shares_n_utt_fig.add_trace(go.Scatter(x=x, y=utt_shares[i],
                                              name='n_utts<=' + str(max_lens[i]),
                                              line={'dash': 'dot'}, marker={'size': 8}), row=1,
                                   col=1)
        for d, r in releases.values:
            if d > start:
                shares_n_utt_fig.add_shape(
                    dict(type="line", x0=d, y0=0, x1=d, y1=1, line=dict(color="RoyalBlue", width=1)),
                    row=1, col=1)
                shares_n_utt_fig.add_annotation(x=d, y=1, text=r, textangle=-90, showarrow=True,
                                            font=dict(color="black", size=10), opacity=0.7, row=1,
                                            col=1)
        #
        shares_n_utt_fig.update_layout(height=500, width=1300, showlegend=True)
        shares_n_utt_fig['layout']['yaxis1']['range'] = [-0.05, 1.05]
        shares_n_utt_fig.update_layout(hovermode='x')
        # shares_n_utt.show()
        # shares_n_utt_div = plot(shares_n_utt, output_type='div', include_plotlyjs=False)


        # return render(request, 'dialogs/skills_sentiment_stacked.html', context_dict)
        return dialog_time_fig, shares_n_utt_fig

    def plot_last_skill_in_dialog(self, dialog_finished_df, skill_names):
        """
        Last skill in dialog, all

        :param dialog_finished_df:
        :return:
        """
        from plotly.subplots import make_subplots
        # import plotly.graph_objects as go
        import datetime as dt

        fig_dialog_finished_skill_all_day = make_subplots(rows=1, cols=1, subplot_titles=(
        'Last skill in dialog, all',))

        now = dt.datetime.now(tz=tz.gettz('UTC'))
        end = now
        start = end - dt.timedelta(days=60)
        # start = end - dt.timedelta(days=14)

        x = dict()
        value_v = dict()
        value_c = dict()


        # skill_names = set(skill_names)
        # skill_names = set(dialog_finished_df['last_skill'].values)
        # print(skill_names)
        for n in skill_names:
            value_c[n] = []
            value_v[n] = []
            x[n] = []

        for date in pd.date_range(start=start, end=end, freq='D'):
            daily_data = dialog_finished_df[
                (dialog_finished_df['date'] < date) & (dialog_finished_df['date'] >= date - date.freq)]
            if len(daily_data)<1:
                continue
            # daily_data = daily_data[daily_data['alexa_command'] == '/alexa_stop_handler']
            for sn in skill_names:
                d = daily_data[daily_data['last_skill'] == sn]
                if len(d) > 2:
                    value_v[sn] += [len(d) / len(daily_data)]
                    # value_c[sn] += [[len(d), d['rating'].mean(), d['n_turns'].mean()]]
                    value_c[sn] += [[len(d), d['n_turns'].mean()]]
                    x[sn] += [date]

        min_v, max_v = 10 ** 10, - 10 ** 10

        for sn in sorted(list(skill_names)):
            if len(value_v[sn]) > 0:
                print(sn)
                # print(sn)

                fig_dialog_finished_skill_all_day.add_scatter(name=sn, x=x[sn], y=value_v[sn],
                                                              customdata=value_c[sn],
                                                              line={'dash': 'dot'},
                                                              hovertemplate='%{y:.2f}: count: %{customdata[0]} n_turns: %{customdata[1]:.2f}',
                                                              row=1, col=1)
                min_v = min(min_v, min(value_v[sn]))
                max_v = max(max_v, max(value_v[sn]))

        # for d, r in releases.values:
        #     if d > start:
        #         fig_dialog_finished_skill_all_day.add_shape(
        #             dict(type="line", x0=d, y0=min_v, x1=d, y1=max_v,
        #                  line=dict(color="RoyalBlue", width=1)), row=1, col=1)
        #         fig_dialog_finished_skill_all_day.add_annotation(x=d, y=max_v, text=r,
        #                                                          textangle=-90, showarrow=True,
        #                                                          font=dict(color="black", size=10),
        #                                                          opacity=0.7, row=1, col=1)

        fig_dialog_finished_skill_all_day.update_layout(height=500, width=1300, showlegend=True)
        # fig_dialog_finished_skill_day['layout']['yaxis1']['range'] = [0, 0.5]
        fig_dialog_finished_skill_all_day.update_layout(hovermode='x')
        # fig_dialog_finished_skill_all_day.show()
        return fig_dialog_finished_skill_all_day

    def prepare_data_for_ratings_plots(self, dialogs):
        """
        Prepares dataframe for some plots
        :param dialogs:
        :return:
        """

        data = []
        for each_dialog in dialogs:
            data.append({
                'id': each_dialog.id,
                'rating': each_dialog.rating,
                'start_time': each_dialog.date_start
            })
        df = pd.DataFrame(data)
        return df

    def plot_ratings_by_releases(self, skills_ratings, releases):
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        import datetime as dt

        fig_versions_ratings = make_subplots(rows=1, cols=1, subplot_titles=('Skills Ratings by releases',))

        now = dt.datetime.now(tz=tz.gettz('UTC'))
        end = now
        start = end - dt.timedelta(days=14)

        min_n_active_skill = 10

        x = dict()
        skill_r = dict()
        skill_z = dict()
        skill_names = set(skills_ratings['active_skill'].unique()) - set(['no_skill_name']) | set(['_total'])
        for n in skill_names:
            skill_r[n] = []
            x[n] = []
            skill_z[n] = []

        min_r, max_r = 5, 0
        releases_reversed = list(reversed(releases.values))
        for i, (d_start, rel) in enumerate(releases_reversed):
            if i == len(releases) - 1:
                d_end = now
            else:
                d_end = releases_reversed[i + 1][0]
            versions = rel.split('/')
            release_ratings = skills_ratings[(skills_ratings['date'] < d_end) & (skills_ratings['date'] >= d_start)]
            if len(release_ratings.groupby('conv_id').first()) < 50:
                continue
            #     release_ratings = release_ratings[release_ratings['version'].isin(versions)]
            for (sn, r), (_, c) in zip(release_ratings.groupby('active_skill')['rating'].mean().items(),
                                       release_ratings.groupby('active_skill')['rating'].count().items()):
                if sn in skill_names:
                    #             if c < min_n_active_skill:
                    #                 continue
                    skill_r[sn] += [r]
                    x[sn] += [f'{d_end.date()} {rel}']
                    skill_z[sn] += [c]
            sn = '_total'
            d = release_ratings.groupby('conv_id').first()
            skill_r[sn] += [d['rating'].mean()]
            x[sn] += [f'{d_end.date()} {rel}']
            skill_z[sn] += [len(d)]

        for n in sorted(list(skill_names), key=str.lower):
            if len(skill_r[n]) > 0:
                fig_versions_ratings.add_trace(go.Scatter(name=n, x=x[n], y=skill_r[n], customdata=skill_z[n],
                                                          hovertemplate='%{y:.2f}: count %{customdata}',
                                                          line_shape='hvh',
                                                          line={'dash': 'dot'}, marker={'size': 8}), row=1, col=1)
                min_r = min(min_r, min(skill_r[n]))
                max_r = max(max_r, max(skill_r[n]))

        fig_versions_ratings.update_layout(height=500, width=1300, showlegend=True, )
        fig_versions_ratings['layout']['yaxis1']['range'] = [min_r - 0.1, max_r + 0.1]
        fig_versions_ratings.update_layout(hovermode='x', xaxis={'type': 'category'})
        # fig_versions_ratings.show()
        return fig_versions_ratings

    def plot_ratings_by_releases_ema_05(self, skills_ratings, releases, dialog_skills_weights_data):
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        import datetime as dt
        ema_alpha = 0.5
        fig_versions_ratings_ema = make_subplots(rows=1, cols=1,
                                                 subplot_titles=(f'Skills Ratings by releases, EMA ({ema_alpha})',))

        now = dt.datetime.now(tz=tz.gettz('UTC'))
        end = now
        start = end - dt.timedelta(days=14)

        x = dict()
        skill_r = dict()
        skill_z = dict()
        skill_names = set(skills_ratings['active_skill'].unique()) - set(['no_skill_name']) | set(['_total'])
        for n in skill_names:
            skill_r[n] = []
            x[n] = []
            skill_z[n] = []

        min_r, max_r = 5, 0
        releases_reversed = list(reversed(releases.values))
        for i, (d_start, rel) in enumerate(releases_reversed):
            if i == len(releases) - 1:
                d_end = now
            else:
                d_end = releases_reversed[i + 1][0]
            versions = rel.split('/')
            release_ratings = dialog_skills_weights_data[
                (dialog_skills_weights_data['date'] < d_end) & (dialog_skills_weights_data['date'] >= d_start)]
            if len(release_ratings) < 50:
                continue
            #     release_ratings = release_ratings[release_ratings['version'].isin(versions)]

            for sn in skill_names:
                if sn != '_total':
                    skill_active_n = release_ratings[f'{sn}_n'].sum()
                    if release_ratings[f'{sn}_{ema_alpha}_w'].sum() > 0:
                        r = (release_ratings[f'{sn}_{ema_alpha}_w'] * release_ratings['rating']).sum() / \
                            release_ratings[f'{sn}_{ema_alpha}_w'].sum()
                        skill_r[sn] += [r]
                        x[sn] += [f'{d_end.date()} {rel}']
                        skill_z[sn] += [skill_active_n]
                else:
                    skill_r[sn] += [release_ratings['rating'].mean()]
                    x[sn] += [f'{d_end.date()} {rel}']
                    skill_z[sn] += [len(release_ratings)]

        for n in sorted(list(skill_names), key=str.lower):
            if len(skill_r[n]) > 0:
                fig_versions_ratings_ema.add_trace(go.Scatter(name=n, x=x[n], y=skill_r[n], customdata=skill_z[n],
                                                              hovertemplate='%{y:.2f}: count %{customdata}',
                                                              line_shape='hvh',
                                                              line={'dash': 'dot'}, marker={'size': 8}), row=1, col=1)
                min_r = min(min_r, min(skill_r[n]))
                max_r = max(max_r, max(skill_r[n]))

        fig_versions_ratings_ema.update_layout(height=500, width=1300, showlegend=True, )
        fig_versions_ratings_ema['layout']['yaxis1']['range'] = [min_r - 0.1, max_r + 0.1]
        fig_versions_ratings_ema.update_layout(hovermode='x', xaxis={'type': 'category'})

        # fig_versions_ratings_ema.show()
        return fig_versions_ratings_ema

    def plot_ratings_by_releases_ema05_gt7(self, skills_ratings, releases, dialog_skills_weights_data):
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        import datetime as dt

        n_turns = 7
        ema_alpha = 0.5
        fig_versions_ratings_ema_more = make_subplots(rows=1, cols=1, subplot_titles=(
        f'Skills Ratings by releases, EMA ({ema_alpha}), n_turns > {n_turns}',))

        now = dt.datetime.now(tz=tz.gettz('UTC'))
        end = now
        start = end - dt.timedelta(days=14)

        x = dict()
        skill_r = dict()
        skill_z = dict()
        skill_names = set(skills_ratings['active_skill'].unique()) - set(['no_skill_name']) | set(['_total'])
        for n in skill_names:
            skill_r[n] = []
            x[n] = []
            skill_z[n] = []

        min_r, max_r = 5, 0
        releases_reversed = list(reversed(releases.values))
        for i, (d_start, rel) in enumerate(releases_reversed):
            if i == len(releases) - 1:
                d_end = now
            else:
                d_end = releases_reversed[i + 1][0]
            versions = rel.split('/')
            release_ratings = dialog_skills_weights_data[
                (dialog_skills_weights_data['date'] < d_end) & (dialog_skills_weights_data['date'] >= d_start)]
            release_ratings = release_ratings[release_ratings['n_turns'] > n_turns]
            if len(release_ratings) < 50:
                continue
            #     release_ratings = release_ratings[release_ratings['version'].isin(versions)]

            for sn in skill_names:
                if sn != '_total':
                    skill_active_n = release_ratings[f'{sn}_n'].sum()
                    if release_ratings[f'{sn}_{ema_alpha}_w'].sum() > 0:
                        r = (release_ratings[f'{sn}_{ema_alpha}_w'] * release_ratings['rating']).sum() / \
                            release_ratings[f'{sn}_{ema_alpha}_w'].sum()
                        skill_r[sn] += [r]
                        x[sn] += [f'{d_end.date()} {rel}']
                        skill_z[sn] += [skill_active_n]
                else:
                    skill_r[sn] += [release_ratings['rating'].mean()]
                    x[sn] += [f'{d_end.date()} {rel}']
                    skill_z[sn] += [len(release_ratings)]

        for n in sorted(list(skill_names), key=str.lower):
            if len(skill_r[n]) > 0:
                fig_versions_ratings_ema_more.add_trace(go.Scatter(name=n, x=x[n], y=skill_r[n], customdata=skill_z[n],
                                                                   hovertemplate='%{y:.2f}: count %{customdata}',
                                                                   line_shape='hvh',
                                                                   line={'dash': 'dot'}, marker={'size': 8}), row=1,
                                                        col=1)
                min_r = min(min_r, min(skill_r[n]))
                max_r = max(max_r, max(skill_r[n]))

        fig_versions_ratings_ema_more.update_layout(height=500, width=1300, showlegend=True, )
        fig_versions_ratings_ema_more['layout']['yaxis1']['range'] = [min_r - 0.1, max_r + 0.1]
        fig_versions_ratings_ema_more.update_layout(hovermode='x', xaxis={'type': 'category'})
        # fig_versions_ratings_ema_more.show()
        return fig_versions_ratings_ema_more

    def plot_ratings_by_releases_ema05_lt7(self, skills_ratings, releases, dialog_skills_weights_data):
        import datetime as dt
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        n_turns = 7
        ema_alpha = 0.5
        fig_versions_ratings_ema_less = make_subplots(rows=1, cols=1, subplot_titles=(
        f'Skills Ratings by releases, EMA ({ema_alpha}), n_turns <= {n_turns}',))

        now = dt.datetime.now(tz=tz.gettz('UTC'))
        end = now
        start = end - dt.timedelta(days=14)

        x = dict()
        skill_r = dict()
        skill_z = dict()
        skill_names = set(skills_ratings['active_skill'].unique()) - set(['no_skill_name']) | set(['_total'])
        for n in skill_names:
            skill_r[n] = []
            x[n] = []
            skill_z[n] = []

        min_r, max_r = 5, 0
        releases_reversed = list(reversed(releases.values))
        for i, (d_start, rel) in enumerate(releases_reversed):
            if i == len(releases) - 1:
                d_end = now
            else:
                d_end = releases_reversed[i + 1][0]
            versions = rel.split('/')
            release_ratings = dialog_skills_weights_data[
                (dialog_skills_weights_data['date'] < d_end) & (dialog_skills_weights_data['date'] >= d_start)]
            release_ratings = release_ratings[release_ratings['n_turns'] <= n_turns]
            if len(release_ratings) < 50:
                continue
            #     release_ratings = release_ratings[release_ratings['version'].isin(versions)]

            for sn in skill_names:
                if sn != '_total':
                    skill_active_n = release_ratings[f'{sn}_n'].sum()
                    if release_ratings[f'{sn}_{ema_alpha}_w'].sum() > 0:
                        r = (release_ratings[f'{sn}_{ema_alpha}_w'] * release_ratings['rating']).sum() / \
                            release_ratings[f'{sn}_{ema_alpha}_w'].sum()
                        skill_r[sn] += [r]
                        x[sn] += [f'{d_end.date()} {rel}']
                        skill_z[sn] += [skill_active_n]
                else:
                    skill_r[sn] += [release_ratings['rating'].mean()]
                    x[sn] += [f'{d_end.date()} {rel}']
                    skill_z[sn] += [len(release_ratings)]

        for n in sorted(list(skill_names), key=str.lower):
            if len(skill_r[n]) > 0:
                fig_versions_ratings_ema_less.add_trace(go.Scatter(name=n, x=x[n], y=skill_r[n], customdata=skill_z[n],
                                                                   hovertemplate='%{y:.2f}: count %{customdata}',
                                                                   line_shape='hvh',
                                                                   line={'dash': 'dot'}, marker={'size': 8}), row=1,
                                                        col=1)
                min_r = min(min_r, min(skill_r[n]))
                max_r = max(max_r, max(skill_r[n]))

        fig_versions_ratings_ema_less.update_layout(height=500, width=1300, showlegend=True, )
        fig_versions_ratings_ema_less['layout']['yaxis1']['range'] = [min_r - 0.1, max_r + 0.1]
        fig_versions_ratings_ema_less.update_layout(hovermode='x', xaxis={'type': 'category'})
        # fig_versions_ratings_ema_less.show()
        return fig_versions_ratings_ema_less


    def plot_ratings_hists(self, skills_ratings):
        """
        Ratings, hist
        :param skills_ratings:
        :return:
        """
        from plotly.subplots import make_subplots
        import datetime as dt

        fig_daily_hist_ratings = make_subplots(rows=1, cols=1,
                                               subplot_titles=('Ratings, hist',))

        now = dt.datetime.now(tz=tz.gettz('UTC'))
        # now = dt.datetime.now()
        end = now
        start = end - dt.timedelta(days=31)

        x = dict()
        skill_r = dict()
        skill_c = dict()
        ratings_values = list(range(6))
        for n in ratings_values:
            skill_r[n] = []
            skill_c[n] = []
            x[n] = []
        for date in pd.date_range(start=start, end=end, freq='D'):
            daily_ratings = skills_ratings[
                (skills_ratings['date'] < date) & (skills_ratings['date'] >= date - date.freq * 1)]
            d = daily_ratings.groupby('conv_id').first()
            d['rating_round'] = d['rating'].apply(round)
            rating_counts = d.groupby('rating_round').count()['rating']
            for r, v in rating_counts.items():
                skill_r[r] += [v / len(d)]
                skill_c[r] += [v]
                x[r] += [date]
        # import ipdb; ipdb.set_trace()
        # for n in skill_names:
        #    fig_daily_hist_ratings.add_trace(go.Scatter(x=x[n], y=skill_r[n], name=n, line={'dash': 'dot'}, marker={'size':8}), row=1, col=1)
        for r in ratings_values:
            fig_daily_hist_ratings.add_bar(name=r, x=x[r], y=skill_r[r], customdata=skill_c[r],
                                           hovertemplate='%{y:.2f}: count: %{customdata}')

        fig_daily_hist_ratings.update_layout(height=500, width=1300, showlegend=True)
        fig_daily_hist_ratings['layout']['yaxis1']['range'] = [0, 1]
        fig_daily_hist_ratings.update_layout(hovermode='x', barmode='stack')
        # fig_daily_hist_ratings.show()
        return fig_daily_hist_ratings

    def plot_ratings_by_version(self, skills_ratings):
        """
        Ratings by version
        :return:
        """

        def make_comparable(v):
            s = []
            if v == 'GOOD_OLD_BOT':
                return [9, 6, 1]
            if v == 'GOOD_NEW_BOT':
                return [10, 0, 0]
            if v is None:
                return [0, 0, 0]
            for c in v.split('.'):
                try:
                    s += [int(c)]
                except ValueError:
                    s += [int(c.split('-')[0])]
                    s += [0]
            return s

        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        fig_version_ratings = make_subplots(rows=1, cols=1, subplot_titles=('Ratings by version',))

        versions = set(skills_ratings['version']) - set(['no_info'])

        x = dict()
        version_r = dict()
        version_c = dict()
        ratings_values = list(range(6))
        for n in ratings_values:
            version_r[n] = []
            version_c[n] = []
            x[n] = []

        for ver in sorted(versions, key=make_comparable):
            version_ratings = skills_ratings[skills_ratings['version'] == ver]
            d = version_ratings.groupby('conv_id').first()
            d['rating_round'] = d['rating'].apply(round)
            if len(d) < 50:
                continue
            rating_counts = d.groupby('rating_round').count()['rating']
            avg_r = d['rating'].mean()
            for r, v in rating_counts.items():
                version_r[r] += [v / len(d)]
                version_c[r] += [[v, avg_r]]
                x[r] += [ver]

        for r in ratings_values:
            fig_version_ratings.add_bar(name=r, x=x[r], y=version_r[r], customdata=version_c[r],
                                        hovertemplate='%{y:.2f}: count: %{customdata[0]} avg_rating: %{customdata[1]:.2f}')

        fig_version_ratings.update_layout(height=500, width=1300, showlegend=True)
        fig_version_ratings.update_layout(hovermode='x', barmode='stack', xaxis={'type': 'category'})
        # fig_version_ratings.show()
        return fig_version_ratings

    def plot_rating_by_turns(self, skills_ratings):
        """
        Rating by n_turns for last 7 days

        :param skills_ratings:
        :return:
        """
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        import datetime as dt

        x = []
        y = []
        z = []
        # n_days = 30
        n_days = 7
        max_n = 30

        now = dt.datetime.now(tz=tz.gettz('UTC'))
        start_date = (now - dt.timedelta(days=n_days))
        start_date = pd.Timestamp(start_date)
        daily_ratings = skills_ratings[skills_ratings['date'] >= start_date]
        # import ipdb; ipdb.set_trace()
        # print("daily_ratings")
        # print(daily_ratings)

        count = daily_ratings.groupby(['n_turns', 'rating']).count()['date']
        print("count var:")
        print(count)
        print(count.max())
        for i in range(1, max_n):
            try:
                print(f"keys for {i} turns: {count.loc[i].keys()}")
                for j in (count.loc[i].keys()):
                    #        if count[i][j] // i > 0:
                    x.append(i)
                    y.append(j)
                    # TODO why do we divide by i (num steps?)?
                    z.append(count[i][j] // i)
            except Exception as e:
                # skip because some keys may absent
                print("missed key?")
                print(e)
                pass
        print("x,y,z:")
        print(x)
        print(y)

        print(z)
        max_z = max(z)


        zs = [j / max_z * 100.0 for j in z]
        print(zs)
        rating_by_n_turns_fig = go.Figure(data=[go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                # TODO fix bug with sizes of markers for the case when we in lack of data
                size=zs,
                # size=[j / 1.5 for j in z],
            ),
            customdata=z,
            hovertemplate='%{y:.2f}: count: %{customdata}',
            name='Rating by n_utt'
        )])
        rating_by_n_turns_fig.update_layout(title='Rating by n_turns for last {:d} days'.format(n_days),
                                            # )
                                            showlegend=False, height=500, width=1300)
        rating_by_n_turns_fig['layout']['yaxis']['range'] = [0.1, 5.9]
        rating_by_n_turns_fig['layout']['xaxis']['title'] = {'text': 'n_turns'}
        rating_by_n_turns_fig['layout']['yaxis']['title'] = {'text': 'rating'}
        # rating_by_n_turns_fig.show()
        return rating_by_n_turns_fig

    def plot_skill_was_selected_relative(self, skills_ratings, skill_names, releases):
        """
        Skill was selected, relative

        :param skills_ratings:
        :param skill_names:
        :return:
        """
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        import datetime as dt

        fig_daily_counts_relative = make_subplots(rows=1, cols=1,
                                                  subplot_titles=('Skill was selected, relative',))

        now = dt.datetime.now(tz=tz.gettz('UTC'))
        end = now
        # start = end - dt.timedelta(days=31)
        start = end - dt.timedelta(days=14)

        x = dict()
        skill_c = dict()
        skill_names = set(skill_names)
        skill_z = dict()
        for n in skill_names:
            skill_c[n] = []
            x[n] = []
            skill_z[n] = []

        # import ipdb; ipdb.set_trace()

        for dt in pd.date_range(start=start, end=end, freq='D'):
            daily_ratings = skills_ratings[(skills_ratings['date'] < dt) & (skills_ratings['date'] >= dt - dt.freq * 1)]
            for sn, c in daily_ratings.groupby('active_skill')['rating'].count().items():
                if sn in skill_names:
                    skill_c[sn] += [c / len(daily_ratings)]
                    x[sn] += [dt]
                    skill_z[sn] += [c]

        min_x, max_x = 1e10, 0
        for n in sorted(list(skill_names)):
            if len(skill_c[n]) > 0:
                fig_daily_counts_relative.add_trace(
                    go.Scatter(x=x[n], y=skill_c[n], customdata=skill_z[n], name=n, line={'dash': 'dot'},
                               marker={'size': 8},
                               line_shape='hvh',
                               hovertemplate='%{y:.3f}: count %{customdata}'),
                    row=1, col=1)
                min_x = min(min_x, min(skill_c[n]))
                max_x = max(max_x, max(skill_c[n]))


        for d, r in releases.values:
            if d > start:
                fig_daily_counts_relative.add_shape(
                    dict(type="line", x0=d, y0=min_x, x1=d, y1=max_x, line=dict(color="RoyalBlue", width=1)), row=1,
                    col=1)
                fig_daily_counts_relative.add_annotation(x=d, y=max_x * 1.1, text=r, textangle=-90, showarrow=True,
                                                         font=dict(color="black", size=10), opacity=0.7, row=1, col=1)

        fig_daily_counts_relative.update_layout(height=500, width=1300, showlegend=True)
        fig_daily_counts_relative.update_layout(hovermode='x')
        # fig_daily_counts_relative.show()
        return fig_daily_counts_relative

    def plot_skills_ratings_ma_dialogs_with_gt_7_turns(self, skills_ratings, skill_names, releases):
        """
        Skills Ratings, moving average over last 100 dialogs with number of turns > 7

        :param skills_ratings:
        :param skill_names:
        :return:
        """
        from tqdm import tqdm as tqdm
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        import datetime as dt

        # avg_n_dialogs = 200
        avg_n_dialogs = 100
        # avg_n_dialogs = 3

        n_turns = 7
        # n_turns = 1

        fig_moving_avg = make_subplots(rows=1, cols=1, subplot_titles=(
        f'Skills Ratings, moving average over last {avg_n_dialogs} dialogs with number of turns > {n_turns}',))

        x = dict()
        skill_c = dict()
        skill_r = dict()
        skill_names = set(skill_names)
        for n in skill_names:
            skill_c[n] = []
            skill_r[n] = []
            x[n] = []
        skill_c['_total'] = []
        skill_r['_total'] = []
        x['_total'] = []

        now = dt.datetime.now(tz=tz.gettz('UTC'))
        end = now
        start = max(end - dt.timedelta(days=35), skills_ratings['date'].min())

        skills_ratings_by_range = skills_ratings[(skills_ratings['date'] <= end) & (skills_ratings['date'] >= start)][
                                  ::-1]
        skills_ratings_by_range = skills_ratings_by_range[skills_ratings_by_range['n_turns'] > n_turns]

        min_r = 5
        max_r = 0

        sr_gb = skills_ratings_by_range.groupby('conv_id', sort=False)
        d = sr_gb.first()
        dates_by_id = d['date'].to_dict()
        d['cnt'] = sr_gb['rating'].count()
        d['r*cnt'] = d['cnt'] * d['rating']
        s_count = d['cnt'].rolling(avg_n_dialogs).sum()
        moving_avg = d['r*cnt'].rolling(avg_n_dialogs).sum() / s_count
        # moving_avg = d['rating'].rolling(avg_n_dialogs).mean()

        for (i, v), (_, c) in zip(moving_avg.items(), s_count.items()):
            date = dates_by_id[i]
            if not pd.isna(v):
                x['_total'] += [pd.to_datetime(date, utc=True)]
                skill_r['_total'] += [v]
                skill_c['_total'] += [c]

        for sn in tqdm(list(skill_names)):
            sr_gb = skills_ratings_by_range[skills_ratings_by_range['active_skill'] == sn].groupby('conv_id',
                                                                                                   sort=False)
            d = sr_gb.first()
            dates_by_id = d['date'].to_dict()
            d['cnt'] = sr_gb['rating'].count()
            d['r*cnt'] = d['cnt'] * d['rating']
            s_count = d['cnt'].rolling(avg_n_dialogs).sum()
            moving_avg = d['r*cnt'].rolling(avg_n_dialogs).sum() / s_count
            for (i, v), (_, c) in zip(moving_avg.items(), s_count.items()):
                date = dates_by_id[i]
                if not pd.isna(v):
                    x[sn] += [pd.to_datetime(date, utc=True)]
                    skill_r[sn] += [v]
                    skill_c[sn] += [c]

        for n in sorted(list(skill_names) + ['_total']):
            if len(skill_r[n]) == 0:
                continue
            fig_moving_avg.add_trace(go.Scatter(x=x[n], y=skill_r[n], name=n, line={'dash': 'dot'}, marker={'size': 8},
                                                customdata=skill_c[n],
                                                hovertemplate='%{y:.2f}: selected: %{customdata}', ), row=1, col=1)
            min_r = min(min_r, min(skill_r[n]))
            max_r = max(max_r, max(skill_r[n]))

        for d, r in releases.values:
            if d > start:
                fig_moving_avg.add_shape(
                    dict(type="line", x0=d, y0=min_r, x1=d, y1=max_r, line=dict(color="RoyalBlue", width=1)), row=1,
                    col=1)
                fig_moving_avg.add_annotation(x=d, y=max_r, text=r, textangle=-90, showarrow=True,
                                              font=dict(color="black", size=10), opacity=0.7, row=1, col=1)

        fig_moving_avg.update_layout(height=500, width=1300, showlegend=True)
        fig_moving_avg.update_layout(hovermode='x')
        fig_moving_avg['layout']['yaxis1']['range'] = [min_r, max_r]
        # fig_moving_avg.show()
        return fig_moving_avg

    def plot_skill_ratings_total_ma_n_turns_gt_7(self, skills_ratings, releases):
        """
        Skills Ratings, -_total, moving average over last 200 dialogs with number of turns > 7

        :param skills_ratings:
        :return:
        """
        from tqdm import tqdm as tqdm
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        import datetime as dt

        # avg_n_dialogs = 200
        avg_n_dialogs = 100
        # avg_n_dialogs = 3
        n_turns = 7
        # n_turns = 1

        fig_moving_avg_d_total = make_subplots(rows=1, cols=1, subplot_titles=(
        f'Skills Ratings, -_total, moving average over last {avg_n_dialogs} dialogs with number of turns > {n_turns}',))

        x = dict()
        skill_c = dict()
        skill_r = dict()
        skill_names = set(skills_ratings['active_skill'].unique()) - set(['no_skill_name'])
        for n in skill_names:
            skill_c[n] = []
            skill_r[n] = []
            x[n] = []
        skill_c['_total'] = []
        skill_r['_total'] = []
        x['_total'] = []

        now = dt.datetime.now(tz=tz.gettz('UTC'))
        end = now
        # start = end - dt.timedelta(days=35)
        start = max(end - dt.timedelta(days=35), skills_ratings['date'].min())

        skills_ratings_by_range = skills_ratings[(skills_ratings['date'] <= end) & (skills_ratings['date'] >= start)][
                                  ::-1]
        skills_ratings_by_range = skills_ratings_by_range[skills_ratings_by_range['n_turns'] > n_turns]

        min_r = 5
        max_r = 0

        sr_gb = skills_ratings_by_range.groupby('conv_id', sort=False)
        d = sr_gb.first()
        dates_by_id = d['date'].to_dict()
        d['cnt'] = sr_gb['rating'].count()
        d['r*cnt'] = d['cnt'] * d['rating']
        s_count = d['cnt'].rolling(avg_n_dialogs).sum()

        # avg_n_dialogs=200 is too much for current data flow
        # TODO fix: adapt for actual numbers
        moving_avg = d['r*cnt'].rolling(avg_n_dialogs).sum() / s_count
        total = dict()
        # import ipdb; ipdb.set_trace()
        for (i, v), (_, c) in zip(moving_avg.items(), s_count.items()):
            date = dates_by_id[i]
            if not pd.isna(v):
                x['_total'] += [pd.to_datetime(date, utc=True)]
                skill_r['_total'] += [v]
                skill_c['_total'] += [c]
                total[pd.to_datetime(date, utc=True)] = v

        for sn in tqdm(list(skill_names)):
            sr_gb = skills_ratings_by_range[skills_ratings_by_range['active_skill'] == sn].groupby('conv_id',
                                                                                                   sort=False)
            d = sr_gb.first()
            dates_by_id = d['date'].to_dict()
            d['cnt'] = sr_gb['rating'].count()
            d['r*cnt'] = d['cnt'] * d['rating']
            s_count = d['cnt'].rolling(avg_n_dialogs).sum()
            moving_avg = d['r*cnt'].rolling(avg_n_dialogs).sum() / s_count
            for (i, v), (_, c) in zip(moving_avg.items(), s_count.items()):
                date = dates_by_id[i]
                if not pd.isna(v):
                    x[sn] += [pd.to_datetime(date, utc=True)]
                    skill_r[sn] += [v - total[pd.to_datetime(date, utc=True)]]
                    skill_c[sn] += [c]
        # import ipdb; ipdb.set_trace()
        for n in sorted(list(skill_names)):
            if len(skill_r[n]) == 0:
                continue
            fig_moving_avg_d_total.add_trace(
                go.Scatter(x=x[n], y=skill_r[n], name=n, line={'dash': 'dot'}, marker={'size': 8},
                           customdata=skill_c[n], hovertemplate='%{y:.2f}: selected: %{customdata}', ), row=1, col=1)
            min_r = min(min_r, min(skill_r[n]))
            max_r = max(max_r, max(skill_r[n]))

        for d, r in releases.values:
            if d > start:
                fig_moving_avg_d_total.add_shape(
                    dict(type="line", x0=d, y0=min_r, x1=d, y1=max_r, line=dict(color="RoyalBlue", width=1)), row=1,
                    col=1)
                fig_moving_avg_d_total.add_annotation(x=d, y=max_r, text=r, textangle=-90, showarrow=True,
                                                      font=dict(color="black", size=10), opacity=0.7, row=1, col=1)

        fig_moving_avg_d_total.update_layout(height=500, width=1300, showlegend=True)
        fig_moving_avg_d_total.update_layout(hovermode='x')
        fig_moving_avg_d_total['layout']['yaxis1']['range'] = [min_r, max_r]
        # fig_moving_avg_d_total.show()
        return fig_moving_avg_d_total

    def plot_dialog_finished_reasons_w_ratings(self, dialog_finished_df):
        """
        Dialog finished reason, with rating
        :return:
        """
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        import datetime as dt
        fig_dialog_finished_day = make_subplots(rows=1, cols=1,
                                                subplot_titles=('Dialog finished reason, with rating',))

        now = dt.datetime.now(tz=tz.gettz('UTC'))
        end = now
        start = end - dt.timedelta(days=14)

        x = dict()
        value_v = dict()
        value_c = dict()
        finished_values = set(dialog_finished_df['alexa_command'].unique()) - {'no_alexa_command'}
        finished_values |= {'no_command_no_goodbye', 'bot_goodbye'}
        # because of some bugged dialog
        finished_values -= {'alexa handler: command logged'}
        for n in finished_values:
            value_v[n] = []
            value_c[n] = []
            x[n] = []

        for dt in pd.date_range(start=start, end=end, freq='D'):
            daily_data = dialog_finished_df[
                (dialog_finished_df['date'] < dt) & (dialog_finished_df['date'] >= dt - dt.freq * 1)]
            daily_data = daily_data[daily_data['has_rating'] == True]
            for v in finished_values:
                if v.startswith('/'):
                    v_count = (daily_data['alexa_command'] == v).sum()
                    avg_rating = daily_data[daily_data['alexa_command'] == v]['rating'].mean()
                    avg_n_turns = daily_data[daily_data['alexa_command'] == v]['n_turns'].mean()
                else:
                    v_count = daily_data[v].sum()
                    avg_rating = daily_data[daily_data[v]]['rating'].mean()
                    avg_n_turns = daily_data[daily_data[v]]['n_turns'].mean()
                if v_count > 0:
                    value_v[v] += [v_count / len(daily_data)]
                    value_c[v] += [[v_count, avg_rating, avg_n_turns]]
                    x[v] += [dt]

        for r in sorted(list(finished_values), reverse=True):
            fig_dialog_finished_day.add_bar(name=r, x=x[r], y=value_v[r], customdata=value_c[r],
                                            hovertemplate='%{y:.2f}: count: %{customdata[0]} rating: %{customdata[1]:.2f} n_turns: %{customdata[2]:.2f}',
                                            row=1, col=1)

        fig_dialog_finished_day.update_layout(height=500, width=1300, showlegend=True)
        fig_dialog_finished_day['layout']['yaxis1']['range'] = [0, 1]
        fig_dialog_finished_day.update_layout(hovermode='x', barmode='stack')
        # fig_dialog_finished_day.show()
        return fig_dialog_finished_day

    def plot_dialog_finished_reason(self, dialog_finished_df):
        """
        Dialog finished reason, all
        :return:
        """
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        import datetime as dt

        fig_dialog_finished_all_day = make_subplots(rows=1, cols=1,
                                                    subplot_titles=('Dialog finished reason, all',))

        now = dt.datetime.now(tz=tz.gettz('UTC'))
        end = now
        start = end - dt.timedelta(days=14)

        x = dict()
        value_v = dict()
        value_c = dict()
        finished_values = set(dialog_finished_df['alexa_command'].unique()) - {'no_alexa_command'}
        finished_values |= {'no_command_no_goodbye', 'bot_goodbye'}
        finished_values -= {'alexa handler: command logged'}
        for n in finished_values:
            value_v[n] = []
            value_c[n] = []
            x[n] = []

        for dt in pd.date_range(start=start, end=end, freq='D'):
            daily_data = dialog_finished_df[
                (dialog_finished_df['date'] < dt) & (dialog_finished_df['date'] >= dt - dt.freq * 1)]
            for v in finished_values:
                if v.startswith('/'):
                    v_count = (daily_data['alexa_command'] == v).sum()
                    avg_n_turns = daily_data[daily_data['alexa_command'] == v]['n_turns'].mean()
                else:
                    v_count = daily_data[v].sum()
                    avg_n_turns = daily_data[daily_data[v]]['n_turns'].mean()

                if v_count > 0:
                    value_v[v] += [v_count / len(daily_data)]
                    value_c[v] += [[v_count, avg_n_turns]]
                    x[v] += [dt]

        for r in sorted(list(finished_values), reverse=True):
            fig_dialog_finished_all_day.add_bar(name=r, x=x[r], y=value_v[r], customdata=value_c[r],
                                                hovertemplate='%{y:.2f}: count: %{customdata[0]} n_turns:  %{customdata[1]:.2f}',
                                                row=1, col=1)

        fig_dialog_finished_all_day.update_layout(height=500, width=1300, showlegend=True)
        fig_dialog_finished_all_day['layout']['yaxis1']['range'] = [0, 1]
        fig_dialog_finished_all_day.update_layout(hovermode='x', barmode='stack')
        # fig_dialog_finished_all_day.show()
        return fig_dialog_finished_all_day

    def plot_last_skill_in_dialog_with_rating(self, dialog_finished_df, skill_names, releases):
        """
        Last skill in dialog, with rating
        :param dialog_finished_df:
        :return:
        """
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        import datetime as dt

        fig_dialog_finished_skill_day = make_subplots(rows=1, cols=1,
                                                      subplot_titles=('Last skill in dialog, with rating',))

        now = dt.datetime.now(tz=tz.gettz('UTC'))
        end = now
        # start = end - dt.timedelta(days=31)
        min_by_dialogs_dt = dialog_finished_df['date'].min().replace(tzinfo=tz.gettz('UTC'))
        start = max(end - dt.timedelta(days=31), min_by_dialogs_dt)

        x = dict()
        value_v = dict()
        value_c = dict()
        skill_names = set(skill_names)
        for n in skill_names:
            value_c[n] = []
            value_v[n] = []
            x[n] = []

        for dt in pd.date_range(start=start, end=end, freq='1D'):
            daily_data = dialog_finished_df[
                (dialog_finished_df['date'] < dt) & (dialog_finished_df['date'] >= dt - dt.freq)]
            daily_data = daily_data[(daily_data['has_rating'])]

            for sn in skill_names:
                d = daily_data[daily_data['last_skill'] == sn]
                if len(d) > 2:
                    value_v[sn] += [len(d) / len(daily_data)]
                    value_c[sn] += [[len(d), d['rating'].mean(), d['n_turns'].mean()]]
                    x[sn] += [dt]

        min_v, max_v = 10 * 10, - 10 ** 10
        for sn in sorted(list(skill_names)):
            if len(value_v[sn]) > 0:
                fig_dialog_finished_skill_day.add_scatter(name=sn, x=x[sn], y=value_v[sn], customdata=value_c[sn],
                                                          line={'dash': 'dot'},
                                                          hovertemplate='%{y:.2f}: count: %{customdata[0]} rating: %{customdata[1]:.2f} n_turns: %{customdata[2]:.2f}',
                                                          row=1, col=1)
                min_v = min(min_v, min(value_v[sn]))
                max_v = max(max_v, max(value_v[sn]))

        for d, r in releases.values:
            if d > start:
                fig_dialog_finished_skill_day.add_shape(
                    dict(type="line", x0=d, y0=min_v, x1=d, y1=max_v, line=dict(color="RoyalBlue", width=1)), row=1,
                    col=1)
                fig_dialog_finished_skill_day.add_annotation(x=d, y=max_v, text=r, textangle=-90, showarrow=True,
                                                             font=dict(color="black", size=10), opacity=0.7, row=1,
                                                             col=1)

        fig_dialog_finished_skill_day.update_layout(height=500, width=1300, showlegend=True)
        # fig_dialog_finished_skill_day['layout']['yaxis1']['range'] = [0, 0.5]
        fig_dialog_finished_skill_day.update_layout(hovermode='x')
        # fig_dialog_finished_skill_day.show()
        return fig_dialog_finished_skill_day

    def plot_last_skill_stop_exit(self, dialog_finished_df, skill_names):
        """
        Last skill in dialog, "Alexa, stop", "Alexa, exit", "Alexa, quit", with rating, Last 24h
        :return:
        """
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        import datetime as dt

        fig_dialog_finished_stop_skill_day = make_subplots(rows=1, cols=1, subplot_titles=(
        'Last skill in dialog, "Alexa, stop", "Alexa, exit", "Alexa, quit", with rating, Last 24h',))

        now = dt.datetime.now(tz=tz.gettz('UTC'))
        end = now
        start = end - dt.timedelta(days=31)

        x = dict()
        value_v = dict()
        value_c = dict()
        skill_names = set(skill_names)
        for n in skill_names:
            value_c[n] = []
            value_v[n] = []
            x[n] = []

        for dt in pd.date_range(start=start, end=end, freq='12H'):
            daily_data = dialog_finished_df[
                (dialog_finished_df['date'] < dt) & (dialog_finished_df['date'] >= dt - dt.freq * 2)]
            daily_data = daily_data[(daily_data['has_rating'] == True) & (
                        (daily_data['alexa_command'] == '/alexa_stop_handler') | (
                            daily_data['alexa_command'] == '/alexa_USER_INITIATED'))]

            for sn in skill_names:
                d = daily_data[daily_data['last_skill'] == sn]
                if len(d) > 2:
                    value_v[sn] += [len(d) / len(daily_data)]
                    value_c[sn] += [[len(d), d['rating'].mean(), d['n_turns'].mean()]]
                    x[sn] += [dt]

        min_v, max_v = 10 ** 10, - 10 ** 10
        for sn in sorted(list(skill_names)):
            if len(value_v[sn]) > 0:
                fig_dialog_finished_stop_skill_day.add_scatter(name=sn, x=x[sn], y=value_v[sn], customdata=value_c[sn],
                                                               line={'dash': 'dot'},
                                                               hovertemplate='%{y:.2f}: count: %{customdata[0]} rating: %{customdata[1]:.2f} n_turns: %{customdata[2]:.2f}',
                                                               row=1, col=1)
                min_v = min(min_v, min(value_v[sn]))
                max_v = max(max_v, max(value_v[sn]))

        # for d, r in releases.values:
        #     if d > start:
        #         fig_dialog_finished_stop_skill_day.add_shape(
        #             dict(type="line", x0=d, y0=min_v, x1=d, y1=max_v, line=dict(color="RoyalBlue", width=1)), row=1,
        #             col=1)
        #         fig_dialog_finished_stop_skill_day.add_annotation(x=d, y=max_v, text=r, textangle=-90, showarrow=True,
        #                                                           font=dict(color="black", size=10), opacity=0.7, row=1,
        #                                                           col=1)

        fig_dialog_finished_stop_skill_day.update_layout(height=500, width=1300, showlegend=True)
        # fig_dialog_finished_skill_day['layout']['yaxis1']['range'] = [0, 0.5]
        fig_dialog_finished_stop_skill_day.update_layout(hovermode='x')
        # fig_dialog_finished_stop_skill_day.show()
        return fig_dialog_finished_stop_skill_day


    def plot_number_of_dialogs_with_ratings_hrly(self, data_df):
        """
        Number of dialogs with ratings, hourly

        data_df DataFrame with ratings, start_time and id fields"""
        import datetime as dt
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        fig = make_subplots(rows=2, cols=1, subplot_titles=(
        'Number of dialogs with ratings, hourly', 'Avg dialog rating, hourly'))

        # now = dt.datetime.now()
        now = dt.datetime.now(tz=tz.gettz('UTC'))
        end = dt.datetime(year=now.year, month=now.month, day=now.day, hour=now.hour)
        # end = dt.datetime(year=now.year, month=now.month, day=now.day, hour=now.hour,
        #                tzinfo=now.tzinfo)
        start = end - dt.timedelta(days=14)

        x = []
        counts = []
        ratings = []

        for date in pd.date_range(start=start, end=end, freq='H'):
            x += [date]
            hourly_dialogs = data_df[(data_df['start_time'] < date) & (
                    data_df['start_time'] > date - date .freq)]
            counts += [len(hourly_dialogs)]
            ratings += [0 if len(hourly_dialogs) == 0 else hourly_dialogs['rating'].mean()]

        fig.add_trace(go.Scatter(x=x, y=counts, fill='tozeroy', name='count', ), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=ratings, fill='tozeroy', name='rating', ), row=2, col=1)


        end = dt.datetime(year=now.year, month=now.month, day=now.day, hour=8)
        # end = dt.datetime(year=now.year, month=now.month, day=now.day, hour=8,
        #                   tzinfo=now.tzinfo)
        start = end - dt.timedelta(days=14)
        x = []
        ratings = []
        if len(data_df)==0:
            return None
        for date in pd.date_range(start=start, end=end, freq='D'):
            x += [date]
            hourly_dialogs = data_df[(data_df['start_time'] <= date) & (
                    data_df['start_time'] > date - date.freq)]
            ratings += [0 if len(hourly_dialogs) == 0 else hourly_dialogs['rating'].mean()]
        fig.add_trace(go.Scatter(x=x, y=ratings, name='rating, 24h'), row=2, col=1)

        fig.update_layout(height=600, width=1200, showlegend=False)

        # first plot start, end
        end = dt.datetime(year=now.year, month=now.month, day=now.day, hour=now.hour)
        # end = dt.datetime(year=now.year, month=now.month, day=now.day, hour=now.hour,
        #                tzinfo=now.tzinfo)
        start = end - dt.timedelta(days=14)
        fig['layout']['xaxis2']['range'] = [start, end]

        fig['layout']['yaxis2']['range'] = [0, 5.5]
        fig.update_layout(hovermode='x')
        # fig.show()
        return fig

    # def prepare_all_data(self, dialogs):
    #     """
    #     Uses Utterances from Table
    #     - prepare_data_for_plotting
    #     -prepare_dialog_finished_df
    #     :param dialogs:
    #     :return:
    #     """
    #     skills_ratings = []
    #     dialog_durations = []
    #     def get_last_skill(dialog, exit_intent=False):
    #         if exit_intent and dialog.utterances.count() >= 3:
    #             return dialog.utterances[-3].active_skill
    #         return dialog.utterances[-1].active_skill
    #
    #     dialog_finished_data = []
    #
    #     rating_data = []
    #     for dialog in tqdm(dialogs, desc="Analyzing conversations"):
    #         if dialog.rating:
    #             rating = float(dialog.rating)
    #             has_rating = True
    #         else:
    #             rating = 'no_rating'
    #             has_rating = False
    #         date = dialog.date_start
    #         alexa_command = 'no_alexa_command'
    #         bot_respond_with_goodbye = False
    #         # n_turns = len(dialog.utterances) // 2
    #         n_utt = len(dialog.raw_utterances)
    #         # n_utt = len(list(dialog.utterances))
    #         n_turns = n_utt // 2
    #         last_skill = None
    #
    #         # if 'alexa_commands' in dialog:
    #         #     alexa_command = dialog['alexa_commands'][0]['text']
    #         #     last_skill = get_last_skill(dialog)
    #
    #         if '#+#exit' in dialog.utterances[-1].text:
    #             bot_respond_with_goodbye = True
    #             last_skill = get_last_skill(dialog, exit_intent=True)
    #
    #         if last_skill is None:
    #             last_skill = get_last_skill(dialog)
    #
    #         no_command_no_goodbye = (alexa_command == 'no_alexa_command') and not bot_respond_with_goodbye
    #
    #         conv_id = dialog.id
    #
    #         dialog_finished_data += [
    #             [date, alexa_command, bot_respond_with_goodbye, no_command_no_goodbye, rating,
    #              has_rating, n_turns, last_skill, conv_id, None]]
    #              # has_rating, n_turns, last_skill, conv_id, dialog['version']]]
    #
    #         if has_rating:
    #             if dialog.date_start:
    #                 rating_data.append({
    #                     'id': dialog.id,
    #                     'rating': dialog.rating,
    #                     'start_time': dialog.date_start
    #                 })
    #             time = (dialog.date_finish - dialog.date_start).seconds
    #
    #             dialog_durations += [[date, time, n_utt]]
    #
    #             for utt in dialog.utterances:
    #                 # if hasattr(utt, 'active_skill'):
    #                 if utt.active_skill:
    #                     #         skills_ratings += [[date, utt.active_skill, rating, conv_id, dialog.version]]
    #                     skills_ratings += [[date, utt.active_skill, rating, conv_id, None]]
    #
    #     ratings_df = pd.DataFrame(rating_data)
    #
    #     dialog_finished_df = pd.DataFrame(dialog_finished_data,
    #                                       columns=['date', 'alexa_command', 'bot_goodbye',
    #                                                'no_command_no_goodbye', 'rating', 'has_rating',
    #                                                'n_turns', 'last_skill', 'conv_id', 'version'])
    #     dialog_finished_df['date'] = pd.to_datetime(dialog_finished_df['date'], utc=True)
    #
    #     skills_ratings = pd.DataFrame(skills_ratings,
    #                                   columns=['date', 'active_skill', 'rating', 'conv_id',
    #                                            'version'])
    #     skills_ratings['date'] = pd.to_datetime(skills_ratings['date'], utc=True)
    #     #
    #     n_turns = skills_ratings['conv_id'].value_counts().to_dict()
    #     skills_ratings['n_turns'] = skills_ratings['conv_id'].apply(lambda x: n_turns[x])
    #     # print("skills_ratings")
    #     # print(skills_ratings)
    #     # skills_ratings = skills_ratings[skills_ratings['n_turns'] > 1]
    #
    #     dialog_durations = pd.DataFrame(dialog_durations, columns=['date', 'time', 'n_utt'])
    #     dialog_durations['date'] = pd.to_datetime(dialog_durations['date'], utc=True)
    #
    #     return dialog_durations, skills_ratings, dialog_finished_df, ratings_df


    # # moved to prepare_all_data
    # def prepare_data_for_plotting(self, dialogs):
    #     skills_ratings = []
    #     dialog_durations = []
    #
    #     for dialog in dialogs:
    #         rating = dialog.rating
    #         if rating == 'no_rating':
    #             continue
    #         conv_id = dialog.id
    #         date = dialog.date_start
    #         time = (dialog.date_finish - dialog.date_start).seconds
    #         n_utt = len(list(dialog.utterances))
    #         dialog_durations += [[date, time, n_utt]]
    #
    #         if dialog.rating:
    #             for utt in dialog.utterances:
    #                 # if hasattr(utt, 'active_skill'):
    #                 if utt.active_skill:
    #             #         skills_ratings += [[date, utt.active_skill, rating, conv_id, dialog.version]]
    #                     skills_ratings += [[date, utt.active_skill, rating, conv_id, None]]
    #     skills_ratings = pd.DataFrame(skills_ratings,
    #                                   columns=['date', 'active_skill', 'rating', 'conv_id',
    #                                            'version'])
    #     skills_ratings['date'] = pd.to_datetime(skills_ratings['date'], utc=True)
    #     #
    #     n_turns = skills_ratings['conv_id'].value_counts().to_dict()
    #     skills_ratings['n_turns'] = skills_ratings['conv_id'].apply(lambda x: n_turns[x])
    #     # print("skills_ratings")
    #     # print(skills_ratings)
    #     # skills_ratings = skills_ratings[skills_ratings['n_turns'] > 1]
    #
    #     dialog_durations = pd.DataFrame(dialog_durations, columns=['date', 'time', 'n_utt'])
    #     dialog_durations['date'] = pd.to_datetime(dialog_durations['date'], utc=True)
    #
    #     return dialog_durations, skills_ratings

    # # moved to prepare_all_data
    # def prepare_dialog_finished_df(self, dialogs):
    #     def get_last_skill(dialog, exit_intent=False):
    #         if exit_intent and dialog.utterances.count() >= 3:
    #             return dialog.utterances[-3].active_skill
    #         return dialog.utterances[-1].active_skill
    #
    #     dialog_finished_data = []
    #     for dialog in dialogs:
    #         # if dialog.utterances[-1]['spk'] == 'Human':
    #         #     # just 2 dialogs in whole dump
    #         #     continue
    #
    #
    #         if dialog.rating:
    #             rating = float(dialog.rating)
    #             has_rating = True
    #         else:
    #             rating = 'no_rating'
    #             has_rating = False
    #         date = dialog.date_start
    #         alexa_command = 'no_alexa_command'
    #         bot_respond_with_goodbye = False
    #         # n_turns = len(dialog.utterances) // 2
    #         n_turns = dialog.utterances.count() // 2
    #         last_skill = None
    #
    #         # if 'alexa_commands' in dialog:
    #         #     alexa_command = dialog['alexa_commands'][0]['text']
    #         #     last_skill = get_last_skill(dialog)
    #
    #         if '#+#exit' in dialog.utterances[-1].text:
    #             bot_respond_with_goodbye = True
    #             last_skill = get_last_skill(dialog, exit_intent=True)
    #
    #         if last_skill is None:
    #             last_skill = get_last_skill(dialog)
    #
    #         no_command_no_goodbye = (alexa_command == 'no_alexa_command') and not bot_respond_with_goodbye
    #
    #         conv_id = dialog.id
    #
    #         dialog_finished_data += [
    #             [date, alexa_command, bot_respond_with_goodbye, no_command_no_goodbye, rating,
    #              has_rating, n_turns, last_skill, conv_id, None]]
    #              # has_rating, n_turns, last_skill, conv_id, dialog['version']]]
    #
    #     dialog_finished_df = pd.DataFrame(dialog_finished_data,
    #                                       columns=['date', 'alexa_command', 'bot_goodbye',
    #                                                'no_command_no_goodbye', 'rating', 'has_rating',
    #                                                'n_turns', 'last_skill', 'conv_id', 'version'])
    #     dialog_finished_df['date'] = pd.to_datetime(dialog_finished_df['date'], utc=True)
    #     return dialog_finished_df

    # def make_skills_freqs_plot(self, skills_ratings_df, skill_names):
    #     # dont work
    #     """
    #     Uses ratings
    #     Skill was selected, relative, Last 24h chart
    #     :return:
    #     """
    #     from plotly.subplots import make_subplots
    #     import plotly.graph_objects as go
    #
    #     fig_daily_counts_relative = make_subplots(rows=1, cols=1, subplot_titles=(
    #         'Skill was selected, relative, Last 24h',))
    #
    #     now = dt.datetime.now(tz=tz.gettz('UTC'))
    #     end = now
    #     start = end - dt.timedelta(days=14)
    #
    #     x = dict()
    #     skill_c = dict()
    #     skill_names = set(skill_names)
    #     skill_z = dict()
    #     for n in skill_names:
    #         skill_c[n] = []
    #         x[n] = []
    #         skill_z[n] = []
    #
    #     for dt in pd.date_range(start=start, end=end, freq='D'):
    #         daily_ratings = skills_ratings_df[
    #             (skills_ratings_df['date'] < dt) & (skills_ratings_df['date'] >= dt - dt.freq * 1)]
    #         for sn, c in daily_ratings.groupby('active_skill')['rating'].count().items():
    #             if sn in skill_names:
    #                 skill_c[sn] += [c / len(daily_ratings)]
    #                 x[sn] += [dt]
    #                 skill_z[sn] += [c]
    #
    #     min_x, max_x = 1e10, 0
    #     for n in sorted(list(skill_names)):
    #         if len(skill_c[n]) > 0:
    #             fig_daily_counts_relative.add_trace(
    #                 go.Scatter(x=x[n], y=skill_c[n], customdata=skill_z[n], name=n,
    #                            line={'dash': 'dot'}, marker={'size': 8},
    #                            hovertemplate='%{y:.3f}: count %{customdata}'),
    #                 row=1, col=1)
    #             min_x = min(min_x, min(skill_c[n]))
    #             max_x = max(max_x, max(skill_c[n]))
    #
    #     # no releases data
    #     # for d, r in releases.values:
    #     #     if d > start:
    #     #         fig_daily_counts_relative.add_shape(
    #     #             dict(type="line", x0=d, y0=min_x, x1=d, y1=max_x,
    #     #                  line=dict(color="RoyalBlue", width=1)), row=1, col=1)
    #     #         fig_daily_counts_relative.add_annotation(x=d, y=max_x, text=r, textangle=-90,
    #     #                                                  showarrow=True,
    #     #                                                  font=dict(color="black", size=10),
    #     #                                                  opacity=0.7, row=1, col=1)
    #
    #     fig_daily_counts_relative.update_layout(height=500, width=1300, showlegend=True)
    #     fig_daily_counts_relative.update_layout(hovermode='x')
    #     return fig_daily_counts_relative
