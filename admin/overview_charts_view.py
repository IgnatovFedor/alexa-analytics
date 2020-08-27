from flask_admin import BaseView, expose
import json
import pandas as pd
import datetime as dt
from dateutil import tz
from plotly.offline import plot
from db.models import Conversation


class OverviewChartsView(BaseView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO fix the shit, fuck the flask:
        from db.db import DBManager, get_session
        with open('core/config.json') as config_file:
            config = json.load(config_file)
        db_config = config['DB']
        self.session = get_session(db_config['user'], db_config['password'], db_config['host'],
                              db_config['dbname'])


    def prepare_data_for_plotting(self, dialogs):
        skills_ratings = []
        dialog_durations = []

        for dialog in dialogs:
            rating = dialog.rating
            if rating == 'no_rating':
                continue
            conv_id = dialog.id
            date = dialog.date_start
            time = (dialog.date_finish - dialog.date_start).seconds
            n_utt = len(list(dialog.utterances))
            dialog_durations += [[date, time, n_utt]]

            if dialog.rating:
                for utt in dialog.utterances:
                    # if hasattr(utt, 'active_skill'):
                    if utt.active_skill:
                #         skills_ratings += [[date, utt.active_skill, rating, conv_id, dialog.version]]
                        skills_ratings += [[date, utt.active_skill, rating, conv_id, None]]
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

        return dialog_durations, skills_ratings

    def prepare_dialog_finished_df(self, dialogs):
        def get_last_skill(dialog, exit_intent=False):
            if exit_intent and dialog.utterances.count() >= 3:
                return dialog.utterances[-3].active_skill
            return dialog.utterances[-1].active_skill

        dialog_finished_data = []
        for dialog in dialogs:
            # if dialog.utterances[-1]['spk'] == 'Human':
            #     # just 2 dialogs in whole dump
            #     continue


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
            n_turns = dialog.utterances.count() // 2
            last_skill = None

            # if 'alexa_commands' in dialog:
            #     alexa_command = dialog['alexa_commands'][0]['text']
            #     last_skill = get_last_skill(dialog)

            if '#+#exit' in dialog.utterances[-1].text:
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

        dialog_finished_df = pd.DataFrame(dialog_finished_data,
                                          columns=['date', 'alexa_command', 'bot_goodbye',
                                                   'no_command_no_goodbye', 'rating', 'has_rating',
                                                   'n_turns', 'last_skill', 'conv_id', 'version'])
        dialog_finished_df['date'] = pd.to_datetime(dialog_finished_df['date'], utc=True)
        return dialog_finished_df

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

    def plot_skills_durations(self, dialog_durations_df):
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
        end = now
        start = end - dt.timedelta(days=50)
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

        # for d, r in releases.values:
        #     dialog_time.add_shape(
        #         dict(type="line", x0=d, y0=0, x1=d, y1=200, line=dict(color="RoyalBlue", width=1)),
        #         row=1, col=1)
        #     dialog_time.add_annotation(x=d, y=200, text=r, textangle=-90, showarrow=True,
        #                                font=dict(color="black", size=10), opacity=0.7, row=1, col=1)
        #     dialog_time.add_shape(
        #         dict(type="line", x0=d, y0=10, x1=d, y1=35, line=dict(color="RoyalBlue", width=1)),
        #         row=2, col=1)
        #     dialog_time.add_annotation(x=d, y=35, text=r, textangle=-90, showarrow=True,
        #                                font=dict(color="black", size=10), opacity=0.7, row=2, col=1)

        dialog_time_fig.update_layout(height=500, width=1300, showlegend=True)
        dialog_time_fig['layout']['yaxis1']['range'] = [0, 2000]
        dialog_time_fig['layout']['yaxis2']['range'] = [0, 35]
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
        # # for d, r in releases.values:
        # #     shares_n_utt.add_shape(
        # #         dict(type="line", x0=d, y0=0, x1=d, y1=1, line=dict(color="RoyalBlue", width=1)),
        # #         row=1, col=1)
        # #     shares_n_utt.add_annotation(x=d, y=1, text=r, textangle=-90, showarrow=True,
        # #                                 font=dict(color="black", size=10), opacity=0.7, row=1,
        # #                                 col=1)
        #
        shares_n_utt_fig.update_layout(height=500, width=1300, showlegend=True)
        shares_n_utt_fig['layout']['yaxis1']['range'] = [-0.05, 1.05]
        shares_n_utt_fig.update_layout(hovermode='x')
        # shares_n_utt.show()
        # shares_n_utt_div = plot(shares_n_utt, output_type='div', include_plotlyjs=False)


        # return render(request, 'dialogs/skills_sentiment_stacked.html', context_dict)
        return dialog_time_fig, shares_n_utt_fig

    def plot_last_skill_in_dialog(self, dialog_finished_df):
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
        skill_names = set(dialog_finished_df['last_skill'].values)
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


    @expose('/')
    def index(self):
        """
        Main page for analytical overview
        :return:
        """
        # retrieve all dialogs
        dialogs = self.session.query(Conversation).order_by(Conversation.date_finish.desc())
        # print(dialogs)

        dialog_durations_df, skills_ratings_df = self.prepare_data_for_plotting(dialogs)

        print("skills_ratings_df")
        print(skills_ratings_df)
        print("dialog_durations_df")
        print(dialog_durations_df)

        dialog_finished_df = self.prepare_dialog_finished_df(dialogs)
        print("dialog_finished_df")
        print(dialog_finished_df)
        # retrieve data for skill frequency chart
        # prepare plot for it
        # TODO prepare data for number_of_dialogs_with_ratings_hrly wit dialogs start time and ratings
        df = self.prepare_data_for_ratings_plots(dialogs)
        hrly_dialogs_ratings_fig = self.plot_number_of_dialogs_with_ratings_hrly(df)


        # retrieve data for Dialog time(sec), Daily chart
        # prepare plot of it
        dialog_time_fig, shares_n_utt_fig = self.plot_skills_durations(dialog_durations_df)
        dialog_time_fig_div = plot(dialog_time_fig, output_type='div', include_plotlyjs=False)
        shares_n_utt_div = plot(shares_n_utt_fig, output_type='div', include_plotlyjs=False)

        hrly_dialogs_ratings_fig_div = plot(hrly_dialogs_ratings_fig, output_type='div', include_plotlyjs=False)

        context_dict = {
            # "plot_title": "Duration analysis",
            "dialog_time_figure_div": dialog_time_fig_div,
            "shares_n_utt_div": shares_n_utt_div,
            "hrly_dialogs_ratings_fig_div": hrly_dialogs_ratings_fig_div
        }

        ########################
        # Last skill in dialog, all
        last_skill_fig = self.plot_last_skill_in_dialog(dialog_finished_df)
        last_skill_fig_div = plot(last_skill_fig, output_type='div', include_plotlyjs=False)
        # return render_template('overview_charts.html', name=name)
        context_dict["last_skill_fig_div"] = last_skill_fig_div

        # ######################################
        # Ratings, hist
        rating_hists_fig = self.plot_ratings_hists(skills_ratings_df)
        rating_hists_fig_div = plot(rating_hists_fig, output_type='div', include_plotlyjs=False)
        context_dict["rating_hists_fig_div"] = rating_hists_fig_div

        # ######################################
        # Rating by n_turns for last 7 days
        rating_by_n_turns_fig = self.plot_rating_by_turns(skills_ratings_df)
        rating_by_n_turns_fig_div = plot(rating_by_n_turns_fig, output_type='div', include_plotlyjs=False)
        context_dict["rating_by_n_turns_fig_div"] = rating_by_n_turns_fig_div

        skill_names = list(set(skills_ratings_df["active_skill"].values))
        # ######################################
        # Skill was selected, relative
        daily_counts_relative_fig = self.plot_skill_was_selected_relative(skills_ratings_df, skill_names)
        daily_counts_relative_fig_div = plot(daily_counts_relative_fig, output_type='div', include_plotlyjs=False)
        context_dict["daily_counts_relative_fig_div"] = daily_counts_relative_fig_div

        #
        moving_avg_fig = self.plot_skills_ratings_ma_dialogs_with_gt_7_turns(skills_ratings_df, skill_names)
        moving_avg_fig_div = plot(moving_avg_fig, output_type='div', include_plotlyjs=False)
        context_dict["moving_avg_fig_div"] = moving_avg_fig_div

        skill_ratings_total_ma_n_turns_gt_7_fig = self.plot_skill_ratings_total_ma_n_turns_gt_7(skills_ratings_df)
        skill_ratings_total_ma_n_turns_gt_7_fig_div = plot(skill_ratings_total_ma_n_turns_gt_7_fig, output_type='div', include_plotlyjs=False)
        context_dict["skill_ratings_total_ma_n_turns_gt_7_fig_div"] = skill_ratings_total_ma_n_turns_gt_7_fig_div


        #
        dialog_finished_reason_fig = self.plot_dialog_finished_reason(dialog_finished_df)
        dialog_finished_reason_fig_div = plot(dialog_finished_reason_fig, output_type='div',
                                                           include_plotlyjs=False)
        context_dict["dialog_finished_reason_fig_div"] = dialog_finished_reason_fig_div

        # ####
        dialog_finished_reason_w_rats_fig = self.plot_dialog_finished_reasons_w_ratings(dialog_finished_df)
        dialog_finished_reason_w_rats_fig_div = plot(dialog_finished_reason_w_rats_fig, output_type='div',
                                              include_plotlyjs=False)
        context_dict["dialog_finished_reason_w_rats_fig_div"] = dialog_finished_reason_w_rats_fig_div

        #
        dialog_finished_skill_rating_day_fig = self.plot_last_skill_in_dialog_with_rating(dialog_finished_df, skill_names)
        dialog_finished_skill_rating_day_fig_div = plot(dialog_finished_skill_rating_day_fig, output_type='div',
                                                     include_plotlyjs=False)
        context_dict["dialog_finished_skill_rating_day_fig_div"] = dialog_finished_skill_rating_day_fig_div

        # ##
        last_skill_stop_exit_info_fig = self.plot_last_skill_stop_exit(dialog_finished_df, skill_names)
        last_skill_stop_exit_info_fig_div = plot(last_skill_stop_exit_info_fig, output_type='div',
                                                        include_plotlyjs=False)
        context_dict["last_skill_stop_exit_info_fig_div"] = last_skill_stop_exit_info_fig_div
        return self.render('overview_charts.html', **context_dict)

    def prepare_data_for_ratings_plots(self, dialogs):
        """
        Prepares dataframe for some plots
        :param dialogs:
        :return:
        """
        import pandas as pd
        data = []
        for each_dialog in dialogs:
            data.append({
                'id': each_dialog.id,
                'rating': each_dialog.rating,
                'start_time': each_dialog.date_start
            })
        df = pd.DataFrame(data)
        return df

    def plot_number_of_dialogs_with_ratings_hrly(self, data_df):
        """
        Number of dialogs with ratings, hourly

        data_df DataFrame with ratings, start_time and id fields"""
        import datetime as dt
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        fig = make_subplots(rows=2, cols=1, subplot_titles=(
        'Number of dialogs with ratings, hourly', 'Avg dialog rating, hourly'))

        now = dt.datetime.now()
        # now = dt.datetime.now(tz=tz.gettz('UTC'))
        end = dt.datetime(year=now.year, month=now.month, day=now.day, hour=now.hour)
                       # tzinfo=now.tzinfo)
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
                          # tzinfo=now.tzinfo)
        start = end - dt.timedelta(days=14)
        x = []
        ratings = []
        for date in pd.date_range(start=start, end=end, freq='D'):
            x += [date]
            hourly_dialogs = data_df[(data_df['start_time'] <= date) & (
                    data_df['start_time'] > date - date.freq)]
            ratings += [0 if len(hourly_dialogs) == 0 else hourly_dialogs['rating'].mean()]
        fig.add_trace(go.Scatter(x=x, y=ratings, name='rating, 24h'), row=2, col=1)

        fig.update_layout(height=600, width=1200, showlegend=False)

        # first plot start, end
        end = dt.datetime(year=now.year, month=now.month, day=now.day, hour=now.hour)
                       # tzinfo=now.tzinfo)
        start = end - dt.timedelta(days=14)
        fig['layout']['xaxis2']['range'] = [start, end]

        fig['layout']['yaxis2']['range'] = [0, 5.5]
        fig.update_layout(hovermode='x')
        # fig.show()
        return fig

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

    def plot_rating_by_turns(self, skills_ratings):
        """
        Rating by n_turns for last 7 days

        :param skills_ratings:
        :return:
        """
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        import datetime as dt
        max_n = 30
        x = []
        y = []
        z = []
        n_days = 30
        now = dt.datetime.now(tz=tz.gettz('UTC'))
        start_date = (now - dt.timedelta(days=n_days))
        start_date = pd.Timestamp(start_date)
        daily_ratings = skills_ratings[skills_ratings['date'] >= start_date]
        # import ipdb; ipdb.set_trace()
        # print("daily_ratings")
        # print(daily_ratings)

        count = daily_ratings.groupby(['n_turns', 'rating']).count()['date']
        # print(count)
        for i in range(1, max_n):
            try:
                for j in (count.loc[i].keys()):
                    #        if count[i][j] // i > 0:
                    x.append(i)
                    y.append(j)
                    z.append(count[i][j] // i)
            except Exception as e:
                # skip because some keys may absent
                print(e)
                pass
        # print(z)
        # print(x)
        # print(y)


        rating_by_n_turns_fig = go.Figure(data=[go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                # TODO fix bug with sizes of markers for the case when we in lack of data
                size=[j / 0.05 for j in z],
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

    def plot_skill_was_selected_relative(self, skills_ratings, skill_names):
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
        start = end - dt.timedelta(days=31)

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
                               hovertemplate='%{y:.3f}: count %{customdata}'),
                    row=1, col=1)
                min_x = min(min_x, min(skill_c[n]))
                max_x = max(max_x, max(skill_c[n]))


        # for d, r in releases.values:
        #     if d > start:
        #         fig_daily_counts_relative.add_shape(
        #             dict(type="line", x0=d, y0=min_x, x1=d, y1=max_x, line=dict(color="RoyalBlue", width=1)), row=1,
        #             col=1)
        #         fig_daily_counts_relative.add_annotation(x=d, y=max_x, text=r, textangle=-90, showarrow=True,
        #                                                  font=dict(color="black", size=10), opacity=0.7, row=1, col=1)

        fig_daily_counts_relative.update_layout(height=500, width=1300, showlegend=True)
        fig_daily_counts_relative.update_layout(hovermode='x')
        # fig_daily_counts_relative.show()
        return fig_daily_counts_relative

    def plot_skills_ratings_ma_dialogs_with_gt_7_turns(self, skills_ratings, skill_names):
        """
        Skills Ratings, moving average over last 200 dialogs with number of turns > 7

        :param skills_ratings:
        :param skill_names:
        :return:
        """
        from tqdm import tqdm as tqdm
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        import datetime as dt

        # avg_n_dialogs = 200
        avg_n_dialogs = 3

        # n_turns = 7
        n_turns = 1

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
        start = end - dt.timedelta(days=35)

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

        # for d, r in releases.values:
        #     if d > start:
        #         fig_moving_avg.add_shape(
        #             dict(type="line", x0=d, y0=min_r, x1=d, y1=max_r, line=dict(color="RoyalBlue", width=1)), row=1,
        #             col=1)
        #         fig_moving_avg.add_annotation(x=d, y=max_r, text=r, textangle=-90, showarrow=True,
        #                                       font=dict(color="black", size=10), opacity=0.7, row=1, col=1)

        fig_moving_avg.update_layout(height=500, width=1300, showlegend=True)
        fig_moving_avg.update_layout(hovermode='x')
        fig_moving_avg['layout']['yaxis1']['range'] = [min_r, max_r]
        # fig_moving_avg.show()
        return fig_moving_avg

    def plot_skill_ratings_total_ma_n_turns_gt_7(self, skills_ratings):
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
        avg_n_dialogs = 3
        # n_turns = 7
        n_turns = 1

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
        start = end - dt.timedelta(days=35)

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

        # for d, r in releases.values:
        #     if d > start:
        #         fig_moving_avg_d_total.add_shape(
        #             dict(type="line", x0=d, y0=min_r, x1=d, y1=max_r, line=dict(color="RoyalBlue", width=1)), row=1,
        #             col=1)
        #         fig_moving_avg_d_total.add_annotation(x=d, y=max_r, text=r, textangle=-90, showarrow=True,
        #                                               font=dict(color="black", size=10), opacity=0.7, row=1, col=1)

        fig_moving_avg_d_total.update_layout(height=500, width=1300, showlegend=True)
        fig_moving_avg_d_total.update_layout(hovermode='x')
        fig_moving_avg_d_total['layout']['yaxis1']['range'] = [min_r, max_r]
        # fig_moving_avg_d_total.show()
        return fig_moving_avg_d_total

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

    def plot_last_skill_in_dialog_with_rating(self, dialog_finished_df, skill_names):
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
        start = end - dt.timedelta(days=31)

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

        # min_v, max_v = 10 * 10, - 10 ** 10
        for sn in sorted(list(skill_names)):
            if len(value_v[sn]) > 0:
                fig_dialog_finished_skill_day.add_scatter(name=sn, x=x[sn], y=value_v[sn], customdata=value_c[sn],
                                                          line={'dash': 'dot'},
                                                          hovertemplate='%{y:.2f}: count: %{customdata[0]} rating: %{customdata[1]:.2f} n_turns: %{customdata[2]:.2f}',
                                                          row=1, col=1)
                # min_v = min(min_v, min(value_v[sn]))
                # max_v = max(max_v, max(value_v[sn]))

        # for d, r in releases.values:
        #     if d > start:
        #         fig_dialog_finished_skill_day.add_shape(
        #             dict(type="line", x0=d, y0=min_v, x1=d, y1=max_v, line=dict(color="RoyalBlue", width=1)), row=1,
        #             col=1)
        #         fig_dialog_finished_skill_day.add_annotation(x=d, y=max_v, text=r, textangle=-90, showarrow=True,
        #                                                      font=dict(color="black", size=10), opacity=0.7, row=1,
        #                                                      col=1)

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