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
            # for utt in dialog.utterances:
            #     if hasattr(utt, 'active_skill'):
            #         skills_ratings += [[date, utt.active_skill, rating, conv_id, dialog.version]]
        # skills_ratings = pd.DataFrame(skills_ratings,
        #                               columns=['date', 'active_skill', 'rating', 'conv_id',
        #                                        'version'])
        # skills_ratings['date'] = pd.to_datetime(skills_ratings['date'], utc=True)
        #
        # n_turns = skills_ratings['conv_id'].value_counts().to_dict()
        # skills_ratings['n_turns'] = skills_ratings['conv_id'].apply(lambda x: n_turns[x])

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

    def make_skills_durations_plot(self, dialog_durations_df):
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
        dialog_time_fig['layout']['yaxis1']['range'] = [50, 200]
        dialog_time_fig['layout']['yaxis2']['range'] = [10, 35]
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

    def make_last_skill_in_dialog_plot(self, dialog_finished_df):
        from plotly.subplots import make_subplots
        # import plotly.graph_objects as go
        import datetime as dt

        fig_dialog_finished_skill_all_day = make_subplots(rows=1, cols=1, subplot_titles=(
        'Last skill in dialog, all, Last 24h',))

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
        # retrieve all dialogs
        dialogs = self.session.query(Conversation).order_by(Conversation.date_finish.desc())
        # print(dialogs)

        dialog_durations_df, skills_ratings = self.prepare_data_for_plotting(dialogs)

        # print(dialog_durations_df)
        # print(dialog_finished_df)

        # retrieve data for skill frequency chart
        # prepare plot for it

        # retrieve data for Dialog time(sec), Daily chart
        # prepare plot of it
        dialog_time_fig, shares_n_utt_fig = self.make_skills_durations_plot(dialog_durations_df)
        dialog_time_fig_div = plot(dialog_time_fig, output_type='div', include_plotlyjs=False)
        shares_n_utt_div = plot(shares_n_utt_fig, output_type='div', include_plotlyjs=False)
        context_dict = {
            # "plot_title": "Duration analysis",
            "dialog_time_figure_div": dialog_time_fig_div,
            "shares_n_utt_div": shares_n_utt_div
        }

        ########################
        # last skill fig
        dialog_finished_df = self.prepare_dialog_finished_df(dialogs)
        last_skill_fig = self.make_last_skill_in_dialog_plot(dialog_finished_df)
        last_skill_fig_div = plot(last_skill_fig, output_type='div', include_plotlyjs=False)
        # return render_template('overview_charts.html', name=name)
        context_dict["last_skill_fig_div"] = last_skill_fig_div

        return self.render('overview_charts.html', **context_dict)
