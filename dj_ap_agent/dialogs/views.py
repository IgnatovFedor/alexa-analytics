from django.shortcuts import render
from django.views.decorators.cache import cache_page
from django.core.cache import cache
from django.shortcuts import render
from django.http import HttpResponse
import numpy as np

# Include the `fusioncharts.py` file which has required functions to embed the charts in html page
# from .fusioncharts import FusionCharts


# Loading Data from a Static JSON String
# The `chart` method is defined to load chart data from an JSON string.

def chart(request):
    sample_data = {
        "1": 1,
        "2": 2,
        "3": 3,
    }
    # returning complete JavaScript and HTML code, which is used to generate chart in the browsers.
    return render(request, 'index.html', {'output': sample_data})


# def show_day_sentiments(request, date=None):
#     if not date:
#         date = "Today!"
#     # prepare data
#
#     # prepare view of the data
#
#     return HttpResponse("You're looking at chart for %s." % date)

# def make_ploly_div():



@cache_page(60 * 60 * 24)
def show_skills_sentiments(request):

    from dialogs.analytics.dialogs_analytics import analyze_skills_sentiments
    from plotly.offline import plot
    import plotly.graph_objs as go
    # prepare data
    skill_stat_dict = cache.get_or_set('analyze_skills_sentiments', analyze_skills_sentiments, timeout=60*60*24)

    for each_skill, sentiments_distribution_dict in skill_stat_dict.items():

        fig = go.Figure()
        # bar_plot = go.Bar(y=[2, 3, 1])
        x_names = list(skill_stat_dict[each_skill].keys())
        ys = [v for k, v in skill_stat_dict[each_skill].items()]
        bar_plot = go.Bar(x=x_names, y=ys)
        fig.add_trace(bar_plot)
        bar_div = plot(fig, output_type='div', include_plotlyjs=False)

        # the html top draw:
        skill_stat_dict[each_skill]['bar_plot'] = bar_div
    # push it to template
    context_dict = {
        "skill_stat_dict": skill_stat_dict
    }
    return render(request, 'dialogs/skills_sentiments.html', context_dict)

def show_skills_sentiments_stacked(request):
    from dialogs.analytics.dialogs_analytics import analyze_skills_sentiments
    from plotly.offline import plot
    import plotly.graph_objs as go
    import pandas as pd

    # prepare data or retrieve from cache
    # skill_stat_dict = analyze_skills_sentiments()
    # cache.set('analyze_skills_sentiments', skill_stat_dict)
    skill_stat_dict = cache.get_or_set('analyze_skills_sentiments', analyze_skills_sentiments, timeout=60*60*24)
    # skill_stat_dict = cache.get_or_set('analyze_skills_sentiments', analyze_skills_sentiments, timeout=60*60)
    # skill_stat_dict = analyze_skills_sentiments()

    xs = list(skill_stat_dict.keys())

    # sentiments:
    negative_sents = []
    neutral_sents = []
    positive_sents = []


    # sentiments distribution per every skill:
    relative_sentiments = []
    for each_skill, sentiments_distribution_dict in skill_stat_dict.items():

        negative_sents.append(skill_stat_dict[each_skill]['negative'])
        neutral_sents.append(skill_stat_dict[each_skill]['neutral'])
        positive_sents.append(skill_stat_dict[each_skill]['positive'])

        # counts:
        counts =[v for k, v in skill_stat_dict[each_skill].items()]

        # per skill statsummas:
        summa = np.sum(counts)

        relative_sentiments_for_skill = [each_count/summa for each_count in counts]
        relative_sentiments.append(relative_sentiments_for_skill)
    
    pos, neut, neg = zip(*relative_sentiments)

    # sort by neg:
    srt_res = sorted(
        zip(neg, pos, neut, negative_sents, positive_sents, neutral_sents, xs),
        key=lambda tup: tup[0],
        reverse=True
    )

    neg, pos, neut, negative_sents, positive_sents, neutral_sents, xs = zip(*srt_res)
    # plotting

    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=neg, name='negative', customdata=negative_sents,
                         hovertemplate='count: %{customdata}',
                         marker_color='tomato'))
    fig.add_trace(go.Bar(x=xs, y=neut, name='neutral', customdata=neutral_sents, hovertemplate='count: %{customdata}',
                         marker_color='lightgrey'))
    #     '#1f77b4',  // muted blue
    #     '#ff7f0e',  // safety orange
    #     '#2ca02c',  // cooked asparagus green
    #     '#d62728',  // brick red
    #     '#9467bd',  // muted purple
    #     '#8c564b',  // chestnut brown
    #     '#e377c2',  // raspberry yogurt pink
    #     '#7f7f7f',  // middle gray
    #     '#bcbd22',  // curry yellow-green
    #     '#17becf'   // blue-teal
    fig.add_trace(go.Bar(x=xs, y=pos, name='positive', customdata=positive_sents, hovertemplate='count: %{customdata}',
                         marker_color='mediumseagreen'))
                         # marker_color='lightgreen'))
    fig.update_layout(barmode='stack', xaxis={'showspikes': True}, hovermode='x')

    stacked_div = plot(fig, output_type='div', include_plotlyjs=False)
    context_dict = {
        "stacked_div": stacked_div
    }

    return render(request, 'dialogs/skills_sentiment_stacked.html', context_dict)


def show_skills_sentiments_bubbles(request):
    from dialogs.analytics.dialogs_analytics import analyze_skills_sentiments
    from plotly.offline import plot
    import plotly.graph_objs as go

    # prepare data or retrieve from cache
    skill_stat_dict = cache.get_or_set('analyze_skills_sentiments', analyze_skills_sentiments, timeout=60*60*24)
    # skill_stat_dict = ()

    xs = []
    ys = []
    zs = []

    circle_sizes = []
    for each_skill, sentiments_distribution_dict in skill_stat_dict.items():

        # bar_plot = go.Bar(y=[2, 3, 1])
        # skill names
        xs += [each_skill]*3

        # category names
        ys += [k for k, v in skill_stat_dict[each_skill].items()]

        # counts:
        counts =[v for k, v in skill_stat_dict[each_skill].items()]
        zs += counts

        summa = np.sum(counts)

        circle_sizes_3 = [each_count/summa*50.0 for each_count in counts]
        circle_sizes +=circle_sizes_3

    print(xs)
    print(ys)
    print(zs)
    print(circle_sizes)

    # max_size = 100
    # # sizes
    # sizes_data = [(j / max_size + 10.0) for j in zs]
    # print(sizes_data)
    fig = go.Figure()
    bubbles_plot = go.Scatter(
        x=xs,
        y=ys,
        mode='markers',
        marker=dict(
            size=circle_sizes,
        ),
        customdata=zs,
        hovertemplate='count: %{customdata}',
        name='Sentiment by skill'
    )


    fig.add_trace(bubbles_plot)
    bubbles_div = plot(fig, output_type='div', include_plotlyjs=False)
    context_dict = {
        "bubbles_div": bubbles_div
    }
    return render(request, 'dialogs/skills_sentiment_bubbles.html', context_dict)


def show_day_sentiments(request, date=None):
    if not date:
        date = "Today!"

    # prepare data
    # dummy_data =
    # prepare plot
    # push it to template

    # import plotly.graph_objects as go
    # fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
    # fig.write_html('first_figure.html', auto_open=True)

    from plotly.offline import plot
    import plotly.graph_objs as go
    fig = go.Figure()
    scatter = go.Scatter(x=[0, 1, 2, 3], y=[0, 1, 2, 3],
                         mode='lines', name='test',
                         opacity=0.8, marker_color='green')
    fig.add_trace(scatter)
    fig.update_layout(
        title="BARBARBAR",
        xaxis_title="x Axis Title",
        yaxis_title="y Axis Title",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )
    plot_div = plot(fig, output_type='div', include_plotlyjs=False)

    fig2 = go.Figure()

    bar_plot = go.Bar(y=[2, 3, 1])
    fig2.add_trace(bar_plot)
    bar_div = plot(fig2, output_type='div', include_plotlyjs=False)

    # fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
    # try:
    #     p = Poll.objects.get(pk=poll_id)
    # except Poll.DoesNotExist:
    #     raise Http404("Poll does not exist")
    context_dict = {'output': date,
                    'plot_div': plot_div,
                    'bar_div': bar_div,
                    }

    return render(request, 'dialogs/chart.html', context_dict)

def show_daily_sentiments_stacked(request):

    from dialogs.analytics.dialogs_analytics import analyze_daily_sentiments_distribution
    from plotly.offline import plot
    import plotly.graph_objs as go
    days=14
    # days=4
    # sent_distribution_df = analyze_daily_sentiments_distribution(days=days)
    sent_distribution_df = cache.get_or_set('analyze_daily_sentiments_distribution',
                                            lambda: analyze_daily_sentiments_distribution(days=days), timeout=60 * 60 * 24)

    # plot
    fig = go.Figure()

    fig.add_trace(go.Bar(x=sent_distribution_df.index, y=sent_distribution_df['negative'], name='negative',
                         # customdata=negative_sents,
                         # hovertemplate='count: %{customdata}',
                         marker_color='tomato'))
    fig.add_trace(go.Bar(x=sent_distribution_df.index, y=sent_distribution_df['neutral'], name='neutral',
                         # customdata=neutral_sents, hovertemplate='count: %{customdata}',
                         marker_color='lightgrey'))
    fig.add_trace(go.Bar(x=sent_distribution_df.index, y=sent_distribution_df['positive'], name='positive',
                         # customdata=positive_sents, hovertemplate='count: %{customdata}',
                         marker_color='mediumseagreen'))
    # marker_color='lightgreen'))
    fig.update_layout(barmode='stack', xaxis={'showspikes': True}, hovermode='x')

    stacked_div = plot(fig, output_type='div', include_plotlyjs=False)
    context_dict = {
        "plot_title": "Daily Sentiments over user utterances",
        "stacked_div": stacked_div,

    }

    return render(request, 'dialogs/skills_sentiment_stacked.html', context_dict)


# ########################################################################################################
# topics views

def show_popular_weekly_topics(request):
    """
    View for presenting weekly popular topics
    :param request:
    :return:
    """
    from dialogs.analytics.dialogs_analytics import collect_topics_statistics
    # stat_df = collect_topics_statistics()
    # cache.set('collect_topics_statistics', stat_df)
    stat_df = cache.get_or_set('collect_topics_statistics', collect_topics_statistics, timeout=60 * 60 * 24)

    context_dict = {
        "title": "Popular weekly topics (cobot noun_phrases)",
        "df": stat_df.to_html(classes='table table-striped table-bordered table-hover table-condensed', table_id='example'),
        "js_table_script": '''
            $(document).ready(function() {
                $("#example").DataTable( {
                        "paging":   false,
                        "order": [[ 1, "desc" ]],
                        "fnRowCallback": function (nRow, aData, iDisplayIndex, iDisplayIndexFull) {
                            if (aData[4] > 0.0) {
                                $("td", nRow).css("background-color", "mediumseagreen");
                            }
                            else if (aData[4] < 0.0) {
                                $("td", nRow).css("background-color", "tomato");
                            }
                        },
                    } );
            } );
        '''

    }

    return render(request, 'dialogs/dataframe_view.html', context_dict)


def show_topics_sentiments(request):
    from dialogs.analytics.dialogs_analytics import calc_topic_sentiments_statistics
    # stat_df = calc_topic_sentiments_statistics()
    # cache.set('calc_topic_sentiments_statistics', stat_df)
    stat_df = cache.get_or_set('calc_topic_sentiments_statistics', calc_topic_sentiments_statistics, timeout=60 * 60 * 24)

    context_dict = {
        "title": "weekly topics sentiments (cobot noun_phrases ~ sentiments)",
        "df": stat_df.to_html(classes='table table-striped table-bordered table-hover table-condensed',
                              table_id='example'),
        "js_table_script": '''
                $(document).ready(function() {
                    $("#example").DataTable( {
                            "paging":   false,
                            "order": [[ 2, "desc" ]],
                            "fnRowCallback": function (nRow, aData, iDisplayIndex, iDisplayIndexFull) {
                                if (aData[6] > 0.40) {
                                    $("td", nRow).css("background-color", "mediumseagreen");
                                }
                                else if (aData[8] > 0.11) {
                                    $("td", nRow).css("background-color", "tomato");
                                }
                            },
                        } );
                } );
            '''

    }

    return render(request, 'dialogs/dataframe_view.html', context_dict)


def show_skills_emotions(request):
    from dialogs.analytics.dialogs_analytics import collect_emotions_after_skill_statistics
    import datetime as dt
    from plotly.offline import plot
    import plotly.graph_objs as go

    # since_dt = dt.datetime.now() - dt.timedelta(days=7)
    since_dt = dt.datetime(2020, 5, 6) - dt.timedelta(days=7)

    # stat_df = collect_emotions_after_skill_statistics(renormalize=True, since_dt=since_dt)
    # cache.set('collect_emotions_after_skill_statistics', stat_df)

    collect_emotions_after_skill_statistics_with_args = lambda: collect_emotions_after_skill_statistics(renormalize=True, since_dt=since_dt)
    stat_df = cache.get_or_set('collect_emotions_after_skill_statistics',
                               collect_emotions_after_skill_statistics_with_args,
                               timeout=60 * 60 * 24)

    # plot
    fig = go.Figure()

    for emotion in stat_df.index:
        fig.add_trace(go.Bar(x=stat_df.columns, y=stat_df.loc[emotion], name=emotion))
    fig.update_layout(barmode='stack', xaxis={'showspikes': True}, hovermode='x')

    stacked_div = plot(fig, output_type='div', include_plotlyjs=False)
    context_dict = {
        "plot_title": "Skills causes emotions for last 7 days",
        "stacked_div": stacked_div,
    }

    return render(request, 'dialogs/skills_sentiment_stacked.html', context_dict)