from django.urls import path

from . import views

app_name = 'dialogs'

urlpatterns = [
    # ex: /polls/
    # path('', views.index, name='index'),
    # ex: /polls/5/
    path('day_sentiments/', views.show_day_sentiments, name='day_sentiments'),
    path('skills_sentiments/', views.show_skills_sentiments, name='skills_sentiments'),
    path('skills_sentiments_bubbles/', views.show_skills_sentiments_bubbles, name='skills_sentiments_bubbles'),
    path('skills_sentiments_stacked/', views.show_skills_sentiments_stacked, name='skills_sentiments_stacked'),

    path('daily_sentiments_stacked/', views.show_daily_sentiments_stacked, name='daily_sentiments_stacked'),

    path('popular_weekly_topics/', views.show_popular_weekly_topics, name='popular_weekly_topics'),
    path('topics_sentiments/', views.show_topics_sentiments, name='topics_sentiments'),
    path('skills_emotions/', views.show_skills_emotions, name='skills_emotions'),
    # # ex: /polls/5/results/
    # path('<int:question_id>/results/', views.results, name='results'),
    # # ex: /polls/5/vote/
    # path('<int:question_id>/vote/', views.vote, name='vote'),
]