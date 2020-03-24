import json
from collections import defaultdict

import requests
from flask import Flask, Response, redirect, flash
from flask_admin import Admin
from flask_admin.actions import action
from flask_admin.contrib.sqla import ModelView
from flask_admin.contrib.sqla.filters import BaseSQLAFilter
from flask_admin.model.filters import BaseBooleanFilter
from flask_basicauth import BasicAuth
from jinja2 import Markup
from sqlalchemy.orm.session import Session
from sqlalchemy.orm.strategy_options import joinedload
from sqlalchemy.sql import func, text
from werkzeug.exceptions import HTTPException

from db.models import Conversation
from db.models.utterance import Utterance

app = Flask(__name__)
basic_auth = BasicAuth(app)


class AuthException(HTTPException):
    def __init__(self, message):
        super().__init__(message, Response(
            message, 401,
            {'WWW-Authenticate': 'Basic realm="Login Required"'}
        ))


class SafeModelView(ModelView):
    can_delete = False
    can_create = False
    can_edit = False

    def is_accessible(self):
        if not basic_auth.authenticate():
            raise AuthException('Not authenticated. Refresh the page.')
        else:
            return True

    def inaccessible_callback(self, name, **kwargs):
        return redirect(basic_auth.challenge())


def _alexa_id_formatter(view, context, model: Conversation, name):
    return Markup(f"<a href='/conversation/{model.id}'>{model.__getattribute__(name)}</a>")


class FilterByActiveSkill(BaseSQLAFilter):
    # Override to create an appropriate query and apply a filter to said query with the passed value from the filter UI
    def apply(self, query, value, alias=None):
        return query.filter(text(f"conversation.id in (select distinct utterance.conversation_id from utterance where utterance.active_skill = '{value}')"))

    # readable operation name. This appears in the middle filter line drop-down
    def operation(self):
        return u'equals'


class FilterByUtteranceText(BaseSQLAFilter):
    # Override to create an appropriate query and apply a filter to said query with the passed value from the filter UI
    def apply(self, query, value, alias=None):
        return query.filter(text(f"conversation.id in (select distinct utterance.conversation_id from utterance where utterance.text LIKE '%{value}%')"))

    # readable operation name. This appears in the middle filter line drop-down
    def operation(self):
        return u'contains'


class FilterByTgId(BaseSQLAFilter):
    # Override to create an appropriate query and apply a filter to said query with the passed value from the filter UI
    def apply(self, query, value, alias=None):
        return query.filter(Conversation.human['user_telegram_id'].astext.contains(value))

    # readable operation name. This appears in the middle filter line drop-down
    def operation(self):
        return u'contains'


class FilterByVersion(BaseSQLAFilter):
    def apply(self, query, value, alias=None):
        return query.filter(text(f"conversation.id in (select distinct utterance.conversation_id from utterance where utterance.attributes->>'version' = '{value}')"))

    def operation(self):
        return u'equals'

class FilterByVersionList(BaseSQLAFilter):
    def apply(self, query, value, alias=None):
        versions = ', '.join([f"'{v.strip()}'" for v in value.split(',')])
        return query.filter(text(f"conversation.id in (select distinct utterance.conversation_id from utterance where utterance.attributes->>'version' in ({versions}))"))

    def operation(self):
        return u'in list'


class ConversationModelView(SafeModelView):
    column_list = ('id',  'length', 'date_start', 'feedback', 'rating', 'tg_id')
    column_filters = (
        'date_start',
        'length',
        'feedback',
        'rating',
        FilterByActiveSkill(column=None, name='active_skill'),
        FilterByUtteranceText(column=None, name='utterance_text'),
        FilterByTgId(column=None, name='tg_id'),
        FilterByVersion(column=None, name='version'),
        FilterByVersionList(column=None, name='version')
    )
    list_template = 'admin/model/custom_list.html'
    column_formatters = {
        'id': _alexa_id_formatter,
        'tg_id': lambda v, c, m, n: m.__getattribute__(n)[:5]
    }
    column_sortable_list = ('length', 'date_start', 'feedback', 'rating', ('tg_id', Conversation.tg_id),)
    column_default_sort = ('date_start', True)

    page_size = 1000

    def render(self, template, **kwargs):
        if template == 'admin/model/custom_list.html':
            search = kwargs['search']
            joins = {}
            count_joins = {}

            query = self.session.query(func.avg(Conversation.rating).label('avg'))
            count_query = self.get_count_query() if not self.simple_list_pager else None

            if hasattr(query, '_join_entities'):
                for entity in query._join_entities:
                    for table in entity.tables:
                        joins[table] = None

            # Apply search criteria
            if self._search_supported and search:
                query, count_query, joins, count_joins = self._apply_search(query,
                                                                            count_query,
                                                                            joins,
                                                                            count_joins,
                                                                            search)

            # Apply filters
            if kwargs['active_filters'] and self._filters:
                query, count_query, joins, count_joins = self._apply_filters(query,
                                                                             count_query,
                                                                             joins,
                                                                             count_joins,
                                                                             kwargs['active_filters'])

            for j in self._auto_joins:
                query = query.options(joinedload(j))
            # we are only interested in the list page
            result = query.first()[0]
            if result is not None:
                result = f'{result:.3f}'
            kwargs['avg_rating'] = result
        return super(ConversationModelView, self).render(template, **kwargs)

    @action('export', 'Export')
    def action_export(self, ids):
        try:
            view_args = self._get_list_extra_args()
            sort_column = self._get_column_by_idx(view_args.sort)

            _, query = self.get_list(view_args.page, sort_column, view_args.sort_desc, view_args.search,
                                     view_args.filters, execute=False)

            conversations = self.session.query(Conversation).filter(Conversation.id.in_(ids))
            if conversations:
                conversations = [{
                    'id': conv.id,
                    'utterances': [
                        {
                            'text': utt.text,
                            'date_time': utt.date_time.strftime('%Y-%m-%d %H:%M:%S.%f'),
                            'active_skill': utt.active_skill
                        } if utt.active_skill is not None else {
                            'text': utt.text,
                            'date_time': utt.date_time.strftime('%Y-%m-%d %H:%M:%S.%f'),
                            'attributes': utt.attributes
                        }
                        for utt in conv.utterances
                    ],
                    'human': conv.human,
                    'bot': conv.bot,
                    'date_start': conv.date_start.strftime('%Y-%m-%d %H:%M:%S.%f'),
                    'date_finish': conv.date_finish.strftime('%Y-%m-%d %H:%M:%S.%f')
                }
                    for conv in conversations]
                return (
                    json.dumps(conversations, indent=4),
                    200,
                    {
                        'Content-Type': 'application/json',
                        'Pragma': 'no-cache',
                        'Cache-Control': 'no-cache, no-store, must-revalidate',
                        'Expires': '0',
                        'Content-Disposition': 'attachment; filename="alexaprize-export.json"'
                    }
                )

            return ''
        except Exception as e:
            if not self.handle_view_exception(e):
                raise

            flash('Failed to export: {}'.format(str(e)), 'error')

# TODO: refactor filters and remove error that occurs when two different rating filters are applied
class FilterByRegExp(BaseSQLAFilter):
    def apply(self, query, value, alias=None):
        return query.filter(Utterance.text.op('~')(value))

    def operation(self):
        return u'regular expression'


class FilterByRatingEquals(BaseSQLAFilter):
    def apply(self, query, value, alias=None):
        return query.join(Conversation).filter(Conversation.rating == value)

    def operation(self):
        return u'equals'


class FilterByRatingGreater(BaseSQLAFilter):
    def apply(self, query, value, alias=None):
        return query.join(Conversation).filter(Conversation.rating > value)

    def operation(self):
        return u'greater than'


class FilterByRatingSmaller(BaseSQLAFilter):
    def apply(self, query, value, alias=None):
        return query.join(Conversation).filter(Conversation.rating > value)

    def operation(self):
        return u'smaller than'


class FilterByRatingEmpty(BaseSQLAFilter, BaseBooleanFilter):
    def apply(self, query, value, alias=None):
        if value == '1':
            return query.join(Conversation).filter(Conversation.rating == None)
        else:
            return query.join(Conversation).filter(Conversation.rating != None)

    def operation(self):
        return u'empty'

def _utt_conv_id_formatter(view, context, model: Utterance, name):
    return Markup(f"<a href='/conversation/{model.conversation_id}'>{model.__getattribute__(name)}</a>")


class UtteranceModelView(SafeModelView):
    page_size = 2000

    column_list = ('conversation_id', 'text', 'date_time', 'active_skill', 'rating')
    column_formatters = {'conversation_id': _utt_conv_id_formatter}

    column_sortable_list = ('text', 'date_time', 'active_skill')
    column_default_sort = ('date_time', True)
    column_filters = (
        FilterByRegExp(column=None, name='Text'),
        'date_time',
        'active_skill',
        FilterByRatingEquals(column=None, name='Rating'),
        FilterByRatingGreater(column=None, name='Rating'),
        FilterByRatingSmaller(column=None, name='Rating'),
        FilterByRatingEmpty(column=None, name='Rating')
    )

    @action('export', 'Export')
    def action_export(self, ids):
        try:
            view_args = self._get_list_extra_args()
            sort_column = self._get_column_by_idx(view_args.sort)

            _, query = self.get_list(view_args.page, sort_column, view_args.sort_desc, view_args.search,
                                     view_args.filters, execute=False)

            utterances = self.session.query(Utterance).filter(Utterance.id.in_(ids))
            if utterances:
                utterances = [{
                    'text': utt.text,
                    'date_time': utt.date_time.strftime('%Y-%m-%d %H:%M:%S.%f'),
                    'active_skill': utt.active_skill,
                    'rating': utt.rating
                }
                    for utt in utterances]
                return (
                    json.dumps(utterances, indent=4),
                    200,
                    {
                        'Content-Type': 'application/json',
                        'Pragma': 'no-cache',
                        'Cache-Control': 'no-cache, no-store, must-revalidate',
                        'Expires': '0',
                        'Content-Disposition': 'attachment; filename="alexaprize-export.json"'
                    }
                )

            return ''
        except Exception as e:
            if not self.handle_view_exception(e):
                raise

            flash('Failed to export: {}'.format(str(e)), 'error')


def start_admin(session: Session, user: str, password: str, port: int, amazon_container: str) -> None:
    @app.route('/conversation/<id>')
    def show(id: str):
        conv = session.query(Conversation).filter_by(id=id).one()
        utts = []
        utterances = list(conv.utterances)

        def format_annotations(utt: dict):
            annotations = utt['annotations']
            annotations = ''.join([f'<tr><td>{key}</td><td>{val}</td></tr>' for key, val in annotations.items()])
            hypotheses = utt.get('hypotheses', [])
            h2 = defaultdict(lambda: '')
            for hyp in hypotheses:
                hyp.pop('annotations', None)
                h2[hyp.pop('skill_name')] += str(hyp) + '<br>'
            hypotheses = ''.join([f'<tr><td>{key}</td><td>{val}</td></tr>' for key, val in h2.items()])
            return f'annotations:<table>{annotations}</table><br>hypotheses:<table>{hypotheses}</table>'

        original_utts = []
        try:
            resp = requests.get(f'{amazon_container}/api/dialogs/{id}')
            if resp.status_code == 200:
                original_utts = resp.json().get('utterances', [])
                if original_utts:
                    original_utts = [format_annotations(utt) for utt in original_utts]
            else:
                original_utts = [f'Error: {resp.status_code}' for _ in utterances]
        except requests.exceptions.ConnectTimeout:
            original_utts = ['Timeout error' for _ in utterances]
        for i, utt in enumerate(utterances):
            if i != 0:
                td = (utt.date_time - utterances[i-1].date_time).total_seconds()
                td = '[{:.2f}] '.format(td)
            else:
                td = ''
            utts.append(
                f'<tr bgcolor={"lightgray" if utt.active_skill else "white"}><td>{utt.active_skill or "Human"}</td><td><details><summary>{td}{utt.text}</summary>{original_utts[i] if original_utts else ""}</details></td></tr>'
            )
        if utterances[0].attributes is not None:
            ver = utterances[0].attributes.get('version')
        else:
            ver = None
        attrs = [
            f'id: {conv.amazon_conv_id}',
            f'user_telegram_id: {conv.human["user_telegram_id"]}',
            f'date_start: {conv.date_start}',
            f'rating: {conv.rating}',
            f'feedback: {conv.feedback}',
            f'version: {ver}'
        ]
        return "<style>details summary {display: block;}"\
               f"</style><table><tr>{'<br>'.join(attrs)}</tr>{''.join(utts)}</table>"

    @app.route('/')
    def index():
        return '<a href="/admin/">Click me to get to Admin!</a>'

    # Without session.expire_all flask-admin is showing old values of rows if they were updated after server start
    @app.teardown_request
    def teardown_request(*args, **kwargs):
        session.expire_all()

    app.config['FLASK_ADMIN_SWATCH'] = 'cerulean'
    app.config['BASIC_AUTH_USERNAME'] = user
    app.config['BASIC_AUTH_PASSWORD'] = password

    admin = Admin(app, name='microblog', template_mode='bootstrap3')
    admin.add_view(ConversationModelView(Conversation, session))
    admin.add_view(UtteranceModelView(Utterance, session))

    app.run(host='0.0.0.0', port=port)
