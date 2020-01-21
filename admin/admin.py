import json

from flask import Flask, Response, redirect, flash
from flask_admin import Admin
from flask_admin.actions import action
from flask_admin.contrib.sqla import ModelView
from flask_admin.contrib.sqla.filters import BaseSQLAFilter
from flask_basicauth import BasicAuth
from jinja2 import Markup
from sqlalchemy import func
from sqlalchemy.orm.session import Session
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
        return query.join(Conversation.utterances).filter(Utterance.active_skill == value)

    # readable operation name. This appears in the middle filter line drop-down
    def operation(self):
        return u'equals'


class FilterByUtteranceText(BaseSQLAFilter):
    # Override to create an appropriate query and apply a filter to said query with the passed value from the filter UI
    def apply(self, query, value, alias=None):
        return query.join(Conversation.utterances).filter(Utterance.text.contains(value))

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


class ConversationModelView(SafeModelView):
    column_list = ('id',  'length', 'date_start', 'feedback', 'rating', 'tg_id')
    column_filters = (
        'date_start',
        'length',
        'feedback',
        'rating',
        FilterByActiveSkill(column=None, name='active_skill'),
        FilterByUtteranceText(column=None, name='utterance_text'),
        FilterByTgId(column=None, name='tg_id')
    )

    column_formatters = {
        'id': _alexa_id_formatter,
        'tg_id': lambda v, c, m, n: m.__getattribute__(n)[:5]
    }
    column_sortable_list = ('length', 'date_start', 'feedback', 'rating', ('tg_id', Conversation.tg_id),)

    page_size = 1000

    def get_count_query(self):
        return self.session.query(func.count(self.model.id.distinct())).select_from(self.model)

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


def start_admin(session: Session, user: str, password: str, port: int) -> None:
    @app.route('/conversation/<id>')
    def show(id: str):
        conv = session.query(Conversation).filter_by(id=id).one()
        utts = []
        utterances = list(conv.utterances)
        for i, utt in enumerate(utterances):
            if i != 0:
                td = (utt.date_time - utterances[i-1].date_time).total_seconds()
                td = '[{:.2f}] '.format(td)
            else:
                td = ''
            utts.append(
                f'<tr bgcolor={"lightgray" if utt.active_skill else "white"}><td>{utt.active_skill or "Human"}</td><td>{td}{utt.text}</td></tr>'
            )
        attrs = [
            f'id: {conv.amazon_conv_id}',
            f'user_telegram_id: {conv.human["user_telegram_id"]}',
            f'date_start: {conv.date_start}',
            f'rating: {conv.rating}',
            f'feedback: {conv.feedback}'
        ]
        return f"<table><tr>{'<br>'.join(attrs)}</tr>{''.join(utts)}</table>"

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

    app.run(host='0.0.0.0', port=port)
