import json

from flask import Flask, Response, redirect, flash
from flask_admin import Admin
from flask_admin.actions import action
from flask_admin.contrib.sqla import ModelView
from flask_basicauth import BasicAuth
from jinja2 import Markup
from sqlalchemy.orm.session import Session
from werkzeug.exceptions import HTTPException

from db.models import Conversation

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
    return Markup(f"<a href='/admin'>{model.__getattribute__(name)[-20:]}</a>")


class ConversationModelView(SafeModelView):
    column_list = ('alexa_conversation_id',  'length', 'started', 'feedback', 'rating')
    column_filters = ('started', 'length', 'feedback', 'rating')
    column_formatters = {'alexa_conversation_id': _alexa_id_formatter}

    page_size = 1000

    @action('export', 'Export')
    def action_export(self, ids):
        try:
            view_args = self._get_list_extra_args()
            sort_column = self._get_column_by_idx(view_args.sort)

            _, query = self.get_list(view_args.page, sort_column, view_args.sort_desc, view_args.search,
                                     view_args.filters, execute=False)

            items = self.session.query(Conversation).filter(Conversation.id.in_(ids))
            if items:
                items = [
                    {
                        'conversation_id': item.alexa_conversation_id,
                        'feedback': item.feedback,
                        'rating': item.rating,
                        'utterances': [
                            f'{utt.active_skill or utt.type}: {utt.text}' for utt in item.utterances
                        ]
                    }
                    for item in items
                ]
                return (
                    json.dumps(items, indent=4),
                    200,
                    {
                        'Content-Type': 'application/json',
                        'Pragma': 'no-cache',
                        'Cache-Control': 'no-cache, no-store, must-revalidate',
                        'Expires': '0',
                        'Content-Disposition': 'attachment; filename="mymodel.json"'
                    }
                )

            return ''
        except Exception as e:
            if not self.handle_view_exception(e):
                raise

            flash('Failed to export: {}'.format(str(e)), 'error')


def start_admin(session: Session, user: str, password: str) -> None:
    app.config['FLASK_ADMIN_SWATCH'] = 'cerulean'
    app.config['BASIC_AUTH_USERNAME'] = user
    app.config['BASIC_AUTH_PASSWORD'] = password

    admin = Admin(app, name='microblog', template_mode='bootstrap3')
    admin.add_view(ConversationModelView(Conversation, session))

    app.run(host='0.0.0.0')
