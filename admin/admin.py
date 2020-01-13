from flask import Flask, Response, redirect
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
from flask_basicauth import BasicAuth
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


class ConversationModelView(SafeModelView):
    pass


def start_admin(session: Session, user: str, password: str) -> None:
    app.config['FLASK_ADMIN_SWATCH'] = 'cerulean'
    app.config['BASIC_AUTH_USERNAME'] = user
    app.config['BASIC_AUTH_PASSWORD'] = password

    admin = Admin(app, name='microblog', template_mode='bootstrap3')
    admin.add_view(ConversationModelView(Conversation, session))

    app.run()
