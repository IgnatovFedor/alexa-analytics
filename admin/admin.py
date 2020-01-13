from flask import Flask
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
from sqlalchemy.orm.session import Session

from db.models import Conversation


def start_admin(session: Session) -> None:
    app = Flask(__name__)
    app.config['FLASK_ADMIN_SWATCH'] = 'cerulean'

    admin = Admin(app, name='microblog', template_mode='bootstrap3')
    admin.add_view(ModelView(Conversation, session))

    app.run()
