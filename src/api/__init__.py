from flask import Flask
from src.api.index_routes import blueprint as index_blueprint
from src.api.inference_routes import blueprint as inference_blueprint
from src.api.database import db
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///C:\\Users\\ADMIN\\test2.db'
db.init_app(app)

with app.app_context():
    db.create_all()
    db.session.commit()

app.register_blueprint(index_blueprint)
app.register_blueprint(inference_blueprint)

if __name__ == "__main__":
    app.run()
