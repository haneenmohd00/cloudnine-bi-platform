from flask import Flask

from extensions import db
from dotenv import load_dotenv
load_dotenv()
from routes.main_routes import main_bp




def create_app():
    app = Flask(__name__)

    app.config["SECRET_KEY"] = "cloudnine-bi-development-key"

    app.config["SQLALCHEMY_DATABASE_URI"] = (
        "sqlite:///cloudnine_bi.db"
    )

    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)

    app.register_blueprint(main_bp)
                                                        #developed by HaNiN
    with app.app_context():
        db.create_all()

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)