from flask import Flask
from flask_restx import Api
from flask_cors import CORS
from api.routes import setup_routes
import urllib3
import logging

urllib3.disable_warnings()
logging.basicConfig(level=logging.DEBUG)

def create_app():
    app = Flask(__name__)
    
    CORS(app, resources={
        r"/*": {
            "origins": "*",
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "X-Fields"]
        }
    })
    
    api = Api(app, 
        version='1.0', 
        title='Confluence RAG API',
        description='使用 Gemini 的 Confluence RAG 系統 API',
        doc='/swagger',
        validate=True  # 添加驗證
    )
    
    setup_routes(api)
    return app

if __name__ == '__main__':
    app = create_app()
    app.run()