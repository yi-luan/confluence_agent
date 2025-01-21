import logging
from flask_restx import Api, Resource
from api.controllers import AskQuestionController
from models.schemas import create_api_models

logger = logging.getLogger(__name__)

def setup_routes(api: Api):
    ns = api.namespace('api', description='RAG 操作')
    
    input_model, source_model, output_model = create_api_models(api)
    
    @ns.route('/ask')
    class AskQuestion(Resource):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.controller = AskQuestionController()
        
        @ns.expect(input_model, validate=True)
        @ns.response(200, 'Success', output_model)
        @ns.response(400, 'Validation Error')
        @ns.response(500, 'Internal Server Error')
        def post(self):
            """向 RAG 系統提問"""
            logger.debug("Handling /ask POST request")
            try:
                return self.controller.post()
            except Exception as e:
                logger.error(f"Error in post: {str(e)}", exc_info=True)
                ns.abort(500, str(e))