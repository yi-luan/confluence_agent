import logging
from flask_restx import Api, Resource, fields
from api.controllers import AskQuestionController, ConversationController
from models.schemas import create_api_models

logger = logging.getLogger(__name__)

def setup_routes(api: Api):
    ns = api.namespace('api', description='RAG 操作')
    
    input_model = api.model('Question', {
        'question': fields.String(required=True, description='用戶的問題'),
        'conversation_id': fields.String(required=False, description='對話 ID'),
        'workspace': fields.String(required=False, description='Confluence workspace 名稱')
    })

    source_model = api.model('Source', {
        'content': fields.String(description='文檔內容片段'),
        'summary': fields.String(description='文檔摘要'),
        'source': fields.String(description='文檔連結'),
        'title': fields.String(description='文檔標題')
    })

    output_model = api.model('Response', {
        'status': fields.String(description='回應狀態'),
        'data': fields.Nested(api.model('Data', {
            'question': fields.String(description='原始問題'),
            'answer': fields.String(description='AI 的回答'),
            'sources': fields.List(fields.Nested(source_model)),
            'conversation_id': fields.String(description='對話 ID')
        }))
    })

    conversation_list_model = api.model('ConversationList', {
        'status': fields.String(description='回應狀態'),
        'data': fields.Nested(api.model('ConversationListData', {
            'conversations': fields.List(fields.Nested(api.model('ConversationInfo', {
                'id': fields.String(description='對話 ID'),
                'title': fields.String(description='對話標題'),
                'last_updated': fields.String(description='最後更新時間'),
                'message_count': fields.Integer(description='消息數量')
            })))
        }))
    })
    
    conversation_detail_model = api.model('ConversationDetail', {
        'status': fields.String(description='回應狀態'),
        'data': fields.Nested(api.model('ConversationDetailData', {
            'conversation_id': fields.String(description='對話 ID'),
            'messages': fields.List(fields.Nested(api.model('Message', {
                'role': fields.String(description='角色'),
                'content': fields.String(description='內容'),
                'timestamp': fields.String(description='時間戳'),
                'sources': fields.List(fields.Nested(source_model))
            })))
        }))
    })

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

    @ns.route('/conversations')
    class ConversationList(Resource):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.controller = ConversationController()
        
        @ns.response(200, 'Success', conversation_list_model)
        @ns.response(500, 'Internal Server Error')
        def get(self):
            """獲取所有對話列表"""
            try:
                return self.controller.get_all()
            except Exception as e:
                logger.error(f"Error getting conversations: {str(e)}", exc_info=True)
                ns.abort(500, str(e))
    
    @ns.route('/conversations/<string:conversation_id>')
    class ConversationDetail(Resource):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.controller = ConversationController()
        
        @ns.response(200, 'Success', conversation_detail_model)
        @ns.response(404, 'Conversation Not Found')
        @ns.response(500, 'Internal Server Error')
        def get(self, conversation_id):
            """獲取特定對話的詳細信息"""
            try:
                return self.controller.get(conversation_id)
            except Exception as e:
                logger.error(f"Error getting conversation: {str(e)}", exc_info=True)
                ns.abort(500, str(e))

