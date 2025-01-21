from flask_restx import fields, Api

def create_api_models(api: Api):
    # 簡化輸入模型
    input_model = api.model('Question', {
        'question': fields.String(required=True, description='用戶的問題'),
        'workspace': fields.String(required=False, description='Confluence workspace 名稱')
    })

    # 保留來源模型
    source_model = api.model('Source', {
        'content': fields.String(description='文檔內容片段'),
        'summary': fields.String(description='文檔摘要'),
        'source': fields.String(description='文檔來源')
    })

    # 簡化輸出模型
    output_model = api.model('Response', {
        'status': fields.String(description='回應狀態'),
        'data': fields.Nested(api.model('Data', {
            'question': fields.String(description='原始問題'),
            'answer': fields.String(description='AI 的回答'),
            'sources': fields.List(fields.Nested(source_model))
        }))
    })

    return input_model, source_model, output_model