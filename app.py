"""Flask API and identical browser UI. Run with waitress-serve --port=8080 app:app."""
from flask import Flask, jsonify, request, send_from_directory
from serving import load_artifact, recommend
from pathlib import Path

def create_app(artifact=None):
    web=Path(__file__).parent/'web'
    app=Flask(__name__,static_folder=str(web),static_url_path='/assets')
    app.config['MAX_CONTENT_LENGTH']=16384
    model=artifact or load_artifact()
    known={p['id'] for p in model['products']}

    @app.get('/')
    def index():return send_from_directory(web,'index.html')

    @app.get('/health')
    def health():return jsonify(status='ok',products=len(model['products']),schema_version=1)

    @app.get('/api/products')
    def products():return jsonify(products=model['products'])

    @app.get('/api/evaluation')
    def evaluation():return jsonify(model['evaluation'])

    @app.post('/api/recommend')
    def recommendations():
        if not request.is_json:return jsonify(error='Send application/json'),415
        body=request.get_json(silent=True)
        if not isinstance(body,dict):return jsonify(error='Expected a JSON object'),400
        for key in ['liked_ids','excluded_ids']:
            ids=body.get(key,[])
            if not isinstance(ids,list) or len(ids)>50 or any(not isinstance(i,str) or i not in known for i in ids):
                return jsonify(error=f'{key} must contain at most 50 known product IDs'),400
        limit=body.get('limit',6)
        if isinstance(limit,bool) or not isinstance(limit,int) or not 1<=limit<=20:return jsonify(error='limit must be an integer from 1 to 20'),400
        for key in ['category','query']:
            if not isinstance(body.get(key,''),str) or len(body.get(key,''))>200:return jsonify(error=f'{key} must be a string of at most 200 characters'),400
        return jsonify(recommendations=recommend(model,body.get('liked_ids',[]),body.get('excluded_ids',[]),limit,body.get('category',''),body.get('query','')),
                       strategy=model['config'])

    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='same-origin'
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'self'"
        return response
    return app

app=create_app()
