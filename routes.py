import pandas as pd
import threading
import uuid

from flask import request, jsonify
from model import train_model, test_model, predict_model
from textutils import find_best_match, compare_contacts

compare_contacts_tasks = {}

def setup_routes(app):

    @app.route('/ping', methods=['GET','POST'])
    def ping():
        return jsonify({
            "Ready": "POST train, test, predict, stringmatch or comparelists. https://github.com/ilya2184/TextClassify"
        })

    @app.route('/train', methods=['POST'])
    def train():
        data = pd.read_excel(request.files['file'])
        model_id = request.form['model_id']
        train_model(data, model_id)
        accuracy = test_model(data, model_id)
        return jsonify({"message": accuracy})

    @app.route('/test', methods=['POST'])
    def test():
        data = pd.read_excel(request.files['file'])
        model_id = request.form['model_id']
        accuracy = test_model(data, model_id)
        return jsonify({"accuracy": accuracy})

    @app.route('/predict', methods=['POST'])
    def predict():
        model_id = request.form['model_id']
        operation = request.form['operation']
        text = request.form['text']
        data = pd.DataFrame({ 'ВидОперации': [operation], 'НазначениеПлатежа': [text] })
        predata = predict_model(data, model_id)
        return jsonify({
            "article": predata.get("prediction"),
            "confidence": predata.get("confidence")
        })
    
    @app.route('/stringmatch', methods=['POST'])
    def stringmatch():
        data = request.get_json()
        strings_list = data.get('string_list', [])
        text = data.get('text', '')
        length_penalty_factor = data.get('length_penalty_factor', -0.01)
        prediction = find_best_match(strings_list, text, length_penalty_factor)
        return jsonify({
            "prediction": prediction
        })
    
    def run_comparison(task_id, contacts, checklist, threshold_high, threshold_low):
        # Запускаем длительную задачу
        compareresult = compare_contacts(contacts, checklist, threshold_high, threshold_low)
        compare_contacts_tasks[task_id]['status'] = 'completed'
        compare_contacts_tasks[task_id]['compareresult'] = compareresult

    @app.route('/comparelists/start', methods=['POST'])
    def comparelists():
        data = request.get_json()
        contacts = data.get('contacts', [])
        checklist = data.get('checklist', [])
        threshold_high = data.get('threshold_high', 97)
        threshold_low = data.get('threshold_low', 91)

        # Генерируем уникальный идентификатор задания
        task_id = str(uuid.uuid4())
        
        # Сохраняем состояние задания
        compare_contacts_tasks[task_id] = {'status': 'in progress', 'compareresult': None}
        
        # Запускаем задачу в отдельном потоке
        threading.Thread(target=run_comparison, args=(task_id, contacts, checklist, threshold_high, threshold_low)).start()

        # Возвращаем идентификатор задания
        return jsonify({"task_id": task_id}), 202

    @app.route('/comparelists/status', methods=['GET'])
    def check_status():
        task_id = request.args.get('taskid')
        if task_id not in compare_contacts_tasks:
            return jsonify({"status": "error"}), 404

        task_info = compare_contacts_tasks[task_id]

        if task_info['status'] == 'completed':
            # Сохраняем результат для ответа
            result = {
                "task_id": task_id,
                "status": task_info['status'],
                "compareresult": task_info['compareresult']
            }
            # Удаляем задачу из словаря
            del compare_contacts_tasks[task_id]
            return jsonify(result)
        else:
            return jsonify({
                "task_id": task_id,
                "status": task_info['status'],
                "compareresult": None
            })