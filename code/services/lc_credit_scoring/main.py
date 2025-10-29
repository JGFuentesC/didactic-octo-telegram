import pickle
import pandas as pd
import functions_framework
from flask import Request, jsonify  # Only for request parsing and response formatting, not running Flask

# Load the model at startup
acs = pickle.load(open('auto_scoring_lc.pkl', 'rb'))

@functions_framework.http
def score_endpoint(request: Request):
    try:
        req_json = request.get_json()
        if not req_json:
            return jsonify({'error': 'No JSON payload provided'}), 400

        # Expecting the request to be a plain dict with the two required variables for the score, not a list
        data = req_json  # Use payload directly as data
        required_vars = ['c_il_util','d_grade']  # Replace with actual required keys
        missing_vars = [var for var in required_vars if var not in data]
        if missing_vars:
            return jsonify({'error': f'Missing required variables in payload: {missing_vars}'}), 400

        df = pd.DataFrame([data])
        score = acs.predict(df)
        score['range_score_5'] = score['range_score_5'].astype(str)
        score['range_score_10'] = score['range_score_10'].astype(str)

        results = score.to_dict(orient='records')
        return jsonify(results[0])

    except Exception as e:
        return jsonify({'error': str(e)}), 500