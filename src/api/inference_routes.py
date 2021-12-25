from flask import Blueprint, request,jsonify
from src.constants import AGGREGATOR_MODEL_PATH,SCALER_PATH
from src.models.aggregator_model import AggregatorModel
import numpy as np
from src.api.database import db
from joblib import load


model = AggregatorModel()
model.load(AGGREGATOR_MODEL_PATH)
scaler = load(SCALER_PATH)
blueprint = Blueprint('api', __name__, url_prefix='/api')

class Transaction(db.Model):

    id = db.Column('transaction_id', db.Integer, primary_key = True)
    v0 = db.Column(db.String(100))
    v1 = db.Column(db.String(100))
    v2 = db.Column(db.String(100))
    v3 = db.Column(db.String(100))
    prediction = db.Column(db.String(100))

    def serialize(self):
        return {
            'id'         : self.id,
            'v0': self.v0,
            'v1'  : self.v1,
            'v2'  : self.v2,
            'Amount'  : self.v3,
            'prediction'  : self.prediction

        }

    def __init__(self, features, prediction):
        self.v0 = features[11]
        self.v1 = features[15]
        self.v2 = features[13]
        self.v3 = features[28]
        self.prediction=prediction


    def __repr__(self):
        return f'<Transaction %r>'%(self.id)

@blueprint.route('/')
@blueprint.route('/inference', methods=['GET', 'POST'])
def run_inference():
    if request.method == 'POST':
        features = np.array(request.json).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        target=prediction[0]
        features = [str(x) for x in features[0]]
        trans = Transaction(features=features, prediction=str(target))
        db.session.add(trans)
        db.session.commit()
        return str(prediction[0])
    elif request.method == 'GET':
        transactions = Transaction.query.all()
        result=[d.serialize() for d in transactions]
        return jsonify(result)

