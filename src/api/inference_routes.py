from flask import Blueprint, request
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


    def __init__(self, features, prediction):
        self.v0 = features[11]
        self.v1 = features[15]
        self.v2 = features[28]
        self.v3 = features[13]
        self.prediction=prediction


    def __repr__(self):
        return f'"id":{self.id}, "V11":{self.v0}, "V15":{self.v1},"V13":{self.v3}, "Amount":{self.v2},  "prediction":{self.prediction}'

@blueprint.route('/')
@blueprint.route('/inference', methods=['GET', 'POST'])
def run_inference():
    if request.method == 'POST':
        features = np.array(request.json).reshape(1, -1)
        features_scaled = scaler.transform(features)
        #prediction = model.predict(features)
        prediction = model.predict(features)
        target=prediction[0]
        #features = [str(x) for x in features[0]]
        features_scaled = [str(x) for x in features_scaled[0]]
        trans = Transaction(features=features_scaled, prediction=str(target))
        db.session.add(trans)
        db.session.commit()
        return str(target)
    elif request.method == 'GET':
        transactions = Transaction.query.all()
        strc="["
        size=len(transactions)
        i=0
        for d in transactions:
            strc=strc+"{"+str(d)+"}"
            if i+1!=size:
                strc+=','
            i+=1

        strc+="]"
        return strc

