import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split


from src.constants import AGGREGATOR_MODEL_PATH, DATASET_PATH, CM_PLOT_PATH
from src.models.aggregator_model import AggregatorModel
from src.models.decision_tree_model import DecisionTreeModel
from src.models.random_forest_model import RandomForestModel
from src.models.svc_model import SVCModel
from src.utils import PlotUtils
import numpy as np

class TrainingPipeline:
    def __init__(self,X_train_scaled: np.ndarray,X_test_scaled: np.ndarray,y_train: np.ndarray,y_test: np.ndarray):

        self.x_train = X_train_scaled
        self.x_test = X_test_scaled
        self.y_train =y_train
        self.y_test = y_test

        self.model = None


    def train(self, serialize: bool = True, model_name: str = 'model.joblib'):
        models_dict = {
            "decisiontree":DecisionTreeModel(),
            "svc":SVCModel(),
            "randomforest":RandomForestModel()
        }

        models = model_name.split(',')
        options=[]
        for i in models:
            options.append(models_dict[i])


        self.model = AggregatorModel(models=options)

        self.model.fit(
            self.x_train,
            self.y_train
        )

        model_path = str(AGGREGATOR_MODEL_PATH)

        if serialize:
            AggregatorModel.save(
                self.model,
                model_path
            )

    def get_model_perfomance(self) -> tuple:
        predictions = self.model.predict(self.x_test)
        return accuracy_score(self.y_test, predictions), f1_score(self.y_test, predictions)

    def render_confusion_matrix(self, plot_name: str = 'plot.png'):
        predictions = self.model.predict(self.x_test)
        cm = confusion_matrix(self.y_test, predictions, labels=[0, 1])
        plt.rcParams['figure.figsize'] = (6, 6)

        PlotUtils.plot_confusion_matrix(
            cm,
            classes=['Clear(0)', 'Fraudulent(1)'],
            normalize=False,
            title=plot_name
        )

        plot_path = str(CM_PLOT_PATH).replace('cm_plot.png', plot_name + '.png')
        print(plot_name)
        plt.savefig(plot_path)
        plt.show()

