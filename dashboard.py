import time
import streamlit as st
import requests
from PIL import Image

from src.constants import INFERENCE_EXAMPLE, CM_PLOT_PATH,DATASET_PATH
from src.training.train_pipeline import TrainingPipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
dataset= pd.read_csv(DATASET_PATH)

st.title("Card Fraud Detection Dashboard")
st.sidebar.title("Data Themes")

sidebar_options = st.sidebar.selectbox(
    "Options",
    ("EDA", "Training", "Inference")
)
def CountPlot():
    fig = plt.figure(figsize=(10, 4))
    sns.countplot(dataset.Class)
    st.pyplot(fig)
def CorrPlot():
    fig=plt.figure(figsize=(8,8))
    heatmap = sns.heatmap(dataset.corr()[['Class']].sort_values(by='Class', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
    heatmap.set_title('Features Correlating with Class', fontdict={'fontsize':18}, pad=16);
    st.pyplot(fig)

def DistribPlot():
    fig = plt.figure(figsize=(25,5))
    box_cols = dataset.columns
    for i,c in enumerate(box_cols):
        plt.subplot(1, len(box_cols), i+1);
        sns.distplot(x=dataset[c]);
        plt.title(c)
    st.pyplot(fig)




if sidebar_options == "EDA":
    '''st.header("Exploratory Data Analysis")

    st.info("In this section, you are invited to create insightful graphs "
            "about the card fraud dataset that you were provided.")
    st.dataframe(dataset.head(10))
    st.write(dataset.describe())
    st.info("The dataset shape : ")
    st.write(dataset.shape)
    nb_col= dataset.columns.size
    nb_rows= len(dataset)
    st.write('It contains %d columns and %d rows'%(nb_col,nb_rows))
    st.write(dataset['Class'].value_counts())
    nb_Frauds = (dataset['Class']==0).sum()
    nb_NotFrauds = (dataset['Class']==1).sum()
    st.write('Number of clean Transactions :%d ------- Number of frauds : %d' % (nb_NotFrauds, nb_Frauds))
    #CountPlot() not showing the small amount of Class values equal to 1
    st.write('missing values')
    st.write(dataset.isna().sum())

    #Correlation Part
    st.header("Correlation")
    corr=dataset.corrwith(dataset['Class']).dropna()
    st.dataframe(corr)
    CorrPlot()

    #Histogramms to display distibutions
    DistribPlot()'''
elif sidebar_options == "Training":
    st.header("Model Training")
    st.info("Before you proceed to training your model. Make sure you "
            "have checked your training pipeline code and that it is set properly.")

    name = st.text_input('Model name', placeholder='decisiontree')
    print("ll", name) # delete
    serialize = st.checkbox('Save model')
    train = st.button('Train Model')

    option = st.selectbox('How would you like to be contacted?',
               ('Email', 'Home phone', 'Mobile phone'))

    st.write('You selected:', option)


    if train:
        with st.spinner('Training model, please wait...'):
            #time.sleep(1) # delete
            try:
                tp = TrainingPipeline()
                tp.train(serialize=serialize, model_name=name)
                tp.render_confusion_matrix(plot_name=name)
                accuracy, f1 = tp.get_model_perfomance()
                col1, col2 = st.columns(2)

                col1.metric(label="Accuracy score", value=str(round(accuracy, 4)))
                col2.metric(label="F1 score", value=str(round(f1, 4)))

                st.image(Image.open(CM_PLOT_PATH))

            except Exception as e:
                st.error('Failed to train model!')
                st.exception(e)


else:
    st.header("Fraud Inference")
    st.info("This section simplifies the inference process. "
            "You can tweak the values of feature 1, 2, 19, "
            "and the transaction amount and observe how your model reacts to these changes.")
    feature_11 = st.slider('Transaction Feature 11', -10.0, 10.0, step=0.001, value=-4.075)
    feature_13 = st.slider('Transaction Feature 13', -10.0, 10.0, step=0.001, value=0.963)
    feature_15 = st.slider('Transaction Feature 15', -10.0, 10.0, step=0.001, value=2.630)
    amount = st.number_input('Transaction Amount', value=1000, min_value=0, max_value=int(1e10), step=100)
    infer = st.button('Run Fraud Inference')

    INFERENCE_EXAMPLE[11] = feature_11
    INFERENCE_EXAMPLE[13] = feature_13
    INFERENCE_EXAMPLE[15] = feature_15
    INFERENCE_EXAMPLE[28] = amount

    if infer:
        with st.spinner('Running inference...'):
            time.sleep(1)
            try:
                result = requests.post(
                    'http://localhost:3333/api/inference',
                    json=INFERENCE_EXAMPLE
                )
                if int(int(result.text) == 1):
                    st.success('Done!')
                    st.metric(label="Status", value="Transaction: Fraudulent")
                else:
                    st.success('Done!')
                    st.metric(label="Status", value="Transaction: Clear")
            except Exception as e:
                st.error('Failed to call Inference API!')
                st.exception(e)
