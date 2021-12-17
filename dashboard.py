import time
import streamlit as st
import requests
from PIL import Image
from src.constants import INFERENCE_EXAMPLE,DATASET_PATH, CM_PLOT_PATH,SCALER_PATH
from src.training.train_pipeline import TrainingPipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.utils import get_dtypes
from joblib import dump

def CountPlot(dataset):
    fig = plt.figure(figsize=(10, 4))
    sns.countplot(dataset.Class)
    st.pyplot(fig)

def CorrPlot(dataset):
    fig=plt.figure(figsize=(8,8))
    heatmap = sns.heatmap(dataset.corr()[['Class']].sort_values(by='Class', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
    heatmap.set_title('Features Correlating with Class', fontdict={'fontsize':18}, pad=16);
    st.pyplot(fig)

def DistribPlot(dataset):
    fig= plt.figure(figsize=(30, 15))
    if("Class" in dataset.columns):
        cols = dataset.drop("Class", axis=1).columns
    else :
        cols = dataset.columns
    for i,c in enumerate(cols):
        plt.subplot(5, 6, i+1)
        plt.hist(dataset[c])
        plt.title(c)
    st.pyplot(fig)

def SplitDataset(dataset):
    y = dataset['Class']
    features = dataset.drop(columns=['Class'])
    X_train,X_test,y_train,y_test = train_test_split(features,y,
                                                     test_size=0.2,
                                                     random_state=0
                                                     )
    return(X_train,X_test,y_train,y_test)

def Scaling(X_train,X_test):
    scaler = StandardScaler()
    cols = dataset.drop("Class", axis=1).columns
    data_scaled = scaler.fit_transform(X_train)
    data_test_scaled = scaler.transform(X_test)
    X_train_scaled=pd.DataFrame(data=data_scaled , columns=cols)
    X_test_scaled=pd.DataFrame(data=data_test_scaled , columns=cols)
    dump(scaler,SCALER_PATH)
    return(X_train_scaled,X_test_scaled)

dataset= pd.read_csv(DATASET_PATH)

st.title("Card Fraud Detection Dashboard")
st.sidebar.title("Data Themes")

sidebar_options = st.sidebar.selectbox(
    "Options",
    ("EDA", "Training", "Inference")
)


if sidebar_options == "EDA":
    st.header("Exploratory Data Analysis")
    st.write("wiwi")
    st.markdown(''' > ## Data Types of Columns''')
    col_types, num_cols, cat_cols = get_dtypes(dataset)
    st.write(col_types)

    st.dataframe(dataset.head(10))
    st.write(dataset.describe())
    st.info("The dataset shape : ")
    st.write(dataset.shape)
    nb_col = dataset.shape[1]
    nb_rows = dataset.shape[0]
    st.write('It contains %d columns and %d rows'%(nb_col,nb_rows))
    st.markdown(''' > ## Values in the target Class''')
    st.write(dataset['Class'].value_counts())
    nb_Frauds = (dataset['Class']==0).sum()
    nb_NotFrauds = (dataset['Class']==1).sum()
    st.write('Number of clean Transactions :%d ------- Number of frauds : %d' % ( nb_Frauds,nb_NotFrauds))


    #CountPlot() not showing the small amount of Class values equal to 1

    st.markdown(''' > ## Missing Data''')
    st.write(dataset.isna().sum())

    #Correlation Part
    st.header("Correlation")
    corr=dataset.corrwith(dataset['Class']).dropna()
    st.dataframe(corr)
    CorrPlot(dataset)

    #Histogramms to display distibutions
    DistribPlot(dataset)
    st.header("Data Cleaning ")
    st.markdown(''' > ## Splitting''')
    X_train,X_test,y_train,y_test=SplitDataset(dataset)
    st.write('It contains %d of train samples and %d test samples'%(X_train.shape[0],X_test.shape[0]))

    st.markdown(''' > ## Scaling''')
    X_train_scaled,X_test_scaled=Scaling(X_train,X_test)
    st.write(X_train_scaled.head(5))
    st.write(X_test_scaled.head(5))
    st.write(X_test_scaled.describe())
    st.write("Plotting the new normalized distributions of features")
    DistribPlot(X_train_scaled)
    #dataset.to_csv(PREPARED_DATASET_PATH)

elif sidebar_options == "Training":
    st.header("Model Training")
    st.info("Before you proceed to training your model. Make sure you "
            "have checked your training pipeline code and that it is set properly.")


    train = False
    selected_options = st.multiselect('Choose Your model(s)?',
                             ['decisiontree', 'svc', 'randomforest'])


    st.write('You selected:')


    for m in selected_options:
        st.markdown(f"- {m}")
    if len(selected_options) != 0:
            train = st.button('Train Model')
            serialize = st.checkbox('Save model')
            options =','.join(selected_options)


    if train:
        with st.spinner('Training model, please wait...'):
            time.sleep(1)
            try:
                tp = TrainingPipeline(X_train_scaled,X_test_scaled,y_train,y_test)
                tp.train(serialize=serialize, model_name=options)
                #tp.render_confusion_matrix(plot_name=options)
                #maccuracy, f1 = tp.get_model_perfomance()
                #col1, col2 = st.columns(2)

                #col1.metric(label="Accuracy score", value=str(round(accuracy, 4)))
                #col2.metric(label="F1 score", value=str(round(f1, 4)))
                #cm_plot_path = str(CM_PLOT_PATH).replace('cm_plot.png', options+'.png')
                #st.image(Image.open(CM_PLOT_PATH))

            except Exception as e:
                st.error('Failed to train model!')
                st.exception(e)


else:
    st.header("Fraud Inference")
    st.info("This section simplifies the inference process. "
            "You can tweak the values of feature 1, 2, 19, "
            "and the transaction amount and observe how your model reacts to these changes.")
    st.markdown(''' > ## History''')

    trans = requests.get(
        "http://localhost:5000/api/inference"
    ).json()
    st.table(trans)

    st.markdown(''' > ## New Fraud Inference''')


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
                    'http://localhost:5000/api/inference',
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
                st.write(e)
                st.exception(e)

