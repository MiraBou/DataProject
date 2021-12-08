import numpy as np
import itertools
import matplotlib.pyplot as plt


class PlotUtils:
    @staticmethod
    def plot_confusion_matrix(cm, classes, title, normalize=False, cmap=plt.cm.Blues):
        title = 'Confusion Matrix of {}'.format(title)

        if normalize:
            cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                horizontalalignment='center',
                color='white' if cm[i, j] > thresh else 'black'
            )

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

def get_dtypes(dataset):
        '''return column data types of a dataframe '''
        cols = dict({})
        data_types = [ 'int64', 'float64', 'bool', 'object' ]
        for dtype in data_types:
            filter = dataset.select_dtypes(include=dtype).columns.values
            #st.write(filter)
            if len(filter)>0:
                cols.update({dtype: filter})

        num_cols = []
        cat_cols = []

        for key, val in cols.items():
            if key == 'float64':
                num_cols.extend(val)
            elif key == 'int64':
                for cat in val:
                    unique = len(dataset[cat].unique())/len(dataset[cat])
                    if unique > 0.1 or len(dataset[cat].unique())>100:
                        num_cols.append(cat)
                    else:
                        cat_cols.append(cat)
            if key == 'object':
                cat_cols.extend(val)
        return cols, num_cols, cat_cols
