import os 
import matplotlib.pylab as plt
import pandas as pd

import datetime, warnings, scipy 
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics, linear_model
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential,Model
from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Activation, Reshape
from keras.layers import Concatenate, Dropout
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow.keras.optimizers
from tensorflow.keras import layers

class EntityEmbedding():
    '''
    :features: features from dataframe
    :target: y - your target variable
    :column: column for which you want to get embedded vector
    :size: dimension of your vector. When set False, it is calculated by int(min(np.ceil((no_of_unique)/2), 50 ))
    
    '''
    def __init__(self, df, features, target, column):
        self.df = df
        self.features = features
        self.target = target
        self.column = column
        self.input_list = self.label_encoder()[0]
        self.num_cols = self.label_encoder()[1]
        self.val_map = self.label_encoder()[2]
        self.temp_val_map = self.label_encoder()[3]
        
    def cat_col(self):
        cats = [cat for cat in self.features.select_dtypes(include=['object'])]
        nunique = [unique for unique in self.features.select_dtypes(include=['object']).nunique()]
        return cats, nunique
  
    def label_encoder(self):
        input_list = []
        val_map = {}
        for cat in self.cat_col()[0]:
            values = np.unique(self.features[cat]) # ზოგიერთ ობჯექთ სვეტში ფლოუთები იყო შერეული, რის გამოც სტრინგად გარდავქმენით
            temp_val_map = {} #აქ ყველა იუნიქ ველიუს შეუსაბამებს ციფრს, რომელიც გადაეწერება საწყის დეითაფრეიმს
            for k in range(len(values)):
                temp_val_map[values[k]] = k
            input_list.append(self.features[cat].map(temp_val_map).fillna(0).values)
            val_map[cat] = temp_val_map
           
        num_cols = [num for num in self.features.columns if not num in self.cat_col()[0]]
        input_list.append(self.features[num_cols].values)
        
        return input_list, num_cols, val_map, temp_val_map
  

    def train_fit(self, activation1='relu', activation2='relu', activation3='relu', loss='mean_squared_error', metrics='mape', dense_size_num=128, dense_size_conc_1=300, dense_size_conc_2=300, alpha=1e-3, epochs=1000, batch_size=512, verbose=1, patience=5):
        input_models = []
        out_models = []
        for cat in self.cat_col()[0]:
            print(cat)
            no_of_unique  = self.features[cat].nunique()
            embed_size = int(min(np.ceil((no_of_unique)/2), 50 ))
            cat_emb_name= cat.replace(" ", "")+'_Embedding'

            input_model = Input(shape=(1,))
            output_model = Embedding(input_dim=no_of_unique+1, output_dim=embed_size, input_length=1, embeddings_initializer='uniform', name=cat_emb_name)(input_model)
            output_model = layers.SpatialDropout1D(0.3)(output_model)
            output_model = Reshape(target_shape=(embed_size,))(output_model)
            input_models.append(input_model)
            out_models.append(output_model)

        input_num = Input(shape=(len(self.features.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns.tolist()),))
        emb_num = Dense(dense_size_num)(input_num)
        input_models.append(input_num)
        out_models.append(emb_num)
        
        output = Concatenate()(out_models)
        output = layers.BatchNormalization()(output)
        output = Dense(dense_size_conc_1, kernel_initializer="glorot_uniform")(output)
        output = Activation(activation1)(output)
        
        output= Dropout(0.4)(output)
        output = layers.BatchNormalization()(output)
        output = Dense(dense_size_conc_2, kernel_initializer="glorot_uniform")(output)
        output = Activation(activation2)(output)
        
        output= Dropout(0.3)(output)
        output = layers.BatchNormalization()(output)
        output = Dense(2, activation=activation3)(output)

        model = Model(input_models, output)
        callback = tensorflow.keras.callbacks.EarlyStopping(monitor='loss', patience=patience)
        model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=alpha), loss=loss, metrics=metrics)
        
        
        model.fit(self.input_list, self.target,epochs = epochs, batch_size = batch_size, callbacks=[callback], verbose=verbose)
        self.ent_emb = model.get_layer(self.column+'_Embedding').get_weights()[0]
        
    
    def transform(self):
        d = {}
        new_d = {}
        for i in range(len(np.unique(self.input_list[self.cat_col()[0].index(self.column)]))):
            d[np.unique(self.input_list[self.cat_col()[0].index(self.column)])[i]] = self.ent_emb[i]
        for cat in self.cat_col()[0]:
            if cat == self.column:
                for k, v in self.val_map[cat].items():
                    new_d[k] = self.ent_emb[v]

        self.features.loc[:,self.column] = self.features.loc[:,self.column].map(new_d).fillna(0)
        return self.features.loc[:,[self.column]]
                
    def compute_pca(self, X, n_components=2):
        """
        Input:
            X: of dimension (m,n) where each row corresponds to a word vector
            n_components: Number of components you want to keep.
        Output:
            X_reduced: data transformed in 2 dims/columns + regenerated original data
        """

        # mean center the data
        X_demeaned = X - np.mean(X,axis=0)

        # calculate the covariance matrix
        covariance_matrix = np.cov(X_demeaned, rowvar=False)

        # calculate eigenvectors & eigenvalues of the covariance matrix
        eigen_vals, eigen_vecs = np.linalg.eigh(covariance_matrix, UPLO='L')

        # sort eigenvalue in increasing order (get the indices from the sort)
        idx_sorted = np.argsort(eigen_vals)
    
        # reverse the order so that it's from highest to lowest.
        idx_sorted_decreasing = idx_sorted[::-1]

        # sort the eigen values by idx_sorted_decreasing
        eigen_vals_sorted = eigen_vals[idx_sorted_decreasing]

        # sort eigenvectors using the idx_sorted_decreasing indices
        eigen_vecs_sorted = eigen_vecs[:,idx_sorted_decreasing]

        # select the first n eigenvectors (n is desired dimension
        # of rescaled data array, or dims_rescaled_data)
        eigen_vecs_subset = eigen_vecs_sorted[:,0:n_components]

        # transform the data by multiplying the transpose of the eigenvectors 
        # with the transpose of the de-meaned data
        # Then take the transpose of that product.
        X_reduced = np.dot(eigen_vecs_subset.transpose(),X_demeaned.transpose()).transpose()

        return X_reduced
        
    def visualize(self):
        # We have done the plotting for you. Just run this cell.
        words = self.df[self.[column]].unique()
        result = self.compute_pca(self.ent_emb)
        plt.scatter(result[:, 0], result[:, 1])
        for i, word in enumerate(words):
            plt.annotate(word, xy=(result[i, 0] - 0.05, result[i, 1] + 0.1))

        plt.show()
        
