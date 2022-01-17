# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 15:41:59 2021

@author: javier.colas
"""
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from tabulate import tabulate
import seaborn as sns




def define_index(df):
    df.index = df.id
    df.drop('id', axis = 1, inplace=True)
    return df

def negative_target(df):
    for i in range(len(df)):
        if pd.isnull(df.Target.iloc[i]) and df.impactos_publicidad.iloc[i] > 0:
            df.Target.iloc[i] = 0
        elif df.Target.iloc[i] == 1:
            pass
        else:
            df.Target.iloc[i] = np.nan
        
    df = define_index(df)
    return df

def target_creation(df_clientes_E, df):
    '''Función para unir ambas bases de datos y crear la variable target (1: Es cliente de la empresa E, 0 si la empresa i
       ha recibido impactos publicitarios y no es cliente, y 'NaN' si no ha recibido impactos y no es cliente)'''
    
    df_clientes_E['Target'] = 1
    df = df.merge(df_clientes_E, on='id', how='left')
    df = negative_target(df)
    return df

def to_category(df, columnas_sin_cambios):
    ''' Modifica el tipo de datos de todas las columnas del dataframe a excepción de la lista columnas_sin_cambios'''
    for columna in df.columns:
        if columna not in columnas_sin_cambios:
            df[columna] = df[columna].astype('category')
            
            
def numero_categorias(df):
    ''' Número de categorías por campo'''
    predictores_cat = list(df.select_dtypes(['category']).columns)
    for columna in predictores_cat:
        print(columna, len(set(df[columna])))
        
def numero_nulos(df):  
    '''Número de valores nulos por campo'''      
    nulos = df.isnull()
    suma_nulos = nulos.sum()
    ordenados = suma_nulos.sort_values(ascending=False)
    print(ordenados)
    
def año_antiguedad_to_dias_antiguedad(año):
    try:
        return int((datetime.now()- datetime.strptime(año, '%d/%m/%Y')).days)
    except:
        pass
    
def columnas_no_binarias(df):
    no_binarias = []
    columnas_binarias = df.select_dtypes(['category']).dropna()
    for columna in columnas_binarias:
        if len(set(columnas_binarias[columna])) > 2:
            no_binarias.append(columna)         
    no_binarias.remove('tipo_de_zona')
    no_binarias.remove('rangoventas')
    no_binarias.remove('nse')
    return no_binarias
    
    
    
def columnas_binarias(df):
    binarias = []
    columnas_binarias = df.select_dtypes(['category']).dropna()
    for columna in columnas_binarias:
        if len(set(columnas_binarias[columna])) == 2:
            binarias.append(columna)
    return binarias

def columnas_numericas(df):
    columnas_numericas = list(df.select_dtypes(['number']).columns)      
    columnas_numericas.remove('Target')
    return columnas_numericas
    
    

def juntar_rangoventas(x):
    if x in [8, 9, 10]:
        return 8
    else:
        return x
    

def juntar_CCAA(x):
    if x in ['Ceuta','Canarias', 'Balears']:
        return 'otra'
    else:
        return x   
    
    
def juntar_nse(x):
    if x in [1, 2]:
        return 2
    else:
        return x  

def descriptivo_categoricas(df, variables_catplot):
    for columna in variables_catplot:
        try:
            print(f'Observaciones no nulas, columna {columna} : {sum(df[columna].notna())} \n')
            
            unique, counts = np.unique(df[columna], return_counts=True)
            df_frecuencias = pd.DataFrame({
                'Clase': [i for i,z in list(zip(unique, counts))],
                'Frecuencia': [z for i,z in list(zip(unique, counts))],
                'F.C Positiva': [str(sum((df[columna] == i) & (df.Target == 1))) for i,z in list(zip(unique, counts))],
                '% Target Positiva': [str(round((sum((df[columna] == i) & (df.Target == 1)) / (sum(df[columna] == i) + 1)) * 100, 2)) + ' %'
                                                                for i,z in list(zip(unique, counts))],
                'F.C Negativa': [str(sum((df[columna] == i) & (df.Target == 0))) for i,z in list(zip(unique, counts))],
                '% Target Negativo': [str(round((sum((df[columna] == i) & (df.Target == 0)) / (sum(df[columna] == i) + 1)) * 100, 2)) + ' %'
                                                                for i,z in list(zip(unique, counts))]
                                          }).sort_values(by ='Frecuencia', ascending=False)
            
            
            print(tabulate(df_frecuencias, headers='keys', tablefmt='pretty', showindex=False))
            sns.catplot(x=columna, kind="count", palette="ch:.25", data=df)
            plt.xticks(rotation=90)
            plt.show()
        except:
            pass


#------------------------------------------------------------------------------
#clustering

def perform_pca(df, n_components=None):
    """
    Returns transformed data with PCA
    """
    pca = PCA(n_components, random_state=42)
    pca.fit(df)
    df_transformed = pca.transform(df)
    return pca, df_transformed
    
    
    
def plot_pca_results(df_entero, pca_df,  cumulative=True, figsize=(10,12)):
    """
    Takes in two PCA models (which are fit on corresponding data) and plots 
    their Explained Variance vs Number of components
    
    """   
    
    if cumulative:
        df_entero_variance = np.cumsum(df_entero.explained_variance_ratio_)
        y_label = "Variance Explained (%)"
    else:
        df_entero_variance = pca_df.explained_variance_ratio_
        y_label = "Explained Variance Ratio"
        
    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(211)
    ax.plot(df_entero_variance, linewidth=2.5, color='dodgerblue')
    ax.set_xlabel("Principal Component", fontsize=14)
    plt.xticks(np.arange(0, 26, 1))
    plt.yticks(np.arange(0.50, 1.05, 0.05))
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_title("Varianza explicada para las 100.000 empresas", fontsize=16)
    ax.grid()
    plt.show()   
    
    
    
def plot_silhoutte(X):
    
    range_n_clusters = list(np.arange(2,15))

    for n_clusters in range_n_clusters:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)
        
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )
    

        sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    
            ith_cluster_silhouette_values.sort()
    
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
    
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
    
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    

            y_lower = y_upper + 10  
    
        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
    
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
        ax1.set_yticks([])  
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )
    
        centers = clusterer.cluster_centers_

        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )
    
        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")
    
        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")
    
        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )
    
    plt.show()
    


def find_distance(array, cluster):
    """
    Returens distances between cluster's centrod of customers and general population
    """
    distances=[]
    for i in range(len(array)):
        dist = np.linalg.norm(array[i] - cluster) 
        distances.append(dist)
    return distances



def resta_consecutivos(Numero_empresas):
    ganancia = []
    for i in range(len(Numero_empresas)-1):
        ganancia.append(Numero_empresas[i+1] - Numero_empresas[i])
    ganancia.insert(0,0)
    return ganancia


def plot_confusion_matrix(matriz): 
    group_names = ['True Neg','False Pos','False Neg','True Pos']

    group_counts = ["{0:0.0f}".format(value) for value in
                    matriz.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                         matriz.flatten()/np.sum(matriz)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names,group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(2,2)

    ax = sns.heatmap(matriz, annot=labels, fmt='', cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])

    ## Display the visualization of the Confusion Matrix.
    plt.show()
    
#-----------------------------------------------------------------------------
## CLASIFICACIÓN
