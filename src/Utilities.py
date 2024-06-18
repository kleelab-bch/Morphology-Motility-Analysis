import numpy as np
import random
import pandas as pd
import umap
from scipy.stats import gaussian_kde, zscore
from scipy import stats
from statsmodels.stats.weightstats import ztest
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from src.Figure_plots import *
import matplotlib.pyplot as plt

## Work Functions ======================================================================================================

def main(Data_file, Feature_type, **kwargs):
    """

    Process the whole machine learning analysis and plot the figures

    Input:
          Data_file : the path of data file
          Feature_type : morphology or motility feature will be worked
          **kwargs: Arbitrary keyword arguments.
                Cell_Type (list): If cell type will be changed, reenter the list of cell type name (str).
                                  Default is ['CTRLs', 'MDAEVs', 'BrEVs', 'Rab7', 'Rab11fip2', 'Rab11fip3', 'Rab11fip5'].
                cluster_parameters (list): These parameters include three variables: [n_neighbors, min_dist, max_cluster_number]
                                  The first two are used for UMAP generation and the last one is for clustering. It is recommended
                                  that the clustering number should be kept below 10. Default is [10, 0, 9].
                balanced_sign (boolean): Balance the sample numbers of different cell type for the analysis. Default is True.

    Output:
          All the figures will be saved in the Result_plot folder.

    """

    Cell_Type = kwargs.get('Cell_Type', ['CTRLs', 'MDAEVs', 'BrEVs', 'Rab7', 'Rab11fip2', 'Rab11fip3', 'Rab11fip5'])
    cluster_parameters = kwargs.get('cluster_parameters', [10, 0, 9])
    balanced_sign = kwargs.get('balanced_sign', True)

    df = pd.read_csv(Data_file)
    col_index = list(df.columns.values)
    All_data = df.iloc[:, 1:len(col_index)]
    All_data = np.array(All_data)
    Cell_Label = Convert_Type_Number(df.iloc[:, 0], Cell_Type)

    # Use UMAP with clustering to automatically identify the number of clusters ============================================
    embedding, Label_list, All_resampling_std = Cluster_generator(All_data, Cell_Type, Cell_Label, cluster_parameters)

    if (Feature_type == 'Morph') and balanced_sign:
        col_index.pop(-2)
        All_data = np.delete(All_data, [-2], axis=1)
    # Plot all figures =====================================================================================================

    # Color Setting
    if len(np.unique(Label_list)) > 9:
        new_color = plt.cm.get_cmap('jet', len(np.unique(Label_list)))
        color = [new_color(i) for i in np.linspace(0, 1, len(np.unique(Label_list)))]
    else:
        color = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
                 "#17becf"]

    # Motility Feature Distribution Map
    Feature_Distribution_generator(embedding, Label_list, color, Feature_type)

    # Cluster Bar Graph Plot in each Cell Type
    Cell_Bar_generator(Cell_Label, Label_list, All_resampling_std, color)

    # Cell Type Distribution Plot
    cell_distribution_plot(embedding, Cell_Label, Cell_Type)

    # Feature Matrix Plot
    Feature_list = []

    for j in range(len(col_index) - 1):

        z = All_data[:, j]
        z_max, z_min = get_boundary(z)
        z[z > z_max] = z_max
        z[z < z_min] = z_min
        f = (z - z.min()) / (z.max() - z.min())

        f_all = [np.mean(f[Label_list == label_i]) for label_i in np.unique(Label_list)]
        f_all = np.array(f_all)

        if len(Feature_list) == 0:
            Feature_list = f_all
        else:
            Feature_list = np.vstack((Feature_list, f_all))

    feature_matrix_generator(Feature_list, col_index, Feature_type, balanced_sign)


def Convert_Type_Number(CT_tmp, Cell_Type):
    """
    Convert the cell type label to label number

    Input:
        CT_tmp: cell type name list of the data
        Cell_Type: the name list of cell type
    Return:
        Cell_Label: the integer label of cell type

    """

    Cell_Label = []
    for c_type in CT_tmp:
        for ct_ind in range(len(Cell_Type)):
            if cmp(c_type, Cell_Type[ct_ind]):
                Cell_Label.append(ct_ind+1)

    return np.array(Cell_Label)

def cmp(a, b):
    """
    Compare two string without space

    Input:
        a: First input string (with space)
        b: Second input string
    Return:
        Boolean (True/False)

    """
    return a.replace(' ', '') == b

def Cluster_generator(All_data, Cell_Type, Cell_Label, cluster_parameters):
    """
    UMAP generation and clustering

    Input:
        All_data: the values of all features
        Cell_Type: the name list of cell types
        Cell_Label: the ground truth cell labels of all sample data
        cluster_parameters: the parameters of UMAP plot and clustering number setting

    Return:
        embedding: the coordinates of UMAP distribution
        Label_list: he labels of clusters of all sample data
        All_resampling_std: the std values of labels from randomly selected sample group

    """
    All_data_Scale_Info = zscore(All_data, axis=0)
    reducer = umap.UMAP(n_components=2, n_neighbors=cluster_parameters[0], min_dist=cluster_parameters[1],
                        metric='correlation')
    embedding = reducer.fit_transform(All_data_Scale_Info)

    potential_clusters = np.arange(3, cluster_parameters[2])
    Silhouette_score_cluster = []
    for cluster_num in range(potential_clusters[0], potential_clusters[-1] + 1):
        clustering = SpectralClustering(n_clusters=cluster_num).fit(embedding)

        Label_list = clustering.labels_
        Label_list = Label_list + 1

        Silhouette_score_cluster.append(silhouette_score(embedding, Label_list))

    Silhouette_score_cluster = np.array(Silhouette_score_cluster)
    max_index = np.where(Silhouette_score_cluster == np.max(Silhouette_score_cluster))
    max_index = int(max_index[0])

    clustering = SpectralClustering(n_clusters=potential_clusters[max_index]).fit(embedding)
    Label_list = clustering.labels_
    Label_list = Label_list + 1

    All_resampling_mean, All_resampling_std = sample_generator(Cell_Type, Cell_Label, Label_list)

    return embedding, Label_list, All_resampling_std

def get_boundary(data):
    """

    Calculate the boundary of the quantiles

    Input:
        data: the values of a feature
    Return:
        max_value, min_value : max and min suitable boundaries of feature values

    """

    q1, medians, q3 = np.percentile(data, [25, 50, 75])
    max_value = q3 + (q3 - q1) * 1.5
    min_value = q1 - (q3 - q1) * 1.5

    return max_value, min_value

def sample_generator(Cell_Type, Cell_Label, Label_list):
    """

    Calculate the mean and std values of randomly selected label group

    Input:
         Cell_Type: the name list of cell types
         Cell_Label: the ground truth cell labels of all sample data
         Label_list: the labels of clusters of all sample data
    Return:
         All_resampling_mean : the mean values of labels from randomly selected sample group
         All_resampling_std : the std values of labels from randomly selected sample group

    """
    sample_list = np.arange(len(Cell_Label))
    cluster_num = np.unique(Label_list)
    All_resampling_mean = []
    All_resampling_std = []
    All_results = []

    for cy_ind in range(1, len(Cell_Type)+1):
      tmp_cell_list = sample_list[Cell_Label == cy_ind]
      tmp_record = []
      tmp_record_ori = []
      for tmp_resample_ind in range(100):
          random.shuffle(tmp_cell_list)
          tmp_cell_list_new = tmp_cell_list[0: round(len(tmp_cell_list)/len(cluster_num))]

          tmp_Label_list = Label_list[tmp_cell_list_new]
          cr_bar = [len(tmp_Label_list[tmp_Label_list == c_i]) for c_i in cluster_num]
          cr_bar_copy = cr_bar
          cr_bar = np.array(cr_bar) / len(tmp_Label_list)

          if len(tmp_record) == 0:
              tmp_record = cr_bar
              tmp_record_ori = cr_bar_copy
          else:
              tmp_record = np.vstack((tmp_record, cr_bar))
              tmp_record_ori = np.vstack((tmp_record_ori, cr_bar_copy))

      All_results.append(tmp_record_ori)
      if len(All_resampling_mean) == 0:
          All_resampling_mean = np.mean(tmp_record, axis=0)
          All_resampling_std = np.std(tmp_record, axis=0)
      else:
          All_resampling_mean = np.vstack((All_resampling_mean, np.mean(tmp_record, axis=0)))
          All_resampling_std = np.vstack((All_resampling_std, np.std(tmp_record, axis=0)))

    return All_resampling_mean*100, All_resampling_std*100