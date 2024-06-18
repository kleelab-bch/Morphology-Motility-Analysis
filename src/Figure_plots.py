import numpy as np
import random
import umap
from scipy.stats import gaussian_kde
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MultipleLocator


## Plot the figures ====================================================================================================

def Feature_Distribution_generator(embedding, Label_list, color, feature_type):
    """

    Plot the feature distribution map

    Input:
        embedding: the coordinates of UMAP distribution
        Label_list: the labels of clusters of all sample data
        color: assign a color to each cluster
        feature_type: select to plot the context of title
    Output:
        save the figure in the Result_plot folder

    """

    groups = ['Cluster_{}'.format(g_i) for g_i in range(1, len(np.unique(Label_list))+1)]
    if feature_type == 'Morph':
        title_name = 'Morphology Feature Distribution'
    elif feature_type == 'Mot':
        title_name = 'Motility Feature Distribution'

    # Motility Feature Distribution Map

    fig, ax = plt.subplots(figsize=(8, 6))
    for c_i in np.unique(Label_list):
        ax.scatter(embedding[Label_list==c_i, 0], embedding[Label_list==c_i, 1], c=color[c_i-1], s=20, label=str(c_i))
    plt.xlabel('UMAP 1', fontsize=18)
    plt.ylabel('UMAP 2', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(title_name, fontsize=24)
    lgnd = plt.legend(title='Clusters')
    for c_i in np.unique(Label_list):
        lgnd.legendHandles[c_i-1]._sizes = [30]
    plt.savefig('Result_plot/Feature_Distribution.png', bbox_inches='tight')

def Cell_Bar_generator(Cell_Label, Label_list, All_resampling_std, color):
    """

    Plot the cell bar for main three cell types and Rab-based cell types

    Input:
        Cell_Label: the ground truth cell labels of all sample data
        Label_list: the labels of clusters of all sample data
        All_resampling_std: the std values of labels from randomly selected sample group
        color: assign a color to each cluster
    Output:
        save the figures in the Result_plot folder

    """

    Cell_Type_Main = ['CTRL', 'MDAEVs', 'BrEVs']
    Cell_Type_Rab = ['Rab7', 'Rab11fip2', 'Rab11fip3', 'Rab11fip5']

    groups = ['Cluster_{}'.format(g_i) for g_i in range(1, len(np.unique(Label_list))+1)]

    ctrl_group = Label_list[Cell_Label == 1]
    ctrl_bar = [len(ctrl_group[ctrl_group == g_i]) for g_i in np.unique(Label_list)]
    MDA_group = Label_list[Cell_Label == 2]
    MDA_bar = [len(MDA_group[MDA_group == g_i]) for g_i in np.unique(Label_list)]
    BrEV_group = Label_list[Cell_Label == 3]
    BrEV_bar = [len(BrEV_group[BrEV_group == g_i]) for g_i in np.unique(Label_list)]
    Rab7_group = Label_list[Cell_Label == 4]
    Rab7_bar = [len(Rab7_group[Rab7_group == g_i]) for g_i in np.unique(Label_list)]
    Rab11_f2_group = Label_list[Cell_Label == 5]
    Rab11_f2_bar = [len(Rab11_f2_group[Rab11_f2_group == g_i]) for g_i in np.unique(Label_list)]
    Rab11_f3_group = Label_list[Cell_Label == 6]
    Rab11_f3_bar = [len(Rab11_f3_group[Rab11_f3_group == g_i]) for g_i in np.unique(Label_list)]
    Rab11_f5_group = Label_list[Cell_Label == 7]
    Rab11_f5_bar = [len(Rab11_f5_group[Rab11_f5_group == g_i]) for g_i in np.unique(Label_list)]

    ctrl_bar = np.array(ctrl_bar)/len(ctrl_group)
    MDA_bar = np.array(MDA_bar)/len(MDA_group)
    BrEV_bar = np.array(BrEV_bar)/len(BrEV_group)
    Rab7_bar = np.array(Rab7_bar)/len(Rab7_group)
    Rab11_f2_bar = np.array(Rab11_f2_bar)/len(Rab11_f2_group)
    Rab11_f3_bar = np.array(Rab11_f3_bar)/len(Rab11_f3_group)
    Rab11_f5_bar = np.array(Rab11_f5_bar)/len(Rab11_f5_group)

    all_bar_main_cell = [ctrl_bar, MDA_bar, BrEV_bar]
    all_bar_Rab_cell = [Rab7_bar, Rab11_f2_bar, Rab11_f3_bar, Rab11_f5_bar]
    all_bar_main_cell = (np.array(all_bar_main_cell).T)*100
    all_bar_Rab_cell = (np.array(all_bar_Rab_cell).T)*100

    # Main Three Cell Type Cluster Bar graph Plot

    x_main = np.arange(1, len(groups)+1)

    figure, ax1 = plt.subplots()
    ax1.bar(x_main-0.25, all_bar_main_cell[:, 0], yerr=All_resampling_std[0, :], width=0.25, edgecolor='black', capsize=5, label='CTRL')
    ax1.bar(x_main, all_bar_main_cell[:, 1], yerr=All_resampling_std[1, :], width=0.25, edgecolor='black', capsize=5, label='MDAEVs')
    ax1.bar(x_main+0.25, all_bar_main_cell[:, 2], yerr=All_resampling_std[2, :], width=0.25, edgecolor='black', capsize=5, label='BrEVs')
    ax1.set_xlabel('Cluster', fontsize=12)
    ax1.set_ylabel('Percentage (%)', fontsize=12)
    ax1.set_xticks(x_main)
    ax1.set_xticklabels(groups)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend()
    ax1.set_ylim(top=65)
    plt.savefig('Result_plot/Main_cell_bar.png', bbox_inches='tight')

    # Rab-based Cluster Bar graph Plot

    fig, ax = plt.subplots(2, 2, figsize=(20, 12))

    ax[0, 0].bar(groups, all_bar_Rab_cell[:, 0], yerr=All_resampling_std[3, :], width=0.5, color=color, capsize=5)
    for t_x, t_y, t_text in zip(groups, all_bar_Rab_cell[:, 0], all_bar_Rab_cell[:, 0]):
        ax[0, 0].text(t_x, t_y+7, round(t_text, 1), ha='center', fontsize=12)
    ax[0, 0].set_xlabel('Cluster', fontsize=14)
    ax[0, 0].set_ylabel('Percentage (%)', fontsize=14)
    ax[0, 0].set_title(Cell_Type_Rab[0], fontsize=18)
    plt.xticks(fontsize=12)
    plt.xticks(fontsize=12)
    ax[0, 0].set_ylim(top=60)

    ax[0, 1].bar(groups, all_bar_Rab_cell[:, 1], yerr=All_resampling_std[4, :], width=0.5, color=color, capsize=5)
    for t_x, t_y, t_text in zip(groups, all_bar_Rab_cell[:, 1], all_bar_Rab_cell[:, 1]):
        ax[0, 1].text(t_x, t_y+7, round(t_text, 1), ha='center', fontsize=12)
    ax[0, 1].set_xlabel('Cluster', fontsize=14)
    ax[0, 1].set_ylabel('Percentage (%)', fontsize=14)
    ax[0, 1].set_title(Cell_Type_Rab[1], fontsize=18)
    plt.xticks(fontsize=12)
    plt.xticks(fontsize=12)
    ax[0, 1].set_ylim(top=60)

    ax[1, 0].bar(groups, all_bar_Rab_cell[:, 2], yerr=All_resampling_std[5, :], width=0.5, color=color, capsize=5)
    for t_x, t_y, t_text in zip(groups, all_bar_Rab_cell[:, 2], all_bar_Rab_cell[:, 2]):
        ax[1, 0].text(t_x, t_y+7, round(t_text, 1), ha='center', fontsize=12)
    ax[1, 0].set_xlabel('Cluster', fontsize=14)
    ax[1, 0].set_ylabel('Percentage (%)', fontsize=14)
    ax[1, 0].set_title(Cell_Type_Rab[2], fontsize=18)
    plt.xticks(fontsize=12)
    plt.xticks(fontsize=12)
    ax[1, 0].set_ylim(top=65)

    ax[1, 1].bar(groups, all_bar_Rab_cell[:, 3], yerr=All_resampling_std[6, :], width=0.5, color=color, capsize=5)
    for t_x, t_y, t_text in zip(groups, all_bar_Rab_cell[:, 3], all_bar_Rab_cell[:, 3]):
        ax[1, 1].text(t_x, t_y+7, round(t_text, 1), ha='center', fontsize=12)
    ax[1, 1].set_xlabel('Cluster', fontsize=14)
    ax[1, 1].set_ylabel('Percentage (%)', fontsize=14)
    ax[1, 1].set_title(Cell_Type_Rab[3], fontsize=18)
    plt.xticks(fontsize=12)
    plt.xticks(fontsize=12)
    ax[1, 1].set_ylim(top=60)
    figure.tight_layout(pad=1)
    plt.savefig('Result_plot/Rab_cell_bar.png', bbox_inches='tight')

def cell_distribution_plot(embedding, Cell_Label, Cell_Type):
    """

    Plot cell type distribution map

    Input:
        embedding: the coordinates of UMAP distribution
        Cell_Label: the ground truth cell labels of all sample data
        Cell_Type: the name list of cell types
    Output:
        save the figure in the Result_plot folder

    """
    for j in range(len(Cell_Type)):

        tmp_embedding = embedding[Cell_Label == (j+1), :]
        tmp_embedding = tmp_embedding.T
        z = gaussian_kde(tmp_embedding)(tmp_embedding)
        cc = cm.jet((z-z.min())/(z.max()-z.min()))

        fig = plt.figure(figsize=(8,6))
        ax = plt.subplot(1,1,1)
        ax.scatter(embedding[Cell_Label != (j+1), 0], embedding[Cell_Label != (j+1), 1], marker='o', facecolors='lightgray', s=20)
        ax.scatter(embedding[Cell_Label == (j+1), 0], embedding[Cell_Label == (j+1), 1], marker='o', facecolors=cc, s=20)
        ax.set_aspect('equal', 'datalim')
        plt.xlabel('UMAP 1', fontsize=18)
        plt.ylabel('UMAP 2', fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title(Cell_Type[j], fontsize=24)

def feature_matrix_generator(Feature_list, col_index, Feature_type, balanced_sign):
    """

    Plot the Feature matrix

    Input:
        Feature_list: the adjusted values of the features
        col_index: the name list of features
        Feature_type: select to plot the order of feature list
        balanced_sign: show the extra morphological feature
    Output:
        save the figure in the Result_plot folder

    """

    feature_id = col_index[1:]
    feature_id = np.array(feature_id)
    if Feature_type == 'Morph':
        if balanced_sign:
            ideal_order = [0, 1, 5, 3, 7, 2, 8, 9, 6, 4]
        else:
            ideal_order = [0, 1, 5, 3, 7, 2, 8, 10, 9, 6, 4]
    elif Feature_type == 'Mot':
        ideal_order = [3, 0, 2, 1, 4, 7, 8, 6, 5]
    Feature_list = Feature_list[ideal_order]
    feature_id_new = feature_id[ideal_order]

    fig = plt.figure(figsize=(6, 5))
    axmatrix = fig.add_axes([0, 0.1, 0.4, 0.6])
    cax = axmatrix.matshow(Feature_list, interpolation='nearest')
    axmatrix.yaxis.set_major_locator(MultipleLocator(1))
    axmatrix.xaxis.set_major_locator(MultipleLocator(1))
    axmatrix.set_yticks(range(len(feature_id_new)))
    axmatrix.set_yticklabels(feature_id_new, fontsize=7)
    axmatrix.set_xlabel('Cluster', fontsize=10)
    axmatrix.set_xticks(range(3))
    axmatrix.set_xticklabels(['1', '2', '3'], fontsize=8)
    plt.gca().yaxis.tick_right()
    plt.gca().xaxis.tick_bottom()

    axcolor = fig.add_axes([0.605, 0.1, 0.02, 0.6])
    clb = plt.colorbar(cax, cax=axcolor)
    clb.ax.set_title('Scaled\n Value', fontsize=8)
    clb.ax.tick_params(labelsize=7)

    plt.savefig('Result_plot/Feature_matrix.png', bbox_inches='tight')
