import numpy as np
from src.Utilities import *
from src.Figure_plots import *

# Parameter Setting ====================================================================================================
# If the user has his/her own dataset, please change the path
# Please select what kind of feature will be analyzed: 'Morph'(morphology) or 'Mot'(motility)

Data_file = 'Testing/qPI_Motility_Features.csv'
Feature_type = 'Mot'

# Main Function ========================================================================================================
# main function has three arbitrary keyword arguments: Cell_Type, cluster_parameters, balanced_sign
# If the user would like to change, please read the details of main function in Utilities.py if the user would like to change
#
# Example:
#   main(Data_file, Feature_type, cluster_parameters=[5, 0, 9])
#

if __name__ == "__main__":
    main(Data_file, Feature_type)