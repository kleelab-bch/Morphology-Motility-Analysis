from src.Utilities import *

# Parameter Setting ====================================================================================================
# If the user has his/her own dataset, please change the path
# Please select what kind of feature will be analyzed: 'Morph'(morphology) or 'Mot'(motility)
# Save(True) or Plot(False) the figures

Data_file = 'Testing/Motility_Features.csv'
Feature_type = 'Mot'
save_figure = True

# Main Function ========================================================================================================
# main function has three arbitrary keyword arguments: Cell_Type, cluster_parameters, balanced_sign
# If the user would like to change, please read the details of main function in Utilities.py if the user would like to change
#
# Example:
#   main(Data_file, Feature_type, cluster_parameters=[5, 0, 9])
#

if __name__ == "__main__":
    main(Data_file, Feature_type, save_figure=save_figure)