from evaluatingDPML.core.attack import save_data
from DPMLadapter.ArgsObject import ArgsObject

# Adapting current working directory
from evaluatingDPML import chdir_to_evaluating
chdir_to_evaluating()

save_data(ArgsObject('purchase_100',save_data=1))
