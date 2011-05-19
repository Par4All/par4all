

import pyps

w = pyps.workspace("cast.c",deleteOnClose=True)

w.props.constant_path_effects = False
w.fun.main.display(activate="print_code_proper_effects")

w.fun.main.display(activate="print_code_cumulated_effects")
