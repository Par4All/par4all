from __future__ import with_statement
from pyps import workspace,module

#Deleting workspace
workspace.delete("hyantes")

#Creating workspace
w = workspace("hyantes/hyantes.c","hyantes/options.c",cppflags='-Ihyantes',name="hyantes",deleteOnClose=True)

#Add some default property
w.props.abort_on_user_error=True

w.activate(module.region_chains)

w.props.constant_path_effects=False

w["hyantes!do_run_AMORTIZED_DISK"].privatize_module()

#Closing workspace
w.close()


