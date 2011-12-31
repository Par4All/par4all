
from pyps import workspace

# We initially create an empty workspace
w = workspace(name="force_create",deleteOnCreate=True);
w.close()

# Now, .database exists and prevent new workspace creation with the same name
try :
	w = workspace(name="force_create");
	w.close()
except RuntimeError:
	print "Create workspace error, as expected ! ;-)"
	

# Try again, and force overwrite
w = workspace(name="force_create",deleteOnCreate=True,deleteOnClose=True);
w.close()


