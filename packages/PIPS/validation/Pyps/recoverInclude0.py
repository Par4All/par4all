from pyps import workspace
w=workspace("recoverInclude0.c",cppflags="-Iinclude",deleteOnClose=True)
w.close()
