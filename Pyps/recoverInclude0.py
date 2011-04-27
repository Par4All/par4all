import os
from pyps import workspace
w=workspace("recoverInclude0.c",cppflags="-Iinclude",deleteOnClose=True,name="recoverInclude0")
(sources, headers)=w.save()
print sources
print headers
for i in sources:os.system(os.environ["PAGER"]+" " + i)

w.close()
