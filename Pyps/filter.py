from __future__ import with_statement # this is to work with python2.5
#!/usr/bin/env python
from pyps import *
import re

#map(lambda x:x.name, w.filter(my_condition))

ws = workspace("filter.c", deleteOnClose=True)

prefix = "toto"
prefix_filter_re = re.compile(prefix + "[0-9]+")
matches = ws.filter(lambda m: prefix_filter_re.match(m.name))


#loop over the modules in that matched
for module in matches:
    print module.name

#get a list using map
res = map(lambda x:x.name, matches)
print res

