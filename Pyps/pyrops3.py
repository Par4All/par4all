from __future__ import with_statement # this is to work with python2.5
#!/usr/bin/env python
#This script is an adaptation of basics0.py for Pyrops

# import everything so that a session looks like tpips one
from pyrops import pworkspace
import shutil,os,pyrops



w2 = pworkspace("basics0.c",deleteOnClose=True)
w1 = pworkspace("cat.c",deleteOnClose=True)


# Should display only functions from w1
w1.all_functions.display()

w1.close()
w2.close()
