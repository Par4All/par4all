from pyps import workspace
workspace.delete("bb")
workspace("a a a /spaces01.c",name="bb",recoverInclude=False).fun.main.display()
