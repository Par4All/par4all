from pyps import workspace

# Create a workspace 
w = workspace('add_source1.c');

w.fun.main.display()

w.add_source("add_source2.c");
w.fun.source2.display()
