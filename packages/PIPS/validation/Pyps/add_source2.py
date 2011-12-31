from pyps import workspace

# Create an empty workspace 
w = workspace();

w.add_source("add_source2.c");
w.fun.source2.display()
