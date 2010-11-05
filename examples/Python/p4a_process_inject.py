"""
This is an example of code injection in p4a_process of p4a with a direct
method modification. Can be seen as a poor man aspect programmin

This is to be used as:

p4a --execute p4a.execute_some_python_code_in_process='"import p4a_process_inject"' ...

and you should adapt the PYTHONPATH to cope with the location where is
this p4a_process_inject.py

Ronan.Keryell@hpc-project.com
"""

# Import the code we want to tweak:
import p4a_process

# Save the old definition of the p4a_process.p4a_processor.parallelize
# method for a later invocation:
old_p4a_processor_parallelize = p4a_process.p4a_processor.parallelize

# Define a new method instead:
def new_p4a_processor_parallelize(self, **args):

    # Apply for example a partial evaluation on all the functions of the
    # programm
    self.workspace.all_functions.partial_eval()

    # Go on by calling the original method with the same parameters:
    old_p4a_processor_parallelize(self, **args)

# Overide the old definition of p4a_process.p4a_processor.parallelize by
# our new one:
p4a_process.p4a_processor.parallelize = new_p4a_processor_parallelize
