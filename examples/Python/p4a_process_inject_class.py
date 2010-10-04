"""
This is an example of code injection in p4a_process of p4a with some class
inheritance.

This is to be used as:

p4a --execute p4a.execute_some_python_code_in_process='"import p4a_process_inject_class"'

and you should adapt the PYTHONPATH to cope with the location where is
this p4a_process_inject.py

Ronan.Keryell@hpc-project.com
"""

# Import the code we want to tweak:
import p4a_process


class my_p4a_processor(p4a_process.p4a_processor):
    """Change the PyPS transit of p4a.

    Since it is done in the p4a_process.p4a_processor, we inherit of this
    class
    """

    def __init__(self, **args):
        "Change the init method just to warn about this new version"

        print "This is now done by class", self.__class__
        # Just call the normal constructors with all the named parameters:
        super(my_p4a_processor, self).__init__(**args)


    # Define a new method instead:
    def parallelize(self, **args):
        "Parallelize the code"

        # Apply for example a partial evaluation on all the functions of the
        # programm at the beginning of the parallelization
        self.workspace.all_functions.partial_eval()

        # Go on by calling the original method with the same parameters:
        # Compatible super() invocation with pre-3 Python
        super(my_p4a_processor, self).parallelize(**args)


# Overide the old class definition by our new one:
p4a_process.p4a_processor = my_p4a_processor
