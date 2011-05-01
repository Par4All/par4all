from __future__ import with_statement # this is to work with python2.5
import pyps




#
# Try getting missing module on the fly using an external resolver
#


# Regular workspace
with pyps.workspace("broker01.c",name="broker01",deleteOnClose=True,deleteOnCreate=True) as w:
    # Ad-hoc properties
    # Use simpleExampleBroker provided by default with pyps which provide module "simpleExampleDynamicLoadedFunction"
    w.props.preprocessor_missing_file_handling="external_resolver"
    w.props.preprocessor_missing_file_generator="python -m broker --brokers=simpleExampleBroker"
    
    # Calling display of cumulated effects will make Pips resolve callees, and there is an undefined module inside...
    w.fun.main.display(pyps.module.print_code_cumulated_effects)
    

