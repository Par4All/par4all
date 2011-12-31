from __future__ import with_statement # this is to work with python2.5
import pyps

# we will use workspace broker, that automatically set a callback for tracking missing module in pips
import broker


#
# Try getting missing module on the fly using an internal resolver
#


# Broker workspace
# Use simpleExampleBroker provided by default with pyps which provide module "simpleExampleDynamicLoadedFunction"
with broker.workspace("broker02.c",brokersList="simpleExampleBroker",name="broker02",deleteOnClose=True,deleteOnCreate=True) as w:
    # Calling display of cumulated effects will make Pips resolve callees, and there is an undefined module inside...
    w.fun.main.display(pyps.module.print_code_cumulated_effects)

