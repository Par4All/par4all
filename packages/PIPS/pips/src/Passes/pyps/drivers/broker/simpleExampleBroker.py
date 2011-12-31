from simpleStubBroker import simpleStubBroker
import os,pypsconfig

class simpleExampleBroker(simpleStubBroker):
    """ Example broker that provide only one stub for simpleExampleDynamicLoadedFunction
    using simpleStubBroker class, which imply that it is located in a file name
    simpleExampleDynamicLoadedFunction.c """

    def __init__(self):
        super(simpleExampleBroker,self).__init__()
        
        # initialize lookup dir, which is a subdir install dir of this broker
        self.__lookupDirs = []
        self.__lookupDirs.append(os.path.join(pypsconfig.pypsruntime,"broker",self.__class__.__name__,"stub"))
        
    def get_broker_dirs(self):
        """called by simpleStubBroker to know where to look for stub files"""
        return self.__lookupDirs
