from optparse import OptionParser
import sys
import pyps
import pypsutils 
import os.path
import shutil
import tempfile




class workspace(pyps.workspace):
    """ This is a broker workspace, it automatically register a callback to PIPS
    and fetch missing module using brokers, it'll also record stub file given to 
    PIPS and interact with Maker for a proper compilation """ 
    
    def __init__(self, *sources, **kwargs):
        # Record which files where given to pips as stub file
        self.stub_files = []

        # Brokers have been initialized by the user
        self.brokers=kwargs.get("brokers",None)
        if self.brokers == None :
            # We allow to pass a comma separated list of broker name
            brokers_list=kwargs.get("brokersList","broker")
            self.brokers = brokers()
            brokers_list = brokers_list.split(",")
            self.brokers.load_brokers_from_list(brokers_list)

        # FIXME : should be done on the fly
        cppflags=kwargs.get("cppflags","")        
        kwargs["cppflags"]=cppflags+self.brokers.cppflags

        super(workspace,self).__init__(*sources,**kwargs)
            
        # register to PIPS !
        self.cpypips.set_python_missing_module_resolver_handler(self.brokers)
        self.props.preprocessor_missing_file_handling="internal_resolver"


class brokers(object):

    def __init__(self):
        # list of loaded brokers
        self.brokers = []
        self.stub_files = []


    def load_brokers_from_list(self,brokers_list):
        for broker_name in brokers_list:
            instanceBroker = getattr(__import__(broker_name),broker_name)()
            self.brokers.append(instanceBroker)

    def append(self,broker):
        self.brokers.append(broker)

    @property
    def cppflags(self):
        flags = ""
        return reduce(lambda x,y: x+y.cppflags(), self.brokers,"")


    def stub_file_for_module(self, module):
        """ Get a stub file using brokers, and register any file given to
        pips here so the we can exclude them from compilation later """
        
        stub_file_name = "";
        
        # Delegate to brokers the sub file retrieving, we stop as soon as 
        # we have one broker that knows this function
        for broker in self.brokers:
            orig_stub_file = broker.stub_file_for_module(module)
            if orig_stub_file != "":
                # Will copy the original stub file to a temporary location and rename
                # it so that we ensure an unique name (no collision with user files)
                stubfileName, stubExtension = os.path.splitext(os.path.basename(orig_stub_file))
                new_name = "stub_broked_by_" + broker.__class__.__name__ + "_" + stubfileName
                
                stub_file=tempfile.NamedTemporaryFile(prefix=new_name,suffix=stubExtension,delete=False)
                stub_file_name=stub_file.name
                shutil.copy2(orig_stub_file,stub_file_name)
        
                # register the file as a stub
                self.stub_files.append(os.path.basename(stub_file_name)); 
                break;


        return stub_file_name

    


class broker(object):
    """ Template class for a broker, to be specialized by implementation"""

    def stub_file_for_module(self, module):
        """ gather stub file for a given module, this is the default stub broker,
        it'll go inside all dirs and try to find a file that name corresponds
        to module name """
        return ""

    def cflags(self,target = "sequential"):
        """ Get CFLAGS for a given target (sequential,openmp,cuda,openCl,...)"""
        return ""

    def ldflags(self,target = "sequential"):
        """ Get LDFLAGS for a given target (sequential,openmp,cuda,openCl,...)"""
        return ""

    def cppflags(self):
        return ""


# main script
if __name__ == '__main__':
    parser = OptionParser(usage = "%prog")
    parser.add_option("--brokers",type="string", default="broker")
    (opts, modules) = parser.parse_args()

    if not modules or len(modules) > 1:
        raise RuntimeError("need one module")

    b = brokers();
    b.load_brokers_from_list(opts.brokers.split(","))
    
    # forge a function name and call it
    try:
        print b.stub_file_for_module(modules[0])
    except KeyError:
        print >> sys.stderr, opts.arch, "is not a supported flag argument"



