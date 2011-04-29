from optparse import OptionParser
import sys
import pyps
from pypsutils import get_runtimefile
import os.path

# you should register the various broker directory here
# Beware, order matters :p

class workspace(pyps.workspace):
    def __init__(self, *sources, **kwargs):
        broker=kwargs.get("broker",Broker())
        cppflags=kwargs.get("cppflags","")
        kwargs["cppflags"]=cppflags+broker.cppflags()
        super(workspace,self).__init__(*sources,**kwargs)
        self.props.preprocessor_missing_file_handling="query"
        self.props.preprocessor_missing_file_generator=\
            "python -m " \
            + broker.__class__.__name__ \
            + " --brokers=" \
            + ",".join(map(lambda x:x+"broker",broker.get_broker_dirs()))

class Broker(object):

    def stub_broker(self, module):
        """ broker to gather stub files """
        for broker_dir in self.get_broker_dirs():
            try:
                print get_runtimefile(module+".c",os.path.join("broker",broker_dir,"stub"))
                break
            except RuntimeError:
                print >> sys.stderr, "function", module, "not found in broker with flavour", broker_dir, ": trying another one"

    def get_broker_dirs(self):
        """ return the list of broker dir to inspect"""
        return list()

    def cppflags(self):
        return ""

def broker_compositor(class_names):
    """ return a class obtained by composing all classes from `class_names'
        warning: each class is found in a module with the same name as the class
        and order matters :)"""
    class_names.reverse()
    class _(Broker):pass
    for class_name in class_names:
        class _(_,getattr(__import__(class_name),class_name)):pass
    class_names.reverse()
    return _

def parse_args():
    parser = OptionParser(usage = "%prog")
    parser.add_option("--brokers",type="string", default="")
    (opts, modules) = parser.parse_args()

    if not modules or len(modules) > 1:
        raise RuntimeError("need one module")

    opts.brokers = opts.brokers.split(",")
    myBroker=broker_compositor(opts.brokers)()
    # forge a function name and call it
    try:
        myBroker.stub_broker(modules[0])
    except KeyError:
        print >> sys.stderr, opts.arch, "is not a supported flag argument"

# main script
if __name__ == '__main__':
    parse_args()



