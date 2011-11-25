from optparse import OptionParser
import sys
import pyps
from pypsutils import get_runtimefile
import os.path
import broker


class simpleStubBroker(broker.broker):
    """ broker that automatically gather stub files by search for file name
    corresponding to module name in a list of directories """

    def stub_file_for_module(self, module):
        for broker_dir in self.get_broker_dirs():
            fname = os.path.join(broker_dir,module+".c")
            if os.path.exists(fname):
              return fname
        return ""

    def get_broker_dirs(self):
        """ return the list of directories to inspect"""
        return list()

