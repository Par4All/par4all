import broker
import os.path

class p4a_cold_stubs_broker(broker.broker):
    """ broker that automatically gather stub files for the par4all runtime
    it extends the default broker and add directory for Cold stubs"""
    def __init__(self):
        super(p4a_cold_stubs_broker,self).__init__()
        self.stubs_dir = os.path.join('stubs','src')

    def stub_file_for_module(self, module):
        for broker_dir in self.get_broker_dirs():
            fname = os.path.join(broker_dir,module+".c")
            if os.path.exists(fname):
              return fname
        return ""

    def get_broker_dirs(self):
	return [self.stubs_dir]
