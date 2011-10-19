import sys
import pyps
import os.path
import broker
import re
import p4a_util

class p4a_stubs_broker(broker.broker):
    """ broker that automatically gather stub files for the par4all runtime
    in some cases it will generate the missing source files """
    def __init__(self):
        super(p4a_stubs_broker,self).__init__()
        self.stubs_dir = os.path.join(os.environ["P4A_ROOT"],'stubs')


    def stub_file_for_module(self, module):
        try:
            return self.generate_p4a_accel_copy(module)
        except:
            pass

        try:
            return self.generate_p4a_atomic(module)
        except:
            pass
        
        
        for broker_dir in self.get_broker_dirs():
            fname = os.path.join(broker_dir,module+".c")
            if os.path.exists(fname):
              return fname
        return ""

    def get_broker_dirs(self):
        """ return the list of directories to inspect"""
        return [
                d for d in (os.path.join(self.stubs_dir, d1) for d1 in os.listdir(self.stubs_dir)
                            ) if os.path.isdir(d)                
                ]

    def generate_p4a_atomic(self,module):
        accepted_name = re.compile(r"atomic(Add|Sub|Inc|Dec|And|Or|Xor)(Int|Float)")
        matched_name = accepted_name.match(module)
        if matched_name == None :
            # will be catched :)
            print "not matched "+module
            raise Exception()

        generated = "void " + module
        if matched_name.group(1)!="Add":
            raise Exception()
        operator = "+"
        
        if matched_name.group(2)!="Int":
            raise Exception()
        type = "int"

        generated += "("+type +" *a, "+type+" val) { *a=*a" + operator + "val; }\n"
        
        return self.generate_stub_for_module(module,generated)
        
    def generate_p4a_accel_copy(self,module):
        accepted_name = re.compile(r"P4A_copy_(to|from)_accel_([0-9]+)d")
        matched_name = accepted_name.match(module)
        if matched_name == None :
            # will be catched :)
            print "not matched "+module
            raise Exception()
        
        ndim = int(matched_name.group(2))
        generated_stub = "void " + module + "(int element_size,\n"
        
        for dim in range(1,ndim+1):
            generated_stub += " int d" + str(dim) + "_size,"
        generated_stub += "\n"
        for dim in range(1,ndim+1):
            generated_stub += " int d" + str(dim) + "_block_size,"
        generated_stub += "\n"
        for dim in range(1,ndim+1):
            generated_stub += " int d" + str(dim) + "_offset,"
        generated_stub += "\n void *host_address,"
        generated_stub += "\n const void *accel_address) {\n"

        # loop indices
        generated_stub += "  int "
        for dim in range(1,ndim):
            generated_stub += " i_" + str(dim) + ","
        generated_stub += "i_"+ str(ndim) + ";\n"
        
        # size in bytes for each dim
        for dim in range(1,ndim+1):
            generated_stub += " int d" + str(dim) + "_byte_block_size = d" + str(dim) + "_block_size * element_size;\n"
            generated_stub += " int d" + str(dim) + "_byte_size = d" + str(dim) + "_size * element_size;\n"
        
        # Copy to or from accel ?
        if matched_name.group(1) == "from":
            generated_stub += " char * cdest = (char*)host_address;\n"
            generated_stub += " const char * csrc = (char*)accel_address;\n"
        else :
            generated_stub += " char * cdest = (char*)accel_address;\n"
            generated_stub += " const char * csrc = (char*)host_address;\n"
            
        #Generate the loop nest
        for dim in range(1,ndim+1):
            generated_stub += " for(i_"+str(dim)+" = 0; i_"+str(dim)+" < d"+str(dim)+"_byte_block_size; i_"+str(dim)+"++) {\n"
        
        # host index
        generated_stub += "  int h_index = "        
        sep = ""
        for dim in range(1,ndim+1):
            generated_stub += "\n      "+ sep + "(i_"+str(dim)+"+d"+str(dim)+"_offset)"
            sep = "+"
            for dim2 in range(dim+1,ndim):
                generated_stub += "*d"+str(dim2)+"_byte_size"
        generated_stub += ";\n"

        # accelerator index
        sep = ""
        generated_stub += "  int a_index = "        
        for dim in range(1,ndim+1):
            generated_stub += "\n      "+sep+"i_"+str(dim)
            sep = "+"
            for dim2 in range(dim+1,ndim+1):
                generated_stub += "*d"+str(dim2)+"_byte_size"
        generated_stub += ";\n"

        # Copy to or from accel ?
        if matched_name.group(1) == "from":
            generated_stub += "cdest[a_index] = csrc[h_index];\n";
        else:
            generated_stub += "cdest[h_index] = csrc[a_index];\n";
            
        #Generate the loop nest
        for dim in range(1,ndim+1):
            generated_stub += "  }\n"

        generated_stub += "}\n"

        return self.generate_stub_for_module(module,generated_stub)
    
    def generate_stub_for_module(self,module,generated_stub):
        fname = os.path.join(self.stubs_dir,"generated",module+".c") 
        f = open(fname, 'w')
        f.write(generated_stub)
        return fname
        