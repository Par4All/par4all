#!/usr/bin/env python 
# -*- coding: utf-8 -*-
#
# Author:
# -  Mehdi Amini <mehdi.amini@hpc-project.com>
#

import os
import sys
import shutil
import re
from p4a_processor import *
from p4a_util import *
import pyps

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
    
    
class p4a_spear_processor(p4a_processor):

    def __init__(self, spear,filter_select="", *sources, **kwargs):
        self.spear=True # flag
        
        if filter_select:
            warn("Override filter_select ("+filter_select+") because of Spear XML file")
        filter_select=self.load_spear_xml(spear)
       
        # Init parent
        p4a_processor.__init__(self, filter_select=filter_select, *sources, **kwargs)
    
    def load_spear_xml(self,xml_file):
        tree = ET.ElementTree(file=xml_file)
        filter=""
        sep=""
        self.tasks = []
        for task in tree.getroot().findall("Task"):
            warn("Filtering Spear Task " + task.attrib['name'])
            self.tasks.append(task)
            filter+=sep+".*"+task.attrib['name']+"$"
            sep="|"
        warn("Filter is : "+filter)
        return filter    

    def parallelize(self, fine_grain = False, *args, **kwargs):
        selected_modules = self.filter_modules("", "")
        #selected_modules.unfold()
        #selected_modules.display()
        #self.workspace.fun.main_segment_P2012.display("print_code_regions")
        #sys.exit()
        for task in self.tasks:
            parallelLoops = int(task.attrib['nbParallelLoops'])
            warn("Parallelizing Task {0} with {1} nested parallel loops".format(task.attrib['name'],parallelLoops))
            if parallelLoops==0:
                self.workspace[task.attrib['name']].one_thread_parallelize()
            else:
                loops = self.workspace[task.attrib['name']].loops()
                while parallelLoops > 0:
                    if len(loops)>1:
                        raise Exception("Expect {0} perfectly nested loop but got only %{1}".format(task.attrib['nbParallelLoops'],int(task.attrib['nbParallelLoops'])-parallelLoops))
                    # Schedule as parallel
                    loops[0].parallel=True
                    parallelLoops=parallelLoops-1
                    loops=loops[0].loops()
        #p4a_processor.parallelize(self,fine_grain = True,  *args, **kwargs)
        
    def save_generated (self, output_dir, subs_dir):
        ret = p4a_processor.save_generated(self,output_dir,subs_dir)
        # 
        warn("XML Post processing " + output_dir + " " + subs_dir)
        tasks_xml = ""
        for task in self.tasks:
            wrapper = "p4a_wrapper_"+task.attrib['name']
            
            warn("Looking for " + wrapper)
            try:
            	fun=self.workspace[wrapper]
            except:
            	warn("No fun for " + wrapper)
            	continue
            
            fun.gpu_xml_dump()
            xml = os.path.join(self.workspace.dirname,self.workspace[wrapper].show("GPU_XML_FILE"))
            # get content
            f = open(xml)
            tasks_xml += f.read()
            f.close()        
            
        # aggregate content
        xml_src = os.path.join(self.workspace.dirname,"P4A","p4a_kernels.xml")
        xml_dst = os.path.join(output_dir,"p4a_kernels.xml")
        f = open(xml_src,mode="w")
        f.write('<!DOCTYPE Tasks SYSTEM "p4a_kernels.dtd">\n<Tasks>\n'+tasks_xml+'</Tasks>\n')
        f.close()
		# install the file to its final dest
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        shutil.copyfile(xml_src, xml_dst)
        
        # Add a DTD
        f = open(os.path.join(output_dir,"p4a_kernels.dtd"),mode="w")
        f.write('''<?xml encoding="UTF-8"?>

<!ELEMENT Tasks (Task)+>
<!ATTLIST Tasks
  xmlns CDATA #FIXED ''>

<!ELEMENT Task (Arg)+>
<!ATTLIST Task
  xmlns CDATA #FIXED ''
  kernel NMTOKEN #REQUIRED
  name NMTOKEN #REQUIRED
  nbParallelLoops CDATA #REQUIRED>

<!ELEMENT Arg EMPTY>
<!ATTLIST Arg
  xmlns CDATA #FIXED ''
  name NMTOKEN #REQUIRED
  type NMTOKEN #REQUIRED
  typeData CDATA #REQUIRED
  value CDATA #IMPLIED>
''')
        f.close()
        return ret
            

        
if __name__ == "__main__":
    print(__doc__)
    print("This module is not directly executable, use 'p4a --spear-xml' instead")
    #proc=p4a_spear_processor("Par4All/p4a_input.xml",project_name = "toto")
    #p4a_spear_processor.accel_post('p4a_wrapper_fusion_F2_Ci.cl')
# Some Emacs stuff:
### Local Variables:
### mode: python
### mode: flyspell
### ispell-local-dictionary: "american"
### tab-width: 4
### End:
