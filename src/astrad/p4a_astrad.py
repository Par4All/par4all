#!/usr/bin/env python 
# -*- coding: utf-8 -*-
#
# Author:
# -  Beatrice Creusillet <beatrice.creusillet@hpc-project.com>
#

import os
import time
import shutil
import re
from p4a_processor import *
from p4a_util import *
import pyps

astrad_dialect_names = {
    "openmp" : "OpenMP",
    "cuda" : "Cuda",
    "opencl" : "OpenCL"}

class p4a_astrad_postprocessor(object):

    # only one module file is supposed to be passed by Astrad to p4a
    sourceName = None

    saveDir = None
    errorCode = "ok"
    outputDialect = None
    outputFileName = None
    mainFunctionName = None
    outputDirectory = None
    generatedHeaderFile = "p4a_new_files_include.h"
    generatedKernelFiles = ""
    p4aVersion = ""

    def __init__(self, dialect="openmp", save_dir=None):
        import p4a_version

        #print("ASTRAD: call to initialization \n")
        global astrad_dialect_names
        self.outputDialect = astrad_dialect_names[dialect]
        #print("output dialect : " + self.outputDialect +"\n")
        self.p4aVersion = p4a_version.VERSION()
        self.saveDir = save_dir or os.getcwd()
        #print("saving in dir: " + self.saveDir + "\n")


    def add_source_name(self, name):

        if self.sourceName:
            self.sourceName += ", "
        else:
            self.sourceName = ""
        self.sourceName += os.path.split(name)[1]

    def set_error_code(self, err):
        self.errorCode = err

    def set_output_file_name(self, name):
        self.outputFileName = os.path.split(name)[1]

    def set_main_function_name(self, name):
        self.mainFunctionName = name

    def set_output_directory(self, dir):
         self.outputDirectory = dir

    def set_generated_kernel_files(self, files):
        first = True
        for name in files:
            if not first:
                self.generatedKernelFiles += ", "
            else:
                first = False
            self.generatedKernelFiles += os.path.split(name)[1]

    def save_dsl_file(self):
        global astrad_dialect_names

        print("ASTRAD: call to save_dsl_file \n")
        dsl_file_name = os.path.join(self.saveDir, self.mainFunctionName + '.dsl')
        print("file name :" + dsl_file_name + "\n")

        f = open(dsl_file_name, 'w')

        content = "optimizeResult request_optimize_"
        content += time.strftime('%y_%m_%d',time.localtime())
        content +="\n"
        content +="{\n"
        content += "sourceName = " + self.outputFileName + ";\n"
        content += "methodName = " + self.mainFunctionName + ";\n"
        content += "errorCode = " + self.errorCode + ";\n"
        content += "kernelFileName = kernel.dsl;\n"
        content += "type = " + self.outputDialect + ";\n"

        # not compliant with dsl specification. To be discussed
        #if (self.outputDialect != astrad_dialect_names['openmp']):
        #    content += "generatedKernelFiles = "
        #    content += self.generatedKernelFiles
        #    content += ";\n"
        #    content += "generatedHeaderFile = " + self.generatedHeaderFile + ";\n"

        content += "Par4All (\"" + self.p4aVersion + "\")\n"
        content += "{\n" + "date = "
        content += time.strftime('%d/%m/%y',time.localtime())
        content += ";\n}\n"
        content += "}\n"

        f.write(content)
        f.close()
