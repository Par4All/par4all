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
    outputDirectory = None
    generatedHeaderFile = "p4a_new_files_include.h"
    generatedKernelFiles = ""
    p4aVersion = ""

    def __init__(self, dialect="openmp", save_dir=None):
        import p4a_version

        print("ASTRAD: call to initialization \n")
        global astrad_dialect_names
        self.outputDialect = astrad_dialect_names[dialect]
        print("output dialect : " + self.outputDialect +"\n")
        self.p4aVersion = p4a_version.VERSION()
        if save_dir:
            self.saveDir = save_dir
        else:
            self.saveDir = os.getcwd()
        print("saving in dir: " + self.saveDir + "\n")


    def set_source_name(self, file):
        if(self.sourceName):
            p4a_util.die("ASTRAD post processor ERROR: there are more than one input modules:\n")
        else:
            self.sourceName = file

    def set_error_code(self, err):
        self.errorCode = err

    def set_output_file_name(self, name):
        self.outputFileName = os.path.split(name)[1];

    def set_output_directory(self, dir):
         self.outputDirectory = dir

    def set_generated_kernel_files(self, files):
        first = True
        for name in files:
            if not first:
                self.generatedKernelFiles += " ,"
            else:
                first = False
            self.generatedKernelFiles += os.path.split(name)[1];

    def save_dsl_file(self):
        global astrad_dialect_names

        print("ASTRAD: call to save_dsl_file \n")
        dsl_file_name = os.path.join(self.saveDir, p4a_util.change_file_ext(self.sourceName, '.dsl'))
        print("file name :" + dsl_file_name + "\n")

        f = open(dsl_file_name, 'w')

        content = "optimizeResult request_optimize\n"
        content +="{\n"
        content += "sourceName = " + self.sourceName + ";\n"
        content += "errorCode = " + self.errorCode + ";\n"
        content += "type = " + self.outputDialect + ";\n"
        content += "outputDirectory = " + self.outputDirectory + ";\n"
        content += "outputFileName = " + self.outputFileName + ";\n"
        if (self.outputDialect != astrad_dialect_names['openmp']):
            content += "generatedKernelFiles = "
            content += self.generatedKernelFiles
            content += ";\n"
            content += "generatedHeaderFile = " + self.generatedHeaderFile + ";\n"

        content += "Par4All (\"" + self.p4aVersion + "\")\n"
        content += "{\n" + "date = "
        content += time.strftime('%d/%m/%y %H:%M',time.localtime())
        content += ";\n}\n"
        content += "}\n"

        f.write(content)
        f.close()
