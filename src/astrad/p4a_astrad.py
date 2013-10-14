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
    moduleName = None
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

    def add_output_file_name(self, name):
        if self.outputFileName:
            self.outputFileName += ", "
        else:
            self.outputFileName = ""
        self.outputFileName += os.path.split(name)[1]

    def set_module_name(self, name):
        self.moduleName = name

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
        dsl_file_name = os.path.join(self.saveDir, self.moduleName + '.dsl')
        print("file name :" + dsl_file_name + "\n")

        f = open(dsl_file_name, 'w')

        content = "optimizeResult request_optimize_"
        content += time.strftime('%y_%m_%d',time.localtime())
        content +="\n"
        content +="{\n"
        content += "sourceName = " + self.outputFileName + ";\n"
        content += "methodName = " + self.moduleName + ";\n"
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

    def save_kernel_dsl_file(self, xml_file):

        # get xml file content as tree
        try:
            import xml.etree.cElementTree as ET
        except ImportError:
            import xml.etree.ElementTree as ET

        tree = ET.ElementTree(file=xml_file)
        root = tree.getroot()
        formal_arrays = root.find("FormalArrays").findall("Array")
        task_parameters = root.find("TaskParameters").findall("TaskParameter")
        loops = root.find('Loops').findall('Loop')

        # save xml file for debugging
        xml_text = read_file(xml_file)
        write_file(os.path.join(self.saveDir, "kernel.xml"), xml_text)

        # generate dsl text
        dsl_text = "kernel {\n"

        # types used
        data_types = {}
        for array in formal_arrays:
            data_type = array.find('DataType').attrib['Type']
            if not (data_type in data_types):
                data_types[data_type] = array.find('DataType').attrib['Size']
        for t, s in data_types.iteritems():
            dsl_text += t +"(" + str(s) + ")\n"

        # module function description
        dsl_text += self.moduleName + "_kernel(openLastDim=false"
        for parameter in task_parameters:
            dsl_text += "," + parameter.attrib['Name']
        dsl_text += ") {\n"

        # parameters
        dsl_text += "parameters("
        first = True
        for parameter in task_parameters:
            if parameter.attrib['ArrayP'] == "FALSE":
                if not first:
                    dsl_text += ","
                else:
                    first = False
                dsl_text += parameter.attrib['Name']
                dsl_text += "(" + parameter.attrib['DataType'] + ')'
        dsl_text += ");\n"

        # I/Os
        for parameter in task_parameters:
            if parameter.attrib['ArrayP'] == "TRUE":
                access_mode = parameter.attrib['AccessMode']
                if access_mode == "USE":
                    dsl_text += "In "
                elif access_mode == "DEF":
                    dsl_text += "Out "
                else:
                    p4a_util.die("ASTRAD post processor ERROR: invalid USE_DEF array:\n")

                dsl_text += parameter.attrib['Name'] + " (datatype="
                dsl_text += parameter.attrib['DataType'] + ", nbdimensions="

                # look for number of dimensions
                for array in formal_arrays:
                    if array.attrib['Name'] == parameter.attrib['Name']:
                        nb_dim = len(array.find('Dimensions').findall('Dimension'))
                        dsl_text += str(nb_dim) + ")\n"
                        break

                # loop consumption
                dim = 0
                dsl_text += "{\n"
                for dim_pavage in parameter.find('Pavage').findall('DimPavage'):
                    dim += 1
                    loop_index_name = dim_pavage.find('RefLoopIndex').attrib['Name']
                    loop_inc = dim_pavage.find('RefLoopIndex').attrib['Inc']

                    # look for loop bounds and stride
                    for loop in loops:
                        if (loop.attrib['Index'] == loop_index_name):
                            loop_stride = loop.find('Stride').find('Symbolic').text
                            loop_lower_bound = loop.find('LowerBound').find('Symbolic').text
                            loop_upper_bound = loop.find('UpperBound').find('Symbolic').text
                            break

                    # format length
                    if loop_lower_bound == "0":
                        length_text = str(loop_upper_bound)
                    else:
                        length_text = str(loop_upper_bound) + "-" + str(loop_lower_bound)
                    # format amplitude
                    if loop_inc == "1":
                        amplitude_text = str(loop_stride)
                    elif loop_stride == "1":
                        amplitude_text = str(loop_inc)
                    else:
                        amplitude_text = str(loop_stride) + '*' + str(loop_inc)

                    dsl_text += "consume(dimension = " + str(dim) + ", "
                    dsl_text += "length = " + length_text  + ", "
                    dsl_text += "amplitude = " + amplitude_text + ")\n"

                dsl_text += "}\n"

        dsl_text += "}\n"
        dsl_text += "}\n"

        # save dsl file
        write_file(os.path.join(self.saveDir, "kernel.dsl"), dsl_text)
