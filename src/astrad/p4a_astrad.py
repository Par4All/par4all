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

        #print("ASTRAD: call to save_dsl_file \n")
        dsl_file_name = os.path.join(self.saveDir, self.moduleName + '.dsl')
        #print("file name :" + dsl_file_name + "\n")

        # file name is *.p4a.c: split it twice to recover base name
        (base, ext1) = os.path.splitext(self.outputFileName)
        (base, ext2) = os.path.splitext(base)
        new_file = base  + '_' + self.outputDialect + ext2 + ext1

        f = open(dsl_file_name, 'w')

        content = "optimizeResult request_optimize_"
        content += time.strftime('%y_%m_%d',time.localtime())
        content +="\n"
        content +="{\n"
        content += "sourceName = " + new_file +";\n"
        content += "methodName = " + self.moduleName + ";\n"
        content += "errorCode = " + self.errorCode + ";\n"
        content += "kernelFileName = kernel.dsl;\n"
        content += "type = " + self.outputDialect + ";\n"

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
        dsl_text += ") {\n\n"

        # parameters
        first_parameter = True
        dsl_text += 'parameters(\n'
        for parameter in task_parameters:
            if parameter.attrib['ArrayP'] == "FALSE":
                if first_parameter:
                    first_parameter = False
                else:
                    dsl_text += ',\n'
                parameter_name = parameter.attrib['Name']
                dsl_text += parameter_name
                dsl_text += "(dataType=" + parameter.attrib['DataType']

                for usage in parameter.findall('TaskParameterUsedFor'):
                    # if the parameter is used in the array pattern,
                    # then don't generate array info
                    # (commented out at Thales request 2013/10/24
                    # not removed in case it may change again)
                    array_name = usage.attrib['ArrayName']
                    found = False
                    # for other_parameter in task_parameters:
                    #     if other_parameter.attrib['ArrayP'] == "TRUE" and other_parameter.attrib['Name'] == array_name:
                    #         for dim_pattern in other_parameter.find('Pattern'):
                    #             if (dim_pattern.tag == 'DimPattern') and (parameter_name in dim_pattern.find('Length').find('Symbolic').text):
                    #                 found = True
                    #                 break
                    #     if found:
                    #         break
                    if not found:
                        dsl_text += ', as_dimension(' + array_name + ',' + str(int(usage.attrib['Dim']) -1) +')'
                dsl_text += ')'
        dsl_text += ");\n"

        # I/Os

        ## first scan task_parameters to check if there are input arrays
        ## if there is no input array, length fields must be systematically
        ## used instead of ALL for paving.
        input_arrays = False
        for parameter in task_parameters:
            access_mode = parameter.attrib['AccessMode']
            if parameter.attrib['ArrayP'] == "TRUE":
                if access_mode == "USE":
                    input_arrays = True
                    break

        ## then generate consumption information
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

                # look for number of array dimensions
                for array in formal_arrays:
                    if array.attrib['Name'] == parameter.attrib['Name']:
                        nb_dim = len(array.find('Dimensions').findall('Dimension'))
                        dsl_text += str(nb_dim) + ")\n"
                        break

                dsl_text += "{\n"

                # origin
                dim = nb_dim
                for dim_pattern in parameter.find('Pattern'):
                    dim -= 1 # array dimensions are in reversed order in dsl file
                    if dim_pattern.tag == 'DimUnitPattern':
                        length_text = "1"
                        amplitude_text = "0"
                    elif dim_pattern.tag == 'DimPattern':
                        length_text = dim_pattern.find('Length').find('Symbolic').text
                        amplitude_text = dim_pattern.find('Offset').find('Symbolic').text
                    else:
                        p4a_util.die("ASTRAD PostProcessor: unexpected DimPattern tag")

                    dsl_text += "origin(dimension=" + str(dim) + ", "
                    dsl_text += "amplitude=" + amplitude_text + ");\n"

                # fitting
                dim = nb_dim
                for dim_pattern in parameter.find('Pattern'):
                    dim -= 1 # array dimensions are in reversed order in dsl file
                    if dim_pattern.tag == 'DimUnitPattern':
                        length_text = "1"
                        amplitude_text = "1"
                    elif dim_pattern.tag == 'DimPattern':
                        length_text = dim_pattern.find('Length').find('Symbolic').text
                        amplitude_text = dim_pattern.find('Stride').find('Symbolic').text
                    else:
                        p4a_util.die("ASTRAD PostProcessor: unexpected DimPattern tag")

                    dsl_text += "consume(dimension=" + str(dim) + ", "
                    dsl_text += "length=" + length_text  + ", "
                    dsl_text += "amplitude=" + amplitude_text + ");\n"

                # paving - scan loops in order
                for index, loop in enumerate(loops):
                    loop_index_name = loop.attrib['Index']
                    loop_stride = loop.find('Stride').find('Symbolic').text
                    loop_lower_bound = loop.find('LowerBound').find('Symbolic').text
                    loop_upper_bound = loop.find('UpperBound').find('Symbolic').text

                    # scan array paving dimensions
                    dim = nb_dim
                    found = False
                    for dim_pavage in parameter.find('Pavage').findall('DimPavage'):
                        dim -= 1 # array dimensions are in reversed order in dsl file
                        # dimPavage may be empty
                        if dim_pavage.find('RefLoopIndex') is not None:
                            ref_loop_index_name = dim_pavage.find('RefLoopIndex').attrib['Name']
                            loop_inc = dim_pavage.find('RefLoopIndex').attrib['Inc']
                            if ref_loop_index_name == loop_index_name:
                                found = True
                                break

                    if found:

                        # format length
                        ## ALL if there are input arrays and loop upper bound is not numeric
                        ## upper_bound - lower_bound + 1 otherwise
                        all_length = input_arrays and not loop.find('UpperBound').find('Numeric')
                        if (not all_length) :
                            if loop_lower_bound == "0":
                                length_text = str(loop_upper_bound) + "+1"
                            else:
                                length_text = str(loop_upper_bound) + "-" + str(loop_lower_bound) + "+1"

                        # format amplitude
                        if loop_inc == "1":
                            amplitude_text = str(loop_stride)
                        elif loop_stride == "1":
                            amplitude_text = str(loop_inc)
                        else:
                            amplitude_text = str(loop_stride) + '*' + str(loop_inc)
                    else:
                        length_text = "All"
                        amplitude_text = "0"

                    dsl_text += ("\t" * index) + "OnIter(consume(dimension = " + str(dim) + ", "
                    if all_length:
                        dsl_text += 'All, '
                    else:
                        dsl_text += "length=" + length_text  + ", "
                    dsl_text += "amplitude=" + amplitude_text + ")"

                    if index < len(loops)-1:
                        dsl_text += "\n"

                dsl_text += (")" * len(loops)) + ";\n"
                dsl_text += "}\n"

        # timing: fake information, not used but necessary
        dsl_text += "timing (prolog=0,core=-1,epilog=0)\n"

        dsl_text += "}\n"
        dsl_text += "}\n"

        # save dsl file
        write_file(os.path.join(self.saveDir, "kernel.dsl"), dsl_text)

    def rename_module(self):
        old_file =  os.path.join(self.saveDir, self.outputFileName )
        # file name is *.p4a.c: split it twice to recover base name
        (base, ext1) = os.path.splitext(self.outputFileName)
        (base, ext2) = os.path.splitext(base)
        new_file = os.path.join(self.saveDir, base  + '_' + self.outputDialect + ext2 + ext1)
        new_module_name = self.moduleName + '_' + self.outputDialect
        old_module_kernel_name = self.moduleName + '_kernel'
        new_module_kernel_name = self.moduleName + '_' + self.outputDialect + '_kernel'

        os.rename(old_file, new_file)
        content = read_file(new_file)
        content = content.replace (self.moduleName + '(', new_module_name + '(')
        content = content.replace (old_module_kernel_name + '(', new_module_kernel_name + '(')

        write_file(new_file, content)
