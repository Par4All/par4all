/*

  $Id$

  Copyright 1989-2011 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/

/*
 * data structures that describe FREIA OpenCL target.
 */

#ifndef HWAC_FREIA_OPENCL_H_
#define HWAC_FREIA_OPENCL_H_

// property name, whether to attempt to merge operations
#define opencl_merge_prop "HWAC_OPENCL_MERGE_OPERATIONS"

// OpenCL types for the generated code
#define OPENCL_PIXEL "PIXEL "
#define OPENCL_IMAGE "GLOBAL " OPENCL_PIXEL "* "

// includes for OpenCL helper
#define FREIA_OPENCL_INCLUDES                                 \
  "// FREIA OpenCL includes\n"                                \
  "#include <CL/OpenCL.h>\n"                                  \
  "#include <private/freiaGatewayOpenCL.h>\n"                 \
  "\n"                                                        \
  "// support function, should be in FREIA runtime?\n"        \
  "static freia_status get_compiled_opencl(\n"                \
  "  const char * source, // opencl source\n"                 \
  "  const char * kname, // expected kernel name\n"           \
  "  const char * option // compiler option\n"                \
  "  cl_kernel * kernel) // where to put the kernel\n"        \
  "{\n"                                                       \
  "  freia_status err;\n"                                     \
  "  char * opts;\n"                                          \
  "  int n = asprint(&ops, \"%s %s \","                       \
                   " frclTarget.compileOptions, option);\n"   \
  "  if (n==-1) return FREIA_UNLISTED_ERROR;\n"               \
  "  cl_program prg =\n"                                      \
  "    freia_op_compile_string_opencl(source, ops, &err);\n"  \
  "  if (err != FREIA_OK) return err;\n"                      \
  "  *kernel = freia_op_get_kernel(prg, kname, &err);\n"      \
  "  return err;\n"                                           \
  "}\n"

#define FREIA_OPENCL_CL_INCLUDES                \
  "// FREIA OpenCL specific includes\n"         \
  "#include <freiaAtomicOpCommonOpenCL.hcl>\n"

// information about OpenCL handling of an operation
typedef struct {
  // whether it can be merged
  bool mergeable;
  // if mergeable, the name of the pixel operation macro
  string macro;
} opencl_hw_t;

#endif // !HWAC_FREIA_OPENCL_H_
