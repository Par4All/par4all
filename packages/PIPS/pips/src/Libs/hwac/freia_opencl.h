/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

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
  "#include <CL/opencl.h>\n"                                  \
  "#include \"freia.h\"\n"                                    \
  "#include \"private/freia_opencl_runtime.h\"\n"             \
  "\n"

// raw include for opencl code generation
#define FREIA_OPENCL_CL_INCLUDES                \
  "#include <freia_opencl_runtime.hcl>"

#define FREIA_SIGMAC_INCLUDES                   \
  "// sigmac includes: not defined yet\n"

// information about OpenCL handling of an operation
typedef struct {
  // whether it can be merged
  bool mergeable;
  // special case for some kernel-based operations
  bool mergeable_kernel;
  // if mergeable, the name of the pixel operation macro
  string macro;
  // and maybe the initial value:
  string init; // not needed?
} opencl_hw_t;

#endif // !HWAC_FREIA_OPENCL_H_
