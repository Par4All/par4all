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

#define opencl_merge_prop "HWAC_OPENCL_MERGE_OPERATIONS"

// types
#define OPENCL_IMAGE "opencl image "
#define OPENCL_PIXEL "register pixel "

// includes for generated opencl helper
#define FREIA_OPENCL_INCLUDES                   \
  "// freia opencl includes\n"

// information about OpenCL handling of an operation
typedef struct {
  // whether it can be merged
  bool mergeable;
  // if mergeable, the name of the pixel operation macro
  string macro;
} opencl_hw_t;

#endif // !HWAC_FREIA_OPENCL_H_
