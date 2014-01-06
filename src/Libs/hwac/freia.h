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
 * FREIA common stuff.
 */
#ifndef HWAC_FREIA_H_
#define HWAC_FREIA_H_

// additionnal hardware specific definitions
#include "freia_spoc.h"
#include "freia_terapix.h"
#include "freia_opencl.h"
#include "freia_sigmac.h"

#define HELPER "_helper_"

// do not require NULL at the end of a vararg.
#define cat(args...) concatenate(args , NULL)
#define sb_cat(args...) string_buffer_cat(args , NULL)

// macros for freia stuff
#define FREIA_IMAGE_TYPE "freia_data2d"
#define FREIA_OUTPUT     "freia_common_tx_image"
#define FREIA_ALLOC      "freia_common_create_data"
#define FREIA_FREE       "freia_common_destruct_data"

#define AIPO "freia_aipo_"
#define FREIA_IMAGE FREIA_IMAGE_TYPE " * "
#define FREIA_DEFAULT_BPP get_int_property("FREIA_PIXEL_SIZE")
#define FREIA_DEFAULT_HEIGHT get_int_property("FREIA_IMAGE_HEIGHT")
#define FREIA_DEFAULT_WIDTH get_int_property("FREIA_IMAGE_WIDTH")

// check the hardware target
#define freia_aipo_p(s) same_string_p((s), "aipo")
#define freia_spoc_p(s) same_string_p((s), "spoc")
#define freia_terapix_p(s) same_string_p((s), "terapix")
#define freia_opencl_p(s) same_string_p((s), "opencl")
#define freia_sigmac_p(s) same_string_p((s), "sigmac")

#define freia_valid_target_p(s)                               \
  (freia_spoc_p(s) || freia_terapix_p(s) ||                   \
   freia_aipo_p(s) || freia_opencl_p(s) || freia_sigmac_p(s))

/* FREIA API function name -> SPoC hardware description (and others?)
 */
typedef struct {
  // function name
  string function_name;
  // something very short for graph nodes
  string compact_name;
  // if the function is commutative/has a corresponding commuted version
  string commutator;
  // expected number of in/out arguments, that we should be able to use...
  unsigned int arg_img_out;  // 0 1
  unsigned int arg_img_in;   // 0 1 2
  // cst, bool, kernel...
  unsigned int arg_misc_out; // 0 1 2 3
  unsigned int arg_misc_in;  // 0 1 2 3
  // mmmh... hardcoded to max 3 in or outs (threshold, global_*_coord...)
  string arg_out_types[3];
  string arg_in_types[3];
  // ...
  // corresponding hardware settings
  spoc_hw_t spoc;
  terapix_hw_t terapix;
  opencl_hw_t opencl;
  sigmac_hw_t sigmac;
  // others? vhdl?:-)
} freia_api_t;

#define dagvtx_freia_api(v) get_freia_api(vtxcontent_opid(dagvtx_content(v)))

#endif /* !HWAC_FREIA_H_ */
