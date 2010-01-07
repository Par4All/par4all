/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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
 * data structures that describe FREIA SPoC target.
 */

#ifndef   HWAC_FREIA_SPOC_H_
# define  HWAC_FREIA_SPOC_H_

#define HELPER "_helper_"

/* all SPoC hardware parts as a bitfield */
typedef enum {
  spoc_nothing = 0x00000000,
  // Inputs
  spoc_input_0 = 0x00000001,
  spoc_input_1 = 0x00000002,
  // Outputs
  spoc_output_0 = 0x00000004,
  spoc_output_1 = 0x00000008,
  // Morpho
  spoc_poc_0 = 0x00000010,
  spoc_poc_1 = 0x00000020,
  // ALU
  spoc_alu = 0x00000040,
  // ThreasHold
  spoc_th_0 = 0x00000100,
  spoc_th_1 = 0x00000200,
  // Measures
  spoc_measure_0 = 0x00004000,
  spoc_measure_1 = 0x00008000,
  // Links...
  spoc_alu_in0 = 0x01000000, // redundant with spoc_alu_op_t.use_in0
  spoc_alu_in1 = 0x02000000, // redundant with spoc_alu_op_t.use_in1
  spoc_alu_out0 = 0x04000000,
  spoc_alu_out1 = 0x08000000
  // others?
} spoc_hw_parts_t;

typedef enum {
  measure_none,
  measure_max,
  measure_max_coord,
  measure_min,
  measure_min_coord,
  measure_vol
} spoc_measure_t;

/* all ALU operations */
typedef enum {
  alu_unused,
  // arithmetic operations
  alu_add,
  alu_add_0cst,
  alu_add_1cst,
  alu_addsat,
  alu_addsat_0cst,
  alu_addsat_1cst,
  alu_sub_01,
  alu_sub_10,
  alu_sub_0cst,
  alu_sub_1cst,
  alu_subsat_01,
  alu_subsat_10,
  alu_subsat_0cst,
  alu_subsat_1cst,
  alu_abssub_01,
  alu_abssub_10,
  alu_abssub_0cst,
  alu_abssub_1cst,
  alu_mul,
  alu_mul_0cst,
  alu_mul_1cst,
  alu_div_01,
  alu_div_10,
  alu_div_0cst,
  alu_div_1cst,
  // comparisons
  alu_inf_01,
  alu_inf_0cst,
  alu_inf_1cst,
  alu_sup_01,
  alu_sup_0cst,
  alu_sup_1cst,
  // logical operations
  alu_and,
  alu_and_0cst,
  alu_and_1cst,
  alu_or,
  alu_or_0cst,
  alu_or_1cst,
  alu_xor,
  alu_xor_0cst,
  alu_xor_1cst,
  alu_not_0,
  alu_not_1,
  // array generation
  alu_copy_cst
  // and so on
} spoc_alu_t;

/* ALU operation full description */
typedef struct {
  spoc_alu_t op;  // operation
  spoc_alu_t flipped; // flipped call
  string setting; // macro the operation
  bool use_cst;   // whether a constant is needed
  bool use_in0;   // whether first input is used
  bool use_in1;   // whether second input is used
} spoc_alu_op_t;

typedef enum {
  spoc_poc_unused,
  spoc_poc_erode,
  spoc_poc_dilate
} spoc_poc_op_t;

typedef struct {
  spoc_poc_op_t op;
  uint32_t connectivity; // 0, 6, 8...
} spoc_poc_t;

/* description of a SPoC hardware configuration
 * should be precise enough to generate a full AIPO function.
 */
typedef struct {
  // what is used within the SPOC (vs strictly necessary???)
  uint32_t used;
  spoc_poc_t poc[2];
  spoc_alu_t alu;
  // threshold: nothing needed
  spoc_measure_t mes[2];
  // links: not sure...
} spoc_hw_t;

/* FREIA API function name -> SPoC hardware description (and others?)
 */
typedef struct {
  // function name
  string function_name;
  string compact_name; // something very short for graph nodes
  string commutator; // if the function is commutative
  // expected number of in/out arguments, that we should be able to use...
  int arg_img_out;  // 0 1
  int arg_img_in;   // 0 1 2
  // cst, bool, kernel...
  int arg_misc_out; // 0 1 2 3
  int arg_misc_in;  // 0 1 2 3
  // mmmh...
  string arg_out_types[3];
  string arg_in_types[3];
  // ...
  // corresponding hardware settings
  spoc_hw_t spoc;
} freia_api_t;

typedef enum {
  // important, in hardware order
  spoc_type_oth = -2, // for anything else...
  spoc_type_nop = -1, // used by copy?
  spoc_type_inp = 0, // used for input
  spoc_type_poc = 1,
  spoc_type_alu = 2,
  spoc_type_thr = 3,
  spoc_type_mes = 4,
  spoc_type_out = 5   // output...
} spoc_hardware_type;

#define FREIA_IMAGE_TYPE "freia_data2d"

/****************************************************** SPOC CODE GENERATION */

#define AIPO "freia_aipo_"

#define spoc_depth_prop "HWAC_SPOC_DEPTH"

// what about something simpler like "freia-spoc.h"?
#define FREIA_SPOC_INCLUDES			\
  "#include <freiaCommon.h>\n"			\
  "#include <freiaCoarseGrain.h>\n"		\
  "#include <spoc.h>\n"

#define FREIA_IMAGE FREIA_IMAGE_TYPE " * "

// ??? could/should be a property? what about asserts to check inputs?
#define FREIA_DEFAULT_BPP "16"

#define FREIA_SPOC_DECL						\
  "  spoc_instr si;\n"						\
  "  spoc_param sp;\n"						\
  "  spoc_reduction reduc;\n"					\
  "  freia_microcode mcode;\n"					\
  "  freia_dynamic_param dynparam;\n"				\
  "  freia_reduction_results redres;\n"				\
  "  freia_op_sel op = 0;\n"					\
  "  freia_op_param param;\n"					\
  "  freia_status ret;\n"					\
  "  int i;\n"							\
  "\n"								\
  "  // init pipe to nop\n"					\
  "  spoc_init_pipe(&si, &sp, " FREIA_DEFAULT_BPP ");\n"	\
  "\n"

#define FREIA_SPOC_CALL						\
  "\n"								\
  "  mcode.raw = (freia_ptr) &si;\n"				\
  "  mcode.size = sizeof(spoc_instr);\n"			\
  "\n"								\
  "  dynparam.raw = (freia_ptr) &sp;\n"				\
  "  dynparam.size = sizeof(spoc_param);\n"			\
  "\n"								\
  "  redres.raw = (freia_ptr) &reduc;\n"			\
  "  redres.size = sizeof(spoc_reduction);\n"			\
  "\n"								\
  "  ret = freia_cg_write_microcode(&mcode);\n"			\
  "  ret |= freia_cg_write_dynamic_param(&dynparam);\n"		\
  "\n"

#endif /* !HWAC_FREIA_SPOC_H_ */
