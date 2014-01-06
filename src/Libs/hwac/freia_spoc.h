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
 * data structures that describe FREIA SPoC target.
 */

#ifndef   HWAC_FREIA_SPOC_H_
# define  HWAC_FREIA_SPOC_H_

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
  spoc_alu_out1 = 0x08000000,
  // others?
  spoc_not_implemented = 0xffffffff
} spoc_hw_parts_t;

typedef enum {
  measure_none,
  measure_max,
  measure_max_coord,
  measure_min,
  measure_min_coord,
  measure_vol
} spoc_measure_t;

/* all SPoC ALU operations */
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
  alu_sub_cst0,
  alu_sub_cst1,
  alu_subsat_01,
  alu_subsat_10,
  alu_subsat_0cst,
  alu_subsat_1cst,
  alu_subsat_cst0,
  alu_subsat_cst1,
  alu_abssub,
  alu_abssub_0cst,
  alu_abssub_1cst,
  alu_mul,
  alu_mul_0cst,
  alu_mul_1cst,
  alu_div_01,
  alu_div_10,
  alu_div_0cst,
  alu_div_1cst,
  alu_div_cst0,
  alu_div_cst1,
  alu_log2_0,
  alu_log2_1,
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
  alu_copy_cst,
  // replace constant
  alu_repcst_0,
  alu_repcst_1
  // and so on
} spoc_alu_t;

/* ALU operation full description */
typedef struct {
  spoc_alu_t op;  // operation
  spoc_alu_t flipped; // flipped call
  string setting; // macro for the operation
  bool use_cst;   // whether a constant is needed
  bool use_in0;   // whether first input is used
  bool use_in1;   // whether second input is used
} spoc_alu_op_t;

typedef enum {
  spoc_poc_unused,
  spoc_poc_erode,
  spoc_poc_dilate,
  spoc_poc_conv
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

typedef enum {
  // important, in hardware order
  spoc_type_sni = -3, // spoc not implemented
  spoc_type_oth = -2, // for anything else...
  spoc_type_nop = -1, // no-operation, used by copy?
  spoc_type_inp = 0, // used for input
  spoc_type_poc = 1,
  spoc_type_alu = 2,
  spoc_type_thr = 3,
  spoc_type_mes = 4,
  spoc_type_out = 5   // output...
} spoc_hardware_type;

/****************************************************** SPOC CODE GENERATION */

#define spoc_depth_prop "HWAC_SPOC_DEPTH"

// what about something simpler like "freia-spoc.h"?
#define FREIA_SPOC_INCLUDES           \
  "#include <freiaCommon.h>\n"        \
  "#include <freiaMediumGrain.h>\n"   \
  "#include <freiaCoarseGrain.h>\n"   \
  "#include <spoc.h>\n"

#define FREIA_SPOC_DECL                 \
  "  spoc_instr si;\n"                  \
  "  spoc_param sp;\n"                  \
  "  freia_microcode mcode;\n"          \
  "  freia_dynamic_param dynparam;\n"   \
  "  freia_op_param param;\n"           \
  "  freia_status ret;\n"

#define FREIA_SPOC_CALL_START                \
  "\n"                                       \
  "  mcode.raw = (freia_ptr) &si;\n"         \
  "  mcode.size = sizeof(spoc_instr);\n"     \
  "\n"                                       \
  "  dynparam.raw = (freia_ptr) &sp;\n"      \
  "  dynparam.size = sizeof(spoc_param);\n"  \
  "\n"

#define FREIA_SPOC_CALL_REDUC                 \
  "  redres.raw = (freia_ptr) &reduc;\n"      \
  "  redres.size = sizeof(spoc_reduction);\n" \
  "\n"

#define FREIA_SPOC_CALL_END                             \
  "  ret = freia_cg_write_microcode(&mcode);\n"         \
  "  ret |= freia_cg_write_dynamic_param(&dynparam);\n" \
  "\n"

#endif /* !HWAC_FREIA_SPOC_H_ */
