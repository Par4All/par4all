/*

  $Id$

  Copyright 1989-2009 MINES ParisTech

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

#ifndef __SAVC_LOCAL_H__
#define __SAVC_LOCAL_H__

#include "sac_private.h"

/* deatomizer.c */
#define FUNC_TO_ATOMIZE_P(call) (type_functional_p(entity_type(call_function(call))) && \
			    (gen_length(functional_parameters(type_functional(entity_type(call_function(call))))) != 0))

extern entity hpfc_new_variable(entity, basic);

/* if conversion */
#define IF_TO_CONVERT "c IF_TO_CONVERT\n"
#define IF_CONV_TO_COMPACT "c IF_CONV_TO_COMPACT\n"

entity get_function_entity(string name);

basic get_basic_from_array_ref(reference ref);

void saCallReplace(call c, reference ref, entity next);

list da_process_list(list seq, bool repOnlyInIndex, bool (*stat_to_process)(statement ));

void init_dep_graph(graph dg);

#define ENTITY_FUNCTION_P(f) (type_functional_p(entity_type(f)) && \
			    (gen_length(functional_parameters(type_functional(entity_type(f)))) != 0))

#define SIMD_PHI_NAME "SIMD_PHI"

/* simd_loop_const_elim.c */

#define SIMD_LOAD_NAME "SIMD_LOAD"
#define SIMD_LOAD_SIZE 9
#define SIMD_SAVE_NAME "SIMD_SAVE"
#define SIMD_SAVE_SIZE 9
#define SIMD_GEN_LOAD_NAME "SIMD_LOAD_GENERIC"
#define SIMD_GEN_LOAD_SIZE 17
#define SIMD_GEN_SAVE_NAME "SIMD_SAVE_GENERIC"
#define SIMD_GEN_SAVE_SIZE 17
#define SIMD_CONS_LOAD_NAME "SIMD_LOAD_CONSTANT"
#define SIMD_CONS_LOAD_SIZE 18
#define SIMD_CONS_SAVE_NAME "SIMD_SAVE_CONSTANT"
#define SIMD_CONS_SAVE_SIZE 18

#define SIMD_NAME "SIMD_"
#define SIMD_SIZE 5

list expression_to_proper_effects(expression e);

#define STATEMENT_INFO_NEWGEN_DOMAIN SIMDSTATEMENTINFO_NEWGEN_DOMAIN
#define gen_STATEMENT_INFO_cons gen_SIMDSTATEMENTINFO_cons

/* simd_loop_unroll.c */

#define SIMD_COMMENT "SIMD_COMMENT_"



#endif /*__SAVC_LOCAL_H__*/
