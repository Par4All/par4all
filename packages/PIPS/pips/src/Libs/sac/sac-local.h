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

#ifndef __SAVC_LOCAL_H__
#define __SAVC_LOCAL_H__

#include "sac_private.h"

/* deatomizer.c */
#define FUNC_TO_ATOMIZE_P(call) (type_functional_p(entity_type(call_function(call))) && \
			    (gen_length(functional_parameters(type_functional(entity_type(call_function(call))))) != 0))

extern entity hpfc_new_variable(entity, basic);

/* if conversion */
#define IF_TO_CONVERT "PIPS IF_TO_CONVERT"
#define IF_CONV_TO_COMPACT "PIPS IF_CONV_TO_COMPACT"

entity get_function_entity(string name);

basic get_basic_from_array_ref(reference ref);

void saCallReplace(call c, reference ref, entity next);

list da_process_list(list seq, bool repOnlyInIndex, bool (*stat_to_process)(statement ));

void init_dep_graph(graph dg);

#define ENTITY_FUNCTION_P(f) (type_functional_p(entity_type(f)) && \
			    (gen_length(functional_parameters(type_functional(entity_type(f)))) != 0))

#define SIMD_PHI_NAME "PHI"

/* simd_loop_const_elim.c */

#define SIMD_LOAD_NAME "SIMD_LOAD"
#define SIMD_SAVE_NAME "SIMD_SAVE"
#define SIMD_MASKED_LOAD_NAME "SIMD_MASKED_LOAD"
#define SIMD_MASKED_SAVE_NAME "SIMD_MASKED_SAVE"
#define SIMD_GEN_LOAD_NAME "SIMD_LOAD_GENERIC"
#define SIMD_GEN_SAVE_NAME "SIMD_SAVE_GENERIC"
#define SIMD_CONS_LOAD_NAME "SIMD_LOAD_CONSTANT"
#define SIMD_CONS_SAVE_NAME "SIMD_SAVE_CONSTANT"
#define SAC_PADDING_ENTITY_NAME "PADDING_VALUE"

#define SIMD_NAME "SIMD_"

list expression_to_proper_effects(expression e);

#define STATEMENT_INFO_NEWGEN_DOMAIN SIMDSTATEMENTINFO_NEWGEN_DOMAIN
#define gen_STATEMENT_INFO_cons gen_SIMDSTATEMENTINFO_cons


#define CHECK_VECTORELEMENT(ve) do {\
    pips_assert("vector Index seems legal",vectorElement_vectorIndex(ve) >= 0 && vectorElement_vectorIndex(ve) < simdStatementInfo_nbArgs(vectorElement_statement(ve)));\
    } while(0)

/* simd_loop_unroll.c */

#define SIMD_COMMENT "SIMD_COMMENT_"



#endif /*__SAVC_LOCAL_H__*/
