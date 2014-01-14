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
typedef hash_table operator_id_sons ;
#include "sac_private.h"
#include "dg.h"

typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;

#include "graph.h"


#define FUNC_TO_ATOMIZE_P(call) (\
        type_functional_p(entity_type(call_function(call))) && \
	    (gen_length(module_functional_parameters(call_function(call))) != 0) && \
        (!ENTITY_DEREFERENCING_P(call_function(call))) && \
        (!ENTITY_POINT_TO_P(call_function(call))) && \
        (!ENTITY_FIELD_P(call_function(call)))\
        )

/* if conversion */
#define IF_TO_CONVERT "PIPS IF_TO_CONVERT"
#define IF_CONV_TO_COMPACT "PIPS IF_CONV_TO_COMPACT"


#define ENTITY_FUNCTION_P(f) (type_functional_p(entity_type(f)) && \
			    (gen_length(module_functional_parameters(f)) != 0))

/* simd_loop_const_elim.c */

#define SIMD_MASKED_SUFFIX "_MASKED"
#define SIMD_GENERIC_SUFFIX "_GENERIC"
#define SIMD_CONSTANT_SUFFIX "_CONSTANT"
#define SIMD_BROADCAST_SUFFIX "_BROADCAST"
#define SAC_PADDING_ENTITY_NAME "PADDING_VALUE"

#define STATEMENT_INFO_NEWGEN_DOMAIN SIMDSTATEMENTINFO_NEWGEN_DOMAIN
#define gen_STATEMENT_INFO_cons gen_SIMDSTATEMENTINFO_cons


#define CHECK_VECTORELEMENT(ve) do {\
    pips_assert("vector Index seems legal",vectorElement_vectorIndex(ve) >= 0 && vectorElement_vectorIndex(ve) < simdStatementInfo_nbArgs(vectorElement_statement(ve)));\
    } while(0)


/* symbols exported by the parser */
extern FILE *patterns_yyin;
extern int patterns_yyparse();
extern int patterns_yylex();
extern void patterns_yyerror(const char*);

