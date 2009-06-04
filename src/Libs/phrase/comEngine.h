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
#ifndef COMENGINE_DEFS
#define COMENGINE_DEFS

#define FIFO_NUM_MAX 100

#define R_EFFECT "R_EFFECT"
#define W_EFFECT "W_EFFECT"
#define RW_EFFECT "RW_EFFECT"

#define GEN_GET_FIFO "GEN_GET_FIFO"
#define GEN_WRITE_FIFO "GEN_WRITE_FIFO"
#define READ_FIFO "READ_FIFO"
#define WRITE_FIFO "WRITE_FIFO"
#define GEN_GET_BUFF "GEN_GET_BUFF"
#define GEN_WRITE_BUFF "GEN_WRITE_BUFF"
#define WAIT_FOR_NEXT_STEP "WAIT_FOR_NEXT_STEP"
#define GEN_LOAD_MMCD "GEN_LOAD_MMCD"
#define GEN_STORE_MMCD "GEN_STORE_MMCD"
#define START_HRE "START_HRE"
#define WAIT_FOR_HRE "WAIT_FOR_HRE"

list identify_analyzed_statements_to_distribute (statement stat);

bool comEngine_feasability (statement externalized_code, graph dg);

statement comEngine_generate_procCode(statement externalized_code,
				      list l_in, list l_out);

statement comEngine_generate_HRECode(statement externalized_code,
				     string new_module_name,
				     list l_in, list l_out, list l_params, list l_priv,
				     string module_name, int hreMemSize);

void create_HRE_module(string new_module_name,
		       string module_name,
		       statement stat, entity new_fun);

expression entity_symbolic_to_expression(entity ent);

void add_private_variable_to_module (reference ref,
				     entity module,
				     statement stat,
				     string new_module_name);

statement comEngine_generate_code(statement stat);

entity get_function_entity(string name);

void comEngine_replace_reference_in_stat(statement stat,
					 reference ref, expression new);

statement make_transStat(statement stat, entity newOuterInd,
			 entity transferSize, expression bufferSizeExp);

list reference_indices_entity_list(reference ref);

bool code_has_write_eff_ref_p(reference ref, statement stat);

statement HRE_distribute(statement stat, string new_module_name, string module_name);

void replace_array_ref_with_fifos(list lRef, expression buffIndExp,
				  entity ind, statement * newStat);

basic get_basic_from_array_ref(reference ref);

list comEngine_expression_to_reference_list(expression e, list lr);


statement generate_stat_from_ref_list_HRE(list lRef, statement stat);

list generate_stat_from_ref_list_HRE_list(list lRef, list lInStats);

void get_supportedRef_HRE(statement stat,
			  list * lSupportedRef, list * lUnSupportedRef);

void process_replacement_HRE(list lRef, expression buffIndExp,
			     statement * stat);

list process_replacement_HRE_OutRef(list lRef, list lStats);

statement get_call_stat_HRE(statement stat);

list make_loop_lStats_HRE(statement stat, entity transferSize,
			  statement innerStat, entity newOuterInd,
			  list lSupportedRef, hash_table htOffset,
			  expression bufferSizeExp);

list process_gLoopToSync_HRE(statement stat, list lInStats);



statement generate_stat_from_ref_list_proc(list lRef, list lToggleEnt,
					   statement newStat);

list generate_stat_from_ref_list_proc_list(list lRef, list lInStats);

void process_innerStat1_proc(statement stat, entity oldInd,
			     entity newOuterInd, entity newInnerInd);

void get_supportedRef_proc(statement stat, hash_table htOffset,
			   list * lSupportedRef, list * lUnSupportedRef);

statement get_call_stat_proc(statement stat);

list make_loop_lStats_proc(statement stat, entity transferSize,
			   statement innerStat, entity newOuterInd,
			   list lSupportedRef, hash_table htOffset,
			   expression bufferSizeExp);

list add_index_out_effect_proc(loop curLoop, list lStats);

list copy_out_to_in_proc(list lOtherRef, statement stat,
			 list lInStats);

list process_gLoopToSync_proc(statement stat, list lInStats);

void create_realFifo_proc(statement stat, list lRef);

statement make_exec_mmcd();

statement generate_code_function(statement stat, bool bCalledFromSeq);

bool has_call_stat_inside(statement stat);

statement comEngine_opt_loop_interchange(statement stat, statement innerStat,
					 entity newInd);

statement make_wait_step_statement();

statement make_loop_step_stat(statement stat, entity newOuterInd);

statement make_init_newInd_stat(statement stat, entity newInd);

expression get_fifo_from_ref(reference ref);

entity get_HRE_buff_ent_from_ref(reference ref);

entity make_new_C_scalar_variable_with_prefix(string prefix,
					      entity module,
					      statement stat,
					      basic b);

statement generate_code_test_HRE(statement stat);

statement generate_code_test_proc(statement stat);

statement make_toggle_mmcd(entity ent);

statement make_step_inc_statement(int incNum);

entity comEngine_make_new_scalar_variable(string prefix,
					  basic bas);

statement make_toggle_inc_statement(entity toggleEnt, int val);

statement make_toggle_init_statement(entity toggleEnt);
#endif
