/*

  $Id$

  Copyright 1989-2010 MINES ParisTech
  Copyright 2010 HPC Project

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

/* pv_context is a structure holding the methods to use during
   pointer values analyses */
typedef struct {

  /* ANALYSIS CONTROL */
  bool initial_pointer_values_p; /* set to true for an initial module analysis */

  /* PIPSDEBM INTERFACES */
  statement_cell_relations (*db_get_pv_func)(const char *);
  void (*db_put_pv_func)(const char * , statement_cell_relations);
  list (*db_get_in_pv_func)(const char *);
  void (*db_put_in_pv_func)(const char * , list);
  list (*db_get_out_pv_func)(const char *);
  void (*db_put_out_pv_func)(const char * , list);
  list (*db_get_initial_pv_func)(const char *);
  void (*db_put_initial_pv_func)(const char * , list);
  list (*db_get_program_pv_func)();
  void (*db_put_program_pv_func)(list);

/*   statement_cell_relations (*db_get_gen_pv_func)(char *); */
/*   void (*db_put_gen_pv_func)(char * , statement_cell_relations); */
/*   statement_effects (*db_get_kill_pv_func)(char *); */
/*   void (*db_put_kill_pv_func)(char * , statement_effects); */

  list (*make_pv_from_effects_func)(effect, effect, cell_interpretation, list);

  /* COMPARISON OPERATORS */
  bool (*cell_preceding_p_func)(cell, descriptor,
				cell, descriptor ,
				transformer, bool, bool *);

  /* TRANSLATION OPERATORS */
  void (*cell_reference_with_value_of_cell_reference_translation_func)
  (reference , descriptor, reference , descriptor, int, reference *, descriptor *, bool *);
  void (*cell_reference_with_address_of_cell_reference_translation_func)
  (reference , descriptor, reference , descriptor, int, reference *, descriptor *, bool *);

  /* UNARY OPERATORS */
  cell_relation (*pv_composition_with_transformer_func)(cell_relation, transformer );

  /* BINARY OPERATORS */
  list (*pvs_must_union_func)(list, list);
  list (*pvs_may_union_func)(list, list);
  bool (*pvs_equal_p_func)(list, list);

  /* STACKS */
  stack stmt_stack;
} pv_context;

/* pv_results is a structure holding the different results of an expression pointer values analysis */
typedef struct {
  list l_out; /* resulting pointer_values */
  list result_paths; /* resulting pointer path of the expression evaluation */
  list result_paths_interpretations; /* interpretation of the resulting pointer path */

} pv_results;

#define pips_debug_pv_results(level, message, pv_res) \
  ifdebug(level) { pips_debug(level, "%s\n", message); \
  print_pv_results(pv_res);}
