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

typedef struct {
  /* PIPSDEBM INTERFACES */
  statement_cell_relations (*db_get_pv_func)(char *);
  void (*db_put_pv_func)(char * , statement_cell_relations);
  statement_cell_relations (*db_get_gen_pv_func)(char *);
  void (*db_put_gen_pv_func)(char * , statement_cell_relations);
  statement_effects (*db_get_kill_pv_func)(char *);
  void (*db_put_kill_pv_func)(char * , statement_effects);
  cell_relation (*make_pv_from_effects_func)(effect, effect, cell_interpretation);
  void (*cell_reference_with_value_of_cell_reference_translation_func)
  (reference , descriptor, reference , descriptor, int, reference *, descriptor *, bool *);
  void (*cell_reference_with_address_of_cell_reference_translation_func)
  (reference , descriptor, reference , descriptor, int, reference *, descriptor *, bool *);
} pv_context;
