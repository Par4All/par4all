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
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
/* -- control.c
 *
 * package atomizer :  Alexis Platonoff, juillet 91
 * --
 *
 * Functions for the manipulations and modifications of the control graph.
 */

#include "local.h"

/*============================================================================*/
/* void modify_blocks(control c): Adds a node to the control graph when the
 * statement of "c" is not an "instruction_block".
 *
 * This node is added just before "c". "c" predecessor became the new node.
 * The latter's predecessors are those of "c", its successor is "c".
 *
 * "l_inst" keep the reference to the instructions for which a new node is
 * added. It is a global variable that will be used when we will generate
 * new statements (see atomizer_of_statement(), in atomizer.c).
 *
 * Called functions :
 *       _ make_empty_statement() : ri-util/statement.c
 */
void modify_blocks(c)
control c;
{
extern list l_inst;

control nc;
instruction inst = statement_instruction(control_statement(c));

if (instruction_tag(inst) != is_instruction_block)
  {
  if (! instruction_in_list_p(inst, l_inst))
    {
    nc = make_control(control_statement(c), NIL, NIL);
    control_statement(c) = make_empty_statement();

    control_successors(nc) = control_successors(c);
    control_predecessors(nc) = CONS(CONTROL, c, NIL);

    control_successors(c) = CONS(CONTROL, nc, NIL);

    l_inst = CONS(INSTRUCTION, inst, l_inst);
    }
  }
}



/*============================================================================*/
/* void atom_get_blocs(control c, cons **l): Almost the get_blocs() of 
 * the file control/control.h; the only modification is the call to
 * modify_blocks() which adds a node to the control graph that will contain
 * the instructions created by the translation of the statement contained
 * in "c".
 */
void atom_get_blocs( c, l )
control c ;
cons **l ;
{
MAPL( cs, {if( CONTROL( CAR( cs )) == c ) return ;}, *l ) ;
*l = CONS( CONTROL, c, *l ) ;

modify_blocks(c);

MAPL( cs, {atom_get_blocs( CONTROL( CAR( cs )), l );},
      control_successors( c )) ;

MAPL( ps, {atom_get_blocs( CONTROL( CAR( ps )), l );},
      control_predecessors( c )) ;
}



/*============================================================================*/
/* control find_control_block(control c): Returns the current control found
 * in the static variable "cont_block".
 *
 * The use of this function is: 
 *      _ first, in order to initialize the static "cont_block" this function is
 *        called with the control we want to memorize. 
 *      _ then, when we need to get this control, this function is called
 *        with the argument "control_undefined".
 *
 * This is used by atomizer_of_statement() when we generate new statements.
 */
control find_control_block(c)
control c;
{
static control cont_block;

if(c != control_undefined)
  {
  instruction inst = statement_instruction(control_statement(c));
  if(instruction_tag(inst) != is_instruction_block)
    cont_block = CONTROL(CAR(control_predecessors(c)));
  else
    cont_block = control_undefined;
  }

return(cont_block);
}

