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
/* Interface with pipsmake for interactive loop transformations: loop
 * interchange, hyperplane method,...
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "constants.h"
#include "misc.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "database.h"
#include "pipsdbm.h"
#include "resources.h"
#include "text-util.h"

#include "control.h"
#include "conversion.h"
#include "properties.h"
#include "pipsmake.h"

#include "transformations.h"

entity selected_label;

bool selected_loop_p(statement s)
{
    /* SG+EC 2010:
        The loop_label(statement_loop(s)) is kept for compatibility reasons
        but is invalid and should eventually be removed */
  return statement_loop_p(s) &&
          (statement_label(s) == selected_label || loop_label(statement_loop(s)) == selected_label);
}

bool interactive_loop_transformation
(const char* module_name,
 statement (*loop_transformation)(list,bool (*)(statement))
)
{
  const char *lp_label=NULL;
  entity module = module_name_to_entity(module_name);
  statement s = statement_undefined;
  bool return_status = false;

  pips_assert("interactive_loop_transformation", entity_module_p(module));

  /* DBR_CODE will be changed: argument "pure" should take false but
     this would be useless since there is only *one* version of code;
     a new version will be put back in the data base after transforming
     the loops */
  s = (statement) db_get_memory_resource(DBR_CODE, module_name, true);
  set_current_module_entity(module);
  set_current_module_statement(s);

  /* Get the loop label from the user */
      lp_label = get_string_property_or_ask("LOOP_LABEL","Which loop do you want to transform?\n"
                            "(give its label): ");
      if( string_undefined_p( lp_label ) )
	{
	  pips_user_error("please set %s  property to a valid label\n");
	}

  if(lp_label)
    {
      selected_label = find_label_entity(module_name, lp_label);
      if (entity_undefined_p(selected_label)) {
	pips_user_error("loop label `%s' does not exist\n", lp_label);
      }

      debug_on("INTERACTIVE_LOOP_TRANSFORMATION_DEBUG_LEVEL");

      look_for_nested_loop_statements(s, loop_transformation, selected_loop_p);

      debug_off();

      /* Reorder the module, because new statements have been generated. */
      module_reorder(s);

      DB_PUT_MEMORY_RESOURCE(DBR_CODE,
			     strdup(module_name),
			     (char*) s);
      return_status = true;
    }
  reset_current_module_entity();
  reset_current_module_statement();

  return return_status;
}

typedef struct {
    list loops;
    bool new_label_created;
} flag_loop_param_t;

static void flag_loop(statement st, flag_loop_param_t *flp)
{
  instruction i = statement_instruction(st);
  if(instruction_loop_p(i) && entity_empty_label_p(statement_label(st)))
    {
        statement_label(st) = make_new_label(get_current_module_entity());
        flp->new_label_created=true;
    }
  if( !get_bool_property("FLAG_LOOPS_DO_LOOPS_ONLY")
      && instruction_forloop_p(i))
    {
        if(entity_empty_label_p(statement_label(st))) {
            statement_label(st)=make_new_label(get_current_module_entity());
            flp->new_label_created=true;
        }
      flp->loops=CONS(STRING,strdup(entity_user_name(statement_label(st))),flp->loops);
    }
}


bool print_loops(const char* module_name)
{
  /* prelude */
  set_current_module_entity(module_name_to_entity( module_name ));
  set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, true) );
  callees loops = (callees)db_get_memory_resource(DBR_LOOPS, module_name, true);

  /* do the job */
  {
    string local = db_build_file_resource_name(DBR_LOOPS_FILE, module_name, ".loops");
    string dir = db_get_current_workspace_directory();
    string full = strdup(concatenate(dir,"/",local, NULL));
    free(dir);
    FILE * fp = safe_fopen(full,"w");
    text r = make_text(NIL);
    FOREACH(STRING,s,callees_callees(loops))
      {
	ADD_SENTENCE_TO_TEXT(r,MAKE_ONE_WORD_SENTENCE(0,s));
      }
    print_text(fp,r);
    free_text(r);
    safe_fclose(fp,full);
    free(full);
    DB_PUT_FILE_RESOURCE(DBR_LOOPS_FILE, module_name, local);
  }

  /*postlude*/
  reset_current_module_entity();
  reset_current_module_statement();
  return true;
}

/**
 * put a label on each doloop without label
 *
 * @param module_name
 *
 * @return
 */
bool flag_loops(const char* module_name)
{
  /* prelude */
  set_current_module_entity(module_name_to_entity( module_name ));
  set_current_module_statement
    ((statement) db_get_memory_resource(DBR_CODE, module_name, true) );
  flag_loop_param_t flp = { .loops = NIL, .new_label_created = false };

  /* run loop labeler */
  gen_context_recurse(get_current_module_statement(),
		      &flp,
		      statement_domain,gen_true,flag_loop);

  /* validate */
  if( flp.new_label_created) DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name,get_current_module_statement());
  DB_PUT_MEMORY_RESOURCE(DBR_LOOPS, module_name,make_callees(flp.loops));

  /*postlude*/
  reset_current_module_entity();
  reset_current_module_statement();
  return true;
}

static bool module_loops_walker(statement s, list *l)
{
  if(statement_loop_p(s))
    {
      string tmp;
      asprintf(&tmp,"%s ",label_local_name(statement_label(s)));
      *l=CONS(STRING,tmp,*l);
      return false;
    }
  return true;
}

char* loop_pragma(const char* module_name, const char* parent_loop)
{
  /* ensure pipsmake is ok with what we ask for */
  safe_make(DBR_LOOPS,module_name);

  /* prelude */
  set_current_module_entity(module_name_to_entity( module_name ));
  set_current_module_statement
    ((statement) db_get_memory_resource(DBR_CODE, module_name, true) );

  entity label = find_label_entity(module_name,parent_loop);
  if(entity_undefined_p(label))
  pips_user_error("label '%s' does not exist\n",parent_loop);
  statement stmt = find_loop_from_label(get_current_module_statement(),label);
  if(statement_undefined_p(stmt))
  pips_user_error("label '%s' is not on a loop\n",parent_loop);
  string s = extensions_to_string(statement_extensions(stmt),false);
  if(string_undefined_p(s)) s = strdup("");
  reset_current_module_entity();
  reset_current_module_statement();
  return s;
}
/**
 * gather the list of enclosing loops
 * expect flag_loops has been called before
 *
 * @param module_name module we want the loops of
 * @param parent_loop null if we wat to gather outer loops, a loop_label if we want to gather enclosed loops
 *
 * @return list of strings, one string per loop label
 */
char* module_loops(const char* module_name, const char* parent_loop)
{
  /* ensure pipsmake is ok with what we ask for */
  safe_make(DBR_LOOPS,module_name);

  /* prelude */
  set_current_module_entity(module_name_to_entity( module_name ));
  set_current_module_statement
    ((statement) db_get_memory_resource(DBR_CODE, module_name, true) );
  list loops = NIL;

  statement seed = statement_undefined;
  if(empty_string_p(parent_loop))
    seed=get_current_module_statement();
  else {
    entity label = find_label_entity(module_name,parent_loop);
    if(entity_undefined_p(label))
        pips_user_error("label '%s' does not exist\n",parent_loop);
    statement stmt = find_loop_from_label(get_current_module_statement(),label);
    if(statement_undefined_p(stmt))
        pips_user_error("label '%s' is not on a loop\n",parent_loop);
    seed=loop_body(statement_loop(stmt));
  }

  /* run loop gatherer */
  gen_context_recurse(seed,
		      &loops,
		      statement_domain,module_loops_walker,gen_null);
  loops=gen_nreverse(loops);

  /*postlude*/
  reset_current_module_entity();
  reset_current_module_statement();
  string out = list_to_string(loops);
  gen_free_list(loops);
  if(out) out[strlen(out)-1]=0;
  return out;
}
