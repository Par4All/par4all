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
#include "ri-util.h"
#include "database.h"
#include "pipsdbm.h"
#include "resources.h"
#include "text-util.h"

#include "control.h"
#include "conversion.h"
#include "properties.h"
/* #include "generation.h" */

#include "transformations.h"

entity selected_label;

bool selected_loop_p(loop l)
{
    return loop_label(l) == selected_label ;
}

bool interactive_loop_transformation(string module_name,
				     statement (*loop_transformation)(list,bool (*)(loop)))
{
    char *lp_label=NULL;
    entity module = module_name_to_entity(module_name);
    statement s = statement_undefined;
    string resp = string_undefined;
    bool return_status = FALSE;

    pips_assert("interactive_loop_transformation", entity_module_p(module));

    /* DBR_CODE will be changed: argument "pure" should take FALSE but
       this would be useless since there is only *one* version of code;
       a new version will be put back in the data base after transforming
       the loops */
    s = (statement) db_get_memory_resource(DBR_CODE, module_name, TRUE);
    set_current_module_entity(module);
    set_current_module_statement(s);

    /* Get the loop label from the user */
    if( empty_string_p(get_string_property("LOOP_LABEL")) )
    {
        resp = user_request("Which loop do you want to transform?\n"
                "(give its label): ");
        if (resp[0] == '\0') {
            user_log("Interactive loop transformation has been cancelled.\n");
            return_status = FALSE;
        }
        else {
            if( (lp_label=malloc(strlen(resp)+1)) == NULL)
                pips_internal_error("not enough memory");
            sscanf(resp, "%s", lp_label);
        }
    }
    else
    {
        lp_label = get_string_property("LOOP_LABEL");
        if( string_undefined_p( lp_label ) )
        {
             pips_user_error("please set %s  property to a valid label\n");
        }
        else
            lp_label=strdup(lp_label); // to keep allocation consistency betwwen interactive / non interactive
    }

    if(lp_label)
    {
        selected_label = find_label_entity(module_name, lp_label);
        if (entity_undefined_p(selected_label)) {
            pips_user_error("loop label `%s' does not exist\n", lp_label);
        }
        free(lp_label);

        debug_on("INTERACTIVE_LOOP_TRANSFORMATION_DEBUG_LEVEL");

        look_for_nested_loop_statements(s, loop_transformation, selected_loop_p);

        debug_off();

        /* Reorder the module, because new statements have been generated. */
        module_reorder(s);

        DB_PUT_MEMORY_RESOURCE(DBR_CODE, 
                strdup(module_name), 
                (char*) s);
        return_status = TRUE;
    }
    reset_current_module_entity();
    reset_current_module_statement();
    
    return return_status;
}

static
void flag_loop(statement st, list *loops) {
    instruction i = statement_instruction(st);
    if(instruction_loop_p(i))
    {
        loop l = instruction_loop(i);
        if(entity_empty_label_p(loop_label(l)))
        {
            if(entity_empty_label_p(statement_label(st)))
                statement_label(st)=loop_label(l)=make_new_label(get_current_module_name());
            else
                loop_label(l)=statement_label(st);
        }
        else if(entity_empty_label_p(statement_label(st)))
            statement_label(st)=loop_label(l);
        else
            pips_assert("same label on loop and statement",same_entity_p(statement_label(st),loop_label(l)));
        *loops=CONS(STRING,strdup(entity_user_name(loop_label(l))),*loops);
    }
    if( !get_bool_property("FLAG_LOOPS_DO_LOOPS_ONLY") && instruction_forloop_p(i))
    {
        if(entity_empty_label_p(statement_label(st)))
            statement_label(st)=make_new_label(get_current_module_name());
        *loops=CONS(STRING,strdup(entity_user_name(statement_label(st))),*loops);
    }
}


bool
print_loops(char *module_name)
{
    /* prelude */
    set_current_module_entity(module_name_to_entity( module_name ));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, TRUE) );
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
bool
flag_loops(char *module_name)
{
    /* prelude */
    set_current_module_entity(module_name_to_entity( module_name ));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, TRUE) );
    list loops = NIL;

    /* run loop labeler */
    gen_context_recurse(get_current_module_statement(),&loops,statement_domain,gen_true,flag_loop);

    /* validate */
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name,get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_LOOPS, module_name,make_callees(loops));

    /*postlude*/
    reset_current_module_entity();
    reset_current_module_statement();
    return true;
}
