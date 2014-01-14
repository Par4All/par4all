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
/* Ansi includes 	*/
#include <stdio.h>
#include <string.h>

/* Newgen includes 	*/
#include "genC.h"

/* C3 includes 		*/
#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "sc.h"
#include "polyedre.h"
#include "union.h"
#include "matrix.h"

/* Pips includes 	*/
#include "boolean.h"
#include "ri.h"
#include "constants.h"
#include "ri-util.h"
#include "misc.h"
#include "complexity_ri.h"
#include "database.h"
#include "graph.h"
#include "dg.h"
#include "paf_ri.h"
#include "parser_private.h"
#include "property.h"
#include "properties.h"
#include "prettyprint.h"
#include "reduction.h"
#include "text.h"
#include "text-util.h"
#include "tiling.h"
#include "text-util.h"
#include "pipsdbm.h"
#include "resources.h"
#include "static_controlize.h"
#include "paf-util.h"
#include "pip.h"
#include "array_dfg.h"
#include "prgm_mapping.h"
#include "reindexing.h"

#define CM_FORTRAN_EXT ".fcm"
#define CRAFT_FORTRAN_EXT ".craft"

/* Local defines */
typedef dfg_vertex_label vertex_label;
typedef dfg_arc_label arc_label;

extern list lparams;
static graph current_dfg;
static bdt current_bdt;


/*=======================================================================*/
/* statement make_layout_statement(entity ae, int serial, news):
 * 
 */

statement make_layout_statement(ae, serial, news)
     entity ae;
     int serial, news;
{
  instruction  ins;
  list         lstat = NIL;
  char         *comment;
  statement    stat;
  int i;

  comment = (char*) malloc(64);
  sprintf(comment,"CMF$  LAYOUT %s(", entity_local_name(ae));
  for(i = 0; i < serial; i++) { 
    sprintf(comment, "%s:SERIAL", comment); 
    if((i < serial-1) || ((i == serial-1) && (news > 0)) ) 
      sprintf(comment, "%s, ", comment);
  } 
  for(i = 0; i < news; i++) { 
    sprintf(comment, "%s:NEWS", comment); 
    if(i < news-1)
      sprintf(comment, "%s, ", comment);
  } 
  sprintf(comment, "%s)\n", comment);
 
  stat = make_nop_statement();
  insert_comments_to_statement(stat, comment);
  lstat = CONS(STATEMENT, stat, lstat);
 
  /* put all the pieces of lstat in one statement */
  stat = make_block_statement(lstat);

  if(get_debug_level() > 1)
    fprintf(stderr, 
	    "\nmake_layout_statement\n=====================\n%s\n", 
	    comment); 

  return(stat);
}
  

/*=======================================================================*/
/* statement make_shared_statement(entity ae, int serial, news):
 * 
 */

statement make_shared_statement(ae, serial, news)
     entity ae;
     int serial, news;
{
  instruction  ins;
  list         lstat = NIL;
  char         *comment;
  statement    stat;
  int i;

  comment = (char*) malloc(64);
  sprintf(comment,"CDIR$ SHARED %s(", entity_local_name(ae));
  for(i = 0; i < serial; i++) { 
    sprintf(comment, "%s:", comment); 
    if((i < serial-1) || ((i == serial-1) && (news > 0)) ) 
      sprintf(comment, "%s, ", comment);
  } 
  for(i = 0; i < news; i++) { 
    sprintf(comment, "%s:BLOCK", comment); 
    if(i < news-1)
      sprintf(comment, "%s, ", comment);
  } 
  sprintf(comment, "%s)\n", comment);
 
  stat = make_nop_statement();
  insert_comments_to_statement(stat, comment);
  lstat = CONS(STATEMENT, stat, lstat);
 
  /* put all the pieces of lstat in one statement */
  stat = make_block_statement(lstat);

  if(get_debug_level() > 1)
    fprintf(stderr, 
	    "\nmake_shared_statement\n=====================\n%s\n", 
	    comment); 

  return(stat);
}
  

/*=======================================================================*/
/* void cmf_layout_align(statement mod_stat):
 * 
 */

void cmf_layout_align(mod_stat)
     statement mod_stat;
{
  list list_body, list_la, lae = NIL, li = NIL, lv;

  for(lv = graph_vertices(current_dfg); !ENDP(lv); POP(lv)) {
    vertex cv;
    int cn, bdim;
    list bv, ciel;
    string name;
    char num[32];
    entity array_ent;
    static_control stco;

    cv = VERTEX(CAR(lv));
    cn = dfg_vertex_label_statement(vertex_vertex_label(cv));

    if (cn == ENTRY_ORDER)
      continue;
    if (cn == EXIT_ORDER)
      continue;

    /* Static control of the node */
    stco = get_stco_from_current_map(adg_number_to_statement(cn));

    ciel = static_control_to_indices(stco);

    if(get_debug_level() > 0) {
      fprintf(stderr, "\nIn cmf_layout_align, CIEL : ");
      fprint_entity_list(stderr, ciel);
    }

    /* We extract the bdt corresponding to this node. */
    bv = extract_bdt(current_bdt, cn);

    /* We count the number of non constant dimension for each
       domain. */
    bdim = 0;
    for (; !ENDP(bv); POP(bv)) { 
      int bcount = 0; 
      schedule sched = SCHEDULE(CAR(bv)); 
      list lexp; 
      
      for (lexp = schedule_dims(sched); !ENDP(lexp); POP(lexp)) { 
 	expression exp = EXPRESSION(CAR(lexp)); 
 	normalized nor = NORMALIZE_EXPRESSION(exp); 
	
 	if(normalized_tag(nor) == is_normalized_linear) { 
 	  Pvecteur vec = normalized_linear(expression_normalized(exp)); 
	  
 	  if (vars_in_vect_p(vec, ciel)) 
 	    bcount++; 
 	} 
 	else 
 	  user_error("cmf_layout_align", 
 		     "Rational case not treated yet !"); 
      } 

      if(bcount > bdim) 
 	bdim = bcount; 
    }
    li = CONS(INT, bdim, li);

    /* We get the array entity corresponding to this node. */
    (void) sprintf(num, "%d", cn-BASE_NODE_NUMBER);
    name = (string) strdup(concatenate(SA_MODULE_NAME, MODULE_SEP_STRING,
				       SAI, num, (string) NULL));
    array_ent = gen_find_tabulated(name, entity_domain);
    if(array_ent == entity_undefined)
      user_error("make_array_bounds",
		 "\nOne ins (%d) has no array entity : %s\n",
		 cn-BASE_NODE_NUMBER, name);

    lae = CONS(ENTITY, array_ent, lae);
  }

  /* Layout */
  list_la = NIL;
  for(; !ENDP(lae); POP(lae), POP(li)) {
    entity ae;
    int serial, nb_dims, news;

    ae = ENTITY(CAR(lae));
    serial = INT(CAR(li));
    nb_dims =
      gen_length(variable_dimensions(type_variable(entity_type(ae))));
    news = nb_dims - serial;

    list_la = CONS(STATEMENT, make_layout_statement(ae, serial, news),
		   list_la);
  }

  /* We add these declarations to the statement of the module */
  /* After reindexing, the first statement is in fact a sequence and
     no longer an unstructured. Trust it to simplify the following: */
  if(instruction_tag(statement_instruction(mod_stat)) !=
     is_instruction_sequence)
    pips_user_error("Body is not a block\n");

  list_body = gen_nconc(list_la,
			sequence_statements(instruction_sequence(statement_instruction(mod_stat))));
  sequence_statements(instruction_sequence(statement_instruction(mod_stat))) =
      list_body;
}


/*=======================================================================*/
/* void craft_layout_align(statement mod_stat):
 * 
 */

void
craft_layout_align(statement mod_stat)
{
  list list_body, list_la, lae = NIL, li = NIL, lv;

  for(lv = graph_vertices(current_dfg); !ENDP(lv); POP(lv)) {
    vertex cv;
    int cn, bdim;
    list bv, ciel;
    string name;
    char num[32];
    entity array_ent;
    static_control stco;

    cv = VERTEX(CAR(lv));
    cn = dfg_vertex_label_statement(vertex_vertex_label(cv));

    if (cn == ENTRY_ORDER)
      continue;
    if (cn == EXIT_ORDER)
      continue;

    /* Static control of the node */
    stco = get_stco_from_current_map(adg_number_to_statement(cn));

    ciel = static_control_to_indices(stco);

    if(get_debug_level() > 0) {
      fprintf(stderr, "\nIn craft_layout_align, CIEL : ");
      fprint_entity_list(stderr, ciel);
    }

    /* We extract the bdt corresponding to this node. */
    bv = extract_bdt(current_bdt, cn);

    /* We count the number of non constant dimension for each
       domain. */
    bdim = 0;
    for (; !ENDP(bv); POP(bv)) { 
      int bcount = 0; 
      schedule sched = SCHEDULE(CAR(bv)); 
      list lexp; 
      
      for (lexp = schedule_dims(sched); !ENDP(lexp); POP(lexp)) { 
 	expression exp = EXPRESSION(CAR(lexp)); 
 	normalized nor = NORMALIZE_EXPRESSION(exp); 
	
 	if(normalized_tag(nor) == is_normalized_linear) { 
 	  Pvecteur vec = normalized_linear(expression_normalized(exp)); 
	  
 	  if (vars_in_vect_p(vec, ciel)) 
 	    bcount++; 
 	} 
 	else 
 	  user_error("craft_layout_align", 
 		     "Rational case not treated yet !"); 
      } 

      if(bcount > bdim) 
 	bdim = bcount; 
    }
    li = CONS(INT, bdim, li);

    /* We get the array entity corresponding to this node. */
    (void) sprintf(num, "%d", cn-BASE_NODE_NUMBER);
    name = (string) strdup(concatenate(SA_MODULE_NAME, MODULE_SEP_STRING,
				       SAI, num, (string) NULL));
    array_ent = gen_find_tabulated(name, entity_domain);
    if(array_ent == entity_undefined)
      user_error("craft_layout_align",
		 "\nOne ins (%d) has no array entity : %s\n",
		 cn-BASE_NODE_NUMBER, name);

    lae = CONS(ENTITY, array_ent, lae);
  }

  /* Shared */
  list_la = NIL;
  for(; !ENDP(lae); POP(lae), POP(li)) {
    entity ae;
    int serial, nb_dims, news;

    ae = ENTITY(CAR(lae));
    serial = INT(CAR(li));
    nb_dims =
      gen_length(variable_dimensions(type_variable(entity_type(ae))));
    news = nb_dims - serial;

    list_la = CONS(STATEMENT, make_shared_statement(ae, serial, news),
		   list_la);
  }

  /* We add these declarations to the statement of the module */
  /* After reindexing, the first statement is in fact a sequence and
     no longer an unstructured. Trust it to simplify the following: */
  if(instruction_tag(statement_instruction(mod_stat)) !=
     is_instruction_sequence)
    pips_user_error("Body is not a block\n");

  list_body = gen_nconc(list_la,
			sequence_statements(instruction_sequence(statement_instruction(mod_stat))));
  sequence_statements(instruction_sequence(statement_instruction(mod_stat))) =
      list_body;
  }


/*=======================================================================*/
bool print_parallelizedCMF_code(mod_name)
char *mod_name;
{
  plc the_plc;

  text r = make_text(NIL);
  entity module;
  statement mod_stat;
  code c;
  string s, pp;
  static_control stco;
  statement_mapping STS;
  bool success;

  debug_on("PRETTYPRINT_DEBUG_LEVEL");
  
  module = local_name_to_top_level_entity(mod_name);
  if (module != get_current_module_entity()) {
    reset_current_module_entity();
    set_current_module_entity(module);
  }
  c = entity_code(module);
  s = code_decls_text(c);

  /* Static controlize code */
  mod_stat = (statement) db_get_memory_resource(DBR_CODE, mod_name, true); 
  STS = (statement_mapping) db_get_memory_resource(DBR_STATIC_CONTROL, 
					    mod_name, true); 
  set_current_stco_map(STS);
  stco = get_stco_from_current_map(mod_stat);

  lparams = static_control_params(stco); 
  if (stco == static_control_undefined)  
    pips_internal_error("This is an undefined static control !");  
  if (!static_control_yes(stco))  
    pips_internal_error("This is not a static control program !");  
  
  /* The DFG, the BDT and the PLC */
  current_dfg = adg_pure_dfg((graph) db_get_memory_resource(DBR_ADFG,
							mod_name, true));
  current_bdt = (bdt) db_get_memory_resource(DBR_BDT, mod_name, true);
  the_plc = (plc) db_get_memory_resource(DBR_PLC, mod_name, true);
  if (get_debug_level() > 0) {
    fprint_dfg(stderr, current_dfg);
    fprint_bdt(stderr, current_bdt);
    fprint_plc(stderr, the_plc);
  }

  pp = strdup(get_string_property(PRETTYPRINT_PARALLEL));
  set_string_property(PRETTYPRINT_PARALLEL, "cmf");
  set_bool_property("PRETTYPRINT_ALL_DECLARATIONS", true);

  init_prettyprint(empty_text);

  mod_stat = (statement)
    db_get_memory_resource(DBR_REINDEXED_CODE, mod_name, true);
    
  insure_declaration_coherency_of_module(module, mod_stat);

  cmf_layout_align(mod_stat);

  MERGE_TEXTS(r, text_module(module, mod_stat));

  success = make_text_resource(mod_name,
			       DBR_PARALLELPRINTED_FILE,
			       CM_FORTRAN_EXT,
			       r);

  reset_current_stco_map();
  reset_current_module_entity();
  close_prettyprint();

  set_string_property(PRETTYPRINT_PARALLEL, pp); free(pp);

  debug_off();
 
  return(success);
}


/*=======================================================================*/
bool print_parallelizedCRAFT_code(mod_name)
char *mod_name;
{
  plc the_plc;

  text r = make_text(NIL);
  entity module;
  statement mod_stat;
  code c;
  string s, pp;
  static_control stco;
  statement_mapping STS;
  bool success;

  debug_on("PRETTYPRINT_DEBUG_LEVEL");
  
  module = local_name_to_top_level_entity(mod_name);
  if (module != get_current_module_entity()) {
    reset_current_module_entity();
    set_current_module_entity(module);
  }
  c = entity_code(module);
  s = code_decls_text(c);

  /* Static controlize code */
  mod_stat = (statement) db_get_memory_resource(DBR_CODE, mod_name, true); 
  STS = (statement_mapping)db_get_memory_resource(DBR_STATIC_CONTROL, 
					   mod_name, true); 
  set_current_stco_map(STS);
  stco = get_stco_from_current_map(mod_stat);

  lparams = static_control_params(stco); 
  if (stco == static_control_undefined)  
    pips_internal_error("This is an undefined static control !");  
  if (!static_control_yes(stco))  
    pips_internal_error("This is not a static control program !");  
  
  /* The DFG, the BDT and the PLC */
  current_dfg = adg_pure_dfg((graph) db_get_memory_resource(DBR_ADFG,
							    mod_name,
							    true));
  current_bdt = (bdt) db_get_memory_resource(DBR_BDT, mod_name, true);
  the_plc = (plc) db_get_memory_resource(DBR_PLC, mod_name, true);
  if (get_debug_level() > 0) {
    fprint_dfg(stderr, current_dfg);
    fprint_bdt(stderr, current_bdt);
    fprint_plc(stderr, the_plc);
  }

  pp = strdup(get_string_property(PRETTYPRINT_PARALLEL));
  set_string_property(PRETTYPRINT_PARALLEL, "craft");
  set_bool_property("PRETTYPRINT_ALL_DECLARATIONS", true);

  init_prettyprint(empty_text);

  mod_stat = (statement) db_get_memory_resource(DBR_REINDEXED_CODE,
						mod_name, true);
    
  insure_declaration_coherency_of_module(module, mod_stat);
  craft_layout_align(mod_stat);
  MERGE_TEXTS(r, text_module(module, mod_stat));

  success = make_text_resource(mod_name,
			       DBR_PARALLELPRINTED_FILE,
			       CRAFT_FORTRAN_EXT,
			       r);

  reset_current_stco_map();
  reset_current_module_entity();
  close_prettyprint();
  set_string_property(PRETTYPRINT_PARALLEL, pp); free(pp);
  debug_off();

  return(success);
}
