/*

 $Id$

 Copyright 1989-2010 MINES ParisTech
 Copyright 2009-2010 HPC-Project

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

/* **
 * This is the Fortran95 pips parser aka gfc2pips
 *
 * Here we convert the gfortran IR in pips IR.
 *
 *
 * The naming of functions is based on the following syntax :
 * - every and each function used to manipulate gfc or pips entities begin
 *  with gfc2pips_
 * - if the function is a translation from one RI to the other it will be:
 * 		gfc2pips_<type_of_the_source_entity>2<type_of_the_target_entity>
 * (arguments)
 * - if a function may need two versions for differents types in arguments, the
 * less frequently used had a _ at the end, the other one doesn't
 * 		list gfc2pips_vars(gfc_namespace *ns);
 * 		list gfc2pips_vars_(gfc_namespace *ns, list variables_p);
 * - functions with similar purpose have similar names, however you need to look
 * at the comments/documentation to know what the difference is
 * 		char* gfc2pips_gfc_char_t2string(gfc_char_t *c,int nb);
 * 		char* gfc2pips_gfc_char_t2string2(gfc_char_t *c);
 * 		char* gfc2pips_gfc_char_t2string_(gfc_char_t *c,int nb);
 *
 */

/**
 * EXPR_STRUCTURE, EXPR_SUBSTRING, EXPR_NULL, EXPR_ARRAY are not dumped
 *
 * EXPR_STRUCTURE is for user defined structures:
 *-  TYPE point
 *-   REAL x,y
 *-  END TYPE point
 * not implemented in PIPS or partially: for C structures only
 *
 * EXPR_SUBSTRING is for a substring of a constant string, but never encountered
 * even with specific test case
 *
 * EXPR_NULL The NULL pointer value (which also has a basic type), but POINTER
 * is not implemented in PIPS
 *
 * EXPR_ARRAY array constructor but seem to be destroyed to the profit of RAW
 * values
 */

#include "gfc2pips-private.h"

#include "c_parser_private.h"
#include "misc.h"
#include "text-util.h"
#include "ri-util.h"

#include <stdio.h>
#include <string.h>
extern char * strdup(const char*);

// globals defined somewhere in pips...
// Temporary HACK, waiting for PIPS to be modified : these have to be moved
int Nbrdo;
list integer_entities = list_undefined;
list real_entities = list_undefined;
list complex_entities = list_undefined;
list logical_entities = list_undefined;
list double_entities = list_undefined;
list char_entities = list_undefined;

#include <stdio.h>

/* Cmd line options */
extern gfc_option_t gfc_option;

// Temp HACK FIXME
typedef loop pips_loop;

/*
 extern type CurrentType;
 */
statement gfc_function_body;
const char *CurrentPackage;

int global_current_offset = 0;

list gfc2pips_format = NULL;//list of expression format
list gfc2pips_format2 = NULL;//list of labels for above

// FIXME may need an other implementation
static int gfc2pips_last_created_label = 95000;
static int gfc2pips_last_created_label_step = 2;

entity gfc2pips_main_entity = entity_undefined;

/**
 * @brief Entry point for gfc2pips translation
 * This will be called each time the parser encounter
 * subroutine, function, or program.
 */
void gfc2pips_namespace(gfc_namespace* ns) {
  gfc_symbol * root_sym;
  gfc_formal_arglist *formal;

  instruction icf = instruction_undefined;
  gfc2pips_format = gfc2pips_format2 = NULL;

  /* Debug level */
  debug_on("GFC2PIPS_DEBUG_LEVEL");

  gfc2pips_debug(2, "Starting gfc2pips dumping\n");
  message_assert( "No namespace to dump.", ns );
  message_assert( "No symtree root.", ns->sym_root );
  message_assert( "No symbol for the root.", ns->sym_root->n.sym );

  /* pips main initialization */
  pips_init();

  // useful ? I don't think so... (Mehdi)
  gfc2pips_shift_comments();

  { // Get symbol for current procedure (function, subroutine, whatever...)
    gfc_symtree * current_proc;
    current_proc = gfc2pips_getSymtreeByName(ns->proc_name->name, ns->sym_root);
    message_assert( "No current symtree to match the name of the namespace",
        current_proc != NULL );
    root_sym = current_proc->n.sym;
  }

  /*
   * No callees at that time !
   */
  gfc_module_callees = NULL;

  /*
   * Get the block_token (what we are working on)
   * and the the return type if it's a function
   */
  gfc2pips_main_entity_type bloc_token = get_symbol_token(root_sym);

  /*
   * Name for entity in PIPS corresponding to current procedure in GFC
   */
  gfc2pips_main_entity = gfc2pips_symbol2entity(ns->proc_name);

  string full_name = entity_name(gfc2pips_main_entity);
  CurrentPackage = global_name_to_user_name(full_name);

  gfc2pips_debug(2, "Currently parsing : %s\n", full_name);
  gfc2pips_debug(2, "CurrentPackage : %s\n", CurrentPackage);
  /*
   * PIPS STUFF Initialization
   */

  // It's safer to declare the current module in PIPS (ri-util)
  set_current_module_entity(gfc2pips_main_entity);

  /* No common has yet been declared */
  initialize_common_size_map();

  /* Generic PIPS areas are created for memory allocation. */
  InitAreas();

  /* declare variables */
  list variables_p, variables;
  variables_p = variables = gfc2pips_vars(ns);
  gfc2pips_debug(2, "%zu variable(s) founded\n",gen_length(variables));

  /* Filter variables so that we keep only local variable */
  variables = NIL;
  FOREACH(entity,e,variables_p) {
    storage s = entity_storage(e);
    if(storage_ram_p(s)) {
      if(ram_function(storage_ram(s)) == gfc2pips_main_entity) {
        variables = CONS(ENTITY,e,variables);
      }
    }
  }

  /*
   * Get USE statements
   */
  list use_entities = get_use_entities_list(ns); // List of entities

  /* Get declarations */
  list decls = code_declarations(EntityCode(gfc2pips_main_entity));
  /* Add variables to declaration */
  decls = gen_nconc(use_entities, gen_nconc(decls, variables));
  code_declarations(EntityCode(gfc2pips_main_entity)) = decls;
  /* Fix Storage for declarations */
  FOREACH( entity, e, decls ) {
    // Fixme insecure
    if(entity_variable_p(e)) {
      ram r = storage_ram(entity_storage(e));
      ram_function(r) = gfc2pips_main_entity;
      const char* name = module_local_name(gfc2pips_main_entity);
      if(ram_section(r) == entity_undefined) {
         entity da = FindOrCreateEntity(name, DYNAMIC_AREA_LOCAL_NAME);
         entity_kind(da)|= ABSTRACT_LOCATION | ENTITY_DYNAMIC_AREA;
         ram_section(r) = da;
      }
    }

  }

  gfc2pips_debug(2, "gfc2pips_main_entity %s %p %s\n",
      entity_name(gfc2pips_main_entity),
      gfc2pips_main_entity,
      CurrentPackage );
  message_assert( "Main entity not created !", gfc2pips_main_entity
      !=entity_undefined );

  // Storage is always ROM for main entity
  entity_storage( gfc2pips_main_entity ) = make_storage_rom();

  gfc2pips_debug(2, "main entity object initialized\n");

  /*
   * we have taken the entities,
   * we need to build user-defined elements
   */
  gfc2pips_getTypesDeclared(ns);

  /*
   * Parameters (if applicable)
   */
  list parameters_name = gfc2pips_parameters(ns, bloc_token);

  if(bloc_token == MET_FUNC || bloc_token == MET_SUB) {
    UpdateFunctionalType(gfc2pips_main_entity, parameters_name);
  }

  int stack_offset = 0;

  // declare commons
  list commons, commons_p, unnamed_commons, unnamed_commons_p, common_entities;
  commons = commons_p = getSymbolBy(ns, ns->common_root, gfc2pips_get_commons);
  unnamed_commons = unnamed_commons_p = getSymbolBy(ns,
                                                    ns->sym_root,
                                                    gfc2pips_get_incommon);

  common_entities = NULL; // NULL != list_undefined !!!


  /*
   * FIXME : outline all the following common stuff
   */
  gfc2pips_debug(2, "%zu explicit common(s) founded\n",gen_length(commons));

  /*
   * We create an entity for each common, and we remove elements
   * inside commons from orphan list
   * We also accumulate the size of each element to compute the
   * total size of the common
   */
  for (; commons_p; POP( commons_p )) {
    gfc_symtree *st = (gfc_symtree*)commons_p->car.e;
    gfc2pips_debug(3, "common founded: /%s/\n",st->name);

    // Check if the common exist first
    char * upper;
    string common_global_name = concatenate(TOP_LEVEL_MODULE_NAME,
                                            MODULE_SEP_STRING
                                            COMMON_PREFIX,
                                            upper=strupper(strdup(st->name),st->name),
                                            NULL);
    entity com = gen_find_tabulated(common_global_name, entity_domain);
    if(com == entity_undefined) {
      com = make_new_common(upper,
                            gfc2pips_main_entity);
      AddEntityToDeclarations(com, get_current_module_entity());
    }

    //we need in the final state a list of entities
    commons_p->car.e = com;

    gfc_symbol *s = st->n.common->head;
    int offset_common = stack_offset;
    /*
     * We loop over elements inside the common :
     *  - first we remove them from the list of orphan common elements
     *  - secondly we create an entity for each element
     */
    for (; s; s = s->common_next) {
      unnamed_commons_p = unnamed_commons;

      /* loop over the orphan list and remove current element from it */
      while(unnamed_commons_p) {
        st = unnamed_commons_p->car.e;
        if(strcmp_(st->n.sym->name, s->name) == 0) {
          gen_remove(&unnamed_commons, st);
          break;
        }
        POP( unnamed_commons_p );
      }
      gfc2pips_debug(4, "element in common founded: %s\t\toffset: %d\n",
          s->name, offset_common );

      /* Create an entity for current element */
      entity in_common_entity = gfc2pips_symbol2entity(s);
      entity_storage(in_common_entity)
          = make_storage_ram(make_ram(get_current_module_entity(),
                                      com,
                                      offset_common,
                                      NIL));

      // Accumulate size of common
      offset_common += array_size(in_common_entity);

      // Add current element to common layout
      area_layout(type_area(entity_type(com)))
          = gen_nconc(area_layout(type_area(entity_type(com))),
                      CONS( ENTITY, in_common_entity, NIL ));
      common_entities = gen_cons(in_common_entity, common_entities);
    }

    // Define the size of the common
    set_common_to_size(com, offset_common);
    gfc2pips_debug(3, "nb of elements in the common: %zu\n\t\tsize of the common: %d\n",
        gen_length(
            area_layout(type_area(entity_type(com)))
        ),
        offset_common
    );
  }

  /*
   * If we still have orphan common elements, we create an "anonymous"
   * common and we put all of them inside
   */
  int unnamed_commons_nb = gen_length(unnamed_commons);
  if(unnamed_commons_nb) {
    gfc2pips_debug(2, "nb of elements in %d unnamed common(s) founded\n",unnamed_commons_nb);

    char local_name[] = COMMON_PREFIX  BLANK_COMMON_LOCAL_NAME;
    entity com =
        FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,
                           strupper(local_name,local_name));
    entity_type(com) = make_type_area(make_area(0, NIL));

    entity_kind(com) |= ABSTRACT_LOCATION ;

    entity_storage(com)
        = make_storage_ram(make_ram(get_current_module_entity(),
                                    StaticArea,
                                    0,
                                    NIL));
    entity_initial(com) = make_value_code(make_code(NIL,
                                                    string_undefined,
                                                    make_sequence(NIL),
                                                    NIL,
                                                    make_language_fortran95()));
    AddEntityToDeclarations(com, get_current_module_entity());
    int offset_common = stack_offset;
    unnamed_commons_p = unnamed_commons;

    // Loop over orphan element
    for (; unnamed_commons_p; POP( unnamed_commons_p )) {
      gfc_symtree* st = unnamed_commons_p->car.e;
      gfc_symbol *s = st->n.sym;
      gfc2pips_debug(4, "element in common founded: %s\t\toffset: %d\n", s->name, offset_common );

      // Create entity
      entity in_common_entity = gfc2pips_symbol2entity(s);
      entity_storage(in_common_entity)
          = make_storage_ram(make_ram(get_current_module_entity(),
                                      com,
                                      offset_common,
                                      NIL));

      // Accumulate size of common
      offset_common += array_size(in_common_entity);

      // Add current element to common layout
      area_layout(type_area(entity_type(com)))
          = gen_nconc(area_layout(type_area(entity_type(com))),
                      CONS( ENTITY, in_common_entity, NIL ));

      common_entities = gen_cons(in_common_entity, common_entities);
    }

    // Define the size of the common
    set_common_to_size(com, offset_common);
    gfc2pips_debug(3, "nb of elements in the common: %zu\n\t\tsize of the common: %d\n",
        gen_length(
            area_layout(type_area(entity_type(com)))
        ),
        offset_common
    );
    commons = gen_cons(com, commons);
  }

  gfc2pips_debug(2, "%zu common(s) founded for %zu entities\n", gen_length(commons), gen_length(common_entities) );

  // declare DIMENSIONS => no need to handle, information
  // already transfered to each and every entity

  // Mehdi : FIXME, unused !!
  // we concatenate the entities from variables, commons and parameters and
  // make sure they are declared only once. It seems parameters cannot be
  // declared implicitly and have to be part of the list
  list complete_list_of_entities = NULL, complete_list_of_entities_p = NULL;

  complete_list_of_entities_p = gen_union(complete_list_of_entities_p,
                                          variables_p);
  commons_p = commons;

  complete_list_of_entities_p = gen_union(commons_p,
                                          complete_list_of_entities_p);

  complete_list_of_entities_p = gen_union(complete_list_of_entities_p,
                                          parameters_name);

  complete_list_of_entities = complete_list_of_entities_p;
  for (; complete_list_of_entities_p; POP( complete_list_of_entities_p )) {
    //force value
    if(entity_initial(ENTITY(CAR(complete_list_of_entities_p)))
        ==value_undefined) {
      entity_initial(ENTITY(CAR(complete_list_of_entities_p)))
          = make_value_unknown();
    }
  }

  /*
   * Get declarations
   */
  list list_of_declarations =
      code_declarations(EntityCode(gfc2pips_main_entity));
  gfc2pips_debug(2, "%zu declaration(s) founded\n",gen_length(list_of_declarations));
  complete_list_of_entities = gen_union(complete_list_of_entities,
                                        list_of_declarations);

  /*
   * Get extern entities
   */
  list list_of_extern_entities = gfc2pips_get_extern_entities(ns);
  gfc2pips_debug(2, "%zu extern(s) founded\n",gen_length(list_of_extern_entities));

  gfc2pips_debug(2, "nb of entities: %zu\n",gen_length(complete_list_of_entities));

  /*
   * Handle subroutines now
   */
  list list_of_subroutines, list_of_subroutines_p;
  list_of_subroutines_p = list_of_subroutines
      = getSymbolBy(ns, ns->sym_root, gfc2pips_test_subroutine);
  for (; list_of_subroutines_p; POP( list_of_subroutines_p )) {
    gfc_symtree* st = list_of_subroutines_p->car.e;

    list_of_subroutines_p->car.e = gfc2pips_symbol2entity(st->n.sym);
    entity check_sub_entity = (entity)list_of_subroutines_p->car.e;
    if(type_functional_p(entity_type(check_sub_entity))
        && strcmp_(st->name, CurrentPackage) != 0) {
      //check list of parameters;
      list check_sub_parameters =
          functional_parameters(type_functional(entity_type(check_sub_entity)));
      if(check_sub_parameters == NULL) {
        // Error ?
      }
      gfc2pips_debug(4,"sub %s has %zu parameters\n",
          entity_name(check_sub_entity),
          gen_length(check_sub_parameters) );
    }
  }
  gfc2pips_debug(2, "%zu subroutine(s) encountered\n",
      gen_length(list_of_subroutines) );

  /*
   * we need variables in commons to be in the list too
   */
  complete_list_of_entities = gen_union(complete_list_of_entities,
                                        common_entities);

  /*
   *  sort the list again to get rid of IMPLICIT, BUT beware
   *  arguments with the current method we have a pb with the ouput
   *  of subroutines/functions arguments even if they are of the right type
   */
  /*complete_list_of_entities_p = complete_list_of_entities;
   while( complete_list_of_entities_p ){
   entity ent = complete_list_of_entities_p->car.e;
   if( ent ){
   gfc2pips_debug(9,"Look for %s %zu\n", entity_local_name(ent), gen_length(complete_list_of_entities_p) );
   POP(complete_list_of_entities_p);
   gfc_symtree* sort_entities = gfc2pips_getSymtreeByName(entity_local_name(ent),ns->sym_root);
   if(
   sort_entities && sort_entities->n.sym
   && (
   sort_entities->n.sym->attr.in_common
   || sort_entities->n.sym->attr.implicit_type
   )
   ){
   gen_remove( &complete_list_of_entities , ent );
   gfc2pips_debug(9,"Remove %s from list of entities, element\n",entity_local_name(ent));
   }
   }else{
   POP(complete_list_of_entities_p);
   }
   }*/

  ifdebug(9) {
    complete_list_of_entities_p = complete_list_of_entities;
    entity ent = entity_undefined;
    for (; complete_list_of_entities_p; POP( complete_list_of_entities_p )) {
      ent = ENTITY(CAR(complete_list_of_entities_p));
      if(ent)
        fprintf(stderr,
                "Complete list of entities, element: %s\n",
                entity_local_name(ent));
    }
  }

  entity_initial(gfc2pips_main_entity)
      = make_value_code(make_code(gen_union(list_of_extern_entities,
                                            complete_list_of_entities),
                                  strdup(""),
                                  make_sequence(NIL),
                                  gen_union(list_of_extern_entities,
                                            list_of_subroutines),
                                  make_language_fortran95()));

  gfc2pips_debug(2, "main entity creation finished\n");

  //get symbols with value, data and explicit-save
  //sym->value is an expression to build the save
  //create data $var /$val/
  //save if explicit save, else nothing
  instruction data_inst = instruction_undefined;

  /***********************************************
   * HERE WE BEGIN THE REAL PARSING OF THE CODE
   */

  // declare code
  gfc2pips_debug(2, "dumping code ...\n");
  icf = gfc2pips_code2instruction__TOP(ns, ns->code);
  gfc2pips_debug(2, "end of dumping code ...\n");
  message_assert( "Dumping instruction failed\n", icf != instruction_undefined );

  /*
   * END
   ***********************************************/

  gfc_function_body = instruction_to_statement(icf);

  //we automatically add a return statement
  //we have got a problem with multiple return in the function
  //and if the last statement is already a return ? badly handled
  // FIXME
  //  insure_return_as_last_statement( gfc2pips_main_entity, &gfc_function_body );


  //  SetChains( ); // ??????????

  /*
   * RR said :
   * using ComputeAddresses() point a problem : entities in *STATIC*
   * are computed two times, however we have to use it !
   *
   * But well... try it anyway !
   */
  gfc2pips_computeAdresses();
  //ComputeAddresses();

  //compute equivalences
  //gfc2pips_computeEquiv(ns->equiv);

  //Syntax !!
  //we have the job done 2 times if debug is at 9, one time if at 8
  //update_common_sizes();
  //print_common_layout(stderr,StaticArea,true);
  gfc2pips_debug(2, "dumping done\n");

  //bad construction when  parameter = subroutine
  //text t = text_module(gfc2pips_main_entity,gfc_function_body);
  //dump_text(t);

  /*gfc2pips_comments com;
   gfc_code *current_code_linked_to_comments=NULL;
   fprintf(stderr,"gfc2pips_comments_stack: %d\n",gfc2pips_comments_stack);
   if( com=gfc2pips_pop_comment() ){
   while(1){
   fprintf(stderr,"comment: %d ",com);
   if(com){
   fprintf(stderr,"linked %s\n",com->done?"yes":"no");
   current_code_linked_to_comments = com->num;
   do{
   fprintf(stderr,"\t %d > %s\n", com->num, com->s );
   com=gfc2pips_pop_comment();
   }while(
   com
   && current_code_linked_to_comments == com->num
   );

   }else{
   break;
   }
   fprintf(stderr,"\n");
   }
   fprintf(stderr,"\n");
   }

   fprintf(stderr,"gfc2pips_list_of_declared_code: %d\n",gfc2pips_list_of_declared_code);
   while( gfc2pips_list_of_declared_code ){
   if(gfc2pips_list_of_declared_code->car.e){
   fprintf(stderr,"gfc_code: %d %d %d %d %d\n",
   gfc2pips_list_of_declared_code->car.e,
   ((gfc_code*)gfc2pips_list_of_declared_code->car.e)->loc.nextc,
   *((gfc_code*)gfc2pips_list_of_declared_code->car.e)->loc.nextc,
   *((gfc_code*)gfc2pips_list_of_declared_code->car.e)->loc.lb->line,
   ((gfc_code*)gfc2pips_list_of_declared_code->car.e)->loc.lb->location
   );
   fprintf(stderr,"%s\n",
   gfc2pips_gfc_char_t2string2( ((gfc_code*)gfc2pips_list_of_declared_code->car.e)->loc.nextc )
   );
   fprintf(stderr,"\n");
   }
   POP(gfc2pips_list_of_declared_code);
   }
   fprintf(stderr,"\n");
   */

  /*
   * It's time to produce the resource in the PIPS workspace
   */

  extern const char *main_input_filename;
  extern const char *aux_base_name;

  // Get workspace directory
  char *dir_name = (char *)aux_base_name;
  char *source_file_orig = strdup(concatenate(dir_name,
                                              "/",
                                              CurrentPackage,
                                              ".f90",// FIXME get correct ext.
                                              NULL));
  char * unsplit_modname = NULL; // Will be followed by a ! for module
  save_entities();

  /*
   * FIXME : The following still need many many cleaning !
   */

  // We don't produce output for module
  if(bloc_token == MET_MODULE) {
    pips_user_warning("Modules are ignored : %s\n", full_name);

    unsplit_modname = (char *)malloc(sizeof(char)
        * (strlen(CurrentPackage) + 2));
    sprintf(unsplit_modname, "%s", CurrentPackage);

    char *module_dir_name = strdup(concatenate(dir_name,
                                               "/",
                                               unsplit_modname,
                                               NULL));
    mkdir(module_dir_name, 0xffffffff);
    pips_debug(2,"Creating module directory : %s\n", module_dir_name);

    // Produce SOURCE_FILE
    string source_file = concatenate(module_dir_name,
                                     "/",
                                     unsplit_modname,
                                     ".f90",// FIXME get correct ext.
                                     NULL);
    fcopy(main_input_filename, source_file);
    source_file = concatenate(CurrentPackage, "/", CurrentPackage, ".f90",// FIXME get correct ext.
                              NULL);

    fcopy(main_input_filename, source_file_orig);

    char *parsedcode_filename = concatenate(module_dir_name,
                                            "/",
                                            "PARSED_CODE",
                                            NULL);
    FILE *parsedcode_file = safe_fopen(parsedcode_filename, "w");
    printf("Write PArsed code in %s \n", parsedcode_filename);
    gen_write(parsedcode_file, (void *)gfc_function_body);
    safe_fclose(parsedcode_file, parsedcode_filename);

    char *callees_filename = concatenate(module_dir_name, "/", "CALLEES", NULL);
    FILE *callees_file = safe_fopen(callees_filename, "w");
    //  printf("Write callees\n");
    gen_write(callees_file, (void *)make_callees(gfc_module_callees));
    safe_fclose(callees_file, callees_filename);

    string unsplit_source_file = concatenate(dir_name,
                                             "/",
                                             unsplit_modname,
                                             ".f90",
                                             NULL);

    printf("Writing %s\n\n", unsplit_source_file);
    FILE *fp = safe_fopen(unsplit_source_file, "w");
    fprintf(fp, "/* module still to be implemented ! */\n\n");
    safe_fclose(fp, unsplit_source_file);

  } else {

    char *module_dir_name = strdup(concatenate(dir_name,
                                               "/",
                                               CurrentPackage,
                                               NULL));
    mkdir(module_dir_name, 0xffffffff);
    pips_debug(2,"Creating module directory : %s\n", module_dir_name);

    // Produce SOURCE_FILE
    string source_file = concatenate(dir_name,
                                     "/",
                                     CurrentPackage,
                                     "/",
                                     CurrentPackage,
                                     ".f90",// FIXME get correct ext.
                                     NULL);
    fcopy(main_input_filename, source_file);
    source_file = concatenate(CurrentPackage, "/", CurrentPackage, ".f90",// FIXME get correct ext.
                              NULL);

    fcopy(main_input_filename, source_file_orig);

    unsplit_modname = strdup(CurrentPackage);

    char *parsedcode_filename = concatenate(module_dir_name,
                                            "/",
                                            "PARSED_CODE",
                                            NULL);
    FILE *parsedcode_file = safe_fopen(parsedcode_filename, "w");
    //  printf("Write PArsed code\n");
    gen_write(parsedcode_file, (void *)gfc_function_body);
    safe_fclose(parsedcode_file, parsedcode_filename);

    char *callees_filename = concatenate(module_dir_name, "/", "CALLEES", NULL);
    FILE *callees_file = safe_fopen(callees_filename, "w");
    //  printf("Write callees\n");
    gen_write(callees_file, (void *)make_callees(gfc_module_callees));
    safe_fclose(callees_file, callees_filename);
  }

  // FIXME free strdup

  ResetChains();
  reset_current_module_entity();
  reset_common_size_map();

  char *file_list = concatenate(dir_name, "/.fsplit_file_list", NULL);
  FILE *fp = safe_fopen(file_list, "a");
  fprintf(fp, "%s %s/%s.f90\n", unsplit_modname, dir_name, unsplit_modname);
  safe_fclose(fp, file_list);

  // Loop over contained procedures
  for (ns = ns->contained; ns; ns = ns->sibling) {
    fprintf(stderr, "CONTAINS\n");
    gfc2pips_namespace(ns);
  }

  //  printf( "nend\n" );
  //  exit( 0 );
}

gfc2pips_main_entity_type get_symbol_token(gfc_symbol *root_sym) {
  gfc2pips_main_entity_type bloc_token;
  if(root_sym->attr.is_main_program) {
    bloc_token = MET_PROG;
  } else if(root_sym->attr.subroutine) {
    bloc_token = MET_SUB;
  } else if(root_sym->attr.function) {
    bloc_token = MET_FUNC;
  } else if(root_sym->attr.flavor == FL_BLOCK_DATA) {
    bloc_token = MET_BLOCK;
  } else if(root_sym->attr.flavor == FL_MODULE) {
    bloc_token = MET_MODULE;
  } else {
    if(root_sym->attr.procedure) {
      fprintf(stderr, "procedure\n");
    }
    pips_user_error("Unknown token !");
  }
  return bloc_token;
}

list gfc2pips_parameters(gfc_namespace * ns,
                         gfc2pips_main_entity_type bloc_token) {

  gfc2pips_debug(2, "Handle the list of parameters\n");
  list parameters = NULL, parameters_name = NULL;
  entity ent = entity_undefined;
  switch(bloc_token) {
    case MET_FUNC: {
      // we add a special entity called "func:func" which
      // is the return variable of the function
      ent = FindOrCreateEntity(CurrentPackage, get_current_module_name());
      entity_type(ent) = copy_type(entity_type(gfc2pips_main_entity));
      entity_initial(ent) = copy_value(entity_initial(gfc2pips_main_entity));
      //don't know were to put it hence StackArea
      entity_storage(ent)
          = make_storage_ram(make_ram(get_current_module_entity(),
                                      StackArea,
                                      UNKNOWN_RAM_OFFSET,
                                      NULL));
    }
      // No break
    case MET_SUB:
      // change it to put both name and namespace in order to catch the parameters
      // of any subroutine ? or create a sub-function for gfc2pips_args
      parameters = gfc2pips_args(ns);

      if(ent != entity_undefined) {
        UpdateFunctionalType(ent, parameters);
      }

      //we need a copy of the list of parameters entities (really ?)
      parameters_name = gen_copy_seq(parameters);
      gfc2pips_generate_parameters_list(parameters);
      //fprintf(stderr,"formal created ?? %d\n", storage_formal_p(entity_storage(ENTITY(CAR(parameters_name)))));
      //ScanFormalParameters(gfc2pips_main_entity, add_formal_return_code(parameters));
      gfc2pips_debug(2, "List of parameters done\t %zu parameters(s)\n", gen_length(parameters_name) );
      break;
    case MET_BLOCK: // Useful ?
      entity_type( gfc2pips_main_entity )
          = make_type_functional(make_functional(parameters,
                                                 make_type_void(NIL)));
      break;
    default:
      break;
  }
  return parameters_name;
}

/**
 * @brief Retrieve the list of names of every argument of the function, if any
 *
 * Since alternate returns are obsoletes in F90 we do not dump them, still
 * there is a start of dump (but crash if some properties are not activated)
 */
list gfc2pips_args(gfc_namespace* ns) {
  gfc_symtree * current = NULL;
  gfc_formal_arglist *formal;
  list args_list = NULL, args_list_p = NULL;
  entity e = entity_undefined;
  set_current_number_of_alternate_returns();

  if(ns && ns->proc_name) {

    current = gfc2pips_getSymtreeByName(ns->proc_name->name, ns->sym_root);
    if(current && current->n.sym) {
      if(current->n.sym->formal) {
        //we have a pb with alternate returns
        formal = current->n.sym->formal;
        if(formal) {
          if(formal->sym) {
            gfc_symtree* sym = gfc2pips_getSymtreeByName(formal->sym->name,
                                                         ns->sym_root);
            e = gfc2pips_symbol2entity(sym->n.sym);
          } else {
            //alternate returns are obsolete in F90 (and since we only want it)
            return NULL;
          }
          args_list = args_list_p = CONS( ENTITY, e, NULL );

          formal = formal->next;
          while(formal) {
            gfc2pips_debug(9,"alt return %s\n", formal->sym?"no":"yes");
            if(formal->sym) {
              gfc_symtree* sym = gfc2pips_getSymtreeByName(formal->sym->name,
                                                           ns->sym_root);
              e = gfc2pips_symbol2entity(sym->n.sym);
              CDR( args_list) = CONS( ENTITY, e, NULL );
              args_list = CDR( args_list );
            } else {
              return args_list_p;
            }
            formal = formal->next;
          }
        }
      }
    }
  }
  return args_list_p;
}

/**
 * @brief replace a list of entities by a list of parameters to those entities
 */
void gfc2pips_generate_parameters_list(list parameters) {
  int formal_offset = 1;
  while(parameters) {
    entity ent = parameters->car.e;
    gfc2pips_debug(8, "parameter founded: %s\n\t\tindice %d\n", entity_local_name(ent), formal_offset );
    entity_storage(ent) = make_storage_formal(make_formal(gfc2pips_main_entity,
                                                          formal_offset));
    //entity_initial(ent) = make_value_unknown();
    type formal_param_type = entity_type(ent);//is the format ok ?
    parameters->car.e = make_parameter(formal_param_type,
                                       make_mode_reference(),
                                       make_dummy_identifier(ent));
    formal_offset++;
    POP( parameters );
  }
}

/**
 * @brief Look for a specific symbol in a tree
 * Check current entry first, then recurse left then right.
 * @param name : the string containing the target symbol name
 * @param st : the current element of the tree beeing processed
 * @return : the tree element if found, else NULL
 */
__attribute__ ((warn_unused_result))
gfc_symtree* gfc2pips_getSymtreeByName(const char* name, gfc_symtree *st) {
  gfc_symtree *return_value = NULL;
  if(!name)
    return NULL;

  if(!st)
    return NULL;
  if(!st->n.sym)
    return NULL;
  if(!st->name)
    return NULL;

  //much much more information, BUT useless (cause recursive)
  gfc2pips_debug(10, "Looking for the symtree called: %s(%zu) %s(%zu)\n",
      name, strlen(name), st->name, strlen(st->name) );

  // FIXME : case insentitive ????
  if(strcmp_(st->name, name) == 0) {
    //much much more information, BUT useless (cause recursive)
    gfc2pips_debug(9, "symbol %s founded\n",name);
    return st;
  }
  return_value = gfc2pips_getSymtreeByName(name, st->left);

  if(return_value == NULL) {
    return_value = gfc2pips_getSymtreeByName(name, st->right);
  }
  return return_value;
}

/**
 * @brief Extract every and each variable from a namespace
 */
list gfc2pips_vars(gfc_namespace *ns) {
  if(ns) {
    return gfc2pips_vars_(ns, gen_nreverse(getSymbolBy(ns,
                                                       ns->sym_root,
                                                       gfc2pips_test_variable)));
  }
  return NULL;
}

/**
 * @brief Convert the list of gfc symbols into a list of pips entities with storage, type, everything
 */
list gfc2pips_vars_(gfc_namespace *ns, list variables_p) {
  list variables = NULL;
  //variables_p = gen_nreverse(getSymbolBy(ns,ns->sym_root, gfc2pips_test_variable));
  //balancer la suite dans une fonction à part afin de pouvoir la réutiliser pour les calls
  //list arguments,arguments_p;
  //arguments = arguments_p = gfc2pips_args(ns);
  while(variables_p) {
    type Type = type_undefined;
    //create entities here
    gfc_symtree *current_symtree = (gfc_symtree*)variables_p->car.e;
    if(current_symtree && current_symtree->n.sym) {
      gfc2pips_debug(3, "translation of entity start\n");
      if(current_symtree->n.sym->attr.in_common) {
        gfc2pips_debug(4, " %s is in a common, skipping\r\n", (current_symtree->name) );
        //we have to skip them, they don't have any place here
        POP( variables_p );
        continue;
      }
      if(current_symtree->n.sym->attr.use_assoc) {
        gfc2pips_debug(4, " %s is in a module, skipping\r\n", (current_symtree->name) );
        //we have to skip them, they don't have any place here
        POP( variables_p );
        continue;
      }
      gfc2pips_debug(4, " symbol: %s size: %d\r\n", (current_symtree->name), current_symtree->n.sym->ts.kind );
      intptr_t TypeSize = gfc2pips_symbol2size(current_symtree->n.sym);
      value Value;// = make_value_unknown();
      Type = gfc2pips_symbol2type(current_symtree->n.sym);
      gfc2pips_debug(3, "Type done\n");

      //handle the value
      //don't ask why it is is_value_constant
      if(Type != type_undefined && current_symtree->n.sym->ts.type
          == BT_CHARACTER) {
        gfc2pips_debug(5, "the symbol is a string\n");
        Value = make_value_constant(make_constant_litteral()//MakeConstant(current_symtree->n.sym->value->value.character.string,is_basic_string)
            );
      } else {
        gfc2pips_debug(5, "the symbol is a constant\n");
        Value = make_value_expression(int_to_expression(TypeSize));
      }

      int i, j = 0;
      //list list_of_dimensions = gfc2pips_get_list_of_dimensions(current_symtree);
      //si allocatable alors on fait qqch d'un peu spécial

      /*
       * we look into the list of arguments to know if the entity is in and thus
       * the offset in the stack
       */
      /*i=0;j=1;
       arguments_p = arguments;
       while(arguments_p){
       //fprintf(stderr,"%s %s\n",entity_local_name((entity)arguments_p->car.e),current_symtree->name);
       if(strcmp_( entity_local_name( (entity)arguments_p->car.e ), current_symtree->name )==0 ){
       i=j;
       break;
       }
       j++;
       POP(arguments_p);
       }*/
      //fprintf(stderr,"%s %d\n",current_symtree->name,i);

      entity
          newEntity =
              FindOrCreateEntity(CurrentPackage,
                                 str2upper(gfc2pips_get_safe_name(current_symtree->name)));
      variables = CONS( ENTITY,newEntity,variables );
      entity_type((entity)variables->car.e) = Type;
      entity_initial((entity)variables->car.e) = Value;
      if(current_symtree->n.sym->attr.dummy) {
        gfc2pips_debug(0,"dummy parameter \"%s\" put in FORMAL\n",current_symtree->n.sym->name);
        //we have a formal parameter (argument of the function/subroutine)
        /*if(entity_storage((entity)variables->car.e)==storage_undefined)
         entity_storage((entity)variables->car.e) = make_storage_formal(
         make_formal(
         gfc2pips_main_entity,
         i
         )
         );*/
      } else if(current_symtree->n.sym->attr.flavor == FL_PARAMETER) {
        gfc2pips_debug(9,"Variable \"%s\" (PARAMETER) put in FORMAL\n",current_symtree->n.sym->name);
        //we have a parameter, we rewrite some attributes of the entity
        entity_type((entity)variables->car.e)
            = make_type_functional(make_functional(NIL,
                                                   entity_type((entity)variables->car.e)));
        entity_initial((entity)variables->car.e)
            = MakeValueSymbolic(gfc2pips_expr2expression(current_symtree->n.sym->value));
        if(entity_storage((entity)variables->car.e) == storage_undefined) {
          gfc2pips_debug(0,"!!!!!! Variable \"%s\" (PARAMETER) put in formal WITHOUT RANK\n",current_symtree->n.sym->name);
          entity_storage((entity)variables->car.e)
              = make_storage_formal(make_formal(gfc2pips_main_entity, 0));
        }
      } else {
        //we have a variable
        entity area = entity_undefined;
        if(gfc2pips_test_save(NULL, current_symtree)) {
          entity da = FindOrCreateEntity(CurrentPackage, STATIC_AREA_LOCAL_NAME);
          entity_kind(da)|= ABSTRACT_LOCATION | ENTITY_STATIC_AREA;
          gfc2pips_debug(9,"Variable \"%s\" put in RAM \"%s\"\n",
              entity_local_name((entity)variables->car.e),
              STATIC_AREA_LOCAL_NAME);
          //set_common_to_size(StaticArea,CurrentOffsetOfArea(StaticArea,(entity)variables->car.e));
        } else {
          if(current_symtree->n.sym->as && current_symtree->n.sym->as->type
              != AS_EXPLICIT && !current_symtree->n.sym->value) {//some other criteria is needed
            if(current_symtree->n.sym->attr.allocatable) {
              /* we do know this entity is allocatable,
               * it's place is *probably* in the heap but it's not so evident
               * we represent it with a struct encaspulating the real array !
               */
              area = HeapArea;
            } else {
              area = StackArea;
            }
          } else {
            area = DynamicArea;
          }
          gfc2pips_debug(
              9,
              "Variable \"%s\" put in RAM \"%s\"\n",
              entity_local_name((entity)variables->car.e),
              area==DynamicArea ? DYNAMIC_AREA_LOCAL_NAME
              :(area==StackArea ?STACK_AREA_LOCAL_NAME
                  :ALLOCATABLE_AREA_LOCAL_NAME)
          );
        }
        ram _r_ = make_ram(get_current_module_entity(),
                           area,
                           UNKNOWN_RAM_OFFSET,
                           NULL);
        if(entity_storage((entity)variables->car.e) == storage_undefined)
          entity_storage( (entity)variables->car.e ) = make_storage_ram(_r_);
      }

      //code for pointers
      //if(Type!=type_undefined){
      //variable_dimensions(type_variable(entity_type( (entity)variables->car.e ))) = gfc2pips_get_list_of_dimensions(current_symtree);
      /*if(current_symtree->n.sym->attr.pointer){
       basic b = make_basic(is_basic_pointer, Type);
       type newType = make_type(is_type_variable, make_variable(b, NIL, NIL));
       entity_type((entity)variables->car.e) = newType;
       }*/
      //}
      gfc2pips_debug(3, "translation for %s end\n",entity_name(newEntity));
    } else {
      variables_p->car.e = NULL;
    }
    POP( variables_p );
  }
  return variables;
}

void gfc2pips_getTypesDeclared(gfc_namespace *ns) {
  if(ns) {
    list variables_p = gen_nreverse(getSymbolBy(ns,
                                                ns->sym_root,
                                                gfc2pips_test_derived));

    while(variables_p) {
      type Type = type_undefined;
      //create entities here
      gfc_symtree *current_symtree = (gfc_symtree*)variables_p->car.e;

      list list_of_components = NULL;

      list members = NULL;
      gfc_component *c;
      gfc2pips_debug(9,"start list of elements in the structure %s\n",current_symtree->name);
      for (c = current_symtree->n.sym->components; c; c = c->next) {
        /*decl_spec_list TK_SEMICOLON struct_decl_list
         {
         //c_parser_context ycontext = stack_head(ContextStack);
         c_parser_context ycontext = GetContext();
         * Create the struct member entity with unique name, the name of the
         * struct/union is added to the member name prefix *
         string istr = i2a(derived_counter++);
         string s = strdup(concatenate("PIPS_MEMBER_",istr,NULL));
         string derived = code_decls_text((code) stack_head(StructNameStack));
         entity ent = CreateEntityFromLocalNameAndPrefix(s,strdup(concatenate(derived,
         MEMBER_SEP_STRING,NULL)),
         is_external);
         gfc2pips_debug(5,"Current derived name: %s\n",derived);
         gfc2pips_debug(5,"Member name: %s\n",entity_name(ent));
         entity_storage(ent) = make_storage_rom();
         entity_type(ent) = c_parser_context_type(ycontext);
         free(s);

         * Temporally put the list of struct/union entities defined in $1 to
         * initial value of ent. FI: where is it retrieved? in TakeDerivedEntities()? *
         entity_initial(ent) = (value) $1;

         $$ = CONS(ENTITY,ent,$3);

         //stack_pop(ContextStack);
         PopContext();
         }*/

        fprintf(stdout, "%s\n", c->name);
        /*show_typespec (&c->ts);
         if(c->attr.pointer) fputs(" POINTER", dumpfile);
         if(c->attr.dimension) fputs(" DIMENSION", dumpfile);
         fputc(' ', dumpfile);
         show_array_spec(c->as);
         if(c->attr.access) fprintf(dumpfile, " %s", gfc_code2string(access_types, c->attr.access) );
         fputc (')', dumpfile);
         if (c->next != NULL) fputc (' ', dumpfile);*/
      }
      entity ent;
      if(current_symtree->n.sym->attr.external) {
          char * local_name;
          asprintf(&local_name,STRUCT_PREFIX "%s",current_symtree->name);
        ent = FindOrCreateEntity(CurrentPackage,local_name);
        free(local_name);
      } else {
          char * local_name;
          asprintf(&local_name,STRUCT_PREFIX "%s"/*scope*/,current_symtree->name);
        ent = FindOrCreateEntity(CurrentPackage,local_name);
        free(local_name);
        /*ent = MakeDerivedEntity(
         current_symtree->n.sym->name,
         list_of_components,
         current_symtree->n.sym->attr.external,
         is_type_struct
         );*/
        //entity_basic(ent) = make_basic_typedef(ent);
        entity_type(ent) = make_type_struct(members);
      }

      /*variable v = make_variable(make_basic_derived(ent),NIL,NIL);
       list le = TakeDerivedEntities(list_of_components);
       variables = gen_cons( gen_nconc(le,CONS(ENTITY,ent,NIL)), variables );

       //get the list of entities in the struct
       entity ent = CreateEntityFromLocalNameAndPrefix(
       s->ts.derived->name,
       STRUCT_PREFIX,
       s->attr.external
       );
       entity_type(ent) = make_type_struct( members );
       return MakeTypeVariable(
       make_basic_derived(ent),
       NULL
       );*/

      POP( variables_p );
    }
  }
}

/**
 * @brief build a list of externals entities
 */
list gfc2pips_get_extern_entities(gfc_namespace *ns) {
  list list_of_extern, list_of_extern_p;
  list_of_extern_p = list_of_extern = getSymbolBy(ns,
                                                  ns->sym_root,
                                                  gfc2pips_test_extern);
  while(list_of_extern_p) {
    gfc_symtree* curr = list_of_extern_p->car.e;
    entity e = gfc2pips_symbol2entity(curr->n.sym);

    // FIXME, is there other extern that calls ???
    gfc2pips_add_to_callees(e);

    if(entity_storage(e) == storage_undefined) {
      gfc2pips_debug(2,"Storage rom !! %s %d\n",entity_name(e),__LINE__);
      entity_storage(e) = make_storage_rom();
    }

    list_of_extern_p->car.e = e;
    POP( list_of_extern_p );
  }
  return list_of_extern;
}
/**
 * @brief return a list of elements needing a DATA statement
 */
list gfc2pips_get_data_vars(gfc_namespace *ns) {
  return getSymbolBy(ns, ns->sym_root, gfc2pips_test_data);
}
/**
 * @brief return a list of SAVE elements
 */
list gfc2pips_get_save(gfc_namespace *ns) {
  return getSymbolBy(ns, ns->sym_root, gfc2pips_test_save);
}

/**
 * @brief build a list - if any - of dimension elements from the gfc_symtree given
 */
list gfc2pips_get_list_of_dimensions(gfc_symtree *st) {
  if(st) {
    return gfc2pips_get_list_of_dimensions2(st->n.sym);
  } else {
    return NULL;
  }
}
/**
 * @brief build a list - if any - of dimension elements from the gfc_symbol given
 */
list gfc2pips_get_list_of_dimensions2(gfc_symbol *s) {
  list list_of_dimensions = NULL;
  int i = 0, j = 0;

  pips_assert("Allocatable should have been handled before !",
      !s->attr.allocatable);

  if(s && s->attr.dimension) {
    gfc_array_spec *as = s->as;
    const char *c;
    gfc2pips_debug(4, "%s is an array\n",s->name);
    if(as != NULL && as->rank != 0) {
      //according to the type of array we create different types of dimensions parameters
      switch(as->type) {
        case AS_EXPLICIT:
          c = strdup("AS_EXPLICIT");
          //create the list of dimensions
          i = as->rank - 1;
          do {
            //check lower ou upper n'est pas une variable dont la valeur est inconnue
            list_of_dimensions
                = gen_cons(make_dimension(gfc2pips_expr2expression(as->lower[i]),
                                          gfc2pips_expr2expression(as->upper[i])),
                           list_of_dimensions);
          } while(--i >= j);
          break;
        case AS_DEFERRED://beware allocatable !!!
          c = strdup("AS_DEFERRED");
          i = as->rank - 1;
          do {
            list_of_dimensions
                = gen_cons(make_dimension(int_to_expression(1),
                                          MakeNullaryCall(CreateIntrinsic(UNBOUNDED_DIMENSION_NAME))),
                           list_of_dimensions);
          } while(--i >= j);
          break;
          //AS_ASSUMED_...  means information come from a dummy argument and the property is inherited from the call
        case AS_ASSUMED_SIZE://means only the last set of dimensions is unknown
          j = 1;
          c = strdup("AS_ASSUMED_SIZE");
          //create the list of dimensions
          i = as->rank - 1;
          while(i > j) {
            //check lower ou upper n'est pas une variable dont la valeur est inconnue
            list_of_dimensions
                = gen_cons(make_dimension(gfc2pips_expr2expression(as->lower[i]),
                                          gfc2pips_expr2expression(as->upper[i])),
                           list_of_dimensions);
            i--;
          }

          list_of_dimensions
              = gen_cons(make_dimension(gfc2pips_expr2expression(as->lower[i]),
                                        MakeNullaryCall(CreateIntrinsic(UNBOUNDED_DIMENSION_NAME))),
                         list_of_dimensions);
          break;
        case AS_ASSUMED_SHAPE:
          c = strdup("AS_ASSUMED_SHAPE");
          break;
        default:
          gfc_internal_error("show_array_spec(): Unhandled array shape "
            "type.");
      }
    }
    gfc2pips_debug(4, "%zu dimensions detected for %s\n",gen_length(list_of_dimensions),s->name);
  }

  return list_of_dimensions;
}

/**
 * @brief Look for a set of symbols filtered by a predicate function
 */
list getSymbolBy(gfc_namespace* ns,
                 gfc_symtree *st,
                 bool(*func)(gfc_namespace*, gfc_symtree *)) {
  list args_list = NULL;

  if(!ns)
    return NULL;
  if(!st)
    return NULL;
  if(!func)
    return NULL;

  if(func(ns, st)) {
    args_list = gen_cons(st, args_list);
  }
  args_list = gen_nconc(args_list, getSymbolBy(ns, st->left, func));
  args_list = gen_nconc(args_list, getSymbolBy(ns, st->right, func));

  return args_list;
}

/*
 * Predicate functions
 */

/**
 * @brief get variables who are not implicit or are needed to be declared for data statements hence variable that should be explicit in PIPS
 */
bool gfc2pips_test_variable(gfc_namespace __attribute__ ((__unused__)) *ns,
                            gfc_symtree *st) {
  if(!st || !st->n.sym)
    return false;
  bool variable_p = TRUE;

  variable_p = (st->n.sym->attr.flavor == FL_VARIABLE || st->n.sym->attr.flavor
      == FL_PARAMETER);

  /*&& (
   (!st->n.sym->attr.implicit_type||st->n.sym->attr.save==SAVE_EXPLICIT)
   || st->n.sym->value//very important
   )*/
  variable_p = variable_p && !st->n.sym->attr.external;
  //&& !st->n.sym->attr.in_common
  variable_p = variable_p && !st->n.sym->attr.pointer;
  variable_p = variable_p && !st->n.sym->attr.dummy;
  variable_p = variable_p && !(st->n.sym->ts.type == BT_DERIVED);

  return variable_p;
}

/*
 * @brief test if it is a variable
 */
bool gfc2pips_test_variable2(gfc_namespace __attribute__ ((__unused__)) *ns,
                             gfc_symtree *st) {
  if(!st || !st->n.sym)
    return false;
  return st->n.sym->attr.flavor == EXPR_VARIABLE && !st->n.sym->attr.dummy;
}

/*
 * @brief test if a variable is a user-defined structure
 */
bool gfc2pips_test_derived(gfc_namespace __attribute__ ((__unused__)) *ns,
                           gfc_symtree *st) {
  if(!st || !st->n.sym)
    return false;
  return st->n.sym->attr.flavor == FL_DERIVED
  //&& !st->n.sym->attr.external
      && !st->n.sym->attr.pointer && !st->n.sym->attr.dummy;
}

/**
 * @brief test if it is an external function
 */
bool gfc2pips_test_extern(gfc_namespace __attribute__ ((__unused__)) *ns,
                          gfc_symtree *st) {
  if(!st || !st->n.sym)
    return false;
  return st->n.sym->attr.external || st->n.sym->attr.proc == PROC_EXTERNAL;
}
bool gfc2pips_test_subroutine(gfc_namespace __attribute__ ((__unused__)) *ns,
                              gfc_symtree *st) {
  if(!st || !st->n.sym)
    return false;
  return (st->n.sym->attr.flavor == FL_PROCEDURE && (st->n.sym->attr.subroutine
      || st->n.sym->attr.function) && strncmp(st->n.sym->name,
                                              "__",
                                              strlen("__")) != 0);
  //return st->n.sym->attr.subroutine && strcmp(str2upper(strdup(ns->proc_name->name)), str2upper(strdup(st->n.sym->name)))!=0;
}

/**
 * @brief test if it is a allocatable entity
 */
bool gfc2pips_test_allocatable(gfc_namespace __attribute__ ((__unused__)) *ns,
                               gfc_symtree *st) {
  if(!st || !st->n.sym)
    return false;
  return st->n.sym->attr.allocatable;
}
/**
 * @brief test if it is a dummy parameter (formal parameter)
 */
bool gfc2pips_test_arg(gfc_namespace __attribute__ ((__unused__)) *ns,
                       gfc_symtree *st) {
  if(!st || !st->n.sym)
    return false;
  return st->n.sym->attr.flavor == EXPR_VARIABLE && st->n.sym->attr.dummy;
}
/**
 * @brief test if there is a value to stock
 */
bool gfc2pips_test_data(gfc_namespace __attribute__ ((__unused__)) *ns,
                        gfc_symtree *st) {
  if(!st || !st->n.sym)
    return false;
  return st->n.sym->value && st->n.sym->attr.flavor != FL_PARAMETER
      && st->n.sym->attr.flavor != FL_PROCEDURE;
}
/**
 * @brief test if there is a SAVE to do
 */
bool gfc2pips_test_save(gfc_namespace __attribute__ ((__unused__)) *ns,
                        gfc_symtree *st) {
  if(!st || !st->n.sym)
    return false;
  return st->n.sym->attr.save != SAVE_NONE;
}
/**
 * @brief test function to know if it is a common, always true because the tree
 * is completely separated therefore the function using it only create a list
 */
bool gfc2pips_get_commons(gfc_namespace __attribute__ ((__unused__)) *ns,
                          gfc_symtree __attribute__ ((__unused__)) *st) {
  return true;
}
bool gfc2pips_get_incommon(gfc_namespace __attribute__ ((__unused__)) *ns,
                           gfc_symtree __attribute__ ((__unused__)) *st) {
  if(!st || !st->n.sym)
    return false;
  return st->n.sym->attr.in_common;
}
/**
 *
 */
bool gfc2pips_test_dimensions(gfc_namespace __attribute__ ((__unused__)) *ns,
                              gfc_symtree* st) {
  if(!st || !st->n.sym)
    return false;
  return st->n.sym->attr.dimension;
}

entity gfc2pips_check_entity_doesnt_exists(char *s) {
  entity e = entity_undefined;
  string full_name;
  //main program
  full_name = concatenate(TOP_LEVEL_MODULE_NAME,
                          MODULE_SEP_STRING,
                          MAIN_PREFIX,
                          str2upper(strdup(s)),
                          NULL);
  e = gen_find_tabulated(full_name, entity_domain);

  //module
  if(e == entity_undefined) {
    full_name = concatenate(TOP_LEVEL_MODULE_NAME,
                            MODULE_SEP_STRING,
                            str2upper(strdup(s)),
                            NULL);
    e = gen_find_tabulated(full_name, entity_domain);
  }

  //simple entity
  if(e == entity_undefined) {
    full_name = concatenate(CurrentPackage,
                            MODULE_SEP_STRING,
                            str2upper(strdup(s)),
                            NULL);
    e = gen_find_tabulated(full_name, entity_domain);
  }
  return e;
}
entity gfc2pips_check_entity_program_exists(char *s) {
  string full_name;
  //main program
  full_name = concatenate(TOP_LEVEL_MODULE_NAME,
                          MODULE_SEP_STRING,
                          MAIN_PREFIX,
                          str2upper(strdup(s)),
                          NULL);
  return gen_find_tabulated(full_name, entity_domain);
}
entity gfc2pips_check_entity_module_exists(char *s) {
  string full_name;
  //module
  full_name = concatenate(TOP_LEVEL_MODULE_NAME,
                          MODULE_SEP_STRING,
                          str2upper(strdup(s)),
                          NULL);

  entity e = gen_find_tabulated(full_name, entity_domain);

  ifdebug(5) {
    if(e==entity_undefined) {
      pips_debug(5,"'%s' doesn't exist.\n",full_name);
    } else {
      pips_debug(5,"'%s' found.\n",full_name);
    }
  }

  return e;
}

entity gfc2pips_check_entity_block_data_exists(char *s) {
  string full_name;
  full_name = concatenate(TOP_LEVEL_MODULE_NAME,
                          MODULE_SEP_STRING,
                          BLOCKDATA_PREFIX,
                          str2upper(strdup(s)),
                          NULL);
  return gen_find_tabulated(full_name, entity_domain);
}
entity gfc2pips_check_entity_exists(const char *s) {
  string full_name;
  //simple entity
  full_name = concatenate(CurrentPackage,
                          MODULE_SEP_STRING,
                          str2upper(strdup(s)),
                          NULL);
  return gen_find_tabulated(full_name, entity_domain);
}

/**
 * @brief translate a gfc symbol to a PIPS entity, check if it is a function,
 * program, subroutine or else
 */
//add declarations of parameters
entity gfc2pips_symbol2entity(gfc_symbol* s) {
  char* name = str2upper(gfc2pips_get_safe_name(s->name));
  entity e = entity_undefined;//gfc2pips_check_entity_doesnt_exists(name);
  bool module = false;

  if(s->attr.flavor == FL_PROGRAM || s->attr.is_main_program) {
    if((e = gfc2pips_check_entity_program_exists(name)) == entity_undefined) {
      gfc2pips_debug(9, "create main program %s\n",name);
      e = make_empty_program(str2upper((name)), make_language_fortran95());
    }
    module = true;
  } else if(s->attr.function) {
    if((e = gfc2pips_check_entity_module_exists(name)) == entity_undefined) {
      if(FindEntity(TOP_LEVEL_MODULE_NAME, str2upper((name)))
          == entity_undefined) {
        gfc2pips_debug(0, "create function %s\n",str2upper( ( name ) ));
        e = make_empty_function(str2upper((name)),
                                gfc2pips_symbol2type(s),
                                make_language_fortran95());
      }
    }
    module = true;
  } else if(s->attr.subroutine) {
    if((e = gfc2pips_check_entity_module_exists(name)) == entity_undefined) {
      gfc2pips_debug(1, "create subroutine %s\n",name);
      e = make_empty_subroutine(str2upper((name)), make_language_fortran95());
    }
    module = true;
  } else if(s->attr.flavor == FL_BLOCK_DATA) {
    if((e = gfc2pips_check_entity_block_data_exists(name)) == entity_undefined) {
      gfc2pips_debug(9, "block data \n");
      e = make_empty_blockdata(str2upper((name)), make_language_fortran95());
    }
    module = true;
  } else if(s->attr.flavor == FL_MODULE) {
    char *module_name = str2upper(strdup(concatenate(name, "!", NULL)));
    if((e = gfc2pips_check_entity_module_exists(module_name))
        ==entity_undefined) {
      gfc2pips_debug(1, "create module %s\n",module_name);
      e = make_empty_f95module(module_name, make_language_fortran95());
    }
    free(module_name);
    module = true;
  } else {
    gfc2pips_debug(9, "Try to resolve entity %s\n",str2upper( ( name ) ));
    if(s->ts.type == BT_DERIVED) {
      if(s->attr.allocatable) {
        /* Allocatable handling : we convert it to a structure */
        e = find_or_create_allocatable_struct(gfc2pips_getbasic(s),
                                              name,
                                              s->as->rank);
        // FIXME This is bad, we have the entity TYPE only. */
      } else {
        pips_user_error( "User-defined variables are not implemented yet\n" );
        //there is still a problem in the check of consistency of the domain names
      }
    } else {
      string location = strdup(CurrentPackage);
      if(s->attr.use_assoc) {
        gfc2pips_debug(2, "Entity %s is located in a module (%s)\n",
            name,
            s->module);
        free(location);
        location = str2upper(strdup(concatenate(s->module, "!", NULL)));
        e = FindEntity(location, name);
        if(e == entity_undefined) {
          pips_internal_error("Entity '%s' located in module '%s' can't be "
              "found in symbol table, are you sure that you parsed the module "
              "first ? Aborting\n",name, s->module);
        }
      } else {
        e = FindOrCreateEntity(location, name);
      }
      if(entity_initial(e) == value_undefined)
        entity_initial(e) = make_value_unknown();
      if(entity_type(e) == type_undefined)
        entity_type(e) = gfc2pips_symbol2type(s);
    }
    //if(entity_storage(e)==storage_undefined) entity_storage(e) = make_storage_rom();
    free(name);
    return e;
  }
  //it is a module and we do not know it yet, so we put an empty content in it
  if(module) {
    //message_assert("arg ! bad handling",entity_initial(e)==value_undefined);
    //fprintf(stderr,"value ... ... ... %s\n",entity_initial(e)==value_undefined?"ok":"nok");
    if(entity_initial(e) == value_undefined) {
      entity_initial(e) = make_value_code(make_code(NULL,
                                                    strdup(""),
                                                    make_sequence(NIL),
                                                    NULL,
                                                    make_language_fortran95()));
    }

  }
  free(name);
  return e;
}

/**
 * @brief translate a gfc symbol to a top-level entity
 */
entity gfc2pips_symbol2top_entity(gfc_symbol* s) {
  char* name = gfc2pips_get_safe_name(s->name);
  entity e = gfc2pips_check_entity_doesnt_exists(name);
  if(e != entity_undefined) {
    gfc2pips_debug(9,"Entity %s already exists\n",name);
    free(name);
    return e;
  }
  e = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, str2upper(name));
  if(entity_initial(e) == value_undefined)
    entity_initial(e) = make_value_unknown();
  if(entity_type(e) == type_undefined)
    entity_type(e) = gfc2pips_symbol2type(s);
  //if(entity_storage(e)==storage_undefined) entity_storage(e) = make_storage_rom();
  free(name);
  return e;
}

/**
 * @brief a little bit more elaborated FindOrCreateEntity
 */
entity gfc2pips_char2entity(char* package, char* s) {
  s = gfc2pips_get_safe_name(s);
  entity e = FindOrCreateEntity(package, str2upper(s));
  if(entity_initial(e) == value_undefined)
    entity_initial(e) = make_value_unknown();
  if(entity_type(e) == type_undefined)
    entity_type(e) = make_type_unknown();
  free(s);
  return e;
}

/**
 * @brief gfc replace some functions by an homemade one, we check and return a
 * copy of the original one if it is the case
 */
char* gfc2pips_get_safe_name(const char* str) {
  if(strncmp_("_gfortran_exit_", str, strlen("_gfortran_exit_")) == 0) {
    return strdup("exit");
  } else if(strncmp_("_gfortran_float", str, strlen("_gfortran_float")) == 0) {
    return strdup("FLOAT");
  } else {
    return strdup(str);
  }
}

/*
 * Functions about the translation of something from gfc into a pips "dimension" object
 */
/**
 * @brief create a <dimension> from the integer value given
 */
dimension gfc2pips_int2dimension(int n) {
  return make_dimension(int_to_expression(1),
                        gfc2pips_int2expression(n));
}

/**
 * @brief translate a int to an expression
 */
expression gfc2pips_int2expression(int n) {
  //return int_expr(n);
  if(n < 0) {
    return MakeFortranUnaryCall(CreateIntrinsic("--"),
                                entity_to_expression(gfc2pips_int_const2entity(-n)));
  } else {
    return entity_to_expression(gfc2pips_int_const2entity(n));
  }
}
/**
 * @brief translate a real to an expression
 */
expression gfc2pips_real2expression(double r) {
  if(r < 0.) {
    return MakeFortranUnaryCall(CreateIntrinsic("--"),
                                entity_to_expression(gfc2pips_real2entity(-r)));
  } else {
    return entity_to_expression(gfc2pips_real2entity(r));
  }
}
/**
 * @brief translate a bool to an expression
 */
expression gfc2pips_logical2expression(bool b) {
  //return int_expr(b!=false);
  return entity_to_expression(gfc2pips_logical2entity(b));
}

/**
 * @brief translate an integer to a PIPS constant, assume n is positive (or it will not be handled properly)
 */
entity gfc2pips_int_const2entity(int n) {
  char str[30];
  sprintf(str, "%d", n);
  return MakeConstant(str, is_basic_int);
}
/**
 * @brief dump an integer to a PIPS label entity
 * @param n the value of the integer
 */
entity gfc2pips_int2label(int n) {
  //return make_loop_label(n,concatenate(TOP_LEVEL_MODULE_NAME,MODULE_SEP_STRING,LABEL_PREFIX,NULL));
  char str[60];
  sprintf(str, LABEL_PREFIX "%d", n);//fprintf(stderr,"new label: %s %s %s %s %d\n",str,TOP_LEVEL_MODULE_NAME,MODULE_SEP_STRING,LABEL_PREFIX,n);
  return make_label(CurrentPackage,str);
}

/**
 * @brief dump reals to PIPS entities
 * @param r the double to create
 * @return the corresponding entity
 *
 * we have a big issue with reals:
 * 16.53 => 16.530001
 * 16.56 => 16.559999
 */
entity gfc2pips_real2entity(double r) {
  //create a more elaborate function to output a fortran format or something like it ?
  char str[60];
  if(r == 0. || r == (double)((int)r)) {//if the real represent an integer, display a string representing an integer then
    sprintf(str, "%d", (int)r);
  } else {
    //we need a test to know if we output in scientific or normal mode
    //sprintf(str,"%.32f",r);
    //sprintf(str,"%.6e",r);
    sprintf(str, "%.16e", r);
    //fprintf(stderr,"copy of the entity name(real) %s\n",str);
  }
  gfc2pips_truncate_useless_zeroes(str);
  return MakeConstant(str, is_basic_float);
}

/**
 * @brief translate a boolean to a PIPS/fortran entity
 */
entity gfc2pips_logical2entity(bool b) {
  return MakeConstant(b ? ".TRUE." : ".FALSE.", is_basic_logical);
}

/**
 * @brief translate a string from a table of integers in gfc to one of chars in PIPS, escape all ' in the string
 * @param c the table of integers in gfc
 * @param nb the size of the table
 *
 * The function only calculate the number of ' to escape and give the information to gfc2pips_gfc_char_t2string_
 * TODO: optimize to know if we should put " or ' quotes
 */
char* gfc2pips_gfc_char_t2string(gfc_char_t *c, int length) {
  if(length) {
    gfc_char_t *p = c;
    while(*p) {
      if(*p == '\'')
        length++;
      p++;
    }
    return gfc2pips_gfc_char_t2string_(c, length);
  } else {
    return strdup("");
  }
}

/**
 * @brief translate a string from a table of integers in gfc to one of chars in PIPS, escape all ' in the string
 * @param c the table of integers in gfc
 * @param nb the size of the table
 *
 * Stupidly add ' before ' and add ' at the beginning and the end of the string
 */
char* gfc2pips_gfc_char_t2string_(gfc_char_t *c, int nb) {
  char *s = malloc(sizeof(char) * (nb + 1 + 2));
  gfc_char_t *p = c;
  int i = 1;
  s[0] = '\'';
  while(i <= nb) {
    if(*p == '\'') {
      s[i++] = '\'';
    }
    s[i++] = *p;
    p++;

  }
  s[i++] = '\'';
  s[i] = '\0';
  return s;
}

/**
 * @brief translate the <nb> first elements of <c> from a wide integer representation to a char representation
 * @param c the gfc integer table
 */
char* gfc2pips_gfc_char_t2string2(gfc_char_t *c) {
  gfc_char_t *p = c;
  char *s = NULL;
  int nb, i;
  //fprintf(stderr,"try gfc2pips_gfc_char_t2string2");

  nb = 0;
  while(p && *p && nb < 132) {
    nb++;
    p++;
  }
  p = c;
  //fprintf(stderr,"continue gfc2pips_gfc_char_t2string2 %d\n",nb);

  if(nb) {

    s = malloc(sizeof(char) * (nb + 1));
    i = 0;
    while(i < nb && *p) {
      //fprintf(stderr,"i:%d *p:(%d=%c)\n",i,*p,*p);
      s[i++] = *p;
      p++;
    }
    s[i] = '\0';
    //fprintf(stderr,"end gfc2pips_gfc_char_t2string2");
    return s;
  } else {
    return NULL;
  }
}

/**
 *
 */
basic gfc2pips_getbasic(gfc_symbol *s) {
  basic b = basic_undefined;
  if(s->attr.pointer) {
    b = make_basic_pointer(type_undefined);
  } else {
    int size = gfc2pips_symbol2size(s);
    switch(s->ts.type) {
      case BT_INTEGER:
        b = make_basic_int(size);
        break;
      case BT_REAL:
        b = make_basic_float(size);
        break;
      case BT_COMPLEX:
        b = make_basic_complex(2 * size);
        break;
      case BT_LOGICAL:
        b = make_basic_logical(size);
        break;
      case BT_CHARACTER:
        b = make_basic_string(make_value_constant(make_constant_int(size)));
        break;
      case BT_UNKNOWN:
        gfc2pips_debug( 5, "Type unknown\n" );
        b = basic_undefined;
        break;
      case BT_DERIVED:
        pips_user_error( "User-def types are not implemented yet\n" );
        b = basic_undefined;
        break;
      case BT_PROCEDURE:
      case BT_HOLLERITH:
      case BT_VOID:
      default:
        pips_user_error( "An error occurred in the type to type translation: impossible to translate the symbol.\n" );
        b = basic_undefined;
        //return make_type_unknown();
    }
  }

  if(s->attr.allocatable) {
    /* Allocatable handling : we convert it to a structure */
    char* name = str2upper(gfc2pips_get_safe_name(s->name));
    entity e = find_or_create_allocatable_struct(b, name, s->as->rank);
    b = make_basic_derived(e);
  }

  if(b != basic_undefined)
    gfc2pips_debug(5, "Basic type is : %d\n",basic_tag(b));
  else
    gfc2pips_debug(5, "Basic type is undefined\n");
  return b;
}

/**
 * @brief try to create the PIPS type that would be associated by the PIPS default parser
 */
type gfc2pips_symbol2type(gfc_symbol *s) {
  //beware the size of strings
  basic b = gfc2pips_getbasic(s);
  if(b == basic_undefined) {
    gfc2pips_debug( 5, "WARNING: unknown type !\n" );
    return MakeTypeUnknown();
  }

  if(basic_derived_p(b)) {
    return MakeTypeVariable(b, NULL);
  } else {
    return MakeTypeVariable(b, gfc2pips_get_list_of_dimensions2(s));
  }
}

type gfc2pips_symbol2specialType(gfc_symbol *s) {

  return type_undefined;
}

/**
 * @brief return the size of an elementary element:  REAL*16 A    CHARACTER*17 B
 * @param s symbol of the entity
 */
int gfc2pips_symbol2size(gfc_symbol *s) {
  if(s->ts.type == BT_CHARACTER && s->ts.cl && s->ts.cl->length) {
    gfc2pips_debug(
        9,
        "size of %s: %zu\n",
        s->name,
        mpz_get_si(s->ts.cl->length->value.integer)
    );
    return mpz_get_si(s->ts.cl->length->value.integer);
  } else {
    gfc2pips_debug(9, "size of %s: %d\n",s->name,s->ts.kind);
    return s->ts.kind;
  }
}
/**
 * @brief calculate the total size of the array whatever the bounds are:  A(-5,5)
 * @param s symbol of the array
 */
int gfc2pips_symbol2sizeArray(gfc_symbol *s) {
  int retour = 1;
  list list_of_dimensions = NULL;
  int i = 0, j = 0;
  if(s && s->attr.dimension) {
    gfc_array_spec *as = s->as;
    const char *c;
    if(as != NULL && as->rank != 0 && as->type == AS_EXPLICIT) {
      i = as->rank - 1;
      do {
        retour *= gfc2pips_expr2int(as->upper[i])
            - gfc2pips_expr2int(as->lower[i]) + 1;
      } while(--i >= j);
    }
  }
  gfc2pips_debug(9, "size of %s: %d\n",s->name,retour);
  return retour;
}

/**
 * @brief convert a list of indices from gfc to PIPS, assume there is no range (dump only the min range element)
 * @param ar the struct with indices
 * only for AR_ARRAY references
 */
list gfc2pips_array_ref2indices(gfc_array_ref *ar) {
  int i;
  list indices = NULL;

  switch(ar->type) {
    case AR_FULL:
      break;
    case AR_SECTION:
      for (i = 0; i < ar->dimen; i++) {
        /* There are two types of array sections: either the
         elements are identified by an integer array ('vector'),
         or by an index range. In the former case we only have to
         get the start expression which contains the vector, in
         the latter case we have to print any of lower and upper
         bound and the stride, if they're present.  */

        if(ar->dimen_type[i] == DIMEN_RANGE) {
          expression start, end, stride;

          /* Get lower bound */
          if(ar->start[i]) {
            start = gfc2pips_expr2expression(ar->start[i]);
          } else {
            start = MakeNullaryCall(CreateIntrinsic(UNBOUNDED_DIMENSION_NAME));
          }

          /* Get upper bound */
          if(ar->end[i]) {
            end = gfc2pips_expr2expression(ar->end[i]);
          } else {
            end = MakeNullaryCall(CreateIntrinsic(UNBOUNDED_DIMENSION_NAME));
          }

          if(ar->stride[i]) {
            stride = gfc2pips_expr2expression(ar->stride[i]);
          } else {
            stride = int_to_expression(1);
          }

          range r = make_range(start, end, stride);
          expression indice = make_expression(make_syntax(is_syntax_range, r),
                                              normalized_undefined);
          indices = CONS( EXPRESSION,indice,indices);
        } else {
          indices = CONS( EXPRESSION,
              gfc2pips_expr2expression( ar->start[i] ),
              indices);
        }

      }
      indices = gen_nreverse(indices);
      break;

    case AR_ELEMENT:
      for (i = 0; i < ar->dimen; i++) {
        if(ar->start[i]) {
          indices = CONS( EXPRESSION,
              gfc2pips_expr2expression( ar->start[i] ),
              indices);
        }
      }
      indices = gen_nreverse(indices);
      break;

    case AR_UNKNOWN:
      pips_user_warning("Ref unknown ?");
      break;

    default:
      pips_internal_error ("Unknown array reference");
  }
  gfc2pips_debug(9,"%zu indice(s)\n", gen_length(indices) );
  return indices;
}

//this is to know if we have to add a continue statement just after the loop statement for (exit/break)
bool gfc2pips_last_statement_is_loop = false;

/**
 * Declaration of instructions
 * @brief We need to differentiate the instructions at the very top of the
 * module from those in other blocks because of the declarations of DATA,
 * SAVE, or simply any variable
 * @param ns the top-level entity from gfc. We need it to retrieve some
 * more informations
 * @param c the struct containing information about the instruction
 */
instruction gfc2pips_code2instruction__TOP(gfc_namespace *ns, gfc_code* c) {
  list list_of_data_symbol, list_of_data_symbol_p;
  list_of_data_symbol_p = list_of_data_symbol = gfc2pips_get_data_vars(ns);

  //create a sequence and put everything into it ? is it right ?
  list list_of_statements, list_of_statements_p;
  list_of_statements_p = list_of_statements = NULL;

  instruction i = instruction_undefined;

  //dump DATA
  //create a list of statements and dump them in one go
  //test for each if there is an explicit save statement
  //fprintf(stderr,"nb of data statements: %d\n",gen_length(list_of_data_symbol_p));
  while(list_of_data_symbol_p) {

    //if there are parts in the DATA statement, we have got a pb !
    instruction ins = instruction_undefined;
    ins
        = gfc2pips_symbol2data_instruction(((gfc_symtree*)list_of_data_symbol_p->car.e)->n.sym);
    //fprintf(stderr,"Got a data !\n");
    //PIPS doesn't tolerate comments here
    //we should shift endlessly the comments number to the first "real" statement
    //string comments  = gfc2pips_get_comment_of_code(c);
    //fprintf(stderr,"comment founded")

    message_assert( "error in data instruction", ins != instruction_undefined );
    list lst = CONS( STATEMENT, make_statement( entity_empty_label( ),
						STATEMENT_NUMBER_UNDEFINED,
						STATEMENT_ORDERING_UNDEFINED,
						//comments,
						empty_comments,
						ins,
						NULL,
						NULL,
						empty_extensions(),
						make_synchronization_none()), NULL );
    code mc = entity_code(get_current_module_entity());
    sequence_statements(code_initializations(mc))
        = gen_nconc(sequence_statements(code_initializations(mc)), lst);

    POP( list_of_data_symbol_p );
  }

  //dump equivalence statements
  //int OffsetOfReference(reference r)
  //int CurrentOffsetOfArea(entity a, entity v)

  /*gfc_equiv * eq, *eq2;
   for (eq = ns->equiv; eq; eq = eq->next){
   //show_indent ();
   //fputs ("Equivalence: ", dumpfile);

   //gfc2pips_handleEquiv(eq);
   eq2 = eq;
   while (eq2){
   fprintf(stderr,"eq: %d %s %d ",eq2->expr, eq2->module, eq2->used);
   //show_expr (eq2->expr);
   eq2 = eq2->eq;
   if(eq2)fputs (", ", stderr);
   else fputs ("\n", stderr);
   }
   }*/
  //StoreEquivChain(<atom>)


  list list_of_save = gfc2pips_get_save(ns);
  //save_all_entities();//Syntax !!!
  gfc2pips_debug(3,"%zu SAVE founded\n",gen_length(list_of_save));
  while(list_of_save) {
    static int offset_area = 0;
    //we should know the current offset of every and each memory area or are equivalence not yet dumped ?
    // ProcessSave(<entity>); <=> MakeVariableStatic(<entity>,true)
    // => balance le storage dans RAM, ram_section(r) = StaticArea
    //fprintf(stderr,"%d\n",list_of_save->car.e);
    gfc2pips_debug(4,"entity to SAVE %s\n",((gfc_symtree*)list_of_save->car.e)->n.sym->name);
    entity curr_save =
        gfc2pips_symbol2entity(((gfc_symtree*)list_of_save->car.e)->n.sym);
    gfc2pips_debug(9,"Size of %s %d\n",STATIC_AREA_LOCAL_NAME, CurrentOffsetOfArea(StaticArea ,curr_save));//Syntax !
    //entity_type(curr_save) = make_type_area(<area>);
    //entity g = local_name_to_top_level_entity(entity_local_name(curr_save));
    if(entity_storage(curr_save) == storage_undefined) {
      //int offset_area = CurrentOffsetOfArea(StaticArea,curr_save);
      entity_storage(curr_save)
          = make_storage_ram(make_ram(get_current_module_entity(),
                                      StaticArea,
                                      UNKNOWN_RAM_OFFSET,
                                      NIL));
      list layout = area_layout(type_area(entity_type(StaticArea)));
      area_layout(type_area(entity_type(StaticArea)))
          =CONS(entity,curr_save,layout);
      //AddVariableToCommon(StaticArea,curr_save);
      //SaveEntity(curr_save);
      //offset_area = CurrentOffsetOfArea(StaticArea,curr_save);
      //set_common_to_size(StaticArea,offset_area);
    } else if(storage_ram_p(entity_storage(curr_save))
        && ram_section(storage_ram(entity_storage(curr_save))) == DynamicArea) {
      //int offset_area = CurrentOffsetOfArea(StaticArea,curr_save);
      ram_section(storage_ram(entity_storage(curr_save))) = StaticArea;
      ram_offset(storage_ram(entity_storage(curr_save))) = UNKNOWN_RAM_OFFSET;
      //ram_offset(storage_ram(entity_storage(curr_save))) = UNKNOWN_RAM_OFFSET;
      //AddVariableToCommon(StaticArea,curr_save);
      //SaveEntity(curr_save);
      //offset_area = CurrentOffsetOfArea(StaticArea,curr_save);
      //set_common_to_size(StaticArea,offset_area);
    } else if(storage_ram_p(entity_storage(curr_save))
        && ram_section(storage_ram(entity_storage(curr_save))) == StaticArea) {
      //int offset_area = CurrentOffsetOfArea(StackArea,curr_save);
      //set_common_to_size(StaticArea,offset_area);
      gfc2pips_debug(9,"Entity %s already in the Static area\n",entity_name(curr_save));
    } else {
      pips_user_warning( "Static entity(%s) not in the correct memory Area: %s\n",
          entity_name(curr_save),
          storage_ram_p(entity_storage(curr_save)) ? entity_name(ram_section(storage_ram(entity_storage(curr_save))))
          : "?" );
    }
    POP( list_of_save );
  }
  if(!c) {
    //fprintf(stderr,"WE HAVE GOT A PROBLEM, SEQUENCE WITHOUT ANYTHING IN IT !\nSegfault soon ...\n");
    return make_instruction_block(CONS( STATEMENT,
        instruction_to_statement( make_instruction_block( NULL ) ),
        NIL ));
  }

  //dump other
  //we know we have at least one instruction, otherwise we would have
  //returned an empty list of statements
  while(i == instruction_undefined && c->op != EXEC_SELECT) {
    i = gfc2pips_code2instruction_(c);
    if(i != instruction_undefined) {
      //string comments  = gfc2pips_get_comment_of_code(c);
      //fprintf(stderr,"comment founded")
      statement s = make_statement(gfc2pips_code2get_label(c),
                                   STATEMENT_NUMBER_UNDEFINED,
                                   STATEMENT_ORDERING_UNDEFINED,
                                   //comments,
                                   empty_comments,
                                   i,
                                   NULL,
                                   NULL,
                                   empty_extensions(),
				   make_synchronization_none());
      /* unlike the classical method, we don't know if we have had
       * a first statement (data inst)
       */
      list_of_statements = gen_nconc(list_of_statements, CONS( STATEMENT,
          s,
          NIL ));
    }
    c = c->next;
  }

  //compter le nombre de statements décalés
  unsigned long first_comment_num = gfc2pips_get_num_of_gfc_code(c);
  unsigned long last_code_num = gen_length(gfc2pips_list_of_declared_code);
  unsigned long curr_comment_num = first_comment_num;
  //for( ; curr_comment_num<first_comment_num ; curr_comment_num++ )
  //  gfc2pips_replace_comments_num( curr_comment_num, first_comment_num );
  for (; curr_comment_num <= last_code_num; curr_comment_num++)
    gfc2pips_replace_comments_num(curr_comment_num, curr_comment_num + 1
        - first_comment_num);

  gfc2pips_assign_gfc_code_to_num_comments(c, 0);
  /*
   unsigned long first_comment_num  = gfc2pips_get_num_of_gfc_code(c);
   unsigned long curr_comment_num=0;
   for( ; curr_comment_num<first_comment_num ; curr_comment_num++ ){
   gfc2pips_replace_comments_num( curr_comment_num, first_comment_num );
   }
   gfc2pips_assign_gfc_code_to_num_comments(c,first_comment_num );
   */

  for (; c; c = c->next) {
    statement s = statement_undefined;
    if(c && c->op == EXEC_SELECT) {
      list_of_statements
          = gen_nconc(list_of_statements, gfc2pips_dumpSELECT(c));
    } else {
      i = gfc2pips_code2instruction_(c);
      if(i != instruction_undefined) {
        string comments = gfc2pips_get_comment_of_code(c);//fprintf(stderr,"comment founded")
        s
            = make_statement(instruction_sequence_p(i) ? entity_empty_label()
                                                       : gfc2pips_code2get_label(c),
                             STATEMENT_NUMBER_UNDEFINED,
                             STATEMENT_ORDERING_UNDEFINED,
                             comments,
                             //empty_comments,
                             i,
                             NULL,
                             NULL,
                             empty_extensions(),
			     make_synchronization_none());
        list_of_statements = gen_nconc(list_of_statements, CONS( STATEMENT,
            s,
            NIL ));

      }
    }
  }

  //FORMAT
  //we have the informations only at the end, (<=>now)
  list gfc2pips_format_p = gfc2pips_format;
  list gfc2pips_format2_p = gfc2pips_format2;
  /*  fprintf( stderr,
   "list of formats: %p %zu\n",
   gfc2pips_format,
   gen_length( gfc2pips_format ) );*/
  list list_of_statements_format = NULL;
  while(gfc2pips_format_p) {
    i
        = MakeZeroOrOneArgCallInst("FORMAT",
                                   (expression)gfc2pips_format_p->car.e);
    statement s =
        make_statement(gfc2pips_int2label((_int)gfc2pips_format2_p->car.e),
                       STATEMENT_NUMBER_UNDEFINED,
                       STATEMENT_ORDERING_UNDEFINED,
                       //comments,
                       empty_comments,
                       i,
                       NULL,
                       NULL,
                       empty_extensions(),
		       make_synchronization_none());
    //unlike the classical method, we don't know if we have had a first statement (data inst)
    list_of_statements_format = gen_nconc(list_of_statements_format,
                                          CONS( STATEMENT, s, NIL ));
    POP( gfc2pips_format_p );
    POP( gfc2pips_format2_p );
  }
  list_of_statements = gen_nconc(list_of_statements_format, list_of_statements);

  if(list_of_statements) {
    return make_instruction_block(list_of_statements);//make a sequence <=> make_instruction_sequence(make_sequence(list_of_statements));
  } else {
    fprintf(stderr, "Warning ! no instruction dumped => very bad\n");
    return make_instruction_block(CONS( STATEMENT,
        instruction_to_statement( make_instruction_block( NULL ) ),
        NIL ));
  }
}
/**
 * Build an instruction sequence
 * @brief same as {func}__TOP but without the declarations
 * @param ns the top-level entity from gfc. We need it to retrieve some more informations
 * @param c the struct containing information about the instruction
 */
instruction gfc2pips_code2instruction(gfc_code* c, bool force_sequence) {
  list list_of_statements;
  instruction i = instruction_undefined;
  force_sequence = true;
  if(!c) {
    if(force_sequence) {
      //fprintf(stderr,"WE HAVE GOT A PROBLEM, SEQUENCE WITHOUT ANYTHING IN IT !\nSegfault soon ...\n");
      return make_instruction_block(CONS( STATEMENT,
          instruction_to_statement( make_instruction_block( NULL ) ),
          NIL ));
    } else {
      //fprintf(stderr,"Undefined code\n");
      return make_instruction_block(NULL);
    }
  }
  //No block, only one instruction
  //if(!c->next && !force_sequence )return gfc2pips_code2instruction_(c);

  //create a sequence and put everything into it ? is it right ?
  list_of_statements = NULL;

  //entity l = gfc2pips_code2get_label(c);
  do {
    if(c && c->op == EXEC_SELECT) {
      list_of_statements
          = gen_nconc(list_of_statements, gfc2pips_dumpSELECT(c));
      c = c->next;
      if(list_of_statements)
        break;
    } else {
      i = gfc2pips_code2instruction_(c);
      if(i != instruction_undefined) {
        string comments = gfc2pips_get_comment_of_code(c);//fprintf(stderr,"comment founded")
        list_of_statements = CONS( STATEMENT,
				   make_statement( gfc2pips_code2get_label( c ),
						   STATEMENT_NUMBER_UNDEFINED,
						   STATEMENT_ORDERING_UNDEFINED,
						   comments,
						   //empty_comments,
						   i,
						   NIL,
						   NULL,
						   empty_extensions( ),
						   make_synchronization_none()),
				   list_of_statements );
      }
    }
    c = c->next;
  } while(i == instruction_undefined && c);

  //statement_label((statement)list_of_statements->car.e) = gfc2pips_code2get_label(c);

  for (; c; c = c->next) {
    statement s = statement_undefined;
    //l = gfc2pips_code2get_label(c);
    //on lie l'instruction suivante à la courante
    //fprintf(stderr,"Dump the following instructions\n");
    //CONS(STATEMENT,instruction_to_statement(gfc2pips_code2instruction_(c)),list_of_statements);
    int curr_label_num = gfc2pips_last_created_label;
    if(c && c->op == EXEC_SELECT) {
      list_of_statements
          = gen_nconc(list_of_statements, gfc2pips_dumpSELECT(c));
    } else {
      i = gfc2pips_code2instruction_(c);
      //si dernière boucle == c alors on doit ajouter un statement continue sur le label <curr_label_num>
      if(i != instruction_undefined) {
        string comments = gfc2pips_get_comment_of_code(c);//fprintf(stderr,"comment founded")
        s = make_statement(gfc2pips_code2get_label(c),
                           STATEMENT_NUMBER_UNDEFINED,
                           STATEMENT_ORDERING_UNDEFINED,
                           comments,
                           //empty_comments,
                           i,
                           NULL,
                           NULL,
                           empty_extensions(),
			   make_synchronization_none());
        if(s != statement_undefined) {
          list_of_statements = gen_nconc(list_of_statements, CONS( STATEMENT,
              s,
              NIL ));
        }
        //TODO: if it is a loop and there is an EXIT statement
        if(gfc2pips_get_last_loop() == c) {
          s = make_continue_statement(gfc2pips_int2label(curr_label_num - 1));
          list_of_statements = gen_nconc(list_of_statements, CONS( STATEMENT,
              s,
              NIL ));
        }
      }
    }
    /*
     * if we have got a label like a
     * ----------------------
     * do LABEL while (expr)
     *    statement
     * LABEL continue
     * ----------------------
     * we do need to make a continue statement BUT this will crash the program in some cases
     * PARSED_PRINTED_FILE is okay, but not PRINTED_FILE so we have to find out why
     */
  }
  return make_instruction_block(list_of_statements);
}

/**
 * @brief this function create an atomic statement, no block of data
 * @param c the instruction to translate from gfc
 * @return the statement equivalent in PIPS
 * never call this function except in gfc2pips_code2instruction or in recursive mode
 */
instruction gfc2pips_code2instruction_(gfc_code* c) {
  instruction return_instruction = instruction_undefined;

  //do we have a label ?
  //if(c->here){}

  gfc2pips_debug(8,"Start\n");
  switch(c->op) {
    /* an instruction without anything => continue statement*/
    case EXEC_NOP:
    case EXEC_CONTINUE:
      gfc2pips_debug(5, "Translation of CONTINUE\n");

      return_instruction = make_continue_instruction();
      break;
    case EXEC_INIT_ASSIGN:
    case EXEC_ASSIGN: {
      /* Beware, left hand side of assignment cannot be a TOP-LEVEL entity
       * If it is an assignment to a function, some complications are to be
       * expected
       */
      expression left_hand_side = gfc2pips_expr2expression(c->expr);
      expression right_hand_side = gfc2pips_expr2expression(c->expr2);
      return_instruction = make_assign_instruction(left_hand_side,
                                                   right_hand_side);
      break;
    }
    case EXEC_POINTER_ASSIGN: {
      gfc2pips_debug(5, "Translation of assign POINTER ASSIGN\n");

      list list_of_arguments = CONS( EXPRESSION,
          gfc2pips_expr2expression( c->expr2 ),
          NIL );

      entity e = CreateIntrinsic(ADDRESS_OF_OPERATOR_NAME);
      call call_ = make_call(e, list_of_arguments);
      expression ex = make_expression(make_syntax_call(call_),
                                      normalized_undefined);
      return_instruction
          = make_assign_instruction(gfc2pips_expr2expression(c->expr), ex);
      break;
    }
    case EXEC_GOTO: {
      gfc2pips_debug(5, "Translation of GOTO\n");
      entity lbl = gfc2pips_code2get_label2(c);
      return_instruction = make_instruction_goto(make_continue_statement(lbl));
      break;
    }
    case EXEC_CALL:
    case EXEC_ASSIGN_CALL: {
      gfc2pips_debug(5,"Translation of %s\n",
          c->op==EXEC_CALL?"CALL":"ASSIGN_CALL");

      gfc_symbol* symbol = NULL;

      /* Get the symbol that represent the function called */
      if(c->resolved_sym) {
        symbol = c->resolved_sym;
      } else if(c->symtree) {
        symbol = c->symtree->n.sym;
      } else {
        pips_user_error( "We do not have a symbol to call!!\n" );
      }

      /* Convert symbol in a pips entity */
      entity called_function = gfc2pips_symbol2entity(symbol);

      /* Get the arglist for the call */
      list list_of_arguments = gfc2pips_arglist2arglist(c->ext.actual);

      /* Fix the storage */
      if(entity_storage(called_function) == storage_undefined) {
        gfc2pips_debug(1,"Storage rom !! %s\n",entity_name(called_function));
        entity_storage(called_function) = make_storage_rom();
      }

      /* Fix the initial value (intrinsic only) */
      if(entity_initial(called_function) == value_undefined) {
        entity_initial(called_function) = make_value(is_value_intrinsic,
                                                     called_function);
        pips_user_warning("Here we make an intrinsic since we "
            "didn't resolve symbol : %s\n",
            gfc2pips_get_safe_name( symbol->name ));
      }

      /* Fix the type of the function */
      if(entity_type(called_function) == type_undefined) {
        gfc2pips_debug(2,"Type is undefined %s\n",entity_name(called_function));
        /* invent list of parameters */
        list param_of_call = gen_copy_seq(list_of_arguments);
        list param_of_call_p = param_of_call;
        /* need to translate list_of_arguments in sthg */
        while(param_of_call_p) {
          entity _new =
              (entity)gen_copy_tree((gen_chunk *)param_of_call_p->car.e);
          entity_name(_new) = "toto"; // hum....
          gfc2pips_debug(2,
              "%s - %s",
              entity_name((entity)param_of_call_p->car.e),
              entity_name(_new) );
          param_of_call_p->car.e = _new;
          POP( param_of_call_p );
        }

        pips_user_warning("Type is undefined for function '%s',"
            " thus we make it 'overloaded'\n",
            gfc2pips_get_safe_name( symbol->name ));

        entity_type(called_function)
            = make_type_functional(make_functional(NULL, MakeOverloadedResult()));
      }

      /* make the call now */
      call call_ = make_call(called_function, list_of_arguments);

      /* add to callees */
      gfc2pips_add_to_callees(called_function);

      return_instruction = make_instruction_call(call_);
      break;
    }
    case EXEC_COMPCALL: {
      /* Function (or subroutine) call of a procedure pointer component or
       * type-bound procedure
       */
      gfc2pips_debug(5, "Translation of COMPCALL\n");
      break;
    }
    case EXEC_RETURN: {
      //we shouldn't even dump that for main entities
      gfc2pips_debug(5, "Translation of RETURN\n");
      entity e = CreateIntrinsic(RETURN_FUNCTION_NAME);

      if(c->expr) {
        expression args = gfc2pips_expr2expression(c->expr);
        return_instruction = MakeUnaryCallInst(e, args);
      } else {
        return_instruction = MakeNullaryCallInst(e);
      }
      break;
    }
    case EXEC_PAUSE: {
      gfc2pips_debug(5, "Translation of PAUSE\n");
      entity e = CreateIntrinsic(PAUSE_FUNCTION_NAME);

      list args = NULL;
      if(c->expr) {
        expression args = gfc2pips_expr2expression(c->expr);
        return_instruction = MakeUnaryCallInst(e, args);
      } else {
        return_instruction = MakeNullaryCallInst(e);
      }
      break;
    }
    case EXEC_STOP: {
      gfc2pips_debug(5, "Translation of STOP\n");
      entity e = CreateIntrinsic(STOP_FUNCTION_NAME);

      if(c->expr) {
        expression args = gfc2pips_expr2expression(c->expr);
        return_instruction = MakeUnaryCallInst(e, args);
      } else {
        return_instruction = MakeNullaryCallInst(e);
      }
      break;
    }
    case EXEC_ARITHMETIC_IF: {
      gfc2pips_debug(5, "Translation of ARITHMETIC IF\n");
      expression e = gfc2pips_expr2expression(c->expr);
      expression e1 = MakeBinaryCall(CreateIntrinsic(LESS_THAN_OPERATOR_NAME),
                                     e,
                                     int_to_expression(0));
      expression e2 = MakeBinaryCall(CreateIntrinsic(EQUAL_OPERATOR_NAME),
                                     e,
                                     int_to_expression(0));
      expression e3 =
          MakeBinaryCall(CreateIntrinsic(LESS_OR_EQUAL_OPERATOR_NAME),
                         e,
                         int_to_expression(0));
      /*
       * we handle the labels doubled because it will never be checked
       * afterwards to combine/fuse
       */
      if(c->label->value == c->label2->value) {
        if(c->label->value == c->label3->value) {
          /* WTF ? Same destination whatever the result of the test is ?
           * let's put a goto !
           */
          entity lbl = gfc2pips_code2get_label2(c);
          statement continue_stmt = make_continue_statement(lbl);
          return_instruction = make_instruction_goto(continue_stmt);
        } else {
          //.LE. / .GT.
          statement
              s2 =
                  instruction_to_statement(make_instruction_goto(make_continue_statement(gfc2pips_code2get_label3(c))));
          statement
              s3 =
                  instruction_to_statement(make_instruction_goto(make_continue_statement(gfc2pips_code2get_label4(c))));
          return_instruction = make_instruction_test(make_test(e3, s2, s3));
        }
      } else if(c->label2->value == c->label3->value) {
        //.LT. / .GE.
        statement
            s2 =
                instruction_to_statement(make_instruction_goto(make_continue_statement(gfc2pips_code2get_label2(c))));
        statement
            s3 =
                instruction_to_statement(make_instruction_goto(make_continue_statement(gfc2pips_code2get_label3(c))));
        return_instruction = make_instruction_test(make_test(e1, s2, s3));
      } else {
        statement
            s1 =
                instruction_to_statement(make_instruction_goto(make_continue_statement(gfc2pips_code2get_label2(c))));
        statement
            s2 =
                instruction_to_statement(make_instruction_goto(make_continue_statement(gfc2pips_code2get_label3(c))));
        statement
            s3 =
                instruction_to_statement(make_instruction_goto(make_continue_statement(gfc2pips_code2get_label4(c))));
        statement s =
            instruction_to_statement(make_instruction_test(make_test(e2,
                        s2,
                        s3)));
        return_instruction = make_instruction_test(make_test(e1, s1, s));
      }
      break;
    }

    case EXEC_IF: {
      gfc2pips_debug(5, "Translation of IF\n");
      /*
       * IF - ELSE IF - ELSE IF - .... - ELSE
       * is handled with a chain list.
       * next is the current "IF" (or ELSEIF or ELSE) and
       * block is the next test (if any)
       *
       * We handled this chain list recursively
       *
       */
      gfc_code* d = c->block;
      if(!d) {
        /* WTF ? */
        return_instruction = make_instruction_block(NULL);
      } else if(!d->expr) {
        /* No test in an if ? we are at the last ELSE statement for an ELSE IF
         */
        if(d->next) {
          return_instruction = gfc2pips_code2instruction(d->next, true);
        } else {
          return_instruction = make_instruction_block(NULL);
        }
      } else if(!d->next) {
        return_instruction = make_instruction_block(NULL);
      } else {
        /* Let's produce an IF statement */

        expression e = gfc2pips_expr2expression(d->expr);
        statement s_if = statement_undefined;
        statement s_else = statement_undefined;

        /* Recursive call */
        instruction s_if_i = gfc2pips_code2instruction(d->next, false);

        message_assert( "Can't produce if instruction !\n",
            s_if_i != instruction_undefined );

        s_if = instruction_to_statement(s_if_i);
        statement_label(s_if) = gfc2pips_code2get_label(d->next);

        /* Handle else */
        if(d->block) {
          instruction s_else_i = instruction_undefined;
          if(d->block->expr) { /* this is an ELSE IF */
            /* Recursive call */
            s_else_i = gfc2pips_code2instruction_(d);
          } else {/* No condition therefore we are in the last ELSE statement */
            /* Recursive call */
            s_else_i = gfc2pips_code2instruction(d->block->next, false);
          }
          message_assert( "s_else_i is defined\n", s_else_i !=instruction_undefined );
          s_else = instruction_to_statement(s_else_i);
          statement_label(s_else) = gfc2pips_code2get_label(d->block->next);
        } else {
          s_else = make_empty_block_statement();
        }

        return_instruction = test_to_instruction(make_test(e, s_if, s_else));

      }
      break;
    }
    case EXEC_DO: {
      gfc2pips_debug(5, "Translation of DO\n");
      //      /* keep trace of enclosing loops *
      gfc2pips_push_loop(c);

      /* recursive call, get loop body */
      instruction do_i = gfc2pips_code2instruction(c->block->next, true);
      message_assert( "first instruction defined", do_i
          !=instruction_undefined );
      statement s = instruction_to_statement(do_i);

      /*
       * it would be perfect if we knew there is a EXIT or a CYCLE in the loop,
       * do not add if already one (then how to stock the label ?)
       * add to s a continue statement at the end to make
       * cycle/continue statements
       */
      /*
       list list_of_instructions =
       sequence_statements(instruction_sequence(statement_instruction(s)));
       entity lbl = gfc2pips_int2label( gfc2pips_last_created_label );
       list_of_instructions = gen_nconc(list_of_instructions,
       CONS(STATEMENT,
       make_continue_statement( lbl ),
       NULL)
       );
       gfc2pips_last_created_label -= gfc2pips_last_created_label_step;
       sequence_statements(instruction_sequence(statement_instruction(s)))
       = list_of_instructions;
       */

      range r = make_range(gfc2pips_expr2expression(c->ext.iterator->start),
                           gfc2pips_expr2expression(c->ext.iterator->end),
                           gfc2pips_expr2expression(c->ext.iterator->step));//lower, upper, increment

      entity loop_index = gfc2pips_expr2entity(c->ext.iterator->var);

      // Fixme : what about parallel loop ? Check gfc...
      pips_loop w = make_loop(loop_index,
                              r,
                              s,
                              gfc2pips_code2get_label2(c),
                              make_execution_sequential(),
                              NULL);
      gfc2pips_pop_loop();
      return_instruction = make_instruction_loop(w);

      break;
    }
    case EXEC_DO_WHILE: {
      gfc2pips_debug(5, "Translation of DO WHILE\n");
      /* keep trace of enclosing loops */
      gfc2pips_push_loop(c);

      /* Get body (recursive call) */
      instruction do_i = gfc2pips_code2instruction(c->block->next, true);
      message_assert( "first instruction defined",
          do_i!=instruction_undefined );
      statement s = instruction_to_statement(do_i);

      /*
       * it would be perfect if we knew there is a EXIT or a CYCLE in the loop,
       * do not add if already one (then how to stock the label ?)
       * add to s a continue statement at the end to make
       * cycle/continue statements
       */
      /*
       list list_of_instructions =
       sequence_statements(instruction_sequence(statement_instruction(s)));
       entity lbl = gfc2pips_int2label( gfc2pips_last_created_label );
       list_of_instructions = gen_nconc(list_of_instructions,
       CONS(STATEMENT,
       make_continue_statement( lbl ),
       NULL)
       );
       gfc2pips_last_created_label -= gfc2pips_last_created_label_step;
       sequence_statements(instruction_sequence(statement_instruction(s)))
       = list_of_instructions;
       */

      statement_label(s) = gfc2pips_code2get_label(c->block->next);
      whileloop w = make_whileloop(gfc2pips_expr2expression(c->expr),
                                   s,
                                   gfc2pips_code2get_label2(c),
                                   make_evaluation_before());
      gfc2pips_pop_loop();
      return_instruction = make_instruction_whileloop(w);
      break;
    }
    case EXEC_CYCLE: {
      gfc2pips_debug(5, "Translation of CYCLE\n");
      gfc_code* loop_c = gfc2pips_get_last_loop();
      entity label = entity_undefined;
      label = gfc2pips_int2label(gfc2pips_last_created_label);
      return_instruction
          = make_instruction_goto(make_continue_statement(label));
      break;
    }
    case EXEC_EXIT: { // FIXME : is the following really an exit ?
      gfc2pips_debug(5, "Translation of EXIT\n");
      gfc_code* loop_c = gfc2pips_get_last_loop();
      entity label = entity_undefined;
      label = gfc2pips_int2label(gfc2pips_last_created_label - 1);
      return_instruction
          = make_instruction_goto(make_continue_statement(label));
      break;
    }
    case EXEC_ALLOCATE:
    case EXEC_DEALLOCATE: {
      gfc2pips_debug(5, "Translation of %s\n",
          c->op==EXEC_ALLOCATE?"ALLOCATE":"DEALLOCATE");
      list lci = NULL;
      gfc_alloc *a;

      entity e =
          CreateIntrinsic(c->op == EXEC_ALLOCATE ? ALLOCATE_FUNCTION_NAME
                                                 : DEALLOCATE_FUNCTION_NAME);

      /* some problem inducted by the prettyprinter output become :
       * DEALLOCATE(variable, STAT=, I)
       * (note STAT= without nothing else)
       */
      if(c->expr) {
        gfc2pips_debug(5,"Handling STAT=\n");
        lci = gfc2pips_exprIO("STAT=", c->expr, NULL);
      }
      for (a = c->ext.alloc_list; a; a = a->next) {
        lci = CONS( EXPRESSION, gfc2pips_expr2expression( a->expr ), lci );
      }
      return_instruction = make_instruction_call(make_call(e, gen_nconc(lci,
                                                                        NULL)));
      break;
    }
    case EXEC_OPEN: {
      gfc2pips_debug(5, "Translation of OPEN\n");

      entity e = CreateIntrinsic(OPEN_FUNCTION_NAME);

      list lci = NULL;
      gfc_open * o = c->ext.open;

      //We have to build the list in the opposite order it should be displayed

      if(o->err)
        lci = gfc2pips_exprIO2("ERR=", o->err->value, lci);
      if(o->asynchronous)
        lci = gfc2pips_exprIO("ASYNCHRONOUS=", o->asynchronous, lci);
      if(o->convert)
        lci = gfc2pips_exprIO("CONVERT=", o->convert, lci);
      if(o->sign)
        lci = gfc2pips_exprIO("SIGN=", o->sign, lci);
      if(o->round)
        lci = gfc2pips_exprIO("ROUND=", o->round, lci);
      if(o->encoding)
        lci = gfc2pips_exprIO("ENCODING=", o->encoding, lci);
      if(o->decimal)
        lci = gfc2pips_exprIO("DECIMAL=", o->decimal, lci);
      if(o->pad)
        lci = gfc2pips_exprIO("PAD=", o->pad, lci);
      if(o->delim)
        lci = gfc2pips_exprIO("DELIM=", o->delim, lci);
      if(o->action)
        lci = gfc2pips_exprIO("ACTION=", o->action, lci);
      if(o->position)
        lci = gfc2pips_exprIO("POSITION=", o->position, lci);
      if(o->blank)
        lci = gfc2pips_exprIO("BLANK=", o->blank, lci);
      if(o->recl)
        lci = gfc2pips_exprIO("RECL=", o->recl, lci);
      if(o->form)
        lci = gfc2pips_exprIO("FORM=", o->form, lci);
      if(o->access)
        lci = gfc2pips_exprIO("ACCESS=", o->access, lci);
      if(o->status)
        lci = gfc2pips_exprIO("STATUS=", o->status, lci);
      if(o->file)
        lci = gfc2pips_exprIO("FILE=", o->file, lci);
      if(o->iostat)
        lci = gfc2pips_exprIO("IOSTAT=", o->iostat, lci);
      if(o->iomsg)
        lci = gfc2pips_exprIO("IOMSG=", o->iomsg, lci);
      if(o->unit)
        lci = gfc2pips_exprIO("UNIT=", o->unit, lci);

      return_instruction = make_instruction_call(make_call(e, gen_nconc(lci,
                                                                        NULL)));

      break;
    }
    case EXEC_CLOSE: {
      gfc2pips_debug(5, "Translation of CLOSE\n");

      entity e = CreateIntrinsic(CLOSE_FUNCTION_NAME);

      list lci = NULL;
      gfc_close * o = c->ext.close;

      if(o->err)
        lci = gfc2pips_exprIO2("ERR=", o->err->value, lci);
      if(o->status)
        lci = gfc2pips_exprIO("STATUS=", o->status, lci);
      if(o->iostat)
        lci = gfc2pips_exprIO("IOSTAT=", o->iostat, lci);
      if(o->iomsg)
        lci = gfc2pips_exprIO("IOMSG=", o->iomsg, lci);
      if(o->unit)
        lci = gfc2pips_exprIO("UNIT=", o->unit, lci);
      return_instruction = make_instruction_call(make_call(e, gen_nconc(lci,
                                                                        NULL)));
      break;
    }
    case EXEC_BACKSPACE:
    case EXEC_ENDFILE:
    case EXEC_REWIND:
    case EXEC_FLUSH: {
      const char* str;
      if(c->op == EXEC_BACKSPACE)
        str = BACKSPACE_FUNCTION_NAME;
      else if(c->op == EXEC_ENDFILE)
        str = ENDFILE_FUNCTION_NAME;
      else if(c->op == EXEC_REWIND)
        str = REWIND_FUNCTION_NAME;
      else if(c->op == EXEC_FLUSH)
        str = "flush"; // FIXME, not implemented
      else
        pips_user_error( "Your computer is mad\n" );//no other possibility

      entity e = CreateIntrinsic((string)str);

      gfc_filepos *fp;
      fp = c->ext.filepos;

      list lci = NULL;
      if(fp->err)
        lci = gfc2pips_exprIO2("ERR=", fp->err->value, lci);
      if(fp->iostat)
        lci = gfc2pips_exprIO("UNIT=", fp->iostat, lci);
      if(fp->iomsg)
        lci = gfc2pips_exprIO("UNIT=", fp->iomsg, lci);
      if(fp->unit)
        lci = gfc2pips_exprIO("UNIT=", fp->unit, lci);
      return_instruction = make_instruction_call(make_call(e, gen_nconc(lci,
                                                                        NULL)));
      break;
    }
    case EXEC_INQUIRE: {
      gfc2pips_debug(5, "Translation of INQUIRE\n");

      entity e = CreateIntrinsic(INQUIRE_FUNCTION_NAME);

      list lci = NULL;
      gfc_inquire *i = c->ext.inquire;

      if(i->err)
        lci = gfc2pips_exprIO2("ERR=", i->err->value, lci);
      if(i->id)
        lci = gfc2pips_exprIO("ID=", i->id, lci);
      if(i->size)
        lci = gfc2pips_exprIO("SIZE=", i->size, lci);
      if(i->sign)
        lci = gfc2pips_exprIO("SIGN=", i->sign, lci);
      if(i->round)
        lci = gfc2pips_exprIO("ROUND=", i->round, lci);
      if(i->pending)
        lci = gfc2pips_exprIO("PENDING=", i->pending, lci);
      if(i->encoding)
        lci = gfc2pips_exprIO("ENCODING=", i->encoding, lci);
      if(i->decimal)
        lci = gfc2pips_exprIO("DECIMAL=", i->decimal, lci);
      if(i->asynchronous)
        lci = gfc2pips_exprIO("ASYNCHRONOUS=", i->asynchronous, lci);
      if(i->convert)
        lci = gfc2pips_exprIO("CONVERT=", i->convert, lci);
      if(i->pad)
        lci = gfc2pips_exprIO("PAD=", i->pad, lci);
      if(i->delim)
        lci = gfc2pips_exprIO("DELIM=", i->delim, lci);
      if(i->readwrite)
        lci = gfc2pips_exprIO("READWRITE=", i->readwrite, lci);
      if(i->write)
        lci = gfc2pips_exprIO("WRITE=", i->write, lci);
      if(i->read)
        lci = gfc2pips_exprIO("READ=", i->read, lci);
      if(i->action)
        lci = gfc2pips_exprIO("ACTION=", i->action, lci);
      if(i->position)
        lci = gfc2pips_exprIO("POSITION=", i->position, lci);
      if(i->blank)
        lci = gfc2pips_exprIO("BLANK=", i->blank, lci);
      if(i->nextrec)
        lci = gfc2pips_exprIO("NEXTREC=", i->nextrec, lci);
      if(i->recl)
        lci = gfc2pips_exprIO("RECL=", i->recl, lci);
      if(i->unformatted)
        lci = gfc2pips_exprIO("UNFORMATTED=", i->unformatted, lci);
      if(i->formatted)
        lci = gfc2pips_exprIO("FORMATTED=", i->formatted, lci);
      if(i->form)
        lci = gfc2pips_exprIO("FORM=", i->form, lci);
      if(i->direct)
        lci = gfc2pips_exprIO("DIRECT=", i->direct, lci);
      if(i->sequential)
        lci = gfc2pips_exprIO("SEQUENTIAL=", i->sequential, lci);
      if(i->access)
        lci = gfc2pips_exprIO("ACCESS=", i->access, lci);
      if(i->name)
        lci = gfc2pips_exprIO("NAME=", i->name, lci);
      if(i->named)
        lci = gfc2pips_exprIO("NAMED=", i->named, lci);
      if(i->number)
        lci = gfc2pips_exprIO("NUMBER=", i->number, lci);
      if(i->opened)
        lci = gfc2pips_exprIO("OPENED=", i->opened, lci);
      if(i->exist)
        lci = gfc2pips_exprIO("EXIST=", i->exist, lci);
      if(i->iostat)
        lci = gfc2pips_exprIO("IOSTAT=", i->iostat, lci);
      if(i->iomsg)
        lci = gfc2pips_exprIO("IOMSG=", i->iomsg, lci);
      if(i->file)
        lci = gfc2pips_exprIO("FILE=", i->file, lci);
      if(i->unit)
        lci = gfc2pips_exprIO("UNIT=", i->unit, lci);
      return_instruction = make_instruction_call(make_call(e, gen_nconc(lci,
                                                                        NULL)));
      break;
    }
    case EXEC_READ:
    case EXEC_WRITE: {
      gfc2pips_debug(5, "Translation of %s\n",c->op==EXEC_WRITE?"PRINT":"READ");
      //yeah ! we've got an intrinsic
      gfc_code *d = c;
      gfc_dt *dt = d->ext.dt;

      pips_assert("dt can't be NULL",dt!=NULL);

      /* Create the entity that will be "called", READ or WRITE only */
      entity e = entity_undefined;
      if(c->op == EXEC_WRITE) {
        //print or write ? print is only a particular case of write
        e = CreateIntrinsic(WRITE_FUNCTION_NAME);
      } else {
        e = CreateIntrinsic(READ_FUNCTION_NAME);
      }

      expression std = expression_undefined;
      expression fmt = expression_undefined;
      list lci = NULL;

      /* lci will be set in reverse order, so we have to put data first */

      /* Get the datas in reverse order */
      list datas = NIL;
      for (c = c->block->next; c; c = c->next) {
        datas = gen_cons(c, datas);
      }
      FOREACH( CHUNKP, _c, datas) {
        gfc_code *c = (gfc_code *)_c;
        if(c->expr) {
          lci = CONS(EXPRESSION,gfc2pips_expr2expression(c->expr),lci);
        } else {
          instruction i = gfc2pips_code2instruction_(c);
          if(instruction_loop_p(i)) {
            /*
             * We have to convert manually a loop to a call to a DO-IMPLIED
             */
            loop l = instruction_loop(i);
            lci = CONS(EXPRESSION,loop_to_implieddo(l),lci);
          } else if(instruction_call_p(i)) {
            lci = CONS(EXPRESSION,call_to_expression(instruction_call(i)),lci);
          } else {
            if(c->op != EXEC_DT_END) {
              pips_user_warning("We don't know how to handle op : %d\n", c->op);
            }
            continue;
          }
        }
        /* Separator is IO_LIST */
        lci = CONS( EXPRESSION,
            MakeCharacterConstantExpression( "IOLIST=" ),
            lci);
      }

      if(dt->format_expr) { //if no format ; it is standard
        fmt = gfc2pips_expr2expression(dt->format_expr);
      } else if(dt->format_label && dt->format_label->value != -1) {
        if(dt->format_label->format) {

          if(dt->format_label->value) {
            /* It is a reference to a FORMAT
             * we do have a label somewhere with a format
             */
            fmt = gfc2pips_int2expression(dt->format_label->value);

            /*
             * we check if we have already the label or not
             * the label is associated to a FORMAT =>
             * we check if we already have this format
             */
            if(gen_find_eq((void *)(_int)dt->format_label->value,
                           gfc2pips_format2) == gen_chunk_undefined) {
              /*
               * we have to push the current FORMAT in a list, we will dump it at
               * the very, very TOP. we need to change the expression, a FORMAT
               * statement doesn't have quotes around it
               */
              expression fmt_expr =
                  gfc2pips_expr2expression(dt->format_label->format);
              /* delete supplementary quotes */
              char *str = entity_name(
                  call_function(syntax_call(expression_syntax(fmt_expr))));
              int curr_char_indice = 0, curr_char_indice_cible = 0,
                  length_curr_format = strlen(str);
              for (; curr_char_indice_cible < length_curr_format - 1; curr_char_indice++, curr_char_indice_cible++) {
                if(str[curr_char_indice_cible] == '\'')
                  curr_char_indice_cible++;
                str[curr_char_indice] = str[curr_char_indice_cible];
              }
              str[curr_char_indice] = '\0';

              gfc2pips_format = gen_cons(fmt_expr, gfc2pips_format);
              gfc2pips_format2
                  = gen_cons((void *)(_int)dt->format_label->value,
                             gfc2pips_format2);
            }
          } else {
            // The format is given as an argument !
            fmt = gfc2pips_expr2expression(dt->format_label->format);
            //delete supplementary quotes
            char * str =
                entity_name(call_function(syntax_call(expression_syntax(fmt))));
            int curr_char_indice = 0, curr_char_indice_cible = 0,
                length_curr_format = strlen(str);
            for (; curr_char_indice_cible < length_curr_format - 1; curr_char_indice++, curr_char_indice_cible++) {
              if(str[curr_char_indice_cible] == '\'')
                curr_char_indice_cible++;
              str[curr_char_indice] = str[curr_char_indice_cible];
            }
            str[curr_char_indice] = '\0';
          }
        } else {
          //error or warning: we have bad code
          pips_user_error( "gfc2pips_code2instruction, No format for label\n" );
        }
      }

      /*
       * Handling of UNIT=
       */
      if(dt->io_unit) {
        bool has_to_generate_unit = FALSE; // Flag

        if(dt->io_unit->expr_type != EXPR_CONSTANT) {
          /* UNIT is not a constant, we have to print it */
          has_to_generate_unit = TRUE;
        } else if(d->op == EXEC_READ && mpz_get_si(dt->io_unit->value.integer)
            != 5 || d->op == EXEC_WRITE
            && mpz_get_si(dt->io_unit->value.integer) != 6) {
          /*
           * if the canal is 6, it is standard for write
           * if the canal is 5, it is standard for read
           */
          has_to_generate_unit = TRUE;
        }

        /* We have to print UNIT= ; it's not implicit */
        if(has_to_generate_unit) {
          std = gfc2pips_expr2expression(dt->io_unit);
        }
      }

      if(dt->err)
        lci = gfc2pips_exprIO2("ERR=", dt->err->value, lci);
      if(dt->end)
        lci = gfc2pips_exprIO2("END=", dt->end->value, lci);
      if(dt->eor)
        lci = gfc2pips_exprIO2("EOR=", dt->end->value, lci);

      if(dt->sign)
        lci = gfc2pips_exprIO("SIGN=", dt->sign, lci);
      if(dt->round)
        lci = gfc2pips_exprIO("ROUND=", dt->round, lci);
      if(dt->pad)
        lci = gfc2pips_exprIO("PAD=", dt->pad, lci);
      if(dt->delim)
        lci = gfc2pips_exprIO("DELIM=", dt->delim, lci);
      if(dt->decimal)
        lci = gfc2pips_exprIO("DECIMAL=", dt->decimal, lci);
      if(dt->blank)
        lci = gfc2pips_exprIO("BLANK=", dt->blank, lci);
      if(dt->asynchronous)
        lci = gfc2pips_exprIO("ASYNCHRONOUS=", dt->asynchronous, lci);
      if(dt->pos)
        lci = gfc2pips_exprIO("POS=", dt->pos, lci);
      if(dt->id)
        lci = gfc2pips_exprIO("ID=", dt->id, lci);
      if(dt->advance)
        lci = gfc2pips_exprIO("ADVANCE=", dt->advance, lci);
      if(dt->rec)
        lci = gfc2pips_exprIO("REC=", dt->rec, lci);
      if(dt->size)
        lci = gfc2pips_exprIO("SIZE=", dt->size, lci);
      if(dt->iostat)
        lci = gfc2pips_exprIO("IOSTAT=", dt->iostat, lci);
      if(dt->iomsg)
        lci = gfc2pips_exprIO("IOMSG=", dt->iomsg, lci);

      if(dt->namelist)
        lci = gfc2pips_exprIO3("NML=", (string)dt->namelist->name, lci);

      if(fmt == expression_undefined && !dt->rec && !dt->namelist) {
        fmt = MakeNullaryCall(CreateIntrinsic(LIST_DIRECTED_FORMAT_NAME));
      }

      if(fmt != expression_undefined) {
        //        fmt = MakeNullaryCall( CreateIntrinsic( LIST_DIRECTED_FORMAT_NAME ) );
        expression format = MakeCharacterConstantExpression("FMT=");
        lci = CONS( EXPRESSION, format, CONS( EXPRESSION, fmt, lci ) );

      }

      if(std == expression_undefined) {
        std = MakeNullaryCall(CreateIntrinsic(LIST_DIRECTED_FORMAT_NAME));
      }
      expression unite = MakeCharacterConstantExpression("UNIT=");
      lci = CONS( EXPRESSION, unite, CONS( EXPRESSION, std, lci ) );

      ifdebug(8) {
        gfc2pips_debug(8,"List of arg : \n");
        FOREACH(expression, e, lci)
        {
          print_expression(e);
        }
      }

      return_instruction = make_instruction_call(make_call(e, lci));

    }
      break;
    case EXEC_TRANSFER: {
      // FIXME, chained transfert ? c->block->next not null ?
      expression transfered = gfc2pips_expr2expression(c->expr);
      return_instruction = make_instruction_expression(transfered);
      break;
    }
    case EXEC_DT_END:
      /*
       * This is normal end of recursion when handling data parameter for
       * READ and WRITE statement. We have handle the data previously in
       * EXEC_TRANSFER.
       */
      break;

    case EXEC_OMP_ATOMIC:
    case EXEC_OMP_BARRIER:
    case EXEC_OMP_CRITICAL:
    case EXEC_OMP_FLUSH:
    case EXEC_OMP_DO:
    case EXEC_OMP_MASTER:
    case EXEC_OMP_ORDERED:
    case EXEC_OMP_PARALLEL:
    case EXEC_OMP_PARALLEL_DO:
    case EXEC_OMP_PARALLEL_SECTIONS:
    case EXEC_OMP_PARALLEL_WORKSHARE:
    case EXEC_OMP_SECTIONS:
    case EXEC_OMP_SINGLE:
    case EXEC_OMP_TASK:
    case EXEC_OMP_TASKWAIT:
    case EXEC_OMP_WORKSHARE:
      pips_user_warning("OpenMP is not supported !!\n");
      break;
    case EXEC_ENTRY:
      pips_user_warning( "Don't know What to do with entry ! (%s)\n",
          c->ext.entry->sym->name );
      break;
    case EXEC_LABEL_ASSIGN:
      pips_user_warning( "Don't know What to do with Label-assign ! (%d)\n",
          c->label->value);
      break;
    case EXEC_IOLENGTH:
      pips_user_warning( "Don't know What to do with IOLENGTH !\n");
      break;
    default:
      pips_user_warning( "not yet dumpable %d\n", (int) c->op );
  }

  if(return_instruction == instruction_undefined) {
    return_instruction = make_instruction_block(NULL);
  }

  return return_instruction;
}

expression gfc2pips_buildCaseTest(gfc_expr *test, gfc_case *cp) {
  expression range_expr = expression_undefined;
  expression tested_variable = gfc2pips_expr2expression(test);
  pips_assert("CASE expr require at least an high OR a low bound !",
      cp->low||cp->high);
  if(cp->low == cp->high) {
    // Exact bound
    range_expr = MakeBinaryCall(CreateIntrinsic(EQUAL_OPERATOR_NAME),
                                tested_variable,
                                gfc2pips_expr2expression(cp->low));
  } else {
    expression low = NULL, high = NULL;
    if(cp->low) {
      low = MakeBinaryCall(CreateIntrinsic(GREATER_OR_EQUAL_OPERATOR_NAME),
                           tested_variable,
                           gfc2pips_expr2expression(cp->low));
    }
    if(cp->high) {
      high = MakeBinaryCall(CreateIntrinsic(LESS_OR_EQUAL_OPERATOR_NAME),
                            tested_variable,
                            gfc2pips_expr2expression(cp->high));
    }

    if(low && !high) {
      range_expr = low;
    } else if(!low && high) {
      range_expr = high;
    } else {
      range_expr
          = MakeBinaryCall(CreateIntrinsic(AND_OPERATOR_NAME), low, high);
    }
  }
  return range_expr;
}

list gfc2pips_dumpSELECT(gfc_code *c) {
  list list_of_statements = NULL;
  gfc_case * cp;
  gfc_code *d = c->block;
  gfc2pips_debug(5,"dump of SELECT\n");

  if(c->here) {
    list_of_statements
        = gen_nconc(CONS( STATEMENT,
			  make_statement( gfc2pips_code2get_label( c ),
					  STATEMENT_NUMBER_UNDEFINED,
					  STATEMENT_ORDERING_UNDEFINED,
					  empty_comments,
					  make_instruction_call( make_call( CreateIntrinsic( CONTINUE_FUNCTION_NAME ),
									    NULL ) ),
					  NULL,
					  NULL,
					  empty_extensions( ),
					  make_synchronization_none()),
			  NULL ),
                    list_of_statements);
  }

  /*list_of_statements = CONS(STATEMENT,
   instruction_to_statement(
   make_assign_instruction(
   gfc2pips_expr2expression(c->expr),
   gfc2pips_expr2expression(c->expr2)
   )
   ),
   NULL
   );*/

  statement selectcase = NULL, current_case = NULL, default_stmt = NULL;
  for (; d; d = d->block) {
    gfc2pips_debug(5,"dump of SELECT CASE\n");
    //create a function with low/high returning a test in one go
    expression test_expr = expression_undefined;
    for (cp = d->ext.case_list; cp; cp = cp->next) {
      if(!cp->low && !cp->high) {
        // Default test case ... or error if test_expr != expression_undefined ?
        pips_assert("We should have default case, but it doesn't seem to be"
            "the case, aborting.\n",test_expr == expression_undefined);
        break;
      }
      if(test_expr == expression_undefined) {
        test_expr = gfc2pips_buildCaseTest(c->expr, cp);
      } else {
        test_expr = MakeBinaryCall(CreateIntrinsic(OR_OPERATOR_NAME),
                                   test_expr,
                                   gfc2pips_buildCaseTest(c->expr, cp));
      }
    }

    instruction s_if = gfc2pips_code2instruction(d->next, false);
    if(s_if != instruction_undefined) {
      statement casetest;
      if(test_expr == expression_undefined) {
        // Default case
        default_stmt = instruction_to_statement(s_if);
      } else {
        casetest = instruction_to_statement(test_to_instruction(make_test(test_expr,
                instruction_to_statement(s_if),
                make_empty_block_statement())));
        if(current_case != NULL) {
          free_statement(test_false(statement_test(current_case)));
          test_false(statement_test(current_case)) = casetest;
        }
        current_case = casetest;
        if(!selectcase) {
          selectcase = casetest;
        }
      }

    } else {
      pips_user_error( "in SELECT : CASE block is empty ?\n" );
    }
  }
  if(default_stmt) {
    if(current_case) {
      free_statement(test_false(statement_test(current_case)));
      test_false(statement_test(current_case)) = default_stmt;
    } else {
      selectcase = default_stmt;
    }
  }

  if(selectcase != NULL) {
    list_of_statements = gen_nconc(list_of_statements, CONS( STATEMENT,
        selectcase,
        NULL ));
  }
  return list_of_statements;
}

/**
 * @brief build a DATA statement, filling blanks with zeroes.
 *
 * TODO:
 * - add variables which tell when split the declaration in parts or not ?
 * - change this function into one returning a set of DATA statements for each sequence instead or filling with zeroes ? (add 0 is what will be done in the memory anyway)
 */
instruction gfc2pips_symbol2data_instruction(gfc_symbol *sym) {
  gfc2pips_debug(3,"%s\n",sym->name);
  entity e1 =
      FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, DATA_LIST_FUNCTION_NAME);
  entity_initial(e1) = make_value_unknown();

  entity tagged_entity = gfc2pips_symbol2entity(sym);
  expression tagged_expr = entity_to_expression(tagged_entity);

  /*
   * FIXME, this is just a fix because
   * entity_to_expression may call reference_to_expression that may
   * normalize the expression. This will further produce an invalid
   * output when using gen_write_tabulated in order to dump entities
   * so we unlink the normalized field, there is probably a leak there...
   */
  //  expression_normalized(tagged_expr) = normalized_undefined;

  list args1 = CONS( EXPRESSION, tagged_expr, NULL );/**/
  //add references for DATA variable(offset +-difference due to offset)
  //if(sym->component_access)

  //list of variables used int the data statement
  list init = CONS( EXPRESSION, make_call_expression( e1, args1 ), NULL );
  //list of values
  /*
   * data (z(i), i = min_off, max_off) /1+max_off-min_off*val/
   * gfc doesn't now the range between min_off and max_off it just state, one by one the value
   * how to know the range ? do we have to ?
   ** data z(offset+1) / value /
   * ? possible ?
   */
  list values = NULL;
  if(sym->value && sym->value->expr_type == EXPR_ARRAY) {
    gfc_constructor *constr = sym->value->value.constructor;
    gfc_constructor *prec = NULL;
    int i, j;
    for (; constr; constr = constr->next) {
      gfc2pips_debug( 9, "offset: %zu\trepeat: %zu\n", mpz_get_si(constr->n.offset), mpz_get_si(constr->repeat) );

      //add 0 to fill the gap at the beginning
      if(prec == NULL && mpz_get_si(constr->n.offset) > 0) {
        gfc2pips_debug(9,"we do not start the DATA statement at the beginning !\n");
        values = CONS( EXPRESSION,
            MakeBinaryCall( CreateIntrinsic( MULTIPLY_OPERATOR_NAME ),
                gfc2pips_int2expression( mpz_get_si( constr->n.offset ) ),
                gfc2pips_make_zero_for_symbol( sym ) ),
            values );
        /*for(i=0 , j=mpz_get_si(constr->n.offset) ; i<j ; i++ ){
         values = CONS( EXPRESSION, gfc2pips_make_zero_for_symbol(sym), values );
         }*/
      }
      //if precedent, we need to know if there has been a repetition of some kind
      if(prec) {
        int offset;
        //if there is a repetition, we need to compare to the end of it
        if(mpz_get_si(prec->repeat)) {
          offset = mpz_get_si(constr->n.offset) - mpz_get_si(prec->n.offset)
              - mpz_get_si(prec->repeat);
        } else {
          offset = mpz_get_si(constr->n.offset) - mpz_get_si(prec->n.offset);
        }

        //add 0 to fill the gaps between the values
        if(offset > 1) {
          gfc2pips_debug(9,"We have a gap in DATA %d\n",offset);
          values = CONS( EXPRESSION,
              MakeBinaryCall( CreateIntrinsic( MULTIPLY_OPERATOR_NAME ),
                  gfc2pips_int2expression( offset - 1 ),
                  gfc2pips_make_zero_for_symbol( sym ) ),
              values );
          /*for( i=1 , j=offset ; i<j ; i++ ){
           values = CONS( EXPRESSION, gfc2pips_make_zero_for_symbol(sym), values );
           }*/
        }
      }
      //if repetition on the current value, repeat, else just add
      if(mpz_get_si(constr->repeat)) {
        //if repeat  =>  offset*value
        values = CONS( EXPRESSION,
            MakeBinaryCall( CreateIntrinsic( MULTIPLY_OPERATOR_NAME ),
                gfc2pips_int2expression( mpz_get_si( constr->repeat ) ),
                gfc2pips_expr2expression( constr->expr ) ),
            values );
        /*for( i=0,j=mpz_get_si(constr->repeat) ; i<j ; i++ ){
         values = CONS( EXPRESSION, gfc2pips_expr2expression(constr->expr), values );
         }*/
      } else {
        values = CONS( EXPRESSION,
            gfc2pips_expr2expression( constr->expr ),
            values );
      }
      prec = constr;
    }

    //add 0 to fill the gap at the end
    //we patch the size of a single object a little
    int size_of_unit = gfc2pips_symbol2size(sym);
    if(sym->ts.type == BT_COMPLEX)
      size_of_unit *= 2;
    if(sym->ts.type == BT_CHARACTER)
      size_of_unit = 1;

    int total_size;
    SizeOfArray(tagged_entity, &total_size);
    gfc2pips_debug(9,"total size: %d\n",total_size);
    int offset_end;
    if(prec) {
      if(mpz_get_si(prec->repeat)) {
        offset_end = mpz_get_si(prec->n.offset) + mpz_get_si(prec->repeat);
      } else {
        offset_end = mpz_get_si(prec->n.offset);
      }
    }

    if(prec && offset_end + 1 < ((double)total_size) / (double)size_of_unit) {
      gfc2pips_debug(9,"We fill all the remaining space in the DATA %d\n",offset_end);
      values = CONS( EXPRESSION,
          MakeBinaryCall( CreateIntrinsic( MULTIPLY_OPERATOR_NAME ),
              gfc2pips_int2expression( total_size
                  / size_of_unit - offset_end - 1 ),
              gfc2pips_make_zero_for_symbol( sym ) ),
          values );
      /*for( i=1 , j=total_size/size_of_unit-offset_end ; i<j ; i++ ){
       values = CONS( EXPRESSION, gfc2pips_make_zero_for_symbol(sym), values );
       }*/
    }
    //fill in the remaining parts
    values = gen_nreverse(values);
  } else if(sym->value) {
    values = CONS( EXPRESSION, gfc2pips_expr2expression( sym->value ), NULL );
  } else {
    pips_user_error( "No value, incoherence\n" );
    return instruction_undefined;
  }

  entity e2 = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,
                                 STATIC_INITIALIZATION_FUNCTION_NAME);
  entity_initial(e2) = make_value_unknown();
  values = gfc2pips_reduce_repeated_values(values);
  list args2 = gen_nconc(init, values);
  call call_ = make_call(e2, args2);
  return make_instruction_call(call_);
}

expression gfc2pips_make_zero_for_symbol(gfc_symbol* sym) {
  int size_of_unit = gfc2pips_symbol2size(sym);
  if(sym->ts.type == BT_CHARACTER)
    size_of_unit = 1;
  if(sym->ts.type == BT_COMPLEX) {
    return MakeComplexConstantExpression(gfc2pips_real2expression(0.),
                                         gfc2pips_real2expression(0.));
  } else if(sym->ts.type == BT_REAL) {
    return gfc2pips_real2expression(0.);
  } else {
    return gfc2pips_int2expression(0);
  }
}

/**
 * @brief look for repeated values (integers or real) in the list (list for DATA instructions) and transform them in a FORTRAN repeat syntax
 *
 * DATA x /1,1,1/ =>  DATA x /3*1/
 *
 * TODO:
 * - look into the consistency issue
 * - add recognition of /3*1, 4*1/ to make it a /7*1/
 */
list gfc2pips_reduce_repeated_values(list l) {
  //return l;
  expression curr = NULL, prec = NULL;
  list local_list = l, last_pointer_on_list = NULL;
  int nb_of_occurences = 0;
  gfc2pips_debug(9, "begin reduce of values\n");
  while(local_list) {
    curr = (expression)local_list->car.e;
    /*gfc2pips_debug( 9,
     "is a call ? %s\n\tis a constant ? %s\n\tfunctional ? %s\n",
     syntax_call_p(expression_syntax(curr))?"true":"false",
     (
     syntax_call_p(expression_syntax(curr))
     && entity_constant_p(call_function(syntax_call(expression_syntax(curr))))
     ) ? "true" : "false",
     type_functional_p(entity_type(call_function(syntax_call(expression_syntax(curr))))) ? "true":"false"
     );*/
    if(expression_is_constant_p(curr)) {
      if(prec && expression_is_constant_p(prec)) {
        if(reference_variable(syntax_reference(expression_syntax(curr)))
            == reference_variable(syntax_reference(expression_syntax(prec)))) {
          gfc2pips_debug(10, "same as before\n");
          nb_of_occurences++;
        } else if(nb_of_occurences > 1) {
          //reduce
          gfc2pips_debug(9, "reduce1 %s %d\n",entity_name(reference_variable(syntax_reference(expression_syntax(prec)))),nb_of_occurences);
          last_pointer_on_list->car.e
              = MakeBinaryCall(CreateIntrinsic(MULTIPLY_OPERATOR_NAME),
                               gfc2pips_int2expression(nb_of_occurences),
                               prec);
          last_pointer_on_list->cdr = local_list;

          nb_of_occurences = 1;
          last_pointer_on_list = local_list;
        } else {
          gfc2pips_debug(10, "skip to next\n");
          nb_of_occurences = 1;
          last_pointer_on_list = local_list;
        }
      } else {
        gfc2pips_debug(10, "no previous\n");
        nb_of_occurences = 1;
        last_pointer_on_list = local_list;
      }
      prec = curr;
    } else {//we will not be able to reduce
      gfc2pips_debug(10, "not a constant\n");
      if(nb_of_occurences > 1) {
        //reduce
        gfc2pips_debug(
            9,
            "reduce2 %s %d %p\n",
            entity_name(reference_variable(syntax_reference(expression_syntax(prec)))),
            nb_of_occurences,
            last_pointer_on_list
        );
        if(last_pointer_on_list) {
          last_pointer_on_list->car.e
              = MakeBinaryCall(CreateIntrinsic(MULTIPLY_OPERATOR_NAME),
                               gfc2pips_int2expression(nb_of_occurences),
                               prec);
          last_pointer_on_list->cdr = local_list;
        } else {
          //an error has been introduced somewhere, last_pointer_on_list is NULL and it should point to the first repeated value
          pips_user_warning( "We don't know what to do ! We do not have a current pointer (and we should).\n" );
        }
      }
      nb_of_occurences = 0;//no dump, thus no increment
      last_pointer_on_list = local_list->cdr;//no correct reference needed
    }
    POP( local_list );
  }
  //a last sequence of data ?
  if(nb_of_occurences > 1) {
    //reduce
    gfc2pips_debug(9, "reduce3 %s %d\n",entity_name(reference_variable(syntax_reference(expression_syntax(prec)))),nb_of_occurences);
    last_pointer_on_list->car.e
        = MakeBinaryCall(CreateIntrinsic(MULTIPLY_OPERATOR_NAME),
                         gfc2pips_int2expression(nb_of_occurences),
                         prec);
    last_pointer_on_list->cdr = local_list;
    last_pointer_on_list = local_list;
  }
  gfc2pips_debug(9, "reduce of values done\n");
  return l;
}

entity gfc2pips_code2get_label(gfc_code *c) {
  if(!c) {
    return entity_empty_label();
  }

  gfc2pips_debug(9,
      "test label: %lu %lu %lu %lu\t"
      "next %lu block %lu %lu\n",
      (_int)(c->label?c->label->value:0),
      (_int)(c->label2?c->label2->value:0),
      (_int)(c->label3?c->label3->value:0),
      (_int)(c->here?c->here->value:0),
      (_int)c->next,
      (_int)c->block,
      (_int)c->expr
  );
  if(c->here) {
    return gfc2pips_int2label(c->here->value);
  }
  return entity_empty_label();
}

entity gfc2pips_code2get_label2(gfc_code *c) {
  if(!c)
    return entity_empty_label();
  gfc2pips_debug(9,
      "test label2: %lu %lu %lu %lu\t"
      "next %lu block %lu %lu\n",
      (_int)(c->label?c->label->value:0),
      (_int)(c->label2?c->label2->value:0),
      (_int)(c->label3?c->label3->value:0),
      (_int)(c->here?c->here->value:0),
      (_int)c->next,
      (_int)c->block,
      (_int)c->expr
  );
  if(c->label)
    return gfc2pips_int2label(c->label->value);
  return entity_empty_label();
}
entity gfc2pips_code2get_label3(gfc_code *c) {
  if(!c)
    return entity_empty_label();
  gfc2pips_debug(9,
      "test label2: %lu %lu %lu %lu\t"
      "next %lu block %lu %lu\n",
      (_int)(c->label?c->label->value:0),
      (_int)(c->label2?c->label2->value:0),
      (_int)(c->label3?c->label3->value:0),
      (_int)(c->here?c->here->value:0),
      (_int)c->next,
      (_int)c->block,
      (_int)c->expr
  );
  if(c->label)
    return gfc2pips_int2label(c->label2->value);
  return entity_empty_label();
}
entity gfc2pips_code2get_label4(gfc_code *c) {
  if(!c)
    return entity_empty_label();
  gfc2pips_debug(9,
      "test label2: %lu %lu %lu %lu\t"
      "next %lu block %lu %lu\n",
      (_int)(c->label?c->label->value:0),
      (_int)(c->label2?c->label2->value:0),
      (_int)(c->label3?c->label3->value:0),
      (_int)(c->here?c->here->value:0),
      (_int)c->next,
      (_int)c->block,
      (_int)c->expr
  );
  if(c->label)
    return gfc2pips_int2label(c->label3->value);
  return entity_empty_label();
}

/*
 * Translate an expression from a RI to another
 * for the moment only care about (a+b)*c   + - * / ** // .AND. .OR. .EQV. .NEQV. .NOT.
 */
expression gfc2pips_expr2expression(gfc_expr *expr) {
  message_assert( "Expr can't be null !\n", expr );
  /*
   * In GFC
   * p->value.op.op is the expression operator
   * p->value.op.op1 is the first argument
   * p->value.op.op2 is the second one
   */

  /* This is the expression that will be returned */
  expression returned_expr = expression_undefined;
  switch(expr->expr_type) {
    case EXPR_OP: {
      gfc2pips_debug(5, "Op : this is an (intrinsic) call)\n");
      const char *c = NULL;
      switch(expr->value.op.op) {
        // FIXME Replace all by PIPS #define like MULTIPLY_OPERATOR_NAME
        case INTRINSIC_UPLUS:
        case INTRINSIC_PLUS:
          c = "+";
          break;
        case INTRINSIC_UMINUS:
        case INTRINSIC_MINUS:
          c = "-";
          break;
        case INTRINSIC_TIMES:
          c = "*";
          break;
        case INTRINSIC_DIVIDE:
          c = "/";
          break;
        case INTRINSIC_POWER:
          c = "**";
          break;
        case INTRINSIC_CONCAT:
          c = "//";
          break;
        case INTRINSIC_AND:
          c = ".AND.";
          break;
        case INTRINSIC_OR:
          c = ".OR.";
          break;
        case INTRINSIC_EQV:
          c = ".EQV.";
          break;
        case INTRINSIC_NEQV:
          c = ".NEQV.";
          break;

        case INTRINSIC_EQ:
        case INTRINSIC_EQ_OS:
          c = ".EQ.";
          break;
        case INTRINSIC_NE:
        case INTRINSIC_NE_OS:
          c = ".NE.";
          break;
        case INTRINSIC_GT:
        case INTRINSIC_GT_OS:
          c = ".GT.";
          break;
        case INTRINSIC_GE:
        case INTRINSIC_GE_OS:
          c = ".GE.";
          break;
        case INTRINSIC_LT:
        case INTRINSIC_LT_OS:
          c = ".LT.";
          break;
        case INTRINSIC_LE:
        case INTRINSIC_LE_OS:
          c = ".LE.";
          break;

        case INTRINSIC_NOT:
          c = ".NOT.";
          break;

        case INTRINSIC_PARENTHESES:
          returned_expr = gfc2pips_expr2expression(expr->value.op.op1);
          break;
        default:
          pips_user_warning( "intrinsic not yet recognized: %d\n",
              (int) expr->value.op.op );
          break;
      }

      if(c) {
        gfc2pips_debug(6, "intrinsic recognized: %s\n",c);
        /* Get 1st arg with a recursive call */
        expression e1 = gfc2pips_expr2expression(expr->value.op.op1);

        /* Assertion */
        if(!e1 || e1 == expression_undefined) {
          pips_user_error( "intrinsic( (string)%s ) : 1st arg "
              "is null or undefined\n", c);
        }
        if(expr->value.op.op2 == NULL) {
          /* Unary call */
          switch(expr->value.op.op) {
            case INTRINSIC_UMINUS:
              returned_expr
                  = MakeFortranUnaryCall(CreateIntrinsic(UNARY_MINUS_OPERATOR_NAME),
                                         e1);
              break;
            case INTRINSIC_UPLUS:
              returned_expr = e1;
              break;
            case INTRINSIC_NOT:
              returned_expr
                  = MakeFortranUnaryCall(CreateIntrinsic(NOT_OPERATOR_NAME), e1);
              break;
            default:
              pips_user_error( "No second expression member for intrinsic %s\n",
                  c );
          }
        } else {
          /* Recursive call, get second expression member */
          expression e2 = gfc2pips_expr2expression(expr->value.op.op2);

          /* Assertion */
          if(!e2 || e2 == expression_undefined) {
            pips_user_error( "intrinsic( (string)%s ) : 2nd arg is null or undefined\n", c);
          }
          returned_expr = MakeBinaryCall(CreateIntrinsic((string)c), e1, e2);
        }
      }
      break;
    }
    case EXPR_VARIABLE: {
      gfc2pips_debug(5, "Variable\n");

      /* Entity corresponding to the Gfortran variable */
      entity ent_ref = gfc2pips_symbol2entity(expr->symtree->n.sym);

      /* Args list is in use for array reference */
      list args_list = NULL;
      syntax s = syntax_undefined;

      if(strcmp(CurrentPackage, entity_name(ent_ref)) == 0) {
        gfc2pips_debug(9,"Variable %s is put in return storage\n",entity_name(ent_ref));
        entity_storage(ent_ref) = make_storage_return(ent_ref);
        entity_type(ent_ref)
            = copy_type(functional_result( type_functional(entity_type(gfc2pips_main_entity)) ));
      } else if(entity_storage(ent_ref) == storage_undefined) {
        gfc2pips_debug(1,"Undefined storage ! Let's put in RAM !! %s\n",
            entity_name(ent_ref));

        int dynamicOffset = area_size(type_area(entity_type(DynamicArea)));
        entity_storage(ent_ref)
            = make_storage_ram(make_ram(get_current_module_entity(),
                                        DynamicArea,
                                        dynamicOffset,
                                        NIL));
        area_size(type_area(entity_type(DynamicArea)))
            += area_size(type_area(entity_type(ent_ref)));
        list layout = area_layout(type_area(entity_type(DynamicArea)));
        area_layout(type_area(entity_type(DynamicArea)))
            = gen_nconc(layout, CONS(entity,ent_ref,NULL));

      }

      if(expr->ref) {
        gfc_ref *r = expr->ref;
        /*
         * We have an array reference, get the indices !
         *
         * I'm not sure a loop is useful here...
         */
        while(r) {
          switch(r->type) {
            case REF_ARRAY: {
              /* It an array ! */
              gfc2pips_debug(9,"ref array : %s\n",entity_name(ent_ref));
              args_list = gfc2pips_array_ref2indices(&r->u.ar);
              break;
            }
            case REF_SUBSTRING: {
              gfc2pips_debug(9,"ref substring\n");
              expression ref =
                  make_expression(make_syntax_reference(make_reference(ent_ref,
                                                                       NULL)),
                                  normalized_undefined);

              /* This is the full list for the call,
               * we put element beginning with the latest
               */
              list lexpr;

              /* Firstly, the upper bound */
              expression upper;
              if(r->u.ss.end) {
                /* We have an upper bound */
                upper = gfc2pips_expr2expression(r->u.ss.end);
              } else {
                upper
                    = MakeNullaryCall(CreateIntrinsic(UNBOUNDED_DIMENSION_NAME));
              }
              lexpr = CONS( EXPRESSION,upper ,NULL );

              /* Now the lower bound */
              lexpr
                  = CONS( EXPRESSION,gfc2pips_expr2expression( r->u.ss.start ),
                      lexpr
                  );

              /* And the string */
              lexpr = CONS( EXPRESSION,ref,lexpr);

              /* substring operator */
              entity substr = entity_intrinsic(SUBSTRING_FUNCTION_NAME);

              /* finally the call is created */
              s = make_syntax_call(make_call(substr, lexpr));
              break;
            }
            default: {
              pips_user_warning("Unable to understand the ref %d\n",
                  (int)r->type);
              break;
            }
          }
          r = r->next;
        }
      }

      if(syntax_undefined_p(s)) {
        /*
         * It doesn't seem to be a substring, it an array or a scalar
         */
        if(entity_allocatable_p(ent_ref)) {
          /* Allocatable array array encapsulated in a structure, we have
           * to produce a subscript
           */
          expression allocatable_data = get_allocatable_data_expr(ent_ref);
          subscript sub = make_subscript(allocatable_data, args_list);
          s = make_syntax_subscript(sub);
        } else {
          s = make_syntax_reference(make_reference(ent_ref, args_list));
        }
      }
      returned_expr = make_expression(s, normalized_undefined);
      break;
    }
    case EXPR_CONSTANT: {
      gfc2pips_debug(5, "Constant expression : %lu %lu\n",(_int)expr,
          (_int)expr->ts.type);
      switch(expr->ts.type) {
        case BT_INTEGER:
          returned_expr = gfc2pips_int2expression(gfc2pips_expr2int(expr));
          break;
        case BT_LOGICAL:
          returned_expr = gfc2pips_logical2expression(expr->value.logical);
          break;
        case BT_REAL: {
          /* we have to use gmp/mpfr function to get the value */
          double value = mpfr_get_d(expr->value.real, GFC_RND_MODE);
          returned_expr = gfc2pips_real2expression(value);
          break;
        }
        case BT_CHARACTER: {
          char *char_expr =
              gfc2pips_gfc_char_t2string(expr->value.character.string,
                                         expr->value.character.length);
          returned_expr = MakeCharacterConstantExpression(char_expr);
        }
          break;
        case BT_COMPLEX: {
          expression real, image;
          real = gfc2pips_real2expression(mpfr_get_d(expr->value.complex.r,
                                                     GFC_RND_MODE));
          image = gfc2pips_real2expression(mpfr_get_d(expr->value.complex.i,
                                                      GFC_RND_MODE));
          returned_expr = MakeComplexConstantExpression(real, image);
          break;
        }
        case BT_HOLLERITH:
          pips_user_error( "Hollerith not implemented\n");
          break;
        default:
          pips_user_warning( "type not implemented %d\n", (int) expr->ts.type );
          break;

      }
      break;
    }
    case EXPR_FUNCTION: {
      gfc2pips_debug(5, "func\n");

      /*
       * beware the automatic conversion here, some conversion functions may be
       * automatically called here, and we do not want them in the code
       * these implicit "cast" have a "__convert_" prefix (gfc internal)
       */
      if(strncmp(expr->symtree->n.sym->name, "__convert_", strlen("__convert_"))
          == 0) {
        /* Recursive call on implicit cast argument */
        gfc_expr *arg = expr->value.function.actual->expr;
        if(arg) {
          returned_expr = gfc2pips_expr2expression(arg);
        } else {
          pips_user_error( "expression null while trying to handle %s\n",
              expr->value.function.name );
        }
      } else {
        /*
         * functions whose name begin with __ should be used by gfc only
         * therefore we put the old name back
         */
        pips_debug(5,"Func name : %s\n",expr->value.function.name);
        if(strncmp(expr->value.function.name, "__", strlen("__")) == 0
            || strncmp(expr->value.function.name,
                       "_gfortran_",
                       strlen("_gfortran_")) == 0) {
          /* FIXME : should we really modify GFC IR ???? */
          expr->value.function.name = expr->symtree->n.sym->name;
        }

        /*
         * actual is the list of actual argument
         */
        list list_of_arguments = NULL, list_of_arguments_p = NULL;
        gfc_actual_arglist *act = expr->value.function.actual;

        if(act) {
          do {
            /*
             * gfc add default parameters for some FORTRAN functions, but it is
             * NULL in this case (could break ?)
             */
            if(act->expr) {
              expression ex = gfc2pips_expr2expression(act->expr);
              if(ex != expression_undefined) {
                if(list_of_arguments_p) {
                  CDR( list_of_arguments_p) = CONS( EXPRESSION, ex, NIL );
                  list_of_arguments_p = CDR( list_of_arguments_p );
                } else {
                  list_of_arguments_p = CONS( EXPRESSION, ex, NIL );
                }
                if(list_of_arguments == NULL) {
                  list_of_arguments = list_of_arguments_p;
                }
              }
            } else {
              break;
            }

          } while((act = act->next) != NULL);
        }

        string func_name = gfc2pips_get_safe_name(expr->value.function.name);
        func_name = str2upper(func_name);
        entity e = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, func_name);

        /*
         *  This is impossible !
         *  We must have found functions when converting symbol table
         *  The only possibility here is a missing intrinsic in PIPS
         */
        if(entity_initial(e) == value_undefined) {
          pips_user_warning("(%d) Entity not declared : '%s' is it a missing"
              "intrinsic ??\n",
              __LINE__,entity_name( e ) );
          entity_initial(e) = make_value(is_value_intrinsic, e);
        }
        call call_ = make_call(e, list_of_arguments);

        returned_expr = make_expression(make_syntax_call(call_),
                                        normalized_undefined);
      }
      break;
    }
    case EXPR_STRUCTURE:
      pips_user_error( "gfc2pips_expr2expression: dump of EXPR_STRUCTURE not "
          "yet implemented\n" );
    case EXPR_SUBSTRING:
      pips_user_error( "gfc2pips_expr2expression: dump of EXPR_SUBSTRING not "
          "yet implemented\n" );
    case EXPR_NULL:
      pips_user_error( "gfc2pips_expr2expression: dump of EXPR_NULL not yet "
          "implemented\n" );
    case EXPR_ARRAY:
      pips_user_error( "gfc2pips_expr2expression: dump of EXPR_ARRAY not yet "
          "implemented\n" );
    default:
      pips_user_error( "gfc2pips_expr2expression: dump not yet implemented, "
          "type of gfc_expr not recognized %d\n",
          (int) expr->expr_type );
      break;
  }
  return returned_expr;
}

/*
 * @brief convert an expression to an integer
 *
 * we assume we have an expression representing an integer, and we translate it
 * this function consider everything is all right: i.e. the expression represent
 * an integer
 */
int gfc2pips_expr2int(gfc_expr *expr) {
  return mpz_get_si(expr->value.integer);
}

bool gfc2pips_exprIsVariable(gfc_expr * expr) {
  return expr && expr->expr_type == EXPR_VARIABLE;
}

/**
 * @brief create an entity based on an expression, assume it is used only for incremented variables in loops
 */
entity gfc2pips_expr2entity(gfc_expr *expr) {
  message_assert( "No expression to dump.", expr );

  if(expr->expr_type == EXPR_VARIABLE) {
    message_assert( "No symtree in the expression.", expr->symtree );
    message_assert( "No symbol in the expression.", expr->symtree->n.sym );
    message_assert( "No name in the expression.", expr->symtree->n.sym->name );
    entity
        e =
            FindOrCreateEntity(CurrentPackage,
                               str2upper(gfc2pips_get_safe_name(expr->symtree->n.sym->name)));
    entity_type(e) = gfc2pips_symbol2type(expr->symtree->n.sym);
    entity_initial(e) = make_value_unknown();
    int dynamicOffset = area_size(type_area(entity_type(DynamicArea)));
    entity_storage(e) = make_storage_ram(make_ram(get_current_module_entity(),
                                                  DynamicArea,
                                                  global_current_offset,
                                                  NIL));
    area_size(type_area(entity_type(DynamicArea)))
        += area_size(type_area(entity_type(e)));
    return e;
  }

  if(expr->expr_type == EXPR_CONSTANT) {
    if(expr->ts.type == BT_INTEGER) {
      return gfc2pips_int_const2entity(mpz_get_ui(expr->value.integer));
    }
    if(expr->ts.type == BT_LOGICAL) {
      return gfc2pips_logical2entity(expr->value.logical);
    }
    if(expr->ts.type == BT_REAL) {
      return gfc2pips_real2entity(mpfr_get_d(expr->value.real, GFC_RND_MODE));
    }
  }

  message_assert( "No entity to extract from this expression", 0 );
}

list gfc2pips_arglist2arglist(gfc_actual_arglist *act) {
  list list_of_arguments = NULL, list_of_arguments_p = NULL;
  while(act) {
    expression ex = gfc2pips_expr2expression(act->expr);

    if(ex != expression_undefined) {

      if(list_of_arguments_p) {
        CDR( list_of_arguments_p) = CONS( EXPRESSION, ex, NIL );
        list_of_arguments_p = CDR( list_of_arguments_p );
      } else {
        list_of_arguments_p = CONS( EXPRESSION, ex, NIL );
      }
    }
    if(list_of_arguments == NULL)
      list_of_arguments = list_of_arguments_p;

    act = act->next;
  }
  return list_of_arguments;
}

list gfc2pips_exprIO(char* s, gfc_expr* e, list l) {
  return CONS( EXPRESSION,
      MakeCharacterConstantExpression( s ),
      CONS( EXPRESSION, gfc2pips_expr2expression( e ), l ) );
}
list gfc2pips_exprIO2(char* s, int e, list l) {
  return CONS( EXPRESSION,
      MakeCharacterConstantExpression( s ),
      CONS( EXPRESSION, MakeNullaryCall( gfc2pips_int2label( e ) ), l ) );
}
list gfc2pips_exprIO3(char* s, string e, list l) {
  return CONS( EXPRESSION,
      MakeCharacterConstantExpression( s ),
      CONS( EXPRESSION,
          MakeNullaryCall( FindOrCreateEntity( CurrentPackage,
                  gfc2pips_get_safe_name( e ) ) ),
          l ) );
}

/**
 * @brief compute addresses of the stack, heap, dynamic and static areas
 */
//aller se balader dans "static segment_info * current_segment;" qui contient miriade d'informations sur la mémoire
void gfc2pips_computeAdresses(void) {
  //check les déclarations, si UNBOUNDED_DIMENSION_NAME dans la liste des dimensions => direction *STACK*
  gfc2pips_computeAdressesHeap();
  gfc2pips_computeAdressesDynamic();
  gfc2pips_computeAdressesStatic();
}
/**
 * @brief compute the addresses of the entities declared in StaticArea
 */
void gfc2pips_computeAdressesStatic(void) {
  gfc2pips_computeAdressesOfArea(StaticArea);
}
/**
 * @brief compute the addresses of the entities declared in DynamicArea
 */
void gfc2pips_computeAdressesDynamic(void) {
  gfc2pips_computeAdressesOfArea(DynamicArea);
}
/**
 * @brief compute the addresses of the entities declared in StaticArea
 */
void gfc2pips_computeAdressesHeap(void) {
  gfc2pips_computeAdressesOfArea(HeapArea);
}

/**
 * @brief compute the addresses of the entities declared in the given entity
 */
int gfc2pips_computeAdressesOfArea(entity _area) {
  //compute each and every addresses of the entities in this area. Doesn't handle equivalences.
  if(!_area || _area == entity_undefined || entity_type(_area)
      == type_undefined || !type_area_p(entity_type(_area))) {
    pips_user_warning( "Impossible to compute the given object as an area\n" );
    return 0;
  }
  int offset = 0;
  list _pcv = code_declarations(EntityCode(get_current_module_entity()));
  list pcv = gen_copy_seq(_pcv);
  gfc2pips_debug(9,"Start \t\t%s %zu element(s) to check\n",entity_local_name(_area),gen_length(pcv));
  for (pcv = gen_nreverse(pcv); pcv != NIL; pcv = CDR( pcv )) {
    entity e = ENTITY(CAR(pcv));//ifdebug(1)fprintf(stderr,"%s\n",entity_local_name(e));
    if(entity_storage(e) != storage_undefined
        && storage_ram_p(entity_storage(e))
        && ram_section(storage_ram(entity_storage(e))) == _area) {
      //we need to skip the variables in commons and commons
      gfc2pips_debug(9,"Compute address of %s - offset: %d\n",entity_name(e), offset);

      entity section = ram_section(storage_ram(entity_storage(e)));
      if(type_tag(entity_type(e)) != is_type_variable || (section != StackArea
          && section != StaticArea && section != DynamicArea && section
          != HeapArea)) {
        pips_user_warning( "We don't know how to do that - skip %s\n",
            entity_local_name( e ) );
        //size = gfc2pips_computeAdressesOfArea(e);
      } else {
        ram_offset(storage_ram(entity_storage(e))) = offset;

        area ca = type_area(entity_type(_area));
        area_layout(ca) = gen_nconc(area_layout(ca), CONS( ENTITY, e, NIL ));

        int size;
        SizeOfArray(e, &size);
        offset += size;
      }
    }
  }

  set_common_to_size(_area, offset);
  gfc2pips_debug( 9, "next offset: %d\t\tEnd %s\n", offset, entity_local_name(_area) );
  return offset;
}

void gfc2pips_computeEquiv(gfc_equiv *eq) {
  //equivalences are not correctly implemented in PIPS: they work on entities instead of expressions
  //there is no point in creating something wrong
  /*
   * example:
   * EQUIVALENCE (a,c(1)),  (b,c(2))
   *
   * |-----a-----|
   *          |-----b-----|
   * |--c(1)--|--c(2)--|--c(3)--|
   * |-------------c------------|-----d-----|
   *
   * It is impossible to represent with only entities !!!
   */

  //ComputeEquivalences();//syntax/equivalence.c
  //offset = calculate_offset(eq->expr);  enlever le static du fichier


  for (; eq; eq = eq->next) {
    gfc_equiv * save = eq;
    gfc_equiv *eq_;
    gfc2pips_debug(9,"sequence of equivalences\n");
    entity storage_area = entity_undefined;
    entity not_moved_entity = entity_undefined;
    int offset = 0;
    int size = -1;
    int not_moved_entity_size;
    for (eq_ = eq; eq_; eq_ = eq_->eq) {
      //check in same memory storage, not formal, not *STACK* ('cause size unknown)
      //take minimum offset
      //calculate the difference in offset to the next variable
      //set the offset to the variable with the greatest offset
      //add if necessary the difference of offset to all variable with an offset greater than the current one (and not equiv too ? or else need to proceed in order of offset ...)
      // ?? gfc2pips_expr2int(eq->expr); ??

      message_assert( "expression to compute in equivalence\n", eq_->expr );
      gfc2pips_debug(9,"equivalence of %s\n",eq_->expr->symtree->name);

      //we have to absolutely know if it is an element in an array or a single variable
      entity e = gfc2pips_check_entity_exists(eq->expr->symtree->name);//this doesn't give an accurate idea for the offset, just an idea about the storage
      message_assert( "entity has been founded\n", e != entity_undefined );
      if(size == -1)
        not_moved_entity = e;

      message_assert( "Storage is defined\n", entity_storage(e)
          != storage_undefined );
      /* FIXME StackArea and HeapArea are entity, not storage... (Mehdi)
       message_assert("Storage is not STACK\n", entity_storage(e) != StackArea );
       message_assert("Storage is not HEAP\n", entity_storage(e) != HeapArea );
       */
      message_assert( "Storage is RAM\n", storage_ram_p(entity_storage(e)) );

      if(!storage_area)
        storage_area = ram_section(storage_ram(entity_storage(e)));
      message_assert( "Entities are in the same area\n",
          ram_section(storage_ram(entity_storage(e)))
          == storage_area );

      storage_area = ram_section(storage_ram(entity_storage(e)));

      //expression ex = gfc2pips_expr2expression(eq_->expr);
      //fprintf(stderr,"toto %x\n",ex);

      //int offset_of_expression = gfc2pips_offset_of_expression(eq_->expr);
      //relative offset from the beginning of the variable (null if simple variable or first value of array)

      // FIXME calculate_offset is static in trans-common.c... (Mehdi)
      //int offset_of_expression = calculate_offset( eq_->expr );//gcc/fortran/trans-common.c
      int offset_of_expression = 0;

      //int offset_of_expression = ram_offset(storage_ram(entity_storage(e)));
      offset_of_expression += ram_offset(storage_ram(entity_storage(e)));

      if(size != -1) {
        //gfc2pips_shiftAdressesOfArea( storage_area, not_moved_entity, e, eq_->expr );
      } else {
        size = 0;
      }
    }
  }
}

//we need 2 offsets, one is the end of the biggest element, another is the cumulated size of each moved element
void gfc2pips_shiftAdressesOfArea(entity _area,
                                  int old_offset,
                                  int size,
                                  int max_offset,
                                  int shift) {
  list _pcv = code_declarations(EntityCode(get_current_module_entity()));
  list pcv = gen_copy_seq(_pcv);
  for (pcv = gen_nreverse(pcv); pcv != NIL; pcv = CDR( pcv )) {
    entity e = ENTITY(CAR(pcv));
    if(entity_storage(e) != storage_undefined
        && storage_ram_p(entity_storage(e))
        && ram_section(storage_ram(entity_storage(e))) == _area) {
      /*
       * put those two lines in one go (to process everything in one loop only)
       when shift, if offset of variable after <c>, retrieve size of <c>
       add to every variable after <a>+sizeof(<a>) the difference of offset

       when shift, if offset of variable after <b>, retrieve size of <b>
       add to every variable after <c(2)>+sizeof(<c>)-sizeof(<c(1)>) the difference of offset

       => when we move an array or a variable, use the full size of the array/variable
       when an element, use the full size of the array minus the offset of the element
       */
      gfc2pips_debug(9,"%s\told_offset: %d\tsize: %d\tmax_offset: %d\tshift: %d\tram_offset: %zu\n", entity_name(e), old_offset, size, max_offset, shift, ram_offset(storage_ram(entity_storage(e))) );
      int personnal_shift = 0;
      //if( ram_offset(storage_ram(entity_storage(e))) > old_offset+size ){
      personnal_shift -= shift;
      //}
      if(ram_offset(storage_ram(entity_storage(e))) > old_offset) {
        personnal_shift -= size;
      }
      ram_offset(storage_ram(entity_storage(e))) += personnal_shift;
      gfc2pips_debug(9,"%s shifted of %d\n",entity_name(e),personnal_shift);
    }
  }
}
