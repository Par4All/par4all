/* package continuation :  Be'atrice Creusillet, 1996
 *
 * This File contains the functions to prettyprint continuation conditions 
 * of a module (over- and under-approximations.
 *
 */
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <string.h>
/* #include <stdlib.h> */


#include "genC.h"

#include "text.h"
#include "text-util.h"

#include "top-level.h"

#include "ri.h"
#include "ri-util.h"

#include "database.h"
#include "pipsdbm.h"
#include "resources.h"

#include "misc.h"
#include "properties.h"

#include "prettyprint.h"

#include "transformer.h"
#include "effects-generic.h"

#include "semantics.h"
#include "continuation.h"

#define PREC_FORESYS_PREFIX "C$PREC"
#define TRAN_FORESYS_PREFIX "C$TRAN"
#define FORESYS_CONTINUATION_PREFIX "C$&"
#define PIPS_NORMAL_PREFIX "C"


static bool is_user_view_p;
static hash_table nts = hash_table_undefined;

static bool print_continuation_conditions(char *module_name);
static text get_continuation_condition_text(char *module_name,bool give_code_p);
static text text_statement_continuation_conditions(entity module, int margin,
						   statement stat);
static text text_continuation_conditions(transformer must_cont_t,
					 transformer may_cont_t);
static text text_continuation(transformer cont, bool is_must);

bool print_code_continuation_conditions(char *module_name)
{
    is_user_view_p = FALSE;
    return print_continuation_conditions(module_name);
}


bool print_source_continuation_conditions(char *module_name)
{
    is_user_view_p = TRUE;
    return print_continuation_conditions(module_name);
}



static bool print_continuation_conditions(char *module_name)
{
    char *file_name, *file_resource_name;
    bool success = TRUE;

    file_name = strdup(concatenate(".cont",
                                  get_bool_property
				  ("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH") ? 
				  GRAPH_FILE_EXT : "",
                                  NULL));
    file_resource_name = get_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH") ?
	DBR_GRAPH_PRINTED_FILE : 
	    (is_user_view_p ? DBR_PARSED_PRINTED_FILE : DBR_PRINTED_FILE);

    success = make_text_resource(module_name,
				 file_resource_name,
				 file_name,
				 get_continuation_condition_text(module_name,TRUE));

    free(file_name);
    return(success);
}

static text get_continuation_condition_text(char *module_name, bool give_code_p)
{
    entity module;
    statement module_stat, user_stat = statement_undefined;
    text txt = make_text(NIL);
    transformer must_sum_cont_t, may_sum_cont_t;   

    set_current_module_entity( local_name_to_top_level_entity(module_name));
    module = get_current_module_entity();
    set_current_module_statement((statement) db_get_memory_resource
				 (DBR_CODE, module_name, TRUE));
    module_stat = get_current_module_statement();

    /* To set up the hash table to translate value into value names */       
    set_cumulated_rw_effects((statement_effects)
	  db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, TRUE));
    module_to_value_mappings(module);


    if(is_user_view_p) 
    {
	user_stat =  (statement)
	    db_get_memory_resource(DBR_PARSED_CODE, module_name, TRUE);

	nts = allocate_number_to_statement();
	nts = build_number_to_statement(nts, module_stat);

	ifdebug(1)
	{
	    print_number_to_statement(nts);
	}
    }

    debug_on("CONTINUATION_DEBUG_LEVEL");

    set_must_continuation_map( (statement_mapping) 
	db_get_memory_resource(DBR_MUST_CONTINUATION, module_name, TRUE) );
    set_may_continuation_map( (statement_mapping) 
	db_get_memory_resource(DBR_MAY_CONTINUATION, module_name, TRUE) );
    must_sum_cont_t = ( (transformer) 
	db_get_memory_resource(DBR_MUST_SUMMARY_CONTINUATION, module_name, TRUE) ); 
    may_sum_cont_t = ( (transformer) 
	db_get_memory_resource(DBR_MAY_SUMMARY_CONTINUATION, module_name, TRUE) );

    /* prepare the prettyprinting */
    init_prettyprint(text_statement_continuation_conditions);
    /* summary information first */
    MERGE_TEXTS(txt,text_continuation(must_sum_cont_t, TRUE)); 
    MERGE_TEXTS(txt,text_continuation(may_sum_cont_t, FALSE)); 

    if (give_code_p)
	/* then code with regions, using text_statement_continuation_conditions */
	MERGE_TEXTS(txt, text_module(module,
				     is_user_view_p? user_stat : module_stat));

    debug_off();

    if(is_user_view_p)
    {
	hash_table_free(nts);
	nts = hash_table_undefined;
    }

    close_prettyprint();

    reset_current_module_entity();
    reset_current_module_statement();
    reset_cumulated_rw_effects();
    reset_must_continuation_map();
    reset_may_continuation_map();
    free_value_mappings();

    return txt;
}

static text text_statement_continuation_conditions(entity module, int margin,
						   statement stat)
{
    transformer must_cont_t, may_cont_t;
    statement s;

    s = is_user_view_p? 
	(statement) hash_get(nts, (char *) statement_number(stat)) :
	stat;

    if (is_user_view_p)
    {
	s = (statement) hash_get(nts, (char *) statement_number(stat));
    }

    
    if (s != (statement) HASH_UNDEFINED_VALUE)
    {
	must_cont_t = load_statement_must_continuation(s);
	may_cont_t = load_statement_may_continuation(s);
    }
    else
    {
	must_cont_t = (transformer) HASH_UNDEFINED_VALUE;
	may_cont_t = (transformer) HASH_UNDEFINED_VALUE;
    }
    
    return text_continuation_conditions(must_cont_t, may_cont_t);
}


static text text_continuation_conditions(transformer must_cont_t,
					 transformer may_cont_t) 
{
    
    text cont_text = make_text(NIL);
    boolean loose_p = get_bool_property("PRETTYPRINT_LOOSE");
    
    if ((must_cont_t ==(transformer) HASH_UNDEFINED_VALUE) &&
	(may_cont_t ==(transformer) HASH_UNDEFINED_VALUE) )
	return(cont_text);

    if (loose_p)
    {
	ADD_SENTENCE_TO_TEXT(cont_text, 
			     make_sentence(is_sentence_formatted, 
					   strdup("\n")));
    }
    
    /* First: must continuation conditions */
    if (must_cont_t !=(transformer) HASH_UNDEFINED_VALUE)
    {
	MERGE_TEXTS(cont_text, text_continuation(must_cont_t,TRUE));
    }
    
    /* Then: may continuation conditions */
    if (may_cont_t !=(transformer) HASH_UNDEFINED_VALUE)
    {
	MERGE_TEXTS(cont_text, text_continuation(may_cont_t,FALSE));
    }
    if (loose_p)
	ADD_SENTENCE_TO_TEXT(cont_text, 
			     make_sentence(is_sentence_formatted, 
					   strdup("\n")));

    return(cont_text);
}



/* The strange argument type is required by qsort(), deep down in the calls */
static int is_inferior_pvarval(Pvecteur * pvarval1, Pvecteur * pvarval2)
{
    /* The constant term is given the highest weight to push constant
       terms at the end of the constraints and to make those easy
       to compare. If not, constant 0 will be handled differently from
       other constants. However, it would be nice to give constant terms
       the lowest weight to print simple constraints first...

       Either I define two comparison functions, or I cheat somewhere else.
       Let's cheat? */
    int is_equal = 0;
    
    if (term_cst(*pvarval1) && !term_cst(*pvarval2))
	is_equal = 1;
    else if (term_cst(*pvarval1) && term_cst(*pvarval2))
	is_equal = 0;
    else if(term_cst(*pvarval2))
	is_equal = -1;
    else
	is_equal = 
	    strcmp(pips_user_value_name((entity) vecteur_var(*pvarval1)),
		   pips_user_value_name((entity) vecteur_var(*pvarval2)));


    return is_equal; 
}


#define LINE_SUFFIX "\n"
#define MAX_LINE_LENGTH 70
#define PIPS_NORMAL_PREFIX "C"



/* text text_continuation(transformer tran) 
 * input    : a transformer representing a transformer or a precondition 
 * output   : a text containing commentaries representing the transformer
 * modifies : nothing.
 *
 * Modification: AP, Nov 10th, 1995. Instead of building a (very long)
 * string, I directly use the transformer to build the prettyprint in text
 * format. This is to avoid the problem occuring when the buffer used in
 * transformer[precondition]_to_string() is too small. I also use a static
 * buffer to build each constraint; we are restricted to constraints of
 * lengths smaller than the line length.
 */
static text text_continuation(transformer cont, bool is_must)
{
  text txt = make_text(NIL);
  string str_prefix;
  static char crt_line[MAX_LINE_LENGTH];
  static char aux_line[MAX_LINE_LENGTH];
  Pcontrainte peq;
  Psysteme ps;
  boolean first_line = TRUE;
  
  str_prefix = PIPS_NORMAL_PREFIX;

  crt_line[0] = '\0'; 
  (void) strcat(crt_line, str_prefix);
  (void) strcat(crt_line, " ");

  if(cont != (transformer) HASH_UNDEFINED_VALUE ) 
  {
      if(cont==transformer_undefined)
      {
	  strcat(crt_line, " CONTINUATION: TRANSFORMER_UNDEFINED");
      }
      else
      {  
	  aux_line[0] = '\0';
	  strcat(aux_line, is_must? "C-MUST-":"C-MAY-");
	  if(strlen(crt_line) + strlen(aux_line) > MAX_LINE_LENGTH - 2)
	      pips_error("text_continuation", "line buffer too small");
	  
	  strcat(crt_line, aux_line);
	  
	  if(strlen(crt_line)+1 > MAX_LINE_LENGTH-2) 
	  {
	      strcat(crt_line, LINE_SUFFIX);
	      ADD_SENTENCE_TO_TEXT(txt, make_sentence(is_sentence_formatted,
						      strdup(crt_line)));
	      
	      if(first_line) first_line = FALSE;	      
	      
	      crt_line[0] = '\0'; (void) strcat(crt_line, str_prefix); 
	      strcat(crt_line, "    ");
	  }
	  else 
	      strcat(crt_line, " ");
	  
	  ps = (Psysteme) predicate_system(transformer_relation(cont));
	  
	  if (ps != NULL)
	  {
	      boolean first_constraint = TRUE, last_constraint = FALSE;
	      
	      sc_lexicographic_sort(ps, is_inferior_pvarval); 
	      
	      for (peq = ps->egalites; peq!=NULL; peq=peq->succ)
	      {
		  last_constraint = ((peq->succ == NULL) &&
				     (ps->inegalites == NULL));
		  aux_line[0] = '\0';
		  
		  if(first_constraint)
		  {
		      strcat(aux_line, "{");
		      first_constraint = FALSE;
		  }
		  egalite_sprint_format(aux_line, peq, pips_user_value_name, FALSE);
		  if(! last_constraint)
		      strcat(aux_line, ", ");
		  else
		      strcat(aux_line, "}");
		  
		  
		  first_line = add_to_current_line(crt_line, aux_line, str_prefix,
						   txt, first_line);
	      }
	      
	      for (peq = ps->inegalites; peq!=NULL; peq=peq->succ)
	      {
		  last_constraint = (peq->succ == NULL);
		  aux_line[0] = '\0';
		  
		  if(first_constraint)
		  {
		      strcat(aux_line, "{");
		      first_constraint = FALSE;
		  }
		  inegalite_sprint_format(aux_line, peq, pips_user_value_name,
					  FALSE);
		  if(! last_constraint)
		      strcat(aux_line, ", ");
		  else
		      strcat(aux_line, "}");
		  
		  first_line = add_to_current_line(crt_line, aux_line, str_prefix,
						   txt, first_line);
	      }
	      
	      /* If there is no constraint */
	      if((ps->egalites == NULL) && (ps->inegalites == NULL))
	      {
		  aux_line[0] = '\0';
		  strcat(aux_line, "{}");
		  first_line = add_to_current_line(crt_line, aux_line, str_prefix,
						   txt, first_line);
	      }
	  }
	  else
	  {
	      aux_line[0] = '\0';
	      strcat(aux_line, "SC_UNDEFINED");
	      first_line = add_to_current_line(crt_line, aux_line, str_prefix,
					       txt, first_line);
	  }
      }
      
      /* Save last line */
      strcat(crt_line, LINE_SUFFIX);
      ADD_SENTENCE_TO_TEXT(txt, make_sentence(is_sentence_formatted,
					      strdup(crt_line)));
  }
  
  /* ADD_SENTENCE_TO_TEXT(txt, make_sentence(is_sentence_formatted,
					  strdup("\n"))); */
  
  return txt; 
}


