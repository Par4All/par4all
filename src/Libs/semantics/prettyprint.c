/*
 * $Id$
 * 
 * package semantics - prettyprint interface 
 */

#ifndef lint
char vcid_semantics_prettyprint[] = "$Id$";
#endif /* lint */

#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <string.h>

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
#include "effects-simple.h"

#include "semantics.h"

#define PREC_FORESYS_PREFIX "C$PREC"
#define TRAN_FORESYS_PREFIX "C$TRAN"
#define FORESYS_CONTINUATION_PREFIX "C$&"
#define PIPS_NORMAL_PREFIX "C"

#define LINE_SUFFIX "\n"
#define MAX_LINE_LENGTH 70

DEFINE_CURRENT_MAPPING(semantic, transformer)

static bool is_transformer;
static bool is_user_view;
static hash_table nts = hash_table_undefined;

static bool print_code_semantics();
static text get_semantic_text();

bool 
print_code_transformers(module_name)
char *module_name;
{
    is_user_view = FALSE;
    is_transformer = TRUE;
    return print_code_semantics(module_name);
}

bool 
print_code_preconditions(module_name)
char *module_name;
{
    is_user_view = FALSE;
    is_transformer = FALSE;
    return print_code_semantics(module_name);
}

bool 
print_source_transformers(module_name)
char *module_name;
{
    is_user_view = TRUE;
    is_transformer = TRUE;
    return print_code_semantics(module_name);
}

bool 
print_source_preconditions(module_name)
char *module_name;
{
    is_user_view = TRUE;
    is_transformer = FALSE;
    return print_code_semantics(module_name);
}

text 
get_text_transformers(module_name)
char *module_name;
{
    is_user_view = FALSE;
    is_transformer = TRUE;
    return get_semantic_text(module_name,FALSE);
}

text 
get_text_preconditions(module_name)
char *module_name;
{
    is_user_view = FALSE;
    is_transformer = FALSE;
    return get_semantic_text(module_name,FALSE);
}

static bool 
print_code_semantics(module_name)
char *module_name;
{
    bool success = TRUE;

    char * file_ext = strdup(concatenate(is_transformer?
		     (is_user_view? USER_TRANSFORMER_SUFFIX :
		      SEQUENTIAL_TRANSFORMER_SUFFIX ) :
		     (is_user_view? USER_PRECONDITION_SUFFIX :
		      SEQUENTIAL_PRECONDITION_SUFFIX),
		     get_bool_property
		     ("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH") ?
		     GRAPH_FILE_EXT : "",
		     NULL));

    char * resource_name =
	get_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH") ?
	    DBR_GRAPH_PRINTED_FILE :
		(is_user_view? DBR_PARSED_PRINTED_FILE :
		 DBR_PRINTED_FILE);

    begin_attachment_prettyprint();
    
    success = make_text_resource(module_name,
				 resource_name,
				 file_ext,
				 get_semantic_text(module_name,TRUE));

    end_attachment_prettyprint();
 
    free(file_ext);
    return success;
}

static text 
get_semantic_text(module_name,give_code_p)
char *module_name;
bool give_code_p;
{
    text r = make_text(NIL), txt_summary;
    entity mod;
    statement mod_stat;
    transformer summary = transformer_undefined;
    statement user_stat = statement_undefined;

    set_current_module_entity( local_name_to_top_level_entity(module_name) );
    mod = get_current_module_entity();

    set_current_module_statement
	((statement)db_get_memory_resource(DBR_CODE, module_name, TRUE) );
    mod_stat = get_current_module_statement();

    /* To set up the hash table to translate value into value names */
    set_cumulated_rw_effects((statement_effects)
			  db_get_memory_resource
			  (DBR_CUMULATED_EFFECTS, module_name, TRUE));

    debug_on("SEMANTICS_PRINT_DEBUG_LEVEL");

    module_to_value_mappings(mod);

    if(is_user_view) {
	user_stat =  (statement)
	    db_get_memory_resource(DBR_PARSED_CODE, module_name, TRUE);

	nts = allocate_number_to_statement();
	nts = build_number_to_statement(nts, mod_stat);

	ifdebug(1) {
	    print_number_to_statement(nts);
	}
	/* debug_off(); */
    }


    /* semantic information to print */
    set_semantic_map( (statement_mapping)
		     db_get_memory_resource(
					    is_transformer? DBR_TRANSFORMERS 
					    : DBR_PRECONDITIONS,
					    module_name,
					    TRUE) );
    summary = (transformer)
	db_get_memory_resource(
			       is_transformer? DBR_SUMMARY_TRANSFORMER
			       : DBR_SUMMARY_PRECONDITION,
			       module_name,
			       TRUE);
    /* The summary precondition may be in another module's frame */
    translate_global_values(mod, summary);

    init_prettyprint(semantic_to_text);

    /* initial version; to be used again when prettyprint really prettyprints*/
    /* print_text(fd, text_statement(mod, 0, mod_stat)); */

    /* summary information first */
    txt_summary = text_transformer(summary);
    ifdebug(7){
	dump_text(txt_summary);
	pips_debug(7, "summary text consistent? %s\n",
		   text_consistent_p(txt_summary)? "YES":"NO"); 
    }
    MERGE_TEXTS(r,txt_summary ); 
    attach_decoration_to_text(r);
    if (is_transformer)
	attach_transformers_decoration_to_text(r);
    else
	attach_preconditions_decoration_to_text(r);
 
    if (give_code_p == TRUE) {
	MERGE_TEXTS(r, text_module(mod, is_user_view? user_stat:mod_stat));
    }

    debug_off();

    if(is_user_view) {
	hash_table_free(nts);
	nts = hash_table_undefined;
    }

    close_prettyprint();

    reset_semantic_map();
    reset_current_module_entity();
    reset_current_module_statement();
    reset_cumulated_rw_effects();

    free_value_mappings();

    return r;
}

/* this function name is VERY misleading - it should be changed, sometime FI */
text 
semantic_to_text(module, margin, stmt)
entity module;
int margin;
statement stmt;
{
    transformer t;
    text txt;

    if(is_user_view) {
	statement i = apply_number_to_statement(nts, statement_number(stmt));

	if(!statement_undefined_p(i)) {
	    t = load_statement_semantic(i);
	}
	else
	    t = (transformer) HASH_UNDEFINED_VALUE;
    }
    else
	t = load_statement_semantic(stmt);

    txt = text_transformer(t);

    if (is_transformer)
	attach_transformers_decoration_to_text(txt);
    else
	attach_preconditions_decoration_to_text(txt);
	
    return txt; 
}


/* It is used to sort arguments preconditions in text_transformer(). */
static int 
wordcmp(s1,s2)
char **s1, **s2;
{
    return strcmp(*s1,*s2);
}

/* The strange argument type is required by qsort(), deep down in the calls */
static int 
is_inferior_pvarval(Pvecteur * pvarval1, Pvecteur * pvarval2)
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

/* text text_transformer(transformer tran) 
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
text 
text_transformer(transformer tran)
{
  text txt = make_text(NIL);
  boolean foresys = get_bool_property("PRETTYPRINT_FOR_FORESYS");
  string str_prefix;
  char crt_line[MAX_LINE_LENGTH];
  char aux_line[MAX_LINE_LENGTH];
  Pcontrainte peq;
  Psysteme ps;
  boolean first_line = TRUE;

  if (is_transformer) {
    if (foresys) 
      str_prefix = TRAN_FORESYS_PREFIX;
    else 
      str_prefix = PIPS_NORMAL_PREFIX;
  }
  else {
    if (foresys) 
      str_prefix = PREC_FORESYS_PREFIX;
    else 
      str_prefix = PIPS_NORMAL_PREFIX;
  }

  /* If in EMACS mode, does not add any separator line: */
  if (!get_bool_property("PRETTYPRINT_ADD_EMACS_PROPERTIES"))
      ADD_SENTENCE_TO_TEXT(txt, make_sentence(is_sentence_formatted,
					      strdup("\n")));

  crt_line[0] = '\0';
  (void) strcat(crt_line, str_prefix);
  (void) strcat(crt_line, " ");

  if(tran != (transformer) HASH_UNDEFINED_VALUE && 
     tran != (transformer) list_undefined)
  {
      if(tran==transformer_undefined)
      {
	  if (is_transformer)
	      (void) strcat(crt_line, " TRANSFORMER: TRANSFORMER_UNDEFINED");
	  else
	      (void) strcat(crt_line, " PRECONDITION: TRANSFORMER_UNDEFINED");
      }
      else
      {
	  list args;
	  int j=0, provi_length = 1;
	  char **provi;
	  
	  aux_line[0] = '\0';
	  if (is_transformer)
	      (void) strcat(aux_line, " T(");
	  else
	      (void) strcat(aux_line, " P(");
	  if(strlen(crt_line) + strlen(aux_line) > MAX_LINE_LENGTH - 2)
	      pips_error("text_transformer", "line buffer too small\n");
	  
	  (void) strcat(crt_line, aux_line);
	  
	  args = transformer_arguments(tran);
	  pips_debug(6, "Number of arguments = %d\n", gen_length(args));

	  if(!ENDP(args))
	  {
	      provi = (char **) malloc(sizeof(char *) * gen_length(args));
	      j = 0;
	      MAP(ENTITY, e,
		   {
		       if (entity_undefined_p(e)) 
		       {
			   pips_debug(7, "undefined entity\n");
			   provi[j] = (char*) "entity_undefined";
		       }
		       else
			   provi[j] = (char*) entity_minimal_name(e); 
		       j++;
		   },
		       args);
	      provi_length = j;
	      
	      qsort(provi, provi_length, sizeof provi[0], wordcmp);
	      pips_debug(7, "Building text for arguments\n");
	      if ( provi_length > 1 )
	      {
		  for (j=0; j < provi_length-1; j++)
		  {
		      aux_line[0] = '\0';
		      (void) strcat(aux_line, provi[j]);
		      strcat(aux_line,",");
		      first_line = add_to_current_line(crt_line, aux_line,
						       str_prefix, txt, first_line);
		      ifdebug(8){
			  pips_debug(8, "%d-th argument %s\n current txt"
				     " (consistent? %s): \n",
				     j+1, provi[j],
				     text_consistent_p(txt)? "YES":"NO");
			  dump_text(txt);
		      }
			  
		  }
	      }
	      aux_line[0] = '\0';
	      (void) strcat(aux_line, provi[provi_length-1]);
	      strcat(aux_line, ")");
	      if (foresys)
		  (void) strcat(aux_line, ",");
	      first_line = add_to_current_line(crt_line, aux_line,
					       str_prefix, txt, first_line);
	      ifdebug(8){
		  pips_debug(8, "%d-th argument %s\n current txt: \n",
			     provi_length-1, provi[provi_length-1]);
		  dump_text(txt);
	      }
	      free(provi);
	  }
	  else
	      strcat(crt_line, ")");
	  
	  if(strlen(crt_line)+1 > MAX_LINE_LENGTH-2) {
	      (void) strcat(crt_line, LINE_SUFFIX);
	      ADD_SENTENCE_TO_TEXT(txt, make_sentence(is_sentence_formatted,
						      strdup(crt_line)));
	      if(first_line) {
		  first_line = FALSE;
		  if(foresys) {
		      str_prefix = strdup(str_prefix);
		      str_prefix[0] = '\0';
		      (void) strcat(str_prefix, FORESYS_CONTINUATION_PREFIX);
		  }
	      }
	      
	      crt_line[0] = '\0';
	      (void) strcat(crt_line, str_prefix); 
	      (void) strcat(crt_line, "    ");
	  }
	  else 
	      (void) strcat(crt_line, " ");
	  ifdebug(7){
	      pips_debug(7, "current txt before dealing with system: \n");
	      dump_text(txt);
	  }	  
	  ps = (Psysteme) predicate_system(transformer_relation(tran));
	  
	  ifdebug(7) {
	      pips_debug(7, "sys %p\n", ps);
	      sc_syst_debug(ps);
	  }
	  
	  if (ps != NULL) {
	      boolean first_constraint = TRUE, last_constraint = FALSE;
	      
	      sc_lexicographic_sort(ps, is_inferior_pvarval);
	      
	      pips_debug(7, " equalities first\n");

	      for (peq = ps->egalites, j=1; peq!=NULL; peq=peq->succ, j=j+1)
	      {
		
		  last_constraint = ((peq->succ == NULL) &&
				     (ps->inegalites == NULL));
		  aux_line[0] = '\0';
		  if (foresys)
		  {
		      (void) strcat(aux_line, "(");
		      (void) egalite_sprint_format(aux_line, peq,
						   pips_user_value_name, foresys);
		      (void) strcat(aux_line, ")");
		      if(! last_constraint)
			  (void) strcat(aux_line, ".AND."); 
		  }
		  else
		  {
		      if(first_constraint)
		      {  
			  first_line = add_to_current_line(crt_line,"{", 
							   str_prefix,txt,first_line);
			  first_constraint = FALSE;
		      }
		      egalite_text_format(crt_line,str_prefix,txt,peq,
					  pips_user_value_name, foresys,
					  first_line);

		      if(! last_constraint)
			  first_line = add_to_current_line(crt_line,", ", 
							   str_prefix,txt,first_line);
		      else
			  first_line = add_to_current_line(crt_line,"}", 
							   str_prefix,txt,first_line);
		  }
		  
		 
		  ifdebug(7){
		      pips_debug(7, "%d-th equality\n current txt: \n", j);
		      dump_text(txt);
		  }  
	      }
	      
	      pips_debug(7, " inequalities \n");
	      
	      for (peq = ps->inegalites, j=1; peq!=NULL; peq=peq->succ,j=j+1)
	      {
		  last_constraint = (peq->succ == NULL);
		  aux_line[0] = '\0';
		  if (foresys)
		  {
		      (void) strcat(aux_line, "(");
		      (void) inegalite_sprint_format(aux_line, peq,
						     pips_user_value_name, foresys);
		      (void) strcat(aux_line, ")");
		      if(! last_constraint)
			  (void) strcat(aux_line, ".AND."); 
		  }
		  else {
		      if(first_constraint)
		      {
			  first_line = add_to_current_line(crt_line,"{", str_prefix,
						   txt, first_line);

			  first_constraint = FALSE;
		      }
		     
		      inegalite_text_format(crt_line,str_prefix,txt,peq,
					    pips_user_value_name, foresys,
					    first_line);
		      if(! last_constraint)
			 first_line = add_to_current_line(crt_line,", ", str_prefix,
						   txt, first_line);
		      else
			 first_line = add_to_current_line(crt_line,"}", str_prefix,
						   txt, first_line);
		  }
		  
		  ifdebug(7){
		      pips_debug(7, "%d-th inequality\n current txt: \n", j);
		      dump_text(txt);
		  }  
	      }
	      
	      /* If there is no constraint */
	      if((ps->egalites == NULL) && (ps->inegalites == NULL))
	      {
		 
		  first_line = add_to_current_line(crt_line,"{}", str_prefix,
					   txt, first_line);
	      }
	  }
	  else
	      first_line = add_to_current_line(crt_line, "SC_UNDEFINED", str_prefix,
					       txt, first_line);
	  
      }
      
      /* Save last line */
      (void) strcat(crt_line, LINE_SUFFIX);
      ADD_SENTENCE_TO_TEXT(txt, make_sentence(is_sentence_formatted,
					    strdup(crt_line)));
  }
  
  if (!get_bool_property("PRETTYPRINT_ADD_EMACS_PROPERTIES"))
      ADD_SENTENCE_TO_TEXT(txt, make_sentence(is_sentence_formatted,
					      strdup("\n")));  

  
  ifdebug(7){
      pips_debug(7, "final txt: \n");
      dump_text(txt);
  }  

  return(txt); 
}


/* ---------------------------------------------------------------- */
/* to convert strings containing predicates to text of commentaries */
/* BA, april 1994                                                   */
/* ---------------------------------------------------------------- */

#define MAX_PRED_COMMENTARY_STRLEN 70


/* text string_predicate_to_commentary(string str_pred, string comment_prefix) 
 * input    : a string, part of which represents a predicate.
 * output   : a text consisting of several lines of commentaries,
 *            containing the string str_pred, and beginning with 
 *            comment_prefix.
 * modifies : str_pred;
 */
text 
string_predicate_to_commentary(str_pred, comment_prefix)
string str_pred;
string comment_prefix;
{
    text t_pred = make_text(NIL);
    string str_suiv = NULL;
    string str_prefix = comment_prefix;
    char str_tmp[MAX_PRED_COMMENTARY_STRLEN];
    int len, new_str_pred_len, longueur_max;
    boolean premiere_ligne = TRUE;
    boolean foresys = get_bool_property("PRETTYPRINT_FOR_FORESYS");
    longueur_max = MAX_PRED_COMMENTARY_STRLEN - strlen(str_prefix) - 2;

    /* if str_pred is too long, it must be splitted in several lines; 
     * the hyphenation must be done only between the constraints of the
     * predicate, when there is a "," or a ")". A space is added at the beginning
     * of extra lines, for indentation. */
    while((len = strlen(str_pred)) > 0) {
	if (len > longueur_max) {

	    /* search the maximal substring which length 
	     * is less than longueur_max */
	    str_tmp[0] = '\0';
	    (void) strncat(str_tmp, str_pred, longueur_max);

	    switch (foresys) {
	    case FALSE : 
		str_suiv = strrchr(str_tmp, ',');
		break;
	    case TRUE : 
		str_suiv = strrchr(str_tmp, ')');
		break;
	    }
	    new_str_pred_len = (strlen(str_tmp) - strlen(str_suiv)) + 1;
	    str_suiv = strdup(&(str_pred[new_str_pred_len]));

	    str_tmp[0] = '\0';
	    if (!premiere_ligne)
		(void) strcat(str_tmp, " "); 
	    (void) strncat(str_tmp, str_pred, new_str_pred_len);

	    /* add it to the text */
	    ADD_SENTENCE_TO_TEXT(t_pred, 
				 make_pred_commentary_sentence(strdup(str_tmp),
							       str_prefix));
	    str_pred =  str_suiv;
	}
	else {
	    /* if the remaining string fits in one line */
	    str_tmp[0] = '\0';
	    if (!premiere_ligne)
		(void) strcat(str_tmp, " "); 
	    (void) strcat(str_tmp, str_pred);

	    ADD_SENTENCE_TO_TEXT(t_pred, 
				 make_pred_commentary_sentence(str_tmp,
							       str_prefix));
	    str_pred[0] = '\0';
	}
	
	if (premiere_ligne) {
	    premiere_ligne = FALSE;
	    longueur_max = longueur_max - 1;
	    if (foresys){
		int i;
		int nb_espaces = strlen(str_prefix) -
		    strlen(FORESYS_CONTINUATION_PREFIX);

		str_prefix = strdup(str_prefix);
		str_prefix[0] = '\0';
		(void) strcat(str_prefix, FORESYS_CONTINUATION_PREFIX);
		for (i=1; i <= nb_espaces; i++)
		    (void) strcat(str_prefix, " ");
	      }
	  }
    }
    
    return(t_pred);
}
    


/* text words_predicate_to_commentary(list w_pred, string comment_prefix)
 * input    : a list of strings, one of them representing a predicate.
 * output   : a text of several lines of commentaries containing 
 *            this list of strings, and beginning with comment_prefix.
 * modifies : nothing.
 */
text 
words_predicate_to_commentary(w_pred, comment_prefix)
list w_pred;
string comment_prefix;
{
    string str_pred;
    text t_pred;

    /* str_pred is the string corresponding to the concatenation
     * of the strings in w_pred */
    str_pred = words_to_string(w_pred);

    t_pred = string_predicate_to_commentary(str_pred, comment_prefix);

    return(t_pred);
}


/* sentence make_pred_commentary_sentence(string str_pred, string comment_prefix) 
 * input    : a substring formatted to be a commentary
 * output   : a sentence, containing the commentary form of this string,
 *            beginning with the comment_prefix.
 * modifies : nothing
 */
sentence 
make_pred_commentary_sentence(str_pred, comment_prefix)
string str_pred;
string comment_prefix;
{
    char str_tmp[MAX_PRED_COMMENTARY_STRLEN + 1];
    sentence sent_pred;

    str_tmp[0] = '\0';
    (void) strcat(str_tmp, comment_prefix); 
    (void) strcat(str_tmp, "  ");
    (void) strcat(str_tmp, str_pred);
    (void) strcat(str_tmp, "\n"); 

    sent_pred = make_sentence(is_sentence_formatted, strdup(str_tmp));
    return(sent_pred);
}




void 
constante_to_textline(
    char * operation_line,
    Value constante,
    boolean is_inegalite, 
    boolean a_la_fortran)
{
    operation_line[0]='\0';
    (void) sprint_operator(operation_line+strlen(operation_line), 
			   is_inegalite, a_la_fortran);
    (void) sprint_Value(operation_line+strlen(operation_line), 
			constante);
}


void
signed_operation_to_textline(
char * operation_line,
char signe,
Value coeff,
Variable var,
char * (*variable_name)(Variable))
{
   
    (void) sprintf(operation_line+strlen(operation_line),"%c",signe);
    unsigned_operation_to_textline(operation_line,coeff,var,variable_name);

}
void
unsigned_operation_to_textline(
char * operation_line,
Value coeff,
Variable var,
char * (*variable_name)(Variable))
{
    if (value_notone_p(ABS(coeff)) || var==TCST)
	(void) sprint_Value(operation_line+strlen(operation_line), coeff);
    (void) sprintf(operation_line+strlen(operation_line),"%s", 
		   variable_name(var));

}


static char * 
contrainte_to_text_1(
string aux_line,
string str_prefix,
text txt,
Pvecteur v,
boolean is_inegalite,
char * (*variable_name)(Variable),
boolean a_la_fortran,
boolean first_line
)
{
    short int debut = 1;
    Value constante = VALUE_ZERO;
    char operation_line[MAX_LINE_LENGTH];
	
    while (!VECTEUR_NUL_P(v)) {
	Variable var = var_of(v);
	Value coeff = val_of(v);
	operation_line[0]='\0';

	if (var!=TCST) {
	    char signe;

	    if (value_notzero_p(coeff)) {
		if (value_pos_p(coeff))
		    signe =  '+';
		else {
		    signe = '-';
		    coeff = value_uminus(coeff);
		};
		if (value_pos_p(coeff) && debut)
		    unsigned_operation_to_textline(operation_line,coeff,var,
						   variable_name);
		else 
		    signed_operation_to_textline(operation_line,signe,coeff,
						 var, variable_name);
		debut = 0;
	    }
	    first_line = add_to_current_line(aux_line,operation_line,
					     str_prefix,txt,first_line);
	}
	else
	    /* on admet plusieurs occurences du terme constant!?! */
	    value_addto(constante, coeff);

	v = v->succ;
    }
    constante_to_textline(operation_line,value_uminus(constante),is_inegalite,
			  a_la_fortran);
    first_line = add_to_current_line(aux_line, operation_line,
				     str_prefix,txt,first_line);
    return aux_line;
}


static char * 
contrainte_to_text_2(
string aux_line,
string str_prefix,
text txt,
Pvecteur v,
boolean is_inegalite,
char * (*variable_name)(Variable),
boolean a_la_fortran,
boolean first_line
)
{
    Pvecteur coord;
    short int debut = TRUE;
    int positive_terms = 0;
    int negative_terms = 0;
    Value const_coeff = 0;
    boolean const_coeff_p = FALSE;
    char signe;
    char operation_line[MAX_LINE_LENGTH];
   
    if(!is_inegalite) {
	for(coord = v; !VECTEUR_NUL_P(coord); coord = coord->succ) {
	    if(vecteur_var(coord)!= TCST) 
		(value_pos_p(vecteur_val(coord))) ? 
		    positive_terms++ :  negative_terms++;   
	}

	if(negative_terms > positive_terms) 
	    vect_chg_sgn(v);
    }

    positive_terms = 0;
    negative_terms = 0;

    for(coord = v; !VECTEUR_NUL_P(coord); coord = coord->succ) {
	Value coeff = vecteur_val(coord);
	Variable var = vecteur_var(coord);
	operation_line[0]='\0';

	if (value_pos_p(coeff)) {
	    positive_terms++;
	     if(!term_cst(coord)|| is_inegalite) {
		 signe =  '+';
		 if (debut)
		     unsigned_operation_to_textline(operation_line,coeff,var,
				       variable_name);
		 else 
		     signed_operation_to_textline(operation_line,signe,
						  coeff,var,variable_name);
		 debut=FALSE;

	    }
	     else  positive_terms--;
	       
	     first_line = add_to_current_line(aux_line,operation_line,
					     str_prefix,txt,first_line);
	}
    }

    operation_line[0]='\0';
    if(positive_terms == 0) 	
	(void) sprintf(operation_line+strlen(operation_line), "0"); 
    
    (void) sprint_operator(operation_line+strlen(operation_line), 
			   is_inegalite, a_la_fortran);
    
    first_line = add_to_current_line(aux_line,operation_line,
					     str_prefix,txt,first_line);

    debut = TRUE;
    for(coord = v; !VECTEUR_NUL_P(coord); coord = coord->succ) {
	Value coeff = vecteur_val(coord);
	Variable var = var_of(coord);
	operation_line[0]='\0';

	if(term_cst(coord) && !is_inegalite) {
	    /* Save the constant term for future use */
	    const_coeff_p = TRUE;
	    const_coeff = coeff;
	    /* And now, a lie... In fact, rhs_terms++ */
	    negative_terms++;
	}
	else if (value_neg_p(coeff)) {
	    negative_terms++;
	    signe = '+';
	    if (debut) {
		unsigned_operation_to_textline(operation_line, 
					       value_uminus(coeff),var,
					       variable_name);
		debut=FALSE;
	    }
	    else 
		signed_operation_to_textline(operation_line,signe,
					     value_uminus(coeff),var,
					     variable_name);
	    
	}
	first_line = add_to_current_line(aux_line, operation_line,
					 str_prefix,txt,first_line); 
    }
    operation_line[0]='\0';
    if(negative_terms == 0) {
	(void) sprintf(operation_line+strlen(operation_line), "0"); 
        first_line = add_to_current_line(aux_line,operation_line,
					     str_prefix,txt,first_line);
    }
      else if(const_coeff_p) {
	assert(value_notzero_p(const_coeff));
	
	if  (!debut && value_neg_p(const_coeff))
	    (void) sprintf(operation_line+strlen(operation_line), "+"); 
	(void) sprint_Value(operation_line+strlen(operation_line), 
			    value_uminus(const_coeff));

	first_line = add_to_current_line(aux_line,operation_line,
					 str_prefix,txt,first_line);
    }
   

    return aux_line;
}


char * 
contrainte_text_format(
char * aux_line,
char * str_prefix,
text txt,
Pcontrainte c,
boolean is_inegalite,
char * (*variable_name)(Variable),
boolean a_la_fortran,
boolean first_line
)
{
    Pvecteur v;
    int heuristique = 2;

    if (!CONTRAINTE_UNDEFINED_P(c))
	v = contrainte_vecteur(c);
    else
	v = VECTEUR_NUL;

    assert(vect_check(v));

    switch(heuristique) {
    case 1: aux_line = contrainte_to_text_1(aux_line,str_prefix,txt,
					    v,is_inegalite, variable_name, 
					    a_la_fortran, first_line);
	break;
    case 2:aux_line = contrainte_to_text_2(aux_line,str_prefix,txt,v,
					   is_inegalite, variable_name, 
					    a_la_fortran, first_line);
	
	break;
    default: contrainte_error("contrainte_sprint", "unknown heuristics\n");
    }

    return aux_line;
}

char  * 
egalite_text_format(aux_line,str_prefix,txt,eg,variable_name,
		    a_la_fortran,first_line)
char *aux_line;
char * str_prefix;
text txt;
Pcontrainte eg;
char * (*variable_name)();
boolean  a_la_fortran, first_line;
{
    return contrainte_text_format(aux_line,str_prefix,txt,eg,FALSE,
				  variable_name,a_la_fortran,first_line);
}

char * 
inegalite_text_format(
char *aux_line,
char * str_prefix,
text txt,
Pcontrainte ineg,
char * (*variable_name)(),
boolean a_la_fortran,
boolean first_line
)
{
    return contrainte_text_format(aux_line,str_prefix,txt,ineg,TRUE, 
				  variable_name,a_la_fortran,first_line);
}
