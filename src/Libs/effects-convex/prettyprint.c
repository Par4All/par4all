/* package convex effects :  Be'atrice Creusillet 5/97
 *
 * File: prettyprint.c
 * ~~~~~~~~~~~~~~~~~~~
 *
 * This File contains the prettyprinting functions.
 *
 */

#include <stdio.h>
#include <string.h>
#include <values.h>

#include "genC.h"
#include "ri.h"
#include "database.h"

#include "misc.h"
#include "ri-util.h"
#include "top-level.h"
#include "text.h"

#include "properties.h"

#include "transformer.h"
#include "semantics.h"
#include "effects-generic.h"
#include "effects-convex.h"

#include "text-util.h"
#include "pipsdbm.h"
#include "resources.h"
#include "prettyprint.h"

#define REGION_BUFFER_SIZE 2048

#define REGION_FORESYS_PREFIX "C$REG"
#define PIPS_NORMAL_PREFIX "C"

/* char * pips_region_user_name(entity ent)
 * output   : the name of entity.
 * modifies : nothing.
 * comment  : allows to "catch" the PHIs entities, else, it works like
 *            pips_user_value_name() (see semantics.c).	
 */
char *
pips_region_user_name(entity ent)
{
    /* external_value_name cannot be used because there is no need for
       the #new suffix, but the #old one is necessary */
    string name;
    if(ent == NULL)
	/* take care of the constant term TCST */
	name = "";
    else {
	char *ent_name = entity_name(ent);

	if (strncmp(ent_name, REGIONS_MODULE_NAME, 7) == 0)
	    /* ent is a PHI entity from the regions module */
	    name = entity_local_name(ent);
	else	
            if (entity_has_values_p(ent))
		name = entity_minimal_name(ent);
	    else
		name = entity_name(ent);
    }

    return name;
}


/* static string region_sc_to_string(string s, Psysteme ps)
 * input    : a string buffer and a region predicate
 * output   : the string buffer filled with a character string representing the
 *            predicate.
 * modifies : nothing.
 * comment  : ps is supposed to be sorted in such a way that in equalities and 
 *            inequalities constraints containing phi variables come first.
 *            equalities with phi variables are printed first, and then 
 *            inequalities with phi variables, and then equalities and 
 *            inequalities with no phi variables.
 */
string
region_sc_to_string(string s, Psysteme ps)
{
    Pcontrainte peg, pineg;
    boolean a_la_fortran = get_bool_property("PRETTYPRINT_FOR_FORESYS");
    bool first = TRUE;
    int passe ;
    
    if (ps == NULL) 
    {
	(void) sprintf(s+strlen(s),"SC_UNDEFINED");
	return(s);
    }

    peg = ps->egalites;
    pineg = ps->inegalites;

    if (!a_la_fortran)
	(void) sprintf(s+strlen(s), "{");

    for(passe = 1; passe <= 2; passe++) {
	bool phis = (passe == 1);

	for (;(peg!=NULL) &&
		 ((!phis) || (phis && vect_contains_phi_p(peg->vecteur))); 
	     peg=peg->succ) 
	{
	    if(first)
		first = FALSE;
	    else
		switch (a_la_fortran) {
		case FALSE :
		    (void) sprintf(s+strlen(s),", ");
		    break;
		case TRUE : 
		    (void) sprintf(s+strlen(s),".AND.");
		    break;
		}
	    if (a_la_fortran)
		(void) sprintf(s+strlen(s),"(");
	    egalite_sprint_format(s,peg,pips_region_user_name, a_la_fortran);
	    if (a_la_fortran)
		(void) sprintf(s+strlen(s),")");
	}
	
	for (;
	     (pineg!=NULL) && ((!phis) || 
			       (phis && vect_contains_phi_p(pineg->vecteur))); 
	     pineg=pineg->succ) {
	    if(first)
	    first = FALSE;
	    else
		switch (a_la_fortran) {
		case FALSE :
		    (void) sprintf(s+strlen(s),", ");
		    break;
		case TRUE : 
		    (void) sprintf(s+strlen(s),".AND.");
		    break;
		}
	    if (a_la_fortran)
		(void) sprintf(s+strlen(s),"(");
	    inegalite_sprint_format(s,pineg, pips_region_user_name, 
				    a_la_fortran);
	    if (a_la_fortran)
		(void) sprintf(s+strlen(s),")");
	}
    
    }
    
    if (!a_la_fortran)
	(void) strcat(s,"}");

    return(s);
}



/* list words_region(effect reg)
 * input    : a region.
 * output   : a list of strings representing the region.
 * modifies : nothing.
 * comment  :	because of 'buffer', this function cannot be called twice
 * before
 * its output is processed. Also, overflows in relation_to_string() 
 * cannot be prevented. They are checked on return.
 */
list
words_region(region reg)
{
    static char buffer[REGION_BUFFER_SIZE];
    
    list pc = NIL;
    reference r = effect_reference(reg);
    action ac = effect_action(reg);
    approximation ap = effect_approximation(reg);
    boolean foresys = get_bool_property("PRETTYPRINT_FOR_FORESYS");
    Psysteme sc = region_system(reg);

    buffer[0] = '\0';

    if(!region_empty_p(reg) && !region_rn_p(reg))
    {
	Pbase sorted_base = region_sorted_base_dup(reg);
	Psysteme sc = sc_dup(region_system(reg));
	
      /* sorts in such a way that constraints with phi variables come first */
	region_sc_sort(sc, sorted_base);

	strcat(buffer, "-");	
	region_sc_to_string(buffer, sc);
	sc_rm(sc);
	base_rm(sorted_base);

    }
    else
    {
	strcat(buffer, "-");	
	region_sc_to_string(buffer, sc);
    }
    pips_assert("words_region", strlen(buffer) < REGION_BUFFER_SIZE );

    if (foresys)
    {
      pc = gen_nconc(pc, words_reference(r));
      pc = CHAIN_SWORD(pc, ", RGSTAT(");
      pc = CHAIN_SWORD(pc, action_read_p(ac) ? "R," : "W,");
      pc = CHAIN_SWORD(pc, approximation_may_p(ap) ? "MAY), " : "EXACT), ");
      pc = CHAIN_SWORD(pc, buffer);
    }
    else /* PIPS prettyprint */
    {
	pc = CHAIN_SWORD(pc, "<");
	pc = gen_nconc(pc, effect_words_reference(r));
	pc = CHAIN_SWORD(pc, "-");
	pc = CHAIN_SWORD(pc, action_interpretation(action_tag(ac)));
	pc = CHAIN_SWORD(pc, approximation_may_p(ap) ? "-MAY" : "-EXACT");
	pc = CHAIN_SWORD(pc, buffer);
	pc = CHAIN_SWORD(pc, ">");
    }

    return pc;
}





/* text text_region(effect reg)
 * input    : a region
 * output   : a text consisting of several lines of commentaries, 
 *            representing the region
 * modifies : nothing
 */
text 
text_region(effect reg)
{
    text t_reg = make_text(NIL);
    boolean foresys = get_bool_property("PRETTYPRINT_FOR_FORESYS");
    string str_prefix;

    if (foresys)
	str_prefix = REGION_FORESYS_PREFIX;
    else
	str_prefix = PIPS_NORMAL_PREFIX;
    
    if(reg == effect_undefined)
    {
	ADD_SENTENCE_TO_TEXT(t_reg, 
			     make_pred_commentary_sentence
			     (strdup("<REGION_UNDEFINED>"),
			      str_prefix));
	user_log("[region_to_string] unexpected effect undefined\n");
    }
    else
    {
	gen_free(t_reg);
	t_reg = words_predicate_to_commentary(words_region(reg), str_prefix);
    }

    return(t_reg);   
}


/* text text_array_regions(list l_reg, string ifread, string ifwrite)
 * input    : a list of regions
 * output   : a text representing this list of regions.
 * comment  : if the number of array regions is not nul, and if 
 *            PRETTYPRINT_LOOSE is TRUE, then empty lines are
 *            added before and after the text of the list of regions.
 */
static text
text_array_regions(list l_reg, string ifread, string ifwrite)
{
    text reg_text = make_text(NIL);
    /* in case of loose_prettyprint, at least one region to print? */
    boolean loose_p = get_bool_property("PRETTYPRINT_LOOSE");
    boolean one_p = FALSE;  

    set_action_interpretation(ifread, ifwrite);

    /* GO: No redundant test anymore, see  text_statement_array_regions */
    if (l_reg != (list) HASH_UNDEFINED_VALUE && l_reg != list_undefined) 
    {
	gen_sort_list(l_reg, effect_compare);
	MAP(EFFECT, reg,
	{
	    entity ent = effect_entity(reg);
	    if ( get_bool_property("PRETTYPRINT_SCALAR_REGIONS") || 
		! entity_scalar_p(ent)) 
	    {
		if (loose_p && !one_p )
		{
		    ADD_SENTENCE_TO_TEXT(reg_text, 
					 make_sentence(is_sentence_formatted, 
						       strdup("\n")));
		    one_p = TRUE;
		}
		MERGE_TEXTS(reg_text, text_region(reg));
	    }	
	},
	    l_reg);

	if (loose_p && one_p)
	    ADD_SENTENCE_TO_TEXT(reg_text, 
				 make_sentence(is_sentence_formatted, 
					       strdup("\n")));
    }

    reset_action_interpretation();
    return reg_text;
}

/* practical interfaces 
 */
text text_inout_array_regions(list l)
{ return text_array_regions(l, ACTION_IN, ACTION_OUT);}

text text_rw_array_regions(list l)
{ return text_array_regions(l, ACTION_READ, ACTION_WRITE);}

text text_copyinout_array_regions(list l)
{ return text_array_regions(l, ACTION_COPYIN, ACTION_COPYOUT);}

text text_private_array_regions(list l)
{ return text_array_regions(l, ACTION_PRIVATE, ACTION_PRIVATE);}

/*********************************************************** ABSOLETE MAYBE? */

/* CALLGRAPH/ICFG stuff (should be OBSOLETE?)
 */
#define is_RW
#define is_IN
#define is_OUT

static text 
get_text_regions_for_module(
    string module_name, 
    string resource_name,
    string ifread,
    string ifwrite)
{
    list /* of effect */ le = effects_effects((effects) 
	db_get_memory_resource(resource_name, module_name, TRUE));
    text t;
    t = text_array_regions(le, ifread, ifwrite);
    return t;
}

text 
get_text_regions(string module_name)
{
    return get_text_regions_for_module
	(module_name, DBR_SUMMARY_REGIONS, ACTION_READ, ACTION_WRITE);
}

text 
get_text_in_regions(string module_name)
{
    return get_text_regions_for_module
	(module_name, DBR_IN_SUMMARY_REGIONS, ACTION_IN, ACTION_OUT);
}

text 
get_text_out_regions(string module_name)
{
    return get_text_regions_for_module
	(module_name, DBR_OUT_SUMMARY_REGIONS, ACTION_IN, ACTION_OUT);
}

/*********************************************************** DEBUG FUNCTIONS */

/* void print_regions(list pc)
 * input    : a list of regions.
 * modifies : nothing.
 * comment  : prints the list of regions on stderr .
 */
static void 
print_regions_with_action(list pc, string ifread, string ifwrite)
{
    list lr;
    set_action_interpretation(ifread, ifwrite);

    if (pc == NIL) {
	fprintf(stderr,"\t<NONE>\n");
    }
    else {
        for (lr = pc ; !ENDP(lr); POP(lr)) {
            effect ef = EFFECT(CAR(lr));
	    print_region(ef);
	    fprintf(stderr,"\n");
        }
    }

    reset_action_interpretation();
}

/* external interfaces
 */
void print_rw_regions(list l)
{ print_regions_with_action(l, ACTION_READ, ACTION_WRITE);}

void print_inout_regions(list l)
{ print_regions_with_action(l, ACTION_IN, ACTION_OUT);}

void print_copyinout_regions(list l)
{ print_regions_with_action(l, ACTION_COPYIN, ACTION_COPYOUT);}

void print_private_regions(list l)
{ print_regions_with_action(l, ACTION_PRIVATE, ACTION_PRIVATE);}

void print_regions(list l) { print_rw_regions(l);}

/* void print_regions(effect r)
 * input    : a region.
 * modifies : nothing.
 * comment  : prints the region on stderr using words_region.
 */
void print_region(effect r)
{
    fprintf(stderr,"\t");
    if(effect_region_p(r)) 
	print_words(stderr, words_region(r));
    /* else print_words(stderr, words_effect(r)); */
    fprintf(stderr,"\n");
}



/************************************************* STATISTICS FOR OPERATORS */


void
print_regions_op_statistics(char *mod_name, int regions_type)
{
    string prefix = string_undefined;

    switch (regions_type) {
    case R_RW : 
	prefix = "rrw-";
	break;
    case R_IN : 
	prefix = "rin-";
	break;
    case R_OUT :
	prefix = "rout-";
	break;
    }

/*    print_proj_op_statistics(mod_name, prefix);
    print_umust_statistics(mod_name, prefix);
    print_umay_statistics(mod_name, prefix);
    print_dsup_statistics(mod_name, prefix); */
    /* print_dinf_statistics(mod_name, prefix); */

}


/***************************************************************** SORTING */


/* Compares two effects for sorting. The first criterion is based on names.
 * Local entities come first; then they are sorted according to the
 * lexicographic order of the module name, and inside each module name class,
 * according to the local name lexicographic order. Then for a given
 * entity name, a read effect comes before a write effect. It is assumed
 * that there is only one effect of each type per entity. bc.
 */
int
effect_compare(effect *peff1, effect *peff2)
{
    entity ent1 = effect_entity(*peff1);
    entity ent2 = effect_entity(*peff2);
    int eff1_pos = 0;

    /* same entity case: sort on action */
    if (same_entity_p(ent1,ent2))
	if (effect_read_p(*peff1)) 
	    return(-1);
	else
	    return(1);
    
    /* sort on module name */
    eff1_pos = strcmp(entity_module_name(ent1), entity_module_name(ent2));
    
    /* if same module name: sort on entity local name */
    if (eff1_pos == 0)
    {
	eff1_pos = strcmp(entity_local_name(ent1), entity_local_name(ent2));	
    }
    /* else: current module comes first, others in lexicographic order */
    else
    {
	entity module = get_current_module_entity();

	if (strcmp(module_local_name(module), entity_module_name(ent1)) == 0)
	    eff1_pos = -1;
	if (strcmp(module_local_name(module), entity_module_name(ent2)) == 0)
	    eff1_pos = 1;	    	
    }

    return(eff1_pos);
}

