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

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "text.h"
#include "database.h"

#include "misc.h"
#include "ri-util.h"
#include "top-level.h"
#include "properties.h"

#include "transformer.h"
#include "semantics.h"
#include "effects-generic.h"
#include "effects-convex.h"

#include "text-util.h"
#include "pipsdbm.h"
#include "resources.h"
#include "prettyprint.h"

#define REGION_FORESYS_PREFIX "C$REG"

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
	    /**** ARGH why using this stuff in transformer... ******/
            /* if (entity_has_values_p(ent)) */
	    /* else name = entity_name(ent); */
	    name = entity_minimal_name(ent);
    }

    return name;
}

string
region_sc_to_string(string s, Psysteme ps)
{
    pips_internal_error("implementation dropped\n");
    return string_undefined;
}

#define append(s) add_to_current_line(line_buffer, s, str_prefix, t_reg)

/* text text_region(effect reg)
 * input    : a region
 * output   : a text consisting of several lines of commentaries, 
 *            representing the region
 * modifies : nothing
 */
text 
text_region(effect reg)
{
    text t_reg;
    boolean foresys = get_bool_property("PRETTYPRINT_FOR_FORESYS");
    string str_prefix = foresys? 
	FORESYS_CONTINUATION_PREFIX: PIPS_COMMENT_CONTINUATION;
    char line_buffer[MAX_LINE_LENGTH];
    reference r;
    action ac;
    approximation ap;
    Psysteme sc;
    Pbase sorted_base;
    list /* of string */ ls;

    if(effect_undefined_p(reg))
    {
	user_log("[text_region] unexpected effect undefined\n");
	return make_text(make_sentence(is_sentence_formatted,
	   strdup(concatenate(str_prefix, "<REGION_UNDEFINED>\n", 0))));
    }
    /* else the effect is defined...
     */

    /* PREFIX
     */
    t_reg = make_text(NIL);
    strcpy(line_buffer, foresys? REGION_FORESYS_PREFIX: PIPS_COMMENT_PREFIX);
    if (!foresys) append("  <");

    /* REFERENCE
     */
    r = effect_reference(reg);
    ls = foresys? words_reference(r): effect_words_reference(r);

    MAP(STRING, s, append(s), ls);
    gen_free_string_list(ls); ls = NIL;

    /* ACTION and APPROXIMATION
     */
    ac = effect_action(reg);
    ap = effect_approximation(reg);
	
    if (foresys)
    {
	append(", RGSTAT(");
	append(action_read_p(ac) ? "R," : "W,");
	append(approximation_may_p(ap) ? "MAY), " : "EXACT), ");
    }
    else /* PIPS prettyprint */
    {
	append("-");
	append(action_interpretation(action_tag(ac)));
	append(approximation_may_p(ap) ? "-MAY" : "-EXACT");
	append("-");
    }

    /* SYSTEM
     * sorts in such a way that constraints with phi variables come first.
     */
    sorted_base = region_sorted_base_dup(reg);
    sc = sc_dup(region_system(reg));
    region_sc_sort(sc, sorted_base);

    system_sorted_text_format(line_buffer, str_prefix, t_reg, sc, 
	       pips_region_user_name, vect_contains_phi_p, foresys);

    sc_rm(sc);
    base_rm(sorted_base);

    /* CLOSE 
     */
    if (!foresys) append(">");
    close_current_line(line_buffer, t_reg,str_prefix);

    return t_reg;   
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
static text 
get_text_regions_for_module(
    string module_name, 
    string resource_name,
    string ifread,
    string ifwrite)
{
    text t;
    entity mod;
    list /* of effect */ le = effects_effects((effects) 
	db_get_memory_resource(resource_name, module_name, TRUE));

    /* the current entity may be used for minimal names... */
    mod = local_name_to_top_level_entity(module_name);
    set_current_module_entity(mod);
    t = text_array_regions(le, ifread, ifwrite);
    reset_current_module_entity();
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
 *
 * NW:
 * before calling
 * "print_inout_regions"
 * or "print_rw_regions"
 * or "print_copyinout_regions"
 * or "print_private_regions"
 *
 * "module_to_value_mappings" must be called to set up the
 * hash table to translate value into value names
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
 *
 * NW:
 * before calling "print_region" or "text_region"
 * 
 * "module_to_value_mappings" must be called to set up the
 * hash table to translate value into value names
 * (see comment for "module_to_value_mappings" for what must be done
 * before that is called)
 * 
 * and also "set_action_interpretation" with arguments:
 * ACTION_READ, ACTION_WRITE to label regions as R/W
 * ACTION_IN, ACTION_OUT to label regions as IN/OUT
 * ACTION_COPYIN, ACTION_COPYOUT to label regions as COPYIN/COPYOUT
 * ACTION_PRIVATE, ACTION_PRIVATE to label regions as PRIVATE
 *
 * like this:
 *
 * string module_name;
 * entity module;
 * ...
 * (set up call to module_to_value_mappings as indicated in its comments)
 * ...
 * module_to_value_mappings(module);
 * set_action_interpretation(ACTION_IN, ACTION_OUT);
 *
 * (that's it, but after the call to "print_region" or "text_region",
 * don't forget to do:)
 *
 * reset_action_interpretation();
 * (resets after call to module_to_value_mappings as indicated in its comments)
 */
void 
print_region(effect r)
{
    fprintf(stderr,"\n");
    if(effect_region_p(r)) {
	text t = text_region(r);
	print_text(stderr, t);
	free_text(t);
    }
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
    {
	if (effect_read_p(*peff1)) 
	    return(-1);
	else
	    return(1);
    }
    
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

