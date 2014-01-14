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
/*
 * (pretty)print of reductions.
 *
 * FC, June 1996.
 */

#include "local-header.h"
#include "text.h"
#include "text-util.h"
#include "semantics.h"
#include "prettyprint.h"

#define PROP_SUFFIX ".proper_reductions"
#define CUMU_SUFFIX ".cumulated_reductions"

#define PROP_DECO "proper reductions"
#define CUMU_DECO "cumulated reductions"

/****************************************************** STATIC INFORMATION */

GENERIC_GLOBAL_FUNCTION(printed_reductions, pstatement_reductions)
static string reduction_decoration = NULL;

/************************************************************* BASIC WORDS */

/* generates a short note to tell about the type of the statement
 * being decorated.
 */
string note_for_statement(statement s)
{
    instruction i = statement_instruction(s);
    switch (instruction_tag(i))
    {
    case is_instruction_sequence: return "seq  ";
    case is_instruction_loop: return "loop ";
    case is_instruction_test: return "test ";
    case is_instruction_call: return "call ";
    case is_instruction_unstructured: return "unst ";
    case is_instruction_whileloop: return "while ";
    case is_instruction_goto: return "goto ";
    case is_instruction_forloop: return "for ";
    case is_instruction_expression: return "expr ";
    default: 
      pips_internal_error("unexpected instruction tag");
      break;
    }
    return "";
}

/* returns a (static) string describing the tag t reduction
 */
string reduction_operator_tag_name(tag t)
{
    switch(t)
    {
    case is_reduction_operator_none:        return "none";
    case is_reduction_operator_sum:         return "sum";
    case is_reduction_operator_csum:        return "csum";
    case is_reduction_operator_prod:        return "prod";
    case is_reduction_operator_min:         return "min";
    case is_reduction_operator_max:         return "max";
    case is_reduction_operator_and:         return "and";
    case is_reduction_operator_or:          return "or";
    case is_reduction_operator_bitwise_or:  return "bitwise_or";
    case is_reduction_operator_bitwise_xor: return "bitwise_xor";
    case is_reduction_operator_bitwise_and: return "bitwise_and";
    case is_reduction_operator_eqv:         return "eqv";
    case is_reduction_operator_neqv:        return "neqv";
    default: pips_internal_error("unexpected reduction operator tag!");
    }

    return NULL;
}

/* allocates and returns the name of the operator
 */
string reduction_operator_name(reduction_operator o)
{
    return strdup(reduction_operator_tag_name(reduction_operator_tag(o)));
}

/* returns the name of the reduction (!!! not allocated)
 */
string reduction_name(reduction r)
{
    return reduction_operator_tag_name(reduction_tag(r));
}

/* allocates and returns a list of strings for reduction r
 */
static list /* of string */ words_reduction(reduction r)
{
    return CONS(STRING, reduction_operator_name(reduction_op(r)),
	   CONS(STRING, strdup("["),
		gen_nconc( words_reference(reduction_reference(r),NIL),
	   CONS(STRING, strdup("],"), NIL))));
}

/* allocates and returns a list of string with note ahead if not empty.
 * it describes the reductions in rs.
 */
static list /* of string */ words_reductions(string note, reductions rs)
{
  list /* of string */ ls = NIL;
  FOREACH (REDUCTION, r,reductions_list(rs)) {
    ls = gen_nconc(words_reduction(r), ls);
  }
  return ls? CONS(STRING, strdup(note), ls): NIL;
}

void print_reduction(reduction r)
{
    reference ref = reduction_reference(r);
    fprintf(stderr, "reduction is %s[",
	    reduction_operator_tag_name
	        (reduction_operator_tag(reduction_op(r))));
    if (!reference_undefined_p(ref)) print_reference(ref);
    fprintf(stderr, "]\n");
}

/************************************************* REDUCTION PRETTY PRINT */

/* function to allocate and returns a text, passed to the prettyprinter
 * uses some static variables:
 * - printed_reductions function
 * -
 */
/************************************************* REDUCTION PRETTY PRINT */

/* function to allocate and returns a text, passed to the prettyprinter
 * uses some static variables:
 * - printed_reductions function
 * -
 */
static text text_reductions(__attribute__((unused)) entity module, __attribute__((unused)) int margin, statement s)
{
    text t;

    debug_on("REDUCTIONS_DEBUG_LEVEL");
    pips_debug(1, "considering statement %p\n", s);

    t = bound_printed_reductions_p(s)? /* unreachable statements? */
	words_predicate_to_commentary
	    (words_reductions(note_for_statement(s),
			      load_printed_reductions(s)),
			          get_comment_sentinel()):
		make_text(NIL);

    debug_off();
    return t;
}

/* returns a reduction-decorated text for statement s
 */
static text text_code_reductions(statement s)
{
    text t;
    debug_on("PRETTYPRINT_DEBUG_LEVEL");
    init_prettyprint(text_reductions);
    t = text_module(get_current_module_entity(), s);
    close_prettyprint();
    debug_off();
    return t;
}

/* handles the required prettyprint
 * ??? what about summary reductions?
 * should be pprinted with cumulated regions.
 * @ return true if succeed
 * @param module_name the module name to process
 * @param resource_name the resource to use
 * @param decoration_name
 * @param summary_name
 * @param file_suffix
 */
static bool
print_any_reductions(
    const char* module_name,
    string resource_name,
    string decoration_name,
    string summary_name,
    string file_suffix)
{
    text t;

    debug_on("REDUCTIONS_DEBUG_LEVEL");
    pips_debug(1, "considering module %s for %s\n",
	       module_name, decoration_name);

    set_current_module_entity(local_name_to_top_level_entity(module_name));
    set_printed_reductions((pstatement_reductions)
	 db_get_memory_resource(resource_name, module_name, true));
    set_current_module_statement((statement)
	 db_get_memory_resource(DBR_CODE, module_name, true));
    reduction_decoration = decoration_name;

    t = text_code_reductions(get_current_module_statement());

    if (summary_name)
    {
	reductions rs = (reductions)
	    db_get_memory_resource(summary_name, module_name, true);
	text p =
	    words_predicate_to_commentary(words_reductions("summary ", rs),
	                                  get_comment_sentinel());
	MERGE_TEXTS(p, t);
	t=p;
    }

    make_text_resource_and_free(module_name, DBR_PRINTED_FILE, file_suffix, t);

    reset_current_module_entity();
    reset_current_module_statement();
    reset_printed_reductions();
    reduction_decoration = NULL;

    debug_off();
    return true;
}

/* Handlers for PIPSMAKE
 */
bool print_code_proper_reductions(const char* module_name)
{
    return print_any_reductions(module_name,
				DBR_PROPER_REDUCTIONS,
				PROP_DECO,
				NULL,
				PROP_SUFFIX);
}

bool print_code_cumulated_reductions(const char* module_name)
{
    return print_any_reductions(module_name,
				DBR_CUMULATED_REDUCTIONS,
				CUMU_DECO,
				DBR_SUMMARY_REDUCTIONS,
				CUMU_SUFFIX);
}

/* end of it
 */
