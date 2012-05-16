/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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
/* Package regions :  Be'atrice Creusillet 11/95
 *
 * array_translation
 * -----------------
 *
 * This File contains general purpose functions that compute the
 * translation of regions from one array to another (e.g. interprocedural
 * translation).
 *
 */

#include <stdio.h>
#include <string.h>

#include <setjmp.h>

#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "sommet.h"
#include "ray_dte.h"
#include "sg.h"
#include "polyedre.h"
#include "union.h"

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "database.h"
#include "constants.h"

#include "ri-util.h"
#include "effects-util.h"
#include "pipsdbm.h"
#include "resources.h"
#include "misc.h"
#include "text.h"
#include "text-util.h"
#include "transformer.h"
#include "preprocessor.h"
#include "properties.h"

#include "effects-generic.h"
#include "effects-convex.h"
//#include "alias-classes.h"

#define BACKWARD true
#define FORWARD false

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))


/***************************************************** LOCAL DEBUG FUNCTIONS */

static void reg_v_debug(v)
Pvecteur v;
{
    vect_fprint(stderr, v, (get_variable_name_t) pips_region_user_name);
}

static void reg_sc_debug(sc)
Psysteme sc;
{
    sc_fprint(stderr, sc, (get_variable_name_t) pips_region_user_name);
}

/*********************************************************************************/
/* STATISTICS                                                                    */
/*********************************************************************************/

static bool statistics_p;

/* inputs */
static int mat_dim_stat[8][8];  /* correspondances between source and target array
                                 * number of dimensions */
static int vect_size_ratio_stat[4]; /* size ratio after normalization */
static int zero_offset_stat; /* number cases in which the offset is nul */

static int scalar_to_scalar_stat;
static int scalar_to_array_stat;
static int array_to_array_stat;

/* translation */

static struct Common_Dimension_Stat
{
    int nb_calls;
    int all_similar;
    int not_same_decl;
    int non_linear_decl;
} common_dimension_stat;

static struct Linearization_Stat
{
    int nb_calls;
    int exact;
    int non_linear_decl;
    int non_linear_system;
} linearization_stat;

static struct Remaining_Dimension_Stat
{
    int nb;
    int exact;
    int non_linear_decl_or_offset;
} remaining_dimension_stat;

static struct Beta_Elimination_Stat
{
    int nb_calls;
    int exact_input;
    int exact;
} beta_elimination_stat;

static struct Phi_Elimination_Stat
{
    int nb_calls;
    int exact_input;
    int exact;
} phi_elimination_stat;

static struct Predicate_Translation
{
    int nb_calls;
    int exact_input;
    int exact;
} predicate_translation_stat;


/* initialization and closing*/

void region_translation_statistics_init(bool stat_p)
{
    int i,j;

    statistics_p = stat_p;

    if (!statistics_p)
	return;

    for (i=0; i<8; i++)
	for (j=0; j<8; j++)
	    mat_dim_stat[i][j] = 0;

    for (i=0; i<4; i++)
	vect_size_ratio_stat[i] = 0;

    zero_offset_stat = 0;
    scalar_to_scalar_stat = 0;
    scalar_to_array_stat = 0;
    array_to_array_stat = 0;

    common_dimension_stat.nb_calls = 0;
    common_dimension_stat.all_similar = 0;
    common_dimension_stat.not_same_decl = 0;
    common_dimension_stat.non_linear_decl = 0;

    linearization_stat.nb_calls = 0;
    linearization_stat.exact = 0;
    linearization_stat.non_linear_decl = 0;
    linearization_stat.non_linear_system = 0;

    remaining_dimension_stat.nb = 0;
    remaining_dimension_stat.exact = 0;
    remaining_dimension_stat.non_linear_decl_or_offset = 0;

    beta_elimination_stat.nb_calls = 0;
    beta_elimination_stat.exact_input = 0;
    beta_elimination_stat.exact = 0;

    phi_elimination_stat.nb_calls = 0;
    phi_elimination_stat.exact_input = 0;
    phi_elimination_stat.exact = 0;

    predicate_translation_stat.nb_calls = 0;
    predicate_translation_stat.exact_input = 0;
    predicate_translation_stat.exact = 0;

}

void
region_translation_statistics_close(const char *mod_name, const char *prefix)
{
    FILE *fp;
    string filename;
    int i,j,total;

    if (!statistics_p) return;

    filename = "inter_trans_stat";
    filename = strdup(concatenate(db_get_current_workspace_directory(), "/",
				  mod_name, ".", prefix, "_", filename, 0));

    fp = safe_fopen(filename, "w");
    fprintf(fp,"%s", mod_name);

    /* inputs */
    fprintf(fp, " %d %d %d %d", scalar_to_scalar_stat, scalar_to_array_stat,
	    array_to_array_stat, zero_offset_stat);

    for (i=0; i<8; i++)
	for (j=0; j<8; j++)
	    fprintf(fp, " %d", mat_dim_stat[i][j]);

    for (i=0; i<4; i++)
	fprintf(fp, " %d", vect_size_ratio_stat[i]);

    /* other ratios */
    for (total = 0, i=0; i<4; i++)
	total = total + vect_size_ratio_stat[i];
    fprintf(fp, " %d", scalar_to_array_stat + array_to_array_stat - total);

    /* translation */
    fprintf(fp, " %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d",
	    common_dimension_stat.nb_calls,
	    common_dimension_stat.all_similar,
	    common_dimension_stat.not_same_decl,
	    common_dimension_stat.non_linear_decl,

	    linearization_stat.nb_calls,
	    linearization_stat.exact,
	    linearization_stat.non_linear_decl,
	    linearization_stat.non_linear_system,

	    remaining_dimension_stat.nb,
	    remaining_dimension_stat.exact,
	    remaining_dimension_stat.non_linear_decl_or_offset,

	    beta_elimination_stat.nb_calls,
	    beta_elimination_stat.exact_input,
	    beta_elimination_stat.exact,

	    phi_elimination_stat.nb_calls,
	    phi_elimination_stat.exact_input,
	    phi_elimination_stat.exact,

	    predicate_translation_stat.nb_calls,
	    predicate_translation_stat.exact_input,
	    predicate_translation_stat.exact);

    fprintf(fp,"\n");
    safe_fclose(fp, filename);
    free(filename);
}

/****************************************************************************/
/* Local variables and functions to avoid multiple computations             */
/****************************************************************************/

static entity array_1, array_2;
static reference ref_1, ref_2;
static bool reference_p;
static Value offset;
static int dim_1, dim_2;
static Value size_elt_1, size_elt_2;
static dimension dims_1[NB_MAX_ARRAY_DIM], dims_2[NB_MAX_ARRAY_DIM];
static bool dim_1_assumed, dim_2_assumed;

static bool dims_array_init(entity array, dimension* dims, int dim_array)
{
    int i;
    bool dim_assumed;

    i = 0;
    dim_assumed = false;
    FOREACH(DIMENSION, dim, variable_dimensions(type_variable(entity_type(array)))) {
	if (i == dim_array -1)
	{
	    normalized nup = NORMALIZE_EXPRESSION(dimension_upper(dim));
	    normalized nlo = NORMALIZE_EXPRESSION(dimension_lower(dim));

	    if(normalized_linear_p(nup) && normalized_linear_p(nlo))
	    {
		Pvecteur pvup = normalized_linear(nup);
		Pvecteur pvlo = normalized_linear(nlo);

		/* FI: special case for the old Fortran habit of using
		   declarations such as D(1) or E(N,1) to declare a
		   pointer to an array of undefined (last) dimension.

		   Such declarations cannot be used for array bound
		   checking.

		   The warning message does not seem to fit the test. */
		if(!VECTEUR_NUL_P(pvup) && !VECTEUR_NUL_P(pvlo))
		if (vect_constant_p(pvup) && value_one_p(val_of(pvup)) &&
		    vect_constant_p(pvlo) && value_one_p(val_of(pvlo)))
		{
		    pips_user_warning("\nvariable (%s): "
				      "last upper dimension equal to lower;"
				      " assuming unbounded upper bound\n",
				      entity_name(array));
		    dim =
			make_dimension(dimension_lower(dim),
				       MakeNullaryCall
				       (CreateIntrinsic(UNBOUNDED_DIMENSION_NAME)));
		    dim_assumed = true;
		}
	    }
	}
	dims[i] = dim;
	i++;
    }

    return dim_assumed;
}

#define IS_EG true
#define NOT_EG false

#define PHI_FIRST true
#define NOT_PHI_FIRST false

static Psysteme entity_assumed_declaration_sc(dimension* dims, int ndim)
{
    Psysteme sc = sc_new();
    int dim;

    for (dim=1; dim<=ndim; dim++)
    {
      (void) sc_add_phi_equation(&sc, dimension_lower(dims[dim-1]),
				 dim, NOT_EG, NOT_PHI_FIRST);
      (void) sc_add_phi_equation(&sc, dimension_upper(dims[dim-1]),
				 dim, NOT_EG, PHI_FIRST);
    }

    return sc;
}


void region_translation_init(entity ent_1, reference rf_1,
			     entity ent_2, reference rf_2,
			     Value offset_1_m_2)
{
    array_1 = ent_1;
    array_2 = ent_2;
    ref_1 = rf_1;
    ref_2 = rf_2;
    reference_p =
      !(reference_undefined_p(ref_1) && reference_undefined_p(ref_2));
    offset = offset_1_m_2;

    dim_1 = NumberOfDimension(array_1);
    dim_2 = NumberOfDimension(array_2);

    if (statistics_p) mat_dim_stat[dim_1][dim_2]++;

    size_elt_1 = int_to_value(SizeOfElements(
	variable_basic(type_variable(entity_type(array_1)))));
    size_elt_2 = int_to_value(SizeOfElements(
	variable_basic(type_variable(entity_type(array_2)))));

    ifdebug(2)
    {
	pips_debug(2,"before conversion:\n");
	fprint_string_Value(stderr, "size_elt_1 = ", size_elt_1);
	fprint_string_Value(stderr, ", size_elt_2 = ", size_elt_2);
	fprintf(stderr, "\n");
	if(!reference_p)
	    fprint_string_Value(stderr, "offset =", offset),
		fprintf(stderr, "\n");
    }
    /* relative sizes of elements */
    if (value_eq(size_elt_1,size_elt_2) &&
	value_zero_p(value_mod(offset,size_elt_1)))
    {
	value_division(offset,size_elt_1);
	size_elt_1 = VALUE_ONE;
	size_elt_2 = VALUE_ONE;
	if (statistics_p) vect_size_ratio_stat[0]++;
    }
    else
	if (value_zero_p(value_mod(size_elt_1,size_elt_2)) &&
	    value_zero_p(value_mod(offset,size_elt_2)))
	{
	    value_division(offset,size_elt_2);
	    value_division(size_elt_1,size_elt_2);
	    size_elt_2 = VALUE_ONE;
	    if (statistics_p)
	    {
		if (value_eq(size_elt_1, VALUE_CONST(1)))
		    vect_size_ratio_stat[0]++;
		else if (value_eq(size_elt_1, VALUE_CONST(2)))
		    vect_size_ratio_stat[1]++;
		else if (value_eq(size_elt_1, VALUE_CONST(4)))
		    vect_size_ratio_stat[3]++;
	    }
	}
	else
	    if (value_zero_p(value_mod(size_elt_2,size_elt_1)) &&
		value_zero_p(value_mod(offset,size_elt_1)))
	    {
		value_division(offset,size_elt_1);
		value_division(size_elt_2,size_elt_1);
		size_elt_1 = VALUE_ONE;
		if (statistics_p)
		{
		    if (value_eq(size_elt_2, VALUE_CONST(1)))
			vect_size_ratio_stat[0]++;
		    else if (value_eq(size_elt_2, VALUE_CONST(2)))
			vect_size_ratio_stat[1]++;
		    else if (value_eq(size_elt_2, VALUE_CONST(4)))
			vect_size_ratio_stat[3]++;
		}
	    }

    if (statistics_p && value_zero_p(offset) && !reference_p) zero_offset_stat++;


    ifdebug(2)
    {
	pips_debug(2,"after conversion:\n");
	fprint_string_Value(stderr, "size_elt_1 = ", size_elt_1);
	fprint_string_Value(stderr, ", size_elt_2 = ", size_elt_2);
	fprintf(stderr, "\n");
	if(!reference_p)
	    fprint_string_Value(stderr, "offset =", offset),
		fprintf(stderr, "\n");
    }

    dim_1_assumed = dims_array_init(array_1, dims_1, dim_1);
    dim_2_assumed = dims_array_init(array_2, dims_2, dim_2);

}

static void region_translation_close()
{
    if (dim_1_assumed)
    {
	/* do not free the real declaration */
	dimension_lower(dims_1[dim_1-1]) = expression_undefined;
	free_dimension(dims_1[dim_1-1]);
    }
    if (dim_2_assumed)
    {
	/* do not free the real declaration */
	dimension_lower(dims_2[dim_2-1]) = expression_undefined;
	free_dimension(dims_2[dim_2-1]);
    }
}

/********************************************************************** MISC */

static bool some_phi_variable(Pcontrainte c)
{
  for (; c; c=c->succ)
    if (vect_contains_phi_p(c->vecteur))
      return true;
  return false;
}

/* if we have a region like: <A(PHI)-EXACT-{}>
 * it means that all *declared* elements are touched, although
 * this is implicit. this occurs with io effects of "PRINT *, A".
 * in such a case, the declaration constraints MUST be appended
 * before the translation, otherwise the result might be false.
 *
 * potential bug : if the declaration system cannot be generated,
 *   the region should be turned to MAY for the translation?
 */
void append_declaration_sc_if_exact_without_constraints(region r)
{
  entity v = reference_variable(region_any_reference(r));
  Psysteme s = region_system(r);

  if (entity_scalar_p(v) || region_may_p(r)) return;
  /* we have an exact array region */

  pips_debug(5, "considering exact region of array %s\n", entity_name(v));

  if (!some_phi_variable(sc_egalites(s)) &&
      !some_phi_variable(sc_inegalites(s)))
  {
    pips_debug(7, "appending declaration system\n");
    region_sc_append(r, entity_declaration_sc(region_entity(r)), false);
  }
}

/***************************************************************** INTERFACE */

static void region_translation_of_predicate(region reg, entity to_func);
static Psysteme array_translation_sc(bool *p_exact_translation_p);

/* region region_translation(region reg1, entity mod1, reference ref1,
 *                         entity ent2, entity mod2, reference ref2,
 *                         Pvecteur offset_1_m_2, bool offset_undef_p)
 * input    : a region reg1, from module mod1; a target entity ent2 in module
 *            mod2 (it is possible to have mod1 = mod2 for equivalences);
 *            references ref1 and ref2 and offset_1_m_2 are provided to
 *            represent the offset between the index of the initial and target
 *            entity; if both entities are in a common or are equivalenced,
 *            then we can only provide offset_1_m_2; when
 *            translating from a formal to a real parameter or from a real to
 *            a formal one, we only know the real reference, the other one being
 *            undefined.
 * output   : a list of regions corresponding to the translation of reg1.
 * modifies : nothing, reg1 is duplicated.
 * comment  :
 *
 * NW:
 * before calling "region_translation" do
 *
 * call "set_interprocedural_translation_context_sc"
 * (see comment for "set_interprocedural_translation_context_sc" for what
 * must be done before that is called)
 *
 * and "set_backward_arguments_to_eliminate" (for translation formals->reals)
 * or "set_forward_arguments_to_eliminate"
 *
 * like this:
 *
 * call call_site;
 * entity callee;
 * list real_args;
 * ...
 * (set up call to "set_interprocedural_translation_context_sc" as
 * indicated in its comments)
 * ...
 * real_args = call_arguments(call_site);
 * set_interprocedural_translation_context_sc(callee, real_args);
 * set_backward_arguments_to_eliminate(callee);
 *
 * (that's it, but after the call to "region_translation", don't forget to do:)
 *
 * reset_translation_context_sc();
 * reset_arguments_to_eliminate();
 * (resets after call to "set_interprocedural_translation_context_sc"
 * as indicated in its comments)
 */
region region_translation(region reg_1, entity func_1, reference rf_1,
			  entity ent_2, entity func_2, reference rf_2,
			  Value offset_1_m_2, bool backward_p)
{
    entity ent_1 = region_entity(reg_1);
    region reg_2 = region_undefined;
    Psysteme trans_sc;
    bool exact_translation_p = true;
    bool exact_input_p = true;

    debug_on("REGION_TRANSLATION_DEBUG_LEVEL");

    pips_debug(1,"initial entity: %s, initial function: %s\n",
	       entity_minimal_name(ent_1), entity_name(func_1));
    pips_debug(1,"target entity: %s, target function: %s\n",
	       entity_minimal_name(ent_2), entity_name(func_2));


    pips_assert("something to translate\n",
		!((ent_1==ent_2) && (func_1== func_2)));
    pips_assert("one reference only\n",
		reference_undefined_p(rf_1) || reference_undefined_p(rf_2));
    pips_assert("non-zero offset, or one reference\n",
		value_zero_p(offset_1_m_2) ||
		(reference_undefined_p(rf_1) && reference_undefined_p(rf_2)) );

    if ((ent_1==ent_2) && (func_1!=func_2))
    {
	reg_2 = region_dup(reg_1);
	pips_debug(1,"same entities.\n");
	region_translation_of_predicate(reg_2, func_2);
	debug_off();
	return(reg_2);
    }

    /* The easiest case first: scalar -> scalar
     * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     * An effect to the initial scalar corresponds to an effect on the
     * target scalar. Even if there is only a partial association (see
     * FORTRAN standard 17.1), writing to ent1 corresponds to a write
     * effect on ent2 (either because it is really written, as for
     * real -> complex, or because it becomes undefined). The only pb
     * is for OUT regions: if the variable becomes undefined, its value
     * cannot be exported. This case is too difficult to handle, and
     * I consider that the value of the variable is exported. BC.
     */

    if (entity_scalar_p(ent_1) && entity_scalar_p(ent_2))
    {
	if (statistics_p) scalar_to_scalar_stat++;
	reg_2 = make_reference_region(make_reference(ent_2, NIL),
				     region_action(reg_1));
	region_approximation_tag(reg_2) = region_approximation_tag(reg_1);
	debug_off();
	return reg_2;
    }

    if (statistics_p)
    {
	if (entity_scalar_p(ent_1) || entity_scalar_p(ent_2))
	    scalar_to_array_stat++;
	else
	    array_to_array_stat++;
    }

    ifdebug(1)
    {
	pips_debug(1,"initial region: \n");
	print_region(reg_1);
    }

    /* We now consider scalars as arrays with zero dimensions */
    region_translation_init(ent_1, rf_1, ent_2, rf_2, offset_1_m_2);

    trans_sc = array_translation_sc(&exact_translation_p);

    if (!SC_UNDEFINED_P(trans_sc))
    {
	trans_sc = sc_safe_append(trans_sc, get_translation_context_sc());

	ifdebug(2)
	{
	    pips_debug(2, " translation context system :\n ");
	    reg_sc_debug(get_translation_context_sc());
	    pips_debug(2, " translation system :\n ");
	    reg_sc_debug(trans_sc);
	}

	reg_2 = region_dup(reg_1);

	/* As soon as possible: This allows to use variable elimination
         * without exactness tests when the translation is not exact.
	 */
	if (!exact_translation_p)
	{
	    pips_user_warning("bad reshaping between %s and %s.\n",
			      entity_name(array_1), entity_name(array_2));
	    region_approximation_tag(reg_2) = is_approximation_may;
	}

	append_declaration_sc_if_exact_without_constraints(reg_2);
	region_sc_append(reg_2, trans_sc, false);

	/* test to avoid the call to region_remove_beta_variables in most usual
	 * cases
	 */
	if (value_ne(size_elt_1,size_elt_2))
	{
	    if (statistics_p)
	    {
		exact_input_p = region_exact_p(reg_2);
		beta_elimination_stat.nb_calls++;
		if (exact_input_p) beta_elimination_stat.exact_input++;
	    }
	    region_remove_beta_variables(reg_2);
	    if (statistics_p && exact_input_p && region_exact_p(reg_2))
		beta_elimination_stat.exact++;
	}

	region_entity(reg_2) = entity_undefined;
	free_reference(region_any_reference(reg_2));
	if(cell_preference_p(region_cell(reg_2))) {
	  preference p_2 = cell_preference(region_cell(reg_2));
	  preference_reference(p_2) = make_regions_psi_reference(array_2);
	}
	else
	  cell_reference(region_cell(reg_2)) = make_regions_psi_reference(array_2);

	if (statistics_p)
	{
	    exact_input_p = region_exact_p(reg_2);
	    phi_elimination_stat.nb_calls++;
	    if (exact_input_p) phi_elimination_stat.exact_input++;
	}
	region_remove_phi_variables(reg_2);
	if (statistics_p && exact_input_p && region_exact_p(reg_2))
	{
	    phi_elimination_stat.exact++;
	}
	debug_region_consistency(reg_2);

	/* should be unnecessary */
	trans_sc = region_system(reg_2);
	trans_sc->base = BASE_NULLE;
	sc_creer_base(trans_sc);
	/* sc_nredund(&trans_sc); */
	region_system_(reg_2) = newgen_Psysteme(trans_sc);
	debug_region_consistency(reg_2);

	psi_to_phi_region(reg_2);
	debug_region_consistency(reg_2);

	if (func_1 != func_2)
	{
	    if (statistics_p)
	    {
		exact_input_p = region_exact_p(reg_2);
		predicate_translation_stat.nb_calls++;
		if (exact_input_p) predicate_translation_stat.exact_input++;
	    }
	    region_translation_of_predicate(reg_2, func_2);
	    if (statistics_p && exact_input_p && region_exact_p(reg_2))
		predicate_translation_stat.exact++;
	    debug_region_consistency(reg_2);
	}

	if (!reference_p || backward_p == BACKWARD)
	{
	  Psysteme sd;

	  if (!storage_formal_p(entity_storage(region_entity(reg_2)))
	      || get_bool_property("REGIONS_WITH_ARRAY_BOUNDS"))
	  {
	    sd = entity_declaration_sc(region_entity(reg_2));
	  }
	  else
	  {
	    /* last dim of formals to be ignored if equal?!
	     * because of bug :
	     *   SUB S1(X), X(1), CALL S2(X)     => Write X(1:1)...
	     *   SUB S2(Y), Y(10) Y(1:10) = ...
	     */
	    sd = entity_assumed_declaration_sc(dims_2, dim_2);
	  }

	  region_sc_append_and_normalize(reg_2, sd, 1);
	  debug_region_consistency(reg_2);
	}
    }
    else
    {
	reg_2 = entity_whole_region(array_2, region_action(reg_1));
	append_declaration_sc_if_exact_without_constraints(reg_2);
	debug_region_consistency(reg_2);
    }

    ifdebug(1)
    {
	pips_debug(1,"final region: \n");
	print_region(reg_2);
    }

    region_translation_close();
    debug_off();
    return reg_2;
}

/*****************************************************************************/
/* INTERFACE FUNCTIONS TO AVOID MULTIPLE COMPUTATIONS                        */
/*****************************************************************************/

/* System of constraints representing the relations between formal
 * and actual parameters.
 */

/* CONTEXT STACK */
/* hack to work around the fact that Psysteme is an external type. should
 * be generalized? bc */
#define Psysteme_domain -1
DEFINE_LOCAL_STACK(translation_context, Psysteme)

void set_region_interprocedural_translation()
{
    make_translation_context_stack();
}

void reset_region_interprocedural_translation()
{
    free_translation_context_stack();
}

/* NW:
 * before calling
 * "set_interprocedural_translation_context_sc"
 * for (entity) module
 * do:
 * (see also comments for "module_to_value_mappings")
 *
 * module_name = module_local_name(module);
 * set_current_module_entity(module);
 * regions_init();
 *
 * (the next call is for IN/OUT regions,
 * otherwise, do get_regions_properties() )
 *
 * get_in_out_regions_properties();
 * set_current_module_statement( (statement)
 * db_get_memory_resource(DBR_CODE, module_name, true) );
 * set_cumulated_rw_effects((statement_effects)
 * db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, true));
 * module_to_value_mappings(module);
 * set_precondition_map( (statement_mapping)
 *               db_get_memory_resource(DBR_PRECONDITIONS, module_name, true));
 *
 * (that's it,
 * but we musn't forget to reset it all again
 * after the call to set_interprocedural_translation_context_sc
 * as below)
 *
 * reset_current_module_statement();
 * reset_cumulated_rw_effects();
 * reset_precondition_map();
 * regions_end();
 * reset_current_module_entity();
 */
void
set_interprocedural_translation_context_sc(entity callee, list real_args)
{
    list /* of entity */ l_formals = module_formal_parameters(callee);
    int arg_num, n_formals = gen_length(l_formals);
    Psysteme sc;

    gen_free_list(l_formals);

    sc = sc_new();

    /* if there are more actuals than formals, they are skipped.
     */
    for(arg_num = 1;
	!ENDP(real_args) && arg_num<=n_formals;
	real_args = CDR(real_args), arg_num++)
    {
	entity formal_ent = find_ith_formal_parameter(callee,arg_num);
	expression real_exp = EXPRESSION(CAR(real_args));

	if (entity_integer_scalar_p(formal_ent))
	{
	    normalized n_real_exp = NORMALIZE_EXPRESSION(real_exp);

	    if (normalized_linear_p(n_real_exp)){
		Pvecteur v1 = normalized_linear(n_real_exp);
		Pvecteur v2 = vect_new((Variable) formal_ent, VALUE_ONE);

		sc_add_egalite(sc, contrainte_make(vect_substract(v1,v2)));
		vect_rm(v2);
	    }

	}
    }

    base_rm(sc_base(sc));
    sc_base(sc) = (Pbase) NULL;
    sc_creer_base(sc);
    set_translation_context_sc(sc_safe_normalize(sc));

}


void set_translation_context_sc(Psysteme sc)
{
    translation_context_push(sc);
}

Psysteme get_translation_context_sc()
{
    return(translation_context_head());
}


void reset_translation_context_sc()
{
    sc_rm(translation_context_head());
    translation_context_pop();
}



/* Formal or actual arguments to eliminate, depending on the direction
 * of propagation.
 */

static list l_arguments_to_eliminate = NIL;

void set_forward_arguments_to_eliminate()
{
    entity module = get_current_module_entity();
    /* FI: Let's hope it's OK for C as well */
    list l_decls = code_declarations(value_code(entity_initial(module)));

    FOREACH(ENTITY, var, l_decls)
     {
	 if (type_variable_p(entity_type(var)) && entity_scalar_p(var))
	     l_arguments_to_eliminate = CONS(ENTITY, var, l_arguments_to_eliminate);
     }

}

void set_backward_arguments_to_eliminate(entity func)
{
    l_arguments_to_eliminate = function_formal_parameters(func);
}

void set_arguments_to_eliminate(list l_args)
{
    l_arguments_to_eliminate = l_args;
}

void reset_arguments_to_eliminate()
{
    l_arguments_to_eliminate = NIL;
}


list get_arguments_to_eliminate()
{
    return l_arguments_to_eliminate;
}



/*********************************************************************************/
/* LOCAL FUNCTIONS: relations between phi and psi variables                      */
/*********************************************************************************/

static Psysteme arrays_same_first_dimensions_sc(int *p_ind_max);
static Psysteme arrays_last_dims_linearization_sc(int dim_min,
						  bool *p_exact_translation_p);

static Psysteme array_translation_sc(bool *p_exact_translation_p)
{
    Psysteme trans_sc;
    int i;

    /* First, search for trivial relation between PHI and PSY variables */
    trans_sc = arrays_same_first_dimensions_sc(&i);

    ifdebug(3)
    {
	pips_debug(3, "same first (%d) dimensions sc:\n", i-1);
	reg_sc_debug(trans_sc);
    }

    /* If we have translated all the dimensions */
    if (i > max(dim_1, dim_2))
    {
	pips_debug(3, "all common dimensions have been translated\n");
	if (statistics_p) common_dimension_stat.all_similar++;

    }
    else /* much more work must be done */
    {
      if (!reference_p || i <= min(dim_1, dim_2))
	{
	  pips_debug(3, "linearization\n");
	    trans_sc = sc_safe_append
		(trans_sc,
		 arrays_last_dims_linearization_sc(i, p_exact_translation_p));
	    if (statistics_p && *p_exact_translation_p)
		linearization_stat.exact++;
	}
	/* for the backward interprocedural propagation only */
	/* if the formal entity is a scalar variable, or if all common dimensions
	 * have already been translated, then we add the equalities
	 * phi_i = i-th index of the reference or phi_i = lower_bound when
	 * the actual reference has no indices.
	 */
	/* vraiment utile? n'est-ce pas fait par arrays_last_dims_linearization_sc
	 * de manie`re plus ge'ne'rale ?? Est-ce que ici je n'oublie aps des cas?
	 */
	else if ((!reference_undefined_p(ref_2)) && (i > dim_1))
	{
	    bool use_ref_p = reference_indices(ref_2) != NIL;

	    pips_debug(3, "the last equations come from the actual array %s.\n",
		       use_ref_p ? "reference" : "declaration");

	    if (statistics_p) remaining_dimension_stat.nb++;

	    for (; i <= dim_2; i++)
	    {
		normalized nind = use_ref_p ?
		    NORMALIZE_EXPRESSION(reference_ith_index(ref_2,i)):
			NORMALIZE_EXPRESSION(dimension_lower(dims_2[i-1]));

		if (normalized_linear_p(nind))
		{
		    entity psi = make_psi_entity(i);
		    Pvecteur v_ind = vect_new((Variable) psi, VALUE_ONE);

		    v_ind = vect_substract(v_ind, normalized_linear(nind));
		    sc_add_egalite(trans_sc, contrainte_make(v_ind));
		}
		else
		{
		    pips_debug(4, "%d-th  %s not linear\n", i,
			       use_ref_p? "index": "lower bound");
		    *p_exact_translation_p = false;
		    if (statistics_p)
			remaining_dimension_stat.non_linear_decl_or_offset++;
		}

	    } /* for */

	    trans_sc->base = BASE_NULLE;
	    sc_creer_base(trans_sc);

	    if (statistics_p && *p_exact_translation_p)
		remaining_dimension_stat.exact++;
	} /* else if */
    } /* else */

    return(trans_sc);
}

/* variables representing the same location in a common are simplified...
 * this should/must exists somewhere?
 * FC 21/06/2000
 */
static void simplify_common_variables(Pcontrainte c)
{
  bool changed;

  do
  {
    Pvecteur v, vp;

    changed = false;
    for (v=c->vecteur; v && !changed; v=v->succ)
    {
      entity var = (entity) var_of(v);
      if (var)
      {
	for (vp=v->succ; vp; vp=vp->succ)
	{
	  entity varp = (entity) var_of(vp);
	  if (varp && entities_may_conflict_p(var, varp))
	  {
	    Value val = val_of(vp);
	    changed = true;
	    vect_add_elem(& c->vecteur, (Variable) varp, value_uminus(val));
	    vect_add_elem(& c->vecteur, (Variable) var, val);
	    break;
	  }
	}
      }
    }

  } while (changed);
}

/* static bool arrays_same_ith_dimension_p(reference array_1_ref,
 *                                            entity array_2,
 *                                            int i)
 * input    : an actual array reference as it appears in a call, the
 *            corresponding formal array (entity), and an array dimension.
 * output   : true if the dimension is identical for both array (see below),
 *            false otherwise.
 * modifies : nothing.
 * comment  :
 *
 *   assumptions :
 *   ~~~~~~~~~~~~~
 *   1- i <= dim(array_1) && i <= dim(array_2) (i is a common dimension)
 *   2- dimensions [1..i-1] are identical
 *
 *   definition :
 *   ~~~~~~~~~~~~
 *   we say that two arrays are identical for dimensions i iff :
 *     1- i is a common dimensions (i <= dim(array_1) && i <= dim(array_2)).
 *     2- the previous dimensions [1..i-1] are identical.
 *     3- the actual reference either has no indices (e.g. A) or the index
 *        of the i-th dimension is equal to the corresponding lower bound.
 *     4- the length of dimension i for both array have the same value.
 *
 */
static bool arrays_same_ith_dimension_p(int i)
{
    bool same_dim = true;
    normalized ndl1 = NORMALIZE_EXPRESSION(dimension_lower(dims_1[i-1]));
    normalized ndu1 = NORMALIZE_EXPRESSION(dimension_upper(dims_1[i-1]));
    normalized ndl2 = NORMALIZE_EXPRESSION(dimension_lower(dims_2[i-1]));
    normalized ndu2 = NORMALIZE_EXPRESSION(dimension_upper(dims_2[i-1]));

    pips_debug(6, "checking dimension %d.\n", i);

    /* FIRST: check the offset with the current dimension */

    /* If there is a reference, we must verify that the offset of this
     * dimension is equal to the lower bound of the declaration.
     */
    if (reference_p)
    {
	normalized nind;
	reference ref = reference_undefined_p(ref_1)? ref_2 : ref_1;

	/* if the reference has no indices, then the offset of this dimension
	 * is equal to the lower bound of the declaration.
	 */
	if (reference_indices(ref) != NIL)
	{
	    normalized ndl = reference_undefined_p(ref_1) ? ndl2: ndl1;

	    nind =  NORMALIZE_EXPRESSION(reference_ith_index(ref, i));

	    if (normalized_linear_p(nind) && normalized_linear_p(ndl))
	    {
		/* nind and ndl are in the same name space */
		same_dim = vect_equal(normalized_linear(nind),
				      normalized_linear(ndl));
		if (statistics_p && !same_dim)
			common_dimension_stat.not_same_decl++;
	    }
	    else
	    {
		same_dim = false;
		if (statistics_p) common_dimension_stat.non_linear_decl++;
	    }
	}

	pips_debug(6, "reference: %ssame lower bound.\n",
		   same_dim? "" : "not ");
    }
    /* If we know the offset, we must verify that it is a multiple of the size
     * of the subarray of dimension i. Else, the dimensions must be considered
     * non-equivalent.
     */
    else
	if(value_notzero_p(offset))
	{
	    static Value dim_cumu_1;
	    static Value dim_cumu_2;

	    if (i == 1)
	    {
		dim_cumu_1 = VALUE_ONE;
		dim_cumu_2 = VALUE_ONE;
	    }

	    if (normalized_linear_p(ndl1) && normalized_linear_p(ndu1) &&
		normalized_linear_p(ndl2) && normalized_linear_p(ndu2))
	    {
		Pvecteur v1 = vect_substract(normalized_linear(ndu1),
					    normalized_linear(ndl1));
		Pvecteur v2 = vect_substract(normalized_linear(ndu2),
					    normalized_linear(ndl2));
		if (vect_constant_p(v1) && vect_constant_p(v2))
		{
		    Value p = value_plus(val_of(v1), VALUE_ONE);
		    value_product(dim_cumu_1, p);
		    p = value_plus(val_of(v2), VALUE_ONE);
		    value_product(dim_cumu_2, p);
		    if (i==1)
		    {
			value_product(dim_cumu_1, size_elt_1);
			value_product(dim_cumu_2, size_elt_2);
		    }
		    same_dim = value_zero_p(value_mod(offset,dim_cumu_1)) &&
			value_zero_p(value_mod(offset,dim_cumu_2));

		    if (statistics_p && !same_dim)
			common_dimension_stat.not_same_decl++;
		}
		else
		{
		    same_dim = false;
		    if (statistics_p) common_dimension_stat.not_same_decl++;
		}
		vect_rm(v1);
		vect_rm(v2);
	    }
	    else
	    {
		same_dim = false;
		if (statistics_p) common_dimension_stat.non_linear_decl++;
	    }
	    pips_debug(6, "offset: %ssame dimension\n", same_dim? "" : "not ");
	}

    /* SECOND: check the size of the current dimension if necessary.
     * It is not necessary if the offset is equal to 0, and it is the last
     * dimension.
     */

    if ( same_dim && !((i==dim_1) && (i==dim_2)) )
    {
      pips_debug(9, "checking size\n");
	if (normalized_linear_p(ndl1) && normalized_linear_p(ndl2) &&
	    normalized_linear_p(ndu1) && normalized_linear_p(ndu2))
	{
	    Pvecteur v1 = vect_substract(normalized_linear(ndu1),
					 normalized_linear(ndl1));
	    Pvecteur v2 = vect_substract(normalized_linear(ndu2),
					 normalized_linear(ndl2));
	    Pcontrainte c;

	    if (i == 1)
	    {
		vect_add_elem(&v1, TCST, VALUE_ONE);
		v1 = vect_multiply(v1, size_elt_1);
		vect_add_elem(&v2, TCST, VALUE_ONE);
		v2 = vect_multiply(v2, size_elt_2);
	    }

	    c = contrainte_make(vect_substract(v1,v2));

	    if (CONTRAINTE_NULLE_P(c))
	      same_dim = true;
	    else {

	      /* if dimensions are declared with common variables,
	       * several entities represent the same value/location.
	       * this must be dealt with somewhere!
	       * maybe this should be handled in some other place?
	       */
	      simplify_common_variables(c);

	      ifdebug(9) {
		pips_debug(9, "linear case: ");
		egalite_debug(c);
	      }

	      same_dim = eq_redund_with_sc_p(get_translation_context_sc(), c);
	    }

	    if (statistics_p && !same_dim)
	      common_dimension_stat.not_same_decl++;
	    vect_rm(v1);
	    vect_rm(v2);
	    contrainte_free(c);
	}
	else
	{
	    same_dim = false;
	    if (statistics_p) common_dimension_stat.non_linear_decl++;
	}
	pips_debug(6, "size: %ssame %d dimension\n", same_dim? "": "not ", i);
    }

    pips_debug(6, "%ssame %d dimension\n", same_dim? "" : "not ", i);
    return same_dim;
}



/* static Psysteme arrays_same_first_dimensions_sc(int *p_ind_max)
 * input    : a non-initialized integer which will represent the rank of the first
 *            non identical dimension of the arrays array_1 and array_2.
 * output   : a system of constraints representing the relations between the first
 *            identical dimensions (see below) of the two arrays, if they are affine.
 * modifies : *p_ind_max. after the call, the value of *p_ind_max is equal to the
 *            index of the first dimension which could not be translated
 *            *p_ind_max is at least equal to 1.
 * comment  :
 *
 *   definition :
 *   ~~~~~~~~~~~~
 *   we say that two arrays are identical for the dimension i iff :
 *     1- i is a common dimensions (i <= dim(array_1) && i <= dim(array_2)).
 *     2- the previous dimensions [1..i-1] are identical.
 *     3- the actual reference either has no indices (e.g. A) or the index
 *        of the i-th dimension is equal to the corresponding lower bound;
 *        or the offset is a multiple of the curretn accumulated sizes.
 *     4- the length of dimension i for both array have the same value.
 *
 *  the elements of array_1 are represented using PHI variables, while those of
 *  array_2 are represented using PSI variables.
 *
 *  algorithm :
 *  ~~~~~~~~~~~
 *
 * dim = 1
 * while (dim <= ndim(array_1) and dim <= ndim(array_2) and
 *                                 dim(array_1) ~ dim (array_2))
 *   sc = sc inter
 *         {PHI_dim - lower_bound(array_2, dim) = PSI_dim - lower_bound(array_1,dim)}
 * endwhile
 *
 */
static Psysteme arrays_same_first_dimensions_sc(int *p_ind_max)
{
    int common_dim = min(dim_1, dim_2); /* max number of common dimensions */
    bool same_shape_p;
    int i; /* dimension index */
    Psysteme trans_sc = sc_new();

    if (statistics_p) common_dimension_stat.nb_calls++;

    i = 1;
    same_shape_p = true;
    while ((i <= common_dim) && same_shape_p)
    {
	/* Is the current dimension identical for both arrays? */
	same_shape_p = arrays_same_ith_dimension_p(i);
	if (same_shape_p)
	{
	    normalized ndl_1 = NORMALIZE_EXPRESSION(dimension_lower(dims_1[i-1]));
	    normalized ndl_2 = NORMALIZE_EXPRESSION(dimension_lower(dims_2[i-1]));

	    if (normalized_linear_p(ndl_2) && normalized_linear_p(ndl_1))
	    {
		/* we add the equality phi - ndl_1 = psi - ndl_2 if i != 1
		 * or s1(phi - ndl_1) + beta_1 = s2(psi - ndl_2) + beta_2
		 * if i == 1
		 */

		Pvecteur v_phi = vect_new((char *) make_phi_entity(i), VALUE_ONE);
		Pvecteur v_psi = vect_new((char *) make_psi_entity(i), VALUE_ONE);
		Pvecteur v_phi_psi;

		v_phi = vect_cl_ofl_ctrl(v_phi, VALUE_MONE, normalized_linear(ndl_1),
					 NO_OFL_CTRL);
		v_psi = vect_cl_ofl_ctrl(v_psi, VALUE_MONE, normalized_linear(ndl_2),
					 NO_OFL_CTRL);

		if (i == 1 && value_ne(size_elt_1,size_elt_2))
		{
		    /* vect_add_elem(&v_phi, TCST, 1); */
		    v_phi = vect_multiply(v_phi, size_elt_1);

		    if (value_notone_p(size_elt_1))
		    {
			Pvecteur pv_beta;
			entity beta = make_beta_entity(1);

			vect_add_elem(&v_phi, (Variable) beta, VALUE_ONE);

			/* add 0 <= beta_1 <= size_elt_1 - 1 to trans_sc */
			pv_beta = vect_make(VECTEUR_NUL,
					    (Variable) beta, VALUE_MONE,
					    TCST, VALUE_ZERO);
			sc_add_inegalite(trans_sc, contrainte_make(pv_beta));
			pv_beta = vect_make(
			    VECTEUR_NUL, (Variable) beta, VALUE_ONE,
			    TCST, value_minus(VALUE_ONE,size_elt_1));
			sc_add_inegalite(trans_sc, contrainte_make(pv_beta));
		    }

		    /* vect_add_elem(&v_psi, TCST, 1); */
		    v_psi = vect_multiply(v_psi, size_elt_2);

		    if (value_notone_p(size_elt_2))
		    {
			Pvecteur pv_beta;
			entity beta = make_beta_entity(2);

			vect_add_elem(&v_psi, (Variable) beta, VALUE_ONE);

			/* add 0 <= beta_2 <= size_elt_2 - 1 to trans_sc */
			pv_beta = vect_make(VECTEUR_NUL,
					    (Variable) beta, VALUE_MONE,
					    TCST, VALUE_ZERO);
			sc_add_inegalite(trans_sc, contrainte_make(pv_beta));
			pv_beta = vect_make(
			    VECTEUR_NUL,
			    (Variable) beta, VALUE_ONE,
			    TCST, value_minus(VALUE_ONE,size_elt_2));
			sc_add_inegalite(trans_sc, contrainte_make(pv_beta));
		    }
		}

		v_phi_psi = vect_substract(v_phi, v_psi);
		vect_rm(v_phi);
		vect_rm(v_psi);

		ifdebug(8)
		{
		    pips_debug(8, "dimension %d, translation vector : \n", i);
		    reg_v_debug(v_phi_psi);
		}

		sc_add_egalite(trans_sc, contrainte_make(v_phi_psi));
		i += 1;
	    }
	    else
	    {
		same_shape_p = false;
		if (statistics_p) common_dimension_stat.non_linear_decl++;
	    }
	}
    }

    *p_ind_max = i;

    trans_sc->base = BASE_NULLE;
    sc_creer_base(trans_sc);

    return(trans_sc);
}


/* static Pvecteur reference_last_indices_offset(reference ref, int ind)
 * input    : an array reference of at least ind dimensions, and an integer
 *            representing one of the array dimension.
 * output   : a vector, representing the offset of last dimensions of
 *            ref, beginning at dimension ind, when it is linear.
 *               offset = (ind_ind - lb_ind) +
 *                        (ind_{ind+1} - lb_{ind+1})* dim_ind  +
 *                        ... +
 *                        (ind_{ind_max} - lb_{ind_max})* dim_ind *... * dim_{ind_max-1}
 *            returns VECTEUR_UNDEFINED, when the offset is not linear.
 * modifies : nothing
 * comment  :
 */
static Pvecteur
reference_last_indices_offset(reference ref, int ind, bool *p_linear_p)
{
    int dim = (ref == ref_1)? dim_1 : dim_2;
    dimension *dims = (ref == ref_1) ? dims_1 : dims_2;
    Value size_elt = (ref == ref_1) ? value_uminus(size_elt_1) : size_elt_2;

    normalized ni, nlb, nub;
    Pvecteur pv_ind = (Pvecteur) NULL;
    Pvecteur pv_ind_plus_1 = VECTEUR_UNDEFINED;

    pips_assert("feasible index", 0 < ind && ind <=dim);

    /* offset of the last dimensions beginning at ind + 1 */
    if (ind < dim)
    {
	pips_debug(8, "ind = %d, dim = %d\n", ind, dim);
	pv_ind_plus_1 = reference_last_indices_offset(ref, ind + 1, p_linear_p);
	if (!*p_linear_p) return(VECTEUR_UNDEFINED);
    }

    pips_debug(8, "ind = %d, dim = %d", ind, dim);

    ni = NORMALIZE_EXPRESSION(reference_ith_index(ref, ind));
    nlb = NORMALIZE_EXPRESSION(dimension_lower(dims[ind-1]));

    if (normalized_linear_p(ni) && normalized_linear_p(nlb))
    {
	Pvecteur vi = normalized_linear(ni);
	Pvecteur vb = normalized_linear(nlb);

	ifdebug(8)
	{
	    pips_debug(8, "current index :\n"); reg_v_debug(vi);
	    pips_debug(8, "current lower bound :\n"); reg_v_debug(vb);
	}

	/* we must multiply the offset of the indices beginning at ind + 1
	 * by the length of the current dimension.
	 */
	if ((ind < dim) && !VECTEUR_UNDEFINED_P(pv_ind_plus_1))
	{
	    nub = NORMALIZE_EXPRESSION(dimension_upper(dims[ind-1]));

	    if (normalized_linear_p(nub))
	    {
		Pvecteur vd = vect_substract(normalized_linear(nub), vb);
		vect_add_elem(&vd, TCST, VALUE_ONE);

		ifdebug(8)
		{
		    pips_debug(8, "lenght of current dimension :\n");
		    reg_v_debug(vd);
		}

		pv_ind = vect_product(&pv_ind_plus_1, &vd);
		if (VECTEUR_UNDEFINED_P(pv_ind))
		{
		    *p_linear_p = false;
		    if (statistics_p) linearization_stat.non_linear_system++;
		}

	    }
	    else
	    {
		*p_linear_p = false;
		if (statistics_p) linearization_stat.non_linear_decl++;
	    }

	    if (*p_linear_p)
		pv_ind = vect_cl_ofl_ctrl(pv_ind, VALUE_ONE,
					  vect_substract(vi, vb),
					  NO_OFL_CTRL);
	}
	else pv_ind = vect_substract(vi, vb);

    } /* if (normalized_linear_p(ni) && normalized_linear_p(nlb)) */

    else
    {
	*p_linear_p = false;
        if (statistics_p) linearization_stat.non_linear_decl++;
    }

    if (! *p_linear_p)
	pv_ind = VECTEUR_UNDEFINED;
    else
	if (ind == 1) pv_ind = vect_multiply(pv_ind, size_elt);

    if (!VECTEUR_UNDEFINED_P(pv_ind_plus_1)) vect_rm(pv_ind_plus_1);

    ifdebug(8)
    {
	pips_debug(8, "result:\n"); reg_v_debug(pv_ind);
    }

    return(pv_ind);
}

/* on entry, offset != 0
 * recursive build of a pvecteur.
 */
static Pvecteur global_to_last_dims_offset(int dim_min, bool *p_linear_p)
{
    Pvecteur pv_offset = VECTEUR_UNDEFINED;

    pips_assert("feasible index", 0 < dim_min && dim_min <=dim_1);

    if (dim_min == 1)
      return vect_make(VECTEUR_NUL, TCST, offset);

    /* here to avoid assert in case of a scalar entity */

    pv_offset = global_to_last_dims_offset(dim_min - 1, p_linear_p);

    if (*p_linear_p)
    {
	normalized nlb, nub;
	Pvecteur vlb, vub;

	nlb = NORMALIZE_EXPRESSION(dimension_lower(dims_1[dim_min-1]));
	nub = NORMALIZE_EXPRESSION(dimension_upper(dims_1[dim_min-1]));

	pips_assert("linear expressions",
		    normalized_linear_p(nlb) && normalized_linear_p(nub));

	vlb = normalized_linear(nlb);
	vub = normalized_linear(nub);

	if (vect_constant_p(vlb) && vect_constant_p(vub))
	{
	  Value dim_size =
	    value_plus(VALUE_ONE, value_minus(vub->val,vlb->val));
	  pv_offset = vect_div(pv_offset, dim_size);
	}
	else
	{
	  *p_linear_p = false;
	  if (!VECTEUR_UNDEFINED_P(pv_offset))
	  {
	    vect_rm(pv_offset);
	    pv_offset = VECTEUR_UNDEFINED;
	  }
	  if (statistics_p) linearization_stat.non_linear_system++;
	}
    }
    return pv_offset;
}

static Pvecteur last_dims_offset(int dim_min, bool *p_linear_p)
{
    Pvecteur pv_offset = VECTEUR_NUL;

    *p_linear_p = true;

    if(reference_p)
    {
      reference ref = reference_undefined_p(ref_1)? ref_2: ref_1;

      if (reference_indices(ref) != NIL)
	pv_offset = reference_last_indices_offset(ref, dim_min, p_linear_p);
    }
    else
    {
      if (value_notzero_p(offset))
      {
	/* FC hack around a bug:
	   this function deal with dims_1[], although dim_min may not be a
	   valid index... It is not obvious to guess what is actually expected.
	 */
	if (dim_min > dim_1) {
	  pips_user_warning("tmp hack, fix me please\n!");
	  *p_linear_p = false;
	  return VECTEUR_NUL;
	}

	pv_offset = global_to_last_dims_offset(dim_min, p_linear_p);
      }
    }
    return pv_offset;
}

/* static Pvecteur array_partial_subscript_value(entity array, int dim_min, dim_max
 *                                               entity (*make_region_entity)(int))
 * input    : an array entity, a dimension, and a function which returns
 *            an entity representing a particular dimension given its rank.
 * output   : a Pvecteur representing the 'subscript value' of the array
 *            consisting of the dimensions higher (>=) than dim_min of the
 *            initial array.
 * modifies : nothing
 * comment  : the form of the subsrctip value is :
 *
 *            [PHI_(dim_min) - lb_(dim_min)]
 *          + [PHI_(dim_min + 1) - lb_(dim_min + 1)] * [ub_(dim_min) - lb_(dim_min)]
 *          + ...
 *          + [PHI_(dim_max + 1) - lb_(dim_max + 1)] * [ub_(dim_min) - lb_(dim_min)]
 *                                                   * ...
 *                                                   * [ub_(dim_max - 1) - lb_(dim_max - 1)]
 *
 *            where lb stands for lower bound, ub for uper bound and dim_max for
 *            the number of dimensions of the array.
 *
 * CAUTION : dim_min must be less or equal than dim_max.
 */
static Pvecteur array_partial_subscript_value(entity array, dimension *dims,
					      int dim_array, int dim_min, int dim_max,
					      entity (*make_region_entity)(int))
{
    Pvecteur v_dim_min = VECTEUR_UNDEFINED;
    Pvecteur v_dim_min_plus_1 = VECTEUR_UNDEFINED;
    dimension d;
    normalized ndl, ndu;

    pips_debug(8, "dim_min = %d, dim_max = %d, dim_array = %d \n",
	       dim_min, dim_max, dim_array);
    pips_assert("dim_max must be less than the dimension of the array\n",
		dim_max <= dim_array);
    pips_assert("dim_min must be less than dim_max\n", dim_min <= dim_max);

    if (dim_min < dim_max)
    {
	v_dim_min_plus_1 = array_partial_subscript_value(array, dims, dim_array,
							 dim_min + 1, dim_max,
							 make_region_entity);
	ifdebug(8)
	{
	    pips_debug(8, "v_dim_min_plus_1 : \n");
	    reg_v_debug(v_dim_min_plus_1);
	}
	if (VECTEUR_UNDEFINED_P(v_dim_min_plus_1))
	    return(VECTEUR_UNDEFINED);
    }

    d = dims[dim_min-1];
    ndl = NORMALIZE_EXPRESSION(dimension_lower(d));
    ndu = NORMALIZE_EXPRESSION(dimension_upper(d));

    if (normalized_linear_p(ndl))
    {
	/* we must multiply the subscript_value of the dimensions beginning
	 * at ind + 1 by the length of the current dimension
	 */
	if (dim_min < dim_max)
	{
	    if (normalized_linear_p(ndu))
	    {
		Pvecteur v_dim_min_length =
		    vect_substract(normalized_linear(ndu),
				   normalized_linear(ndl));
		vect_add_elem(&v_dim_min_length, TCST, VALUE_ONE);

		ifdebug(8)
		{
		    pips_debug(8, "length of current dimension: \n");
		    reg_v_debug(v_dim_min_length);
		}
		v_dim_min_plus_1 = vect_product(&v_dim_min_plus_1,
						&v_dim_min_length);

		if (VECTEUR_UNDEFINED_P(v_dim_min_plus_1))
		{
		    pips_debug(8, "non linear multiplication\n");
		    v_dim_min = VECTEUR_UNDEFINED;
		    if (statistics_p) linearization_stat.non_linear_system++;
		}
	    }
	    else
	    {
		pips_debug(8, "uper bound not linear \n");
		vect_rm(v_dim_min_plus_1);
		v_dim_min_plus_1 = VECTEUR_UNDEFINED;
		if (statistics_p) linearization_stat.non_linear_decl++;
	    }
	}

	if (!VECTEUR_UNDEFINED_P(v_dim_min_plus_1) || (dim_min == dim_max))
	{
	    v_dim_min = vect_new((Variable) make_region_entity(dim_min), VALUE_ONE);
	    v_dim_min = vect_cl_ofl_ctrl(v_dim_min, VALUE_MONE,
					 normalized_linear(ndl),
					 NO_OFL_CTRL);
	      /* works even if v_dim_min_plus_1 == VECTEUR_UNDEFINED  */
	      /* vect_add(v_dim_min_plus_1, v_dim_min) would not work */
	    v_dim_min = vect_cl_ofl_ctrl(v_dim_min, VALUE_ONE,
					 v_dim_min_plus_1, NO_OFL_CTRL);
	}

    } /* if (normalized_linear_p(ndl)) */
    else
    {
	pips_debug(8, "lower bound not linear : \n");
	v_dim_min = VECTEUR_UNDEFINED;
	if (statistics_p) linearization_stat.non_linear_decl++;
    }

    if (!VECTEUR_UNDEFINED_P(v_dim_min_plus_1)) vect_rm(v_dim_min_plus_1);

    ifdebug(8)
    {
	pips_debug(8, "result: \n");
	reg_v_debug(v_dim_min);
    }

    return(v_dim_min);
}


/* static Psysteme arrays_last_dims_linearization_sc( int dim_min,
 *                                                    bool *p_exact_translation_p)
 * input    : an array reference, an array entity, an integer corresponding to an
 *            array dimension, and a pointer to a boolean.
 * output   : the translation systeme from real_ref (PSI variables) to func_ent
 *            (PHI variables) for the dimensions uper or equal to dim_min; it
 *            is based on the linearization equation, and the formal_real_sc is
 *            added. *exact_translation_p is set to true if the translation is exact,
 *            false otherwise
 * modifies : nothing.
 * comment  : for the moment, only affine linearization equations are handled.
 */
static Psysteme arrays_last_dims_linearization_sc(int dim_min,
						  bool *p_exact_translation_p)
{
    Psysteme trans_sc = sc_new();
    Pvecteur pv_1 = VECTEUR_UNDEFINED, pv_2 = VECTEUR_UNDEFINED;
    Pvecteur pv_offset = VECTEUR_UNDEFINED;

    pips_debug(8, " dim_min = %d, dim_1 = %d, dim_2 = %d \n",
	       dim_min, dim_1, dim_2);
    /* pips_assert("dim_min must be less that the dimensions of both arrays.",
		(dim_min <= dim_1) && ( dim_min <= dim_2)); */

    if (statistics_p) linearization_stat.nb_calls++;

    *p_exact_translation_p = true;

    /* subscript_value of first entity */
    if (dim_min <= dim_1)
    {
	pv_1 = array_partial_subscript_value(array_1, dims_1, dim_1, dim_min, dim_1,
					     make_phi_entity);
	ifdebug(6){ pips_debug(6, "array_1: \n"); reg_v_debug(pv_1); }
	*p_exact_translation_p = !VECTEUR_UNDEFINED_P(pv_1);
    }

    if (*p_exact_translation_p)
    {
	/* subscript_value of second entity */

	if (dim_min <= dim_2)
	{
	    pv_2 = array_partial_subscript_value(array_2, dims_2, dim_2, dim_min,
						 dim_2, make_psi_entity);
	    ifdebug(6){ pips_debug(6, "array_2: \n"); reg_v_debug(pv_2); }
	    *p_exact_translation_p = !VECTEUR_UNDEFINED_P(pv_2);
	}

	if (*p_exact_translation_p)
	{
	    bool linear_offset_p = true;

	    pv_offset = last_dims_offset(dim_min, &linear_offset_p);

	    ifdebug(6) { pips_debug(6, "offset: \n"); reg_v_debug(pv_offset);}

	    *p_exact_translation_p = linear_offset_p;

	}
    }

    if (*p_exact_translation_p)
    {
	if ((dim_min == 1) && !VECTEUR_UNDEFINED_P(pv_1) &&
	    value_notone_p(size_elt_1))
	{
	    Pvecteur pv_beta;
	    entity beta = make_beta_entity(1);

	    pv_1 = vect_multiply(pv_1, size_elt_1);
	    vect_add_elem(&pv_1, (Variable) beta, VALUE_ONE);

	    /* add 0 <= beta_1 <= size_elt_1 - 1 to trans_sc */
	    pv_beta = vect_make(VECTEUR_NUL, (Variable) beta, VALUE_MONE,
				TCST, VALUE_ZERO);
	    sc_add_inegalite(trans_sc, contrainte_make(pv_beta));
	    pv_beta = vect_make(VECTEUR_NUL, (Variable) beta, VALUE_ONE,
				TCST, value_minus(VALUE_ONE,size_elt_1));
	    sc_add_inegalite(trans_sc, contrainte_make(pv_beta));
	}

	if ((dim_min == 1) && !VECTEUR_UNDEFINED_P(pv_2) &&
	    value_notone_p(size_elt_2))
	{
	    Pvecteur pv_beta;
	    entity beta = make_beta_entity(2);

	    pv_2 = vect_multiply(pv_2, size_elt_2);
	    vect_add_elem(&pv_2, (Variable) beta, VALUE_ONE);

	    /* add 0 <= beta_1 <= size_elt_1 - 1 to trans_sc */
	    pv_beta = vect_make(VECTEUR_NUL, (Variable) beta, VALUE_MONE,
				TCST, VALUE_ZERO);
	    sc_add_inegalite(trans_sc, contrainte_make(pv_beta));
	    pv_beta = vect_make(VECTEUR_NUL, (Variable) beta, VALUE_ONE,
				TCST, value_minus(VALUE_ONE,size_elt_2));
	    sc_add_inegalite(trans_sc, contrainte_make(pv_beta));
	}

	pv_2 = vect_cl_ofl_ctrl(pv_2, VALUE_MONE, pv_offset, NO_OFL_CTRL);
	sc_add_egalite(trans_sc, contrainte_make(vect_substract(pv_1, pv_2)));
	/* trans_sc = sc_safe_normalize(trans_sc); */
	trans_sc->base = BASE_NULLE;
	sc_creer_base(trans_sc);
    }

    if(!VECTEUR_UNDEFINED_P(pv_1)) vect_rm(pv_1);
    if(!VECTEUR_UNDEFINED_P(pv_2)) vect_rm(pv_2);
    if(!VECTEUR_UNDEFINED_P(pv_offset)) vect_rm(pv_offset);

    return(trans_sc);
}

/*********************************************************************************/
/* LOCAL FUNCTIONS: translation of the region predicate into the target function */
/*                  name space                                                   */
/*********************************************************************************/



/* Try to convert an value on a non-local variable into an value
 * on a local variable using a guessed name (instead of a location
 * identity: M and N declared as COMMON/FOO/M and COMMON/FOO/N
 * are not identified as a unique variable/location).
 *
 * Mo more true: It might also fail to translate variable C:M into A:M if C is
 * indirectly called from A thru B and if M is not defined in B.
 *
 * This routine is not too safe. It accepts non-translatable variable
 * as input and does not refuse them, most of the time.
 */
static void region_translate_global_value(module, reg, val)
entity module;
region reg;
entity val;
{
    storage store = storage_undefined;
    ram r = ram_undefined;
    entity rf = entity_undefined;
    entity section = entity_undefined;

    ifdebug(8)
    {
	pips_debug(8, "begin v = %s and reg =\n", entity_name(val));
	print_region(reg);
    }

    if(val == NULL)
    {
	pips_internal_error("Trying to translate TCST");
	return;
    }

    if(value_entity_p(val))
    {
	/* FI: to be modified to account for global values that have a name
	 * but that should nevertheless be translated on their canonical
	 * representant; this occurs for non-visible global variables
	 */
	/* FI: to be completed later... 3 December 1993
	    entity var = value_to_variable(v);
	    debug(8, "translate_global_value", "%s is translated into %s\n",
		  entity_name(v), entity_name(e));
	    transformer_value_substitute(tf, val, e);
	 */

	pips_debug(8, "No need to translate %s\n",entity_name(val));
	return;
    }

    pips_debug(8, "Trying to translate %s\n", entity_name(val));

    store = entity_storage(val);
    if(!storage_ram_p(store))
    {
	if(storage_rom_p(store))
	{
	  pips_debug(8, "%s is not translatable: store tag %d\n",
		     entity_name(val), storage_tag(store));
	  /* Should it be projected? No, this should occur later for xxx#init
	   * variables when the xxx is translated. Or before if xxx has been
	   * translated
	   */
	  return;
	}
	else if(storage_formal_p(store))
	{
	  pips_debug(8, "formal %s is not translatable\n", entity_name(val));
	  return;
	}
	/* FC 2001/03/22: some obscure bug workaround...
	 * The 'return' of a function could be translated into the
	 * assigned variable? well, it should be added anyway somewhere.
	 */
        else if (storage_return_p(store))
	{
	  pips_debug(8, "return %s does not need to be translated.\n",
		     entity_name(val));
	  return;
	}
	else
	{
	  pips_internal_error("%s is not translatable: store tag %d",
				entity_name(val), storage_tag(store));
	}
    }

    r = storage_ram(store);
    rf = ram_function(r);
    section = ram_section(r);

    if(rf != module && top_level_entity_p(section))
    {
	/* must be a common; dynamic and static area must have been
	 * filtered out before */
	entity e;
	entity v_init = entity_undefined;
	Psysteme sc = SC_UNDEFINED;
	Pbase b = BASE_UNDEFINED;

	/* try to find an equivalent entity by its name
	   (whereas we should use locations) */
	/*
	e = FindEntity(module_local_name(m),
				  entity_local_name(v));
	e = value_alias(value_to_variable(v));
	*/
	e = value_alias(val);
	if((e == entity_undefined) || !same_scalar_location_p(val, e))
	{
	    list l = CONS(ENTITY, val, NIL);
	    /* no equivalent name found, get rid of val */
	    ifdebug(8)
	    {
		if (e == entity_undefined)
		{
		    pips_debug(8, "No equivalent for %s in %s: project %s\n",
			       entity_name(val), entity_name(module),
			       entity_name(val));
		}
		else
		{
		    pips_debug(8,
			       "No equivalent location for %s and %s: project %s\n",
			       entity_name(val), entity_name(e), entity_name(val));
		}
	    }
	    if (must_regions_p())
		region_exact_projection_along_parameters(reg, l);
	    else
		region_non_exact_projection_along_parameters(reg, l);
	    gen_free_list(l);
	    sc = region_system(reg);
	    base_rm(sc->base);
	    sc->base = BASE_NULLE;
	    sc_creer_base(sc);
	    return;
	}

	sc = region_system(reg);
	b = sc_base(sc);
	if(base_contains_variable_p(b, (Variable) e) )
	{
	    /* e has already been introduced and val eliminated; this happens
	     * when a COMMON variable is also passed as real argument */
	    pips_debug(8, "%s has already been translated into %s\n",
		       entity_name(val), entity_name(e));
	    /*  sc_base_remove_variable(sc,(Variable) val);*/
	    base_rm(sc->base);
	    sc->base = BASE_NULLE;
	    sc_creer_base(sc);
	}
	else
	{
	    pips_debug(8, "%s is translated into %s\n",
		       entity_name(val), entity_name(e));
	    region_value_substitute(reg, val, e);
	    sc = region_system(reg);
	    base_rm(sc->base);
	    sc->base = BASE_NULLE;
	    sc_creer_base(sc);
	}

	v_init = (entity) gen_find_tabulated
	    (concatenate(entity_name(val), OLD_VALUE_SUFFIX, (char *) NULL),
	     entity_domain);

	if(v_init != entity_undefined)
	{
	    entity e_init = (entity) gen_find_tabulated
		(concatenate(entity_name(e), OLD_VALUE_SUFFIX, (char *) NULL),
		 entity_domain);

	    if(e_init == entity_undefined)
	    {
		/* this cannot happen when the summary transformer of a called
		 * procedure is translated because the write effect in the callee
		 * that is implied by v_init existence must have been passed
		 * upwards and must have led to the creation of e_init
		 */
		/* this should not happen when a caller precondition at a call site
		 * is transformed into a piece of a summary precondition for
		 * the callee because v_init becomes meaningless; at the callee's
		 * entry point, by definition, e == e_init; v_init should have been
		 * projected before
		 */
		Psysteme r = region_system(reg);

		if(base_contains_variable_p(sc_base(r), (Variable) v_init))
		    pips_internal_error("Cannot find value %s",
			       strdup(
				      concatenate(
						  module_local_name(module),
						  MODULE_SEP_STRING,
						  entity_local_name(val),
						  OLD_VALUE_SUFFIX,
						  (char *) NULL)));
		else
		{
		    /* forget e_init: there is no v_init in tf */
		    pips_debug(8, "%s is not used in tf\n", entity_name(v_init));
		}
	    }
	    else
	    {
		pips_debug(8, "%s is translated into %s\n",
			   entity_name(val), entity_name(e));
		region_value_substitute(reg, v_init, e_init);
		sc = region_system(reg);
		base_rm(sc->base);
		sc->base = BASE_NULLE;
		sc_creer_base(sc);
	    }
	}
	/* else : there is no v_init to worry about; v is not changed in
	   the caller (or its subtree of callees) */
    }
    /* else : this value does not need to be translated */
}



/* static void region_translate_global_values(module, reg)
 * input    : a region, reg, and a module.
 * output   : nothing.
 * modifies : the global values are translated into global values for
 *            the frame of module.
 * comment  : same as translate_global_values, but deals with regions
 *            (projection is not the same)
 */
static void region_translate_global_values(module, reg)
entity module;
region reg;
{
    Psysteme s = region_system(reg);
    /* a copy of sc_base(s) is needed because region_translate_global_value()
       modifies it at the same time */
    Pbase b = (Pbase) vect_dup(sc_base(s));
    Pbase bv;

    pips_debug(8, "initial region: \n%s\n", region_to_string(reg));


    for(bv = b; bv != NULL; bv = bv->succ)
    {
	region_translate_global_value(module, reg, (entity) vecteur_var(bv));
    }

    base_rm(b);
}



/* static void region_translation_of_predicate(region reg, entity to_func,
 *                                             list real_args)
 * input    : a region, a function towards which the region must be translated,
 *            and the real arguments of the current call site.
 * output   : nothing
 * modifies : the region reg. The integer scalar variables that appear in the
 *            predicate of the region are translated into their corresponding
 *            counterparts in the target function to_func, using the affine
 *            relations between real and formal parameters when they exist, and
 *            between common variables in the two functions. When such relations
 *            don't exist, the variables are simply eliminated. The approximation
 *            of the region can become may if an elimination is not exact
 *            (see report E/185).
 * comment  : Also eliminates global values (commons).
 */
static void region_translation_of_predicate(region reg, entity to_func)
{
  convex_region_descriptor_translation(reg);
    region_translate_global_values(to_func, reg);
    debug_region_consistency(reg);

    ifdebug(8)
      {
	pips_debug(8, "region after translation of globals: \n");
	print_region(reg);
      }
}

void convex_region_descriptor_translation(effect eff)
{
  /* FI: regions were are not store regions do not need translation */
  if(store_effect_p(eff)) {
    ifdebug(8)
      {
	pips_debug(8, "region before translation: \n");
	print_region(eff);
      }

    if (!sc_rn_p(region_system(eff)))
      {
	/* we add the system representing the association between
	 * actual and formal parameters to the region */
	region_sc_append_and_normalize(eff, get_translation_context_sc(),2);

	/* then, we eliminate all the scalar variables that appear in the formal
	 * parameters */
	ifdebug(8)
	  {
	    pips_debug(8, "variables to eliminate: \n");
	    print_arguments(get_arguments_to_eliminate());
	  }
	if (must_regions_p())
	  region_exact_projection_along_parameters
	    (eff, get_arguments_to_eliminate());
	else
	  region_non_exact_projection_along_parameters
	    (eff, get_arguments_to_eliminate());


	debug_region_consistency(eff);
      }

    ifdebug(8)
      {
	pips_debug(8, "region after translation of arguments: \n");
	print_region(eff);
      }
  }
}



/************************************************** PATH TRANSLATION */

/** @brief translates a convex memory access path reference from given indices
           using an address_of memory access path reference

    This function is used when we want to translate a cell or an effect on a[i][j][k] as input_ref,
    knowing that a[i] = &address_of_ref. In this case the list of remaning_input_indices is [j][k].

    @param input_ref is the input convex cell reference
    @param input_desc is the descriptor describing the input reference
    @param input_remaining_indices is the list of indices from input_ref which have to be translated.

    @param address_of_ref is the convex cell reference giving the output base memory access path.
    @param address_of_desc is the descriptor describing address_of_ref.

    @param output_ref is a pointer on the resulting convex reference
    @param output_desc is a pointer on teh resulting descriptor describing output_ref.
    @param exact_p is a pointer on a bool which is set to true if the translation is exact, false otherwise.

    input_remaining_indices does not need to be a copy of a part of the indices of input_ref, because it is not modified.
    In a first version of this function, it was replaced by a integer giving the rank of the beginning of this list
    in the input_ref indices list. However, in generic_eval_cell_with_points_to,
    convex_cell_reference_with_address_of_cell_reference_translation may be called
    several times with the same input_ref and rank, so it was more efficient to pass the input_remaning_indices list
    directly as an argument.

 */
void convex_cell_reference_with_address_of_cell_reference_translation
(reference input_ref, descriptor input_desc,
 reference address_of_ref, descriptor address_of_desc,
 int nb_common_indices,
 reference *output_ref, descriptor *output_desc,
 bool *exact_p)
{
  Psysteme sc_output = SC_UNDEFINED;

  pips_debug(6, "beginning with input_ref = %s and address_of_ref = %s, nb_common_indices = %d\n",
	     entity_name(reference_variable(input_ref)), entity_name(reference_variable(address_of_ref)), nb_common_indices);

  *output_ref = copy_reference(address_of_ref);

  if (entity_all_locations_p(reference_variable(address_of_ref)))
    {
      *output_desc = descriptor_undefined;
      *exact_p = false;
    }
  else
    {
      *exact_p = true;
      int nb_phi_address_of_ref = (int) gen_length(reference_indices(address_of_ref));
      list volatile input_remaining_indices = reference_indices(input_ref);
      int nb_phi_input_ref = (int) gen_length(input_remaining_indices);

      for(int i = 0; i<nb_common_indices; i++, POP(input_remaining_indices));

      if(!ENDP(input_remaining_indices))
	{
	  Psysteme sc_input = descriptor_convex(input_desc);
	  int i;

	  if (nb_phi_address_of_ref !=0)
	    {
	      /* the first index of address_of_ref is added to the last index
		 of input_ref, and the other indexes of address_of_ref are
		 appended to those of input_ref
		 We first check that if the last index of address_of_ref is a field entity
		 then the first non-common index of input_ref is equal to zero. If not we
		 issue a warning and return an anywhere effect.
	      */
	      expression last_address_of_index =
		EXPRESSION(CAR(gen_last(reference_indices(address_of_ref))));

	      if(entity_field_p(expression_variable(last_address_of_index)))
		{
		  Psysteme volatile sc = sc_dup(sc_input);
		  entity phi_first_non_common = make_phi_entity(nb_common_indices+1);
		  Pvecteur v1 = vect_new(TCST, VALUE_ONE);
		  Pvecteur v_phi_first_non_common = vect_new((Variable) phi_first_non_common, VALUE_ONE);
		  bool feasible = true;
		  Pvecteur v = vect_substract(v1, v_phi_first_non_common);
		  vect_rm(v1);
		  vect_rm(v_phi_first_non_common);
		  sc_constraint_add(sc, contrainte_make(v), false);

		  CATCH(overflow_error)
		  {
		    pips_debug(3, "overflow error \n");
		    feasible = true;
		  }
		  TRY
		    {
		      feasible = sc_integer_feasibility_ofl_ctrl(sc,
								 FWD_OFL_CTRL, true);
		      UNCATCH(overflow_error);
		    }
		  if (feasible)
		    {
		      pips_user_warning("potential memory overflow -> returning anywhere\n");
		      free_reference(*output_ref);
		      *output_ref = make_reference(entity_all_locations(), NIL);
		      *output_desc = descriptor_undefined;
		      *exact_p = false;
		    }
		  sc_rm(sc);
		}

	      if(!entity_all_locations_p(reference_variable(*output_ref)))
		{
		  /* preparing the part of sc_input which is to be added to sc_output */
		  /* first eliminate common dimensions in sc_input (well a copy, do not modify the original) */
		  sc_input = sc_dup(sc_input);
		  if (nb_common_indices >0)
		    {
		      list l_phi = phi_entities_list(1,nb_common_indices);
		      FOREACH(ENTITY,phi, l_phi)
			{
			  bool exact_projection;
			  sc_input = cell_reference_sc_exact_projection_along_variable(*output_ref, sc_input, phi, &exact_projection);
			  *exact_p = *exact_p && exact_projection;
			}
		    }

		  /* then rename phi_nb_common_indices+1 into psi1 in sc_input */
		  entity phi_first_non_common = make_phi_entity(nb_common_indices+1);
		  entity psi1 = make_psi_entity(1);
		  sc_input = sc_variable_rename(sc_input, (Variable) phi_first_non_common, (Variable) psi1);
		  ifdebug(8)
		    {
		      pips_debug(8, "sc_input after phi_first_non_common variable renaming: \n");
		      sc_print(sc_input, (get_variable_name_t) entity_local_name);
		    }

		  /* then shift other phi variables if necessary */
		  if (nb_phi_address_of_ref != nb_common_indices + 1)
		    {
		      for(i=nb_phi_input_ref; i>(nb_common_indices+1); i--)
			{
			  entity old_phi = make_phi_entity(i);
			  entity new_phi = make_phi_entity(nb_phi_address_of_ref+i-(nb_common_indices+1));

			  pips_debug(8, "renaming %s into %s\n", entity_name(old_phi), entity_name(new_phi));
			  sc_input = sc_variable_rename(sc_input, (Variable) old_phi, (Variable) new_phi);
			}
		    }

		  /* preparing the system of sc_output from sc_address_of */
		  sc_output = sc_dup(descriptor_convex(address_of_desc));
		  entity phi_max_output = make_phi_entity(nb_phi_address_of_ref);
		  entity rho_max_output = make_rho_entity(nb_phi_address_of_ref);

		  sc_output = sc_variable_rename(sc_output, (Variable) phi_max_output, (Variable) rho_max_output);

		  ifdebug(8)
		    {
		      pips_debug(8, "sc_output after variable renaming: \n");
		      sc_print(sc_output, (get_variable_name_t) entity_local_name);
		    }

		  /* then we append sc_input to sc_output
		   */
		  sc_output = cell_system_sc_append_and_normalize(sc_output, sc_input, true);

		  ifdebug(8)
		    {
		      pips_debug(8, "sc_output after appending sc_input: \n");
		      sc_print(sc_output, (get_variable_name_t) entity_local_name);
		    }

		  /* then we add the constraint phi_max_output = psi1 + rho_max_output
		     and we eliminate psi1 and rho_max_output
		  */
		  Pvecteur v_phi_max_output = vect_new((Variable) phi_max_output, VALUE_ONE);
		  Pvecteur v_psi1 = vect_new((Variable) psi1, VALUE_ONE);
		  Pvecteur v_rho_max_output = vect_new((Variable) rho_max_output, VALUE_ONE);
		  v_phi_max_output = vect_substract(v_phi_max_output, v_psi1);
		  v_phi_max_output = vect_substract(v_phi_max_output, v_rho_max_output);
		  sc_constraint_add(sc_output, contrainte_make(v_phi_max_output), true);

		  bool exact_removal;
		  sc_output = cell_reference_system_remove_psi_variables(*output_ref, sc_output, &exact_removal);
		  *exact_p = *exact_p && exact_removal;
		  sc_output = cell_reference_system_remove_rho_variables(*output_ref, sc_output, &exact_removal);
		  *exact_p = *exact_p && exact_removal;

		  ifdebug(8)
		    {
		      pips_debug(8, "sc_output after appending removing psi and rho variables: \n");
		      sc_print(sc_output, (get_variable_name_t) entity_local_name);
		    }
		  *output_desc = make_descriptor_convex(sc_output);

		  /* finally we must add the additional PHI variables or the field entities
		     to the indices of the output reference */
		  for(i = nb_phi_address_of_ref +1; i<nb_phi_address_of_ref + nb_phi_input_ref - nb_common_indices; i++)
		    {
		      POP(input_remaining_indices);
		      expression input_remaining_indices_exp = EXPRESSION(CAR(input_remaining_indices));
		      if (entity_field_p(expression_variable(input_remaining_indices_exp)))
			reference_indices(*output_ref) = gen_nconc(reference_indices(*output_ref),
								   CONS(EXPRESSION,
									copy_expression(input_remaining_indices_exp),
									NIL));
		      else
			reference_indices(*output_ref) = gen_nconc(reference_indices(*output_ref),
								   CONS(EXPRESSION,
									make_phi_expression(i),
									NIL));
		    }
		  pips_debug(8, "*output_ref after adding phi: %s\n", words_to_string(effect_words_reference(*output_ref)));
		} /* if(!anywhere_effect_p(n_eff))*/

	    } /*  if (nb_phi_address_of_ref !=0) */
	  else
	    {
	      /* here nb_phi_address_of_ref is equal to 0 */
	      /* if it's a scalar, but not a pointer, output_ref is OK */
	      /* if it's a pointer, output_ref is equal to input_ref but for the first
		 non-common dimension, which should be equal to 0 in input_ref (I should check
		 that as in the previous case).
	      */
	      entity output_ent = reference_variable(*output_ref);
	      type bct = entity_basic_concrete_type(output_ent);

	      if (!entity_scalar_p(output_ent) || derived_type_p(bct) || pointer_type_p(bct))
	      {
	      sc_output = sc_dup(sc_input);
	      pips_debug(8, "derived or pointer_type\n");

	      /* preparing the part of sc_input which is to be added to sc_output */
	      /* first eliminate common dimensions in sc_input (well a copy, do not modify the original) */
	      // sc_input = sc_dup(sc_input);
	      if (nb_common_indices >0)
		{
		  list l_phi = phi_entities_list(1,nb_common_indices);
		  FOREACH(ENTITY,phi, l_phi)
		    {
		      bool exact_projection;
		      sc_output = cell_reference_sc_exact_projection_along_variable(*output_ref, sc_output, phi, &exact_projection);
		      *exact_p = *exact_p && exact_projection;
		    }
		}

	      /* first remove the first common phi variable which should be equal to zero */
	      entity phi_first_non_common = make_phi_entity(nb_common_indices+1);
	      bool exact_projection;
	      sc_output = cell_reference_sc_exact_projection_along_variable(*output_ref, sc_output, phi_first_non_common, &exact_projection);
	      *exact_p = *exact_p && exact_projection;

	      /* then rename all the phi variables in reverse order */
	      for(i=nb_common_indices+2; i<=nb_phi_input_ref; i++)
		{
		  entity old_phi = make_phi_entity(i);
		  entity new_phi = make_phi_entity(i-(nb_common_indices+1));

		  sc_variable_rename(sc_output, old_phi, new_phi);
		}
	      *output_desc = make_descriptor_convex(sc_output);

	      /* int output_ref_inds = (int) gen_length(reference_indices(*output_ref));*/
	      for(int i = 1; i<nb_phi_input_ref-nb_common_indices; i++)
		{
		  POP(input_remaining_indices);
		  expression input_remaining_indices_exp = EXPRESSION(CAR(input_remaining_indices));
		  if ( entity_field_p(expression_variable(input_remaining_indices_exp)))
		    reference_indices(*output_ref) = gen_nconc(reference_indices(*output_ref),
							       CONS(EXPRESSION, copy_expression(input_remaining_indices_exp), NIL));
		  else
		    reference_indices(*output_ref) = gen_nconc( reference_indices(*output_ref),
								CONS(EXPRESSION,
								     make_phi_expression(i),
								     NIL));
		} /* for */
	      } /* if (derived_type_p(bct) || pointer_type_p(bct)) */
	      else
	      {
		*output_desc = make_descriptor_convex(sc_new());
		*exact_p = true;
	      }
	    }

	} /* if(!ENDP(input_remaining_indices))*/

    } /* else du if (effect_undefined_p(eff_real) || ...) */


}

/** @brief translates a convex memory access path reference from given indices
           using a value_of memory access path reference

    This function is used when we want to translate a cell or an effect on a[i][j][k] as input_ref,
    knowing that a[i] = value_of_ref. In this case the list of remaning_input_indices is [j][k].

    @param input_ref is the input convex cell reference
    @param input_desc is the descriptor describing the input reference
    @param input_remaining_indices is the list of indices from input_ref which have to be translated.

    @param value_of_ref is the convex cell reference giving the output base memory access path.
    @param value_of_desc is the descriptor describing value_of_ref.

    @param output_ref is a pointer on the resulting convex reference
    @param output_desc is a pointer on teh resulting descriptor describing output_ref.
    @param exact_p is a pointer on a bool which is set to true if the translation is exact, false otherwise.

    input_remaining_indices does not need to be a copy of a part of the indices of input_ref, because it is not modified.
    In a first version of this function, it was replaced by a integer giving the rank of the beginning of this list
    in the input_ref indices list. However, in generic_eval_cell_with_points_to,
    convex_cell_reference_with_value_of_cell_reference_translation may be called
    several times with the same input_ref and rank, so it was more efficient to pass the input_remaning_indices list
    directly as an argument.

 */
void convex_cell_reference_with_value_of_cell_reference_translation
(reference input_ref, descriptor input_desc,
 reference value_of_ref, descriptor value_of_desc,
 int nb_common_indices,
 reference *output_ref, descriptor *output_desc,
 bool *exact_p)
{
  pips_debug(8, "input_ref = %s\n", words_to_string(effect_words_reference(input_ref)));
  pips_debug(8, "value_of_ref = %s\n", words_to_string(effect_words_reference(value_of_ref)));
  pips_debug(8, "nb_common_indices = %d\n", nb_common_indices);

  /* assume exactness */
  *exact_p = true;

  /* we do not handle yet the cases where the type of value_of_ref does not match
     the type of a[i]. I need a special function to test if types are compatible,
     because type_equal_p is much too strict.
     moreover the signature of the function may not be adapted in case of the reshaping of a array
     of structs into an array of char for instance.
  */

  /* first build the output reference */
  list input_inds = reference_indices(input_ref);
  int nb_phi_input = (int) gen_length(input_inds);

  list value_of_inds = reference_indices(value_of_ref);
  int nb_phi_value_of = (int) gen_length(value_of_inds);

  *output_ref = copy_reference(value_of_ref);

  /* we add the indices of the input reference past the nb_common_indices
     (they have already be poped out) to the copy of the value_of reference */

  for(int i = 0; i<nb_common_indices; i++, POP(input_inds));

  int i = nb_phi_value_of+1; /* current index to be handled */
  FOREACH(EXPRESSION, input_ind, input_inds)
    {
      if (entity_field_p(expression_variable(input_ind)))
	reference_indices(*output_ref) = gen_nconc(reference_indices(*output_ref),
						   CONS(EXPRESSION,
							copy_expression(input_ind),
							NIL));
      else
	reference_indices(*output_ref) = gen_nconc(reference_indices(*output_ref),
						   CONS(EXPRESSION,
							 make_phi_expression(i),
							NIL));
      i++;
    }

  /* Then deal with the output descriptor*/
  Psysteme input_sc2 = sc_dup(descriptor_convex(input_desc));
  Psysteme value_of_sc = descriptor_convex(value_of_desc);

  Psysteme output_sc = sc_dup(value_of_sc);

  /* preparing the part of sc_input which is to be added to sc_output */
  /* first eliminate common dimensions in sc_input (well a copy, do not modify the original) */
  if (nb_common_indices >0)
    {
      list l_phi = phi_entities_list(1,nb_common_indices);
      FOREACH(ENTITY,phi, l_phi)
	{
	  bool exact_projection;
	  input_sc2 = cell_reference_sc_exact_projection_along_variable(input_ref, input_sc2, phi, &exact_projection);
	  *exact_p = *exact_p && exact_projection;
	}
    }

  if(nb_phi_value_of - nb_common_indices != 0) /* avoid useless renaming */
    {
      if (nb_phi_value_of<nb_common_indices)
	for(i= nb_common_indices+1; i<=nb_phi_input; i++)
	  {
	    entity old_phi = make_phi_entity(i);
	    entity new_phi = make_phi_entity(nb_phi_value_of+i-nb_common_indices);
	    pips_debug(8, "case 1: i = %d, nb_phi_value_of+i-nb_common_indices = %d\n",
		       i, nb_phi_value_of+i-nb_common_indices);
	    sc_variable_rename(input_sc2, old_phi, new_phi);
	  }
      else
	for(i= nb_phi_input; i>nb_common_indices; i--)
	{
	  entity old_phi = make_phi_entity(i);
	  entity new_phi = make_phi_entity(nb_phi_value_of+i-nb_common_indices);

	  pips_debug(8, "case 2: i = %d, nb_phi_value_of+i-nb_common_indices = %d\n",
		       i, nb_phi_value_of+i-nb_common_indices);
	  sc_variable_rename(input_sc2, old_phi, new_phi);
	}

    }
  output_sc = cell_system_sc_append_and_normalize(output_sc, input_sc2, 1);

  *output_desc = make_descriptor_convex(output_sc);
  sc_rm(input_sc2);
}

void convex_cell_with_address_of_cell_translation
(cell input_cell, descriptor input_desc,
 cell address_of_cell, descriptor address_of_desc,
 int nb_common_indices,
 cell *output_cell, descriptor * output_desc,
 bool *exact_p)
{
  reference input_ref = cell_any_reference(input_cell);
  reference address_of_ref = cell_any_reference(address_of_cell);
  reference output_ref;
  convex_cell_reference_with_address_of_cell_reference_translation(input_ref, input_desc,
								   address_of_ref, address_of_desc,
								   nb_common_indices, &output_ref,
								   output_desc,
								   exact_p);

  *output_cell = make_cell_reference(output_ref);
}

void convex_cell_with_value_of_cell_translation
(cell input_cell, descriptor input_desc,
 cell value_of_cell, descriptor  value_of_desc,
 int nb_common_indices,
 cell *output_cell, descriptor * output_desc,
 bool *exact_p)
{
  reference input_ref = cell_any_reference(input_cell);
  reference value_of_ref = cell_any_reference(value_of_cell);
  reference output_ref;
  convex_cell_reference_with_value_of_cell_reference_translation(input_ref, input_desc,
								 value_of_ref, value_of_desc,
								 nb_common_indices, &output_ref,
								 output_desc,
								 exact_p);

  *output_cell = make_cell_reference(output_ref);
}
