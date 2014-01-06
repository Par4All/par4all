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
/* this file describe a few functions usefull to the compiler
 * to manage the hpfc data structures.
 *
 * Fabien Coelho, May 1993.
 */

#include "defines-local.h"
#include "effects.h"
#include "effects-util.h"
#include "effects-generic.h"
#include "effects-simple.h"

/* Predicates
 */

/* true if there is a reference to a distributed array within obj
 *
 * ??? not very intelligent, should use the regions, the problem is
 * that I should normalize the code *before* the pips analysis...
 */
bool ref_to_dist_array_p(void * obj)
{
    list l = FindRefToDistArray(obj);
    bool b = (l!=NIL);

    gen_free_list(l); return(b);
}

/* written_effects_to_dist_arrays_p
 */
bool written_effect_p(entity var,
		      list le)
{
  FOREACH(EFFECT, e, le)
    {
      if(store_effect_p(e)) {
	if (reference_variable(effect_any_reference(e))==var &&
	    action_write_p(effect_action(e)))
	    return true;
      }
    }

    return false;
}

bool written_effects_to_dist_arrays_p(expression expr)
{
    list l, leffects_to_dist_arrays = DistArraysEffects(expr);

    // FI: looks like a FOREACH to me...
    for(l=leffects_to_dist_arrays; !ENDP(l); POP(l))
      if(store_effect_p(EFFECT(CAR(l)))) {
	if  (action_write_p(effect_action(EFFECT(CAR(l)))))
	{
	    gen_free_list(leffects_to_dist_arrays);
	    return true;
	}
      }

    gen_free_list(leffects_to_dist_arrays);
    return false;
}

/* replicated_p
 *
 * check whether the distributed array e
 * is replicated or not.
 */
bool replicated_p(entity e)
{
    int i, ntdim;
    align a;
    list la, ld;
    entity template;
    distribute d;

    pips_assert("distributed array", array_distributed_p(e));

    a = load_hpf_alignment(e);
    la = align_alignment(a);    
    template = align_template(a);
    d = load_hpf_distribution(template);
    ld = distribute_distribution(d);
    ntdim = NumberOfDimension(template);

    for(i=1; i<=ntdim; i++, POP(ld))
	if (ith_dim_replicated_p(template, i, la, DISTRIBUTION(CAR(ld))))
	    return true;

    return false;
}

/* bool ith_dim_replicated_p(template, i, la, dist)
 *
 * true if template dimension i distributed with dist leads to 
 * a replication for array align al.
 */
bool ith_dim_replicated_p(template, i, la, dist)
entity template;
int i;
list la;
distribution dist;
{
    if (style_none_p(distribution_style(dist))) return false;

    /* select the relevent alignment if exists.
     * could be some kind of gen_find_if()...
     */
    MAP(ALIGNMENT, a, if (alignment_templatedim(a)==i) return false, la);

    return true;
}

/* true if array a is replicated on processors p i-th dimension.
 */
bool processors_dim_replicated_p(p, a, i)
entity p, a;
int i;
{
    int tdim;
    align al = load_hpf_alignment(a);
    entity t = align_template(al);
    distribute d = load_hpf_distribution(t);
    distribution di =
	FindDistributionOfProcessorDim(distribute_distribution(d), i, &tdim);
    alignment ali = 
	FindAlignmentOfTemplateDim(align_alignment(al), tdim);

    return(!style_none_p(distribution_style(di)) &&
	   alignment_undefined_p(ali));
}

/* whether a dimension is distributed or not.
 */
bool ith_dim_distributed_p(array, i, pprocdim)
entity array;
int i, *pprocdim;
{
    align       al = load_hpf_alignment(array);
    list       lal = align_alignment(al);
    alignment  alt = FindAlignmentOfDim(lal, i);
    entity template = align_template(al);
    distribute dis = load_hpf_distribution(template);
    list ld = distribute_distribution(dis);
    distribution d;

    if (alignment_undefined_p(alt)) return(false);
    d = FindDistributionOfDim(ld, alignment_templatedim(alt), pprocdim);
    return(!style_none_p(distribution_style(d)));
}

bool 
ith_dim_overlapable_p(
    entity array,
    int i)
{
    align       al = load_hpf_alignment(array);
    list       lal = align_alignment(al);
    alignment  alt = FindAlignmentOfDim(lal, i);
    entity template = align_template(al);
    distribute dis = load_hpf_distribution(template);
    list ld = distribute_distribution(dis);
    distribution d;
    int p;

    if (alignment_undefined_p(alt)) return false;
    d = FindDistributionOfDim(ld, alignment_templatedim(alt), &p);

    return style_block_p(distribution_style(d));
}


/* creates a new statement for the given module
 * that looks like the stat one, i.e. same comment, same 
 * label, and so on. The goto table is updated. The instruction
 * is also created. (is that really a good idea?)
 */
statement MakeStatementLike(stat, the_tag)
statement stat;
int the_tag;
{
    void* x = loop_undefined;
    string c = statement_comments(stat);
    statement new_s;

    if (the_tag==is_instruction_sequence)
        x = make_sequence(NIL);


    new_s = make_statement(statement_label(stat),
			   STATEMENT_NUMBER_UNDEFINED,
			   STATEMENT_ORDERING_UNDEFINED,
			   string_undefined_p(c)? c: strdup(c),
			   make_instruction(the_tag, x),NIL,NULL,
			   copy_extensions (statement_extensions(stat)), make_synchronization_none());
    fix_statement_attributes_if_sequence(new_s);
    return new_s;
}

static void stmt_rwt(s)
statement s;
{
    statement_number(s) = STATEMENT_NUMBER_UNDEFINED;
    statement_ordering(s) = STATEMENT_ORDERING_UNDEFINED;
}

void kill_statement_number_and_ordering(s)
statement s;
{
    gen_recurse(s, statement_domain, gen_true, stmt_rwt);
}

/* effects' action in an expression are here supposed to be read one's
 * but that may not be correct?
 */
list DistArraysEffects(expr)
expression expr;
{
    list le = proper_effects_of_expression(expr), lde = NIL;

    FOREACH(EFFECT, e, le) {
      if(store_effect_p(e)) {
	if (array_distributed_p(effect_variable(e)))
	  lde=CONS(EFFECT,e,lde);
      }
    }

    gen_free_list(le);
    return(lde);
}

/* FindRefToDistArrayFromList
 *
 * these functions compute the list of syntax that are
 * references to a distributed variable.
 */
list FindRefToDistArrayFromList(lexpr)
list lexpr;
{
    list l=NIL;
    MAP(EXPRESSION, e,{l=gen_nconc(FindRefToDistArray(e),l);},lexpr);
    return(l);
}

static list found_syntaxes = NIL;

static void FindRefToDistArray_syntax_rewrite(s)
syntax s;
{
    if (syntax_reference_p(s))
	if (array_distributed_p
	    (reference_variable(syntax_reference(s))))
	    found_syntaxes = 
		CONS(SYNTAX, s, found_syntaxes);
}

list FindRefToDistArray(void * obj)
{
    list result = NIL, saved = found_syntaxes;
    found_syntaxes = NIL;
    gen_multi_recurse(obj,
		      syntax_domain,
		      gen_true,
		      FindRefToDistArray_syntax_rewrite, NULL);

    result = found_syntaxes, found_syntaxes = saved;

    return result;
}

/* hmmm...
 */
entity FindOrCreateEntityLikeModel(package, name, model)
const char* package, *name;
entity model;
{
    string new_name = concatenate(package, 
				  MODULE_SEP_STRING, 
				  name, 
				  NULL);
    entity new = gen_find_tabulated(new_name, entity_domain);
    area tmp_area = area_undefined;

    pips_debug(8, "entity %s to be made after %s\n",
	       new_name, entity_name(model));

    ifdebug(9)
	pips_assert("consistent model", entity_consistent_p(model));

    return(!entity_undefined_p(new) ? new :
	   make_entity(copy_string(new_name),
		       /*
			* ??? some bug in copy_type disable the possibility
			* of copying area for instance...
			*
			* moreover I do not wish to copy the layout list
			* for commons.
			*/
		       (!type_area_p(entity_type(model)) ?
			copy_type(entity_type(model)) :
			make_type(is_type_area,
				  (tmp_area = type_area(entity_type(model)),
				   make_area(area_size(tmp_area), NIL)))),
		       copy_storage(entity_storage(model)),
		       copy_value(entity_initial(model))));
}

/*   !!! caution, it may not be a module, but a common...
 */
entity AddEntityToModule(e, module)
entity e, module;
{
    entity new = FindOrCreateEntityLikeModel(module_local_name(module),
					     entity_local_name(e),e);

    pips_debug(7, "adding %s to module %s\n",
	       entity_name(new), entity_name(module));
    
    if (entity_module_p(module))
	AddEntityToDeclarations(new, module);

    return(new);
}

/*   AddEntityToHostAndNodeModules
 */
void AddEntityToHostAndNodeModules(e)
entity e;
{
    entity new_node = AddEntityToModule(e, node_module),
	   new_host = entity_undefined;

    if (!bound_new_node_p(e))
	store_new_node_variable(new_node, e);
    else
	AddEntityToDeclarations(load_new_node(e), node_module);
    
    if (!array_distributed_p(e))
    {
	new_host = AddEntityToModule(e, host_module);

	if (!bound_new_host_p(e))
	    store_new_host_variable(new_host, e),
	    /* 
	     * added because of some entity errors.
	     */
	    store_new_host(new_node, new_host),
	    store_new_node(new_host, new_node); 
	else
	    AddEntityToDeclarations(load_new_host(e), host_module);
    }
}

/* The common name is changed to distinguish the current, host and
 * node instances of the common. 
 */
void AddCommonToModule(common, module, update, suffix)
entity common, module;
void (*update)();
string suffix;
{
    string name = strdup(concatenate(entity_local_name(common),
				     "_", suffix, NULL));
    entity new_common = 
	FindOrCreateEntityLikeModel(HPFC_PACKAGE, name, common);
    list 
	lref = area_layout(type_area(entity_type(common))),
	lold = area_layout(type_area(entity_type(new_common))),
	lnew = NIL;

    free(name);
    update(new_common, common);

    /* The layout list must be updated to the right entities
     */

    MAP(ENTITY, e,
    {
	entity new_e;
	
	if (local_entity_of_module_p(e, common)) /* !!! not in current  */
	{
	    new_e = AddEntityToModule(e, new_common);
	    
	    if (gen_find_eq(new_e, lold)==entity_undefined)
	    {
		lnew = CONS(ENTITY, new_e, lnew);
		update(new_e, e);
	    }
	}
    },
	lref);

    AddEntityToDeclarations(new_common, module);

    area_layout(type_area(entity_type(new_common))) = 
	gen_nconc(gen_nreverse(lnew), lold);
}

void AddCommonToHostAndNodeModules(entity common)
{
    AddCommonToModule(common, node_module, (void (*)())store_new_node_variable, NODE_NAME);
    AddCommonToModule(common, host_module, (void (*)())store_new_host_variable, HOST_NAME);
}

alignment FindAlignmentOfDim(lal, dim)
list lal;
int dim;
{
    list l=lal;

    while ((!ENDP(l)) && (alignment_arraydim(ALIGNMENT(CAR(l))) != dim))
	POP(l);

    return (l==NIL) ? alignment_undefined : ALIGNMENT(CAR(l));
}

alignment FindAlignmentOfTemplateDim(lal, dim)
list lal;
int dim;
{
    list l=lal;

    while ((!ENDP(l)) && (alignment_templatedim(ALIGNMENT(CAR(l))) != dim))
	POP(l);

    return ((l==NULL)?(alignment_undefined):(ALIGNMENT(CAR(l))));
}

distribution FindDistributionOfDim(ldi, dim, pdim)
list ldi;
int dim, *pdim;
{
    list l = ldi;
    int i, procdim = 1;

    pips_assert("valid dimension", dim>=1 && dim<=gen_length(ldi));

    for (i=1; i<dim; i++) 
    {
	if (!style_none_p(distribution_style(DISTRIBUTION(CAR(l)))))
	    procdim++;
	POP(l);
    }

    (*pdim) = procdim;
    return(DISTRIBUTION(CAR(l)));
}

distribution FindDistributionOfProcessorDim(ldi, dim, tdim)
list ldi;
int dim, *tdim;
{
    int i = 1, procdim = 0;
    
    MAP(DISTRIBUTION, d,
    {
	if (!style_none_p(distribution_style(d)))
	    procdim++;
	
	if (procdim==dim)
	{
	    (*tdim) = i;
	    return(d);
	}
	
	i++;
    },
	ldi);

    pips_internal_error("dimension %d not found", dim);

    return(distribution_undefined);
}    

int
template_dimension_of_array_dimension(
    entity array, 
    int dim)
{
    align a;
    alignment al;

    a = load_hpf_alignment(array);
    al = FindAlignmentOfDim(align_alignment(a), dim);
    
    return (al==alignment_undefined) ? -1 : alignment_templatedim(al);
}

int
processor_dimension_of_template_dimension(
    entity template,
    int dim)
{
    int pdim = 0, n;
    if (dim>=0) get_distribution(template, dim, &pdim, &n);
    return pdim;
}

int
DistributionParameterOfArrayDim(
    entity array,
    int dim,
    int *pprocdim)
{
    entity template = array_to_template(array);
    distribute d = load_hpf_distribution(template);
    distribution
	di = FindDistributionOfDim
	    (distribute_distribution(d),
	     alignment_templatedim(FindArrayDimAlignmentOfArray(array, dim)),
	     pprocdim);
    
    return HpfcExpressionToInt(distribution_parameter(di));
}

/* int processor_number(template, tdim, tcell, pprocdim)
 *
 * the processor number of a template cell, on dimension *pprocdim
 */
int processor_number(template, tdim, tcell, pprocdim)
entity template;
int tdim, tcell, *pprocdim; /* template dimension, template cell */
{
    distribute d = load_hpf_distribution(template);
    list ld = distribute_distribution(d);
    entity procs = distribute_processors(d);
    distribution di = FindDistributionOfDim(ld, tdim, pprocdim);
    style st = distribution_style(di);
    int	n, tmin, pmin, psiz;

    if (style_none_p(st))
    {
	*pprocdim = -1;
	return -1;
    }

    tmin = HpfcExpressionToInt
	(dimension_lower(FindIthDimension(template, tdim)));
    pmin = HpfcExpressionToInt
	(dimension_lower(FindIthDimension(procs, *pprocdim)));
    psiz = SizeOfIthDimension(procs, *pprocdim);
    n    = HpfcExpressionToInt(distribution_parameter(di));

    if (style_block_p(st))
	return ((tcell-tmin)/n)+pmin;

    if (style_cyclic_p(st))
	return (((tcell-tmin)/n)%psiz)+pmin;

    *pprocdim = -1; /* just to avoid a gcc warning */
    return -1; 
}    


/* int template_cell_local_mapping(array, dim, tc)
 *
 * ??? should check that it is indeed block distributed !
 * or should implement all the formulas...
 */
int template_cell_local_mapping(array, dim, tc)
entity array;
int dim, tc;
{
    alignment a = FindArrayDimAlignmentOfArray(array, dim);
    int	p, tmin, n = DistributionParameterOfArrayDim(array, dim, &p);
    dimension d = FindIthDimension(array_to_template(array), 
				   alignment_templatedim(a));

    tmin = HpfcExpressionToInt(dimension_lower(d));

    return (tc-tmin)%n+1;
}
	
/* int global_array_cell_to_local_array_cell(array, dim, acell)
 *
 * ??? not enough general a function
 */
int global_array_cell_to_local_array_cell(array, dim, acell)
entity array;
int dim, acell;
{
    alignment a = FindArrayDimAlignmentOfArray(array, dim);
    int rate, constant;

    pips_assert("aligned", a!=alignment_undefined);

    rate     = HpfcExpressionToInt(alignment_rate(a));
    constant = HpfcExpressionToInt(alignment_constant(a));

    return template_cell_local_mapping(array, dim, rate*acell+constant);
}

/* HpfcExpressionToInt(e)
 *
 * uses the normalized value if possible. 
 */
int HpfcExpressionToInt(e)
expression e;
{
    normalized n = expression_normalized(e);
    intptr_t val = 0;

    ifdebug(8) print_expression(e);

    if ((n!=normalized_undefined) && (normalized_linear_p(n)))
    {
	Pvecteur v = normalized_linear(n);
	int s = vect_size(v), val;
	Value vval = vect_coeff(TCST, v);

	val = VALUE_TO_INT(vval);
	if (s==0) return 0;
	if ((s==1) && (val!=0)) return val;
    }
    
    /*
    if (expression_integer_constant_p(e))
	return ExpressionToInt(e);
    */
    if(expression_integer_value(e, &val))
	return val;
    else
	pips_internal_error("can't return anything, sorry");

    return -1; /* just to avoid a gcc warning */
}

/* -------------------------------------------------------
 *
 * a nicer interface to extract the needed information. FC 29/03/94
 *
 */

void get_alignment(array, dim, ptdim, pa, pb)
entity array;
int dim, *ptdim, *pa, *pb;
{ 
    align al = load_hpf_alignment(array);
    alignment a;
    
    pips_assert("distributed array", array_distributed_p(array));
    
    *ptdim = template_dimension_of_array_dimension(array, dim);

    if (*ptdim==-1) /* replication */
    {
	*pa=0, *pb=0;
	return;
    }

    a = FindAlignmentOfTemplateDim(align_alignment(al), *ptdim);

    if (a==alignment_undefined)
    {
	pips_assert("not aligned", *ptdim==0);
	*pa = 0; *pb = 0;
    }
    else
    {
	pips_assert("aligned", *ptdim>=1);
	*pa = HpfcExpressionToInt(alignment_rate(a));
	*pb = HpfcExpressionToInt(alignment_constant(a));
    }
}

void get_distribution(template, dim, ppdim, pn)
entity template;
int dim, *ppdim, *pn;
{
    distribution
        d = FindDistributionOfDim
	    (distribute_distribution(load_hpf_distribution(template)), 
				     dim, ppdim);

    *pn = (distribution_undefined_p(d) ?
	   -1: HpfcExpressionToInt(distribution_parameter(d)));
}

void get_entity_dimensions(e, dim, plow, pup)
entity e;
int dim, *plow, *pup;
{
    dimension d = dimension_undefined;

    pips_assert("valid variable and dimension",
		entity_variable_p(e) && dim>0 && dim<=7);

    d = entity_ith_dimension(e, dim),
    *plow = ExpressionToInt(dimension_lower(d)),
    *pup = ExpressionToInt(dimension_upper(d));
}

/* bool alignments_compatible_p(entity e1, int dim1,
 *                             entity e2, int dim2)
 *
 * what: whether e1 and e2 dimensions dim1 and dim2 are aligned.
 * how: basic low level comparison
 * input: entities and dimension numbers
 * output: the bool result
 * side effects:
 *  - uses alignment internal descriptions
 * bugs or features:
 *  - ??? should be relative to a reference, instead of assuming mere indexes
 */
bool 
alignments_compatible_p(entity e1, int dim1,
		       entity e2, int dim2)
{
    int tdim1, rate1, shift1, tdim2, rate2, shift2;

    get_alignment(e1, dim1, &tdim1, &rate1, &shift1);
    get_alignment(e2, dim2, &tdim2, &rate2, &shift2);

    if (tdim1!=tdim2) return false;
    if (!tdim1 && !tdim2) return true;
    
    return rate1==rate2 && shift1==shift2;
}

/* bool references_aligned_p(reference r1, reference r2)
 *
 * what: tells whether the references are aligned or not
 * how: quite basic and low level
 * input: both references
 * output: the returned boolean
 * side effects:
 *  - uses alignment internal descriptions
 * bugs or features:
 *  - assumes that both references are in the same store.
 *  - ??? indices must be simple references to indexes.
 */
/* returns 0 if not found */
static int 
expression_number_for_index(entity index, list /* of expression */ le)
{
    int dim=1;

    MAP(EXPRESSION, e,
    {
	if (expression_reference_p(e) && 
	    reference_variable(syntax_reference(expression_syntax(e)))==index)
	    return dim;
	dim++;
    },
	le);

    return 0;
}

#define XDEBUG(msg) \
  pips_debug(6, "%s and %s: " msg "\n", entity_name(e1), entity_name(e2))

bool 
references_aligned_p(reference r1, reference r2)
{
    entity e1 = reference_variable(r1),
           e2 = reference_variable(r2);
    list /* of expression */ le1 = reference_indices(r1),
                             le2 = reference_indices(r2);
    int dim1 = 1, dim2;
    entity index;
    align a1, a2;

    if (!array_distributed_p(e1) || !array_distributed_p(e2))
    {
	XDEBUG("not distributed"); return false;
    }

    a1 = load_hpf_alignment(e1);
    a2 = load_hpf_alignment(e2);

    /* both references must be aligned to the same template
     * and be of the same arity.
     */
    if (align_template(a1)!=align_template(a2))
    {
	XDEBUG("template is different"); return false;
    }

    if (gen_length(le1)!=gen_length(le2))
    {
	XDEBUG("arities are different"); return false;
    }
    
    MAP(EXPRESSION, ind, 
    {
	if (!expression_reference_p(ind))
	    return false;

	index = reference_variable(syntax_reference(expression_syntax(ind)));
	dim2 = expression_number_for_index(index, le2);

	if (!alignments_compatible_p(e1, dim1, e2, dim2))
	{
	    XDEBUG("alignments are incompatible"); return false;
	}
	
	dim1++;
    },
	le1);

    XDEBUG("aligned!"); return true;
}

/*************************************************** IR STRUCTURAL cleaning */

/* removes IF (.TRUE.) THEN
 * and DO X=n, n
 */

DEFINE_LOCAL_STACK(current_stmt, statement)

void hpfc_util_error_handler()
{
    error_reset_current_stmt_stack();
}

static void test_rewrite(test t)
{
    entity e = expression_to_entity(test_condition(t));

    if (ENTITY_TRUE_P(e))
    {
	statement s = current_stmt_head();
	/* instruction i = statement_instruction(s); */
	pips_debug(5, "true test simplified\n");

	statement_instruction(s) = statement_instruction(test_true(t));
	/* Fix attributes if it is a sequence: */
	fix_statement_attributes_if_sequence(s);

	statement_instruction(test_true(t)) = instruction_undefined;
	/* free_instruction(i); */ /* ??? */
    }
    else if (ENTITY_FALSE_P(e))
    {	
	statement s = current_stmt_head();
	/* instruction i = statement_instruction(s); */
	pips_debug(5, "false test simplified\n");

	statement_instruction(s) = statement_instruction(test_false(t));
	/* Fix attributes if it is a sequence: */
	fix_statement_attributes_if_sequence(s);

	statement_instruction(test_false(t)) = instruction_undefined;
	/* free_instruction(i); */ /* ??? */
    }
}

static void loop_rewrite(loop l)
{
    range r = loop_range(l);
    if (expression_equal_p(range_lower(r), range_upper(r)))
    {
	statement s = current_stmt_head();
	/* instruction i = statement_instruction(s); */
	pips_debug(5, "loop on %s simplified\n", entity_name(loop_index(l)));

	statement_instruction(s) = 
	    make_instruction_block(
	      CONS(STATEMENT, make_assign_statement
		   (entity_to_expression(loop_index(l)),
		    copy_expression(range_lower(r))),
	      CONS(STATEMENT, loop_body(l), NIL)));
	/* Do not forget to move forbidden information associated with
	   block: */
	fix_sequence_statement_attributes(s);
    
	loop_body(l) = statement_undefined;
	/* free_instruction(i); */ /* ??? memory leak, cores on AIX */
    }
}

void 
statement_structural_cleaning(statement s)
{
    make_current_stmt_stack();

    gen_multi_recurse(s,
	statement_domain, current_stmt_filter, current_stmt_rewrite,
        test_domain,      gen_true,            test_rewrite,
        loop_domain,      gen_true,            loop_rewrite,
		      NULL);  
  
    free_current_stmt_stack();
}

/*   that is all
 */
