/*
 * init.c
 *
 * Fabien Coelho, June 1993
 *
 * in this file there are functions to generate the 
 * run-time resolution parameters.
 *
 * SCCS stuff:
 * $RCSfile: inits.c,v $ ($Date: 1994/09/01 15:47:44 $, ) version $Revision$,
 * got on %D%, %T%
 * $Id$
 *
 */

#include <stdio.h>
extern int fprintf();
extern int vfprintf();

#include "genC.h"
#include "ri.h"
#include "hpf.h"

#include "misc.h"
#include "ri-util.h"
#include "hpfc.h"
#include "defines-local.h"

void create_common_parameters_h(file)
FILE* file;
{
    fprintf(file, "      integer\n");
    fprintf(file, "     $     REALNBOFARRAYS,\n");
    fprintf(file, "     $     REALNBOFTEMPLATES,\n");
    fprintf(file, "     $     REALNBOFPROCESSORS,\n");
    fprintf(file, "     $     REALMAXSIZEOFPROCS\n\n");

    fprintf(file, "c\nc parameters\nc\n");

    fprintf(file, "      parameter(REALNBOFARRAYS = %d)\n", 
	    number_of_distributed_arrays());

    fprintf(file, "      parameter(REALNBOFTEMPLATES = %d)\n", 
	    number_of_templates());

    fprintf(file, "      parameter(REALNBOFPROCESSORS = %d)\n", 
	    number_of_processors());

    fprintf(file, "      parameter(REALMAXSIZEOFPROCS = %d)\n", 
	    max_size_of_processors());
}

/*
 * create_parameters_h
 *
 * to be called after the declaration_with_overlaps() call.
 * it generates the parameters for module.
 */
void create_parameters_h(file, module)
FILE* file;
entity module;
{
    entity 
	array = entity_undefined,
	newarray = entity_undefined;
    dimension
	d = dimension_undefined;
    int i;
    list l = list_of_distributed_arrays_for_module(module);

    fprintf(file, 
	    "c\nc parameters generated for %s\nc\n",
	    module_local_name(module));

    MAPL(ce,
     {
	 array = ENTITY(CAR(ce));
	 newarray = load_entity_node_new(array);

	 debug(8, "create_parameters_h",
	       "considering array %s (new is %s)\n",
	       entity_name(array), entity_name(newarray));
	 
	 for (i=1 ; i<=NumberOfDimension(newarray) ; i++)
	 {
	     d = FindIthDimension(newarray, i);
	     
	     fprintf(file, "      integer %s, %s\n",
		     bound_parameter_name(newarray, LOWER, i),
		     bound_parameter_name(newarray, UPPER, i));
	     
	     fprintf(file, 
		     "      parameter(%s = %d)\n      parameter(%s = %d)\n",
		     bound_parameter_name(newarray, LOWER, i),
		     HpfcExpressionToInt(dimension_lower(d)),
		     bound_parameter_name(newarray, UPPER, i),
		     HpfcExpressionToInt(dimension_upper(d)));
	 }
     },
	 l);

    gen_free_list(l);
}

/*
 * max_size_of_processors
 */
int max_size_of_processors()
{
    int 
	current_max = 1;

    MAPL(ce, 
     {
	 entity
	     e = ENTITY(CAR(ce));
	 variable 
	     a;
	 
	 pips_assert("max_size_of_processors", type_variable_p(entity_type(e)));
	 a = type_variable(entity_type(e));

	 current_max = max(current_max,
			   NumberOfElements(variable_dimensions(a)));
     }, 
	 list_of_processors());

    return(current_max);
}

void create_init_common_param_for_arrays(file, module)
FILE* file;
entity module;
{
    list l = list_of_distributed_arrays_for_module(module);
    
    fprintf(file, "c\nc Arrays Initializations for %s\nc\n",
	    module_local_name(module));

    MAPL(ce,
     {
	 entity
	     array = ENTITY(CAR(ce));
	 int an = load_entity_hpf_number(array);
	 int nd = NumberOfDimension(array);
	 align
	     al = load_entity_align(array);
	 entity
	     template = align_template(al);
	 int tn = load_entity_hpf_number(template);
	 distribute
	     di = load_entity_distribute(template);
	 int i;
	 
	 fprintf(file, "c\nc initializing array %s, number %d\nc\n",
		 entity_local_name(array), an);
	 
	 /*
	  * NODIMA: Number Of  DIMensions of an Array
	  * ATOT: Array TO Template
	  */
	 fprintf(file, "      NODIMA(%d) = %d\n", an, nd);
	 fprintf(file, "      ATOT(%d) = %d\n", an, tn);
	 
	 /*
	  * RANGEA: lower, upper, size and declaration, aso
	  */
	 i = 1;
	 for (i=1; i<=nd; i++)
	 {
	     dimension
		 d = entity_ith_dimension(array, i);
	     int lb = HpfcExpressionToInt(dimension_lower(d));
	     int ub = HpfcExpressionToInt(dimension_upper(d));
	     int sz = (ub-lb+1);
	     int decl = new_declaration(array, i);
	     alignment 
		 a = FindAlignmentOfDim(align_alignment(al), i);
	     
	     /*
	      * RANGEA contents:
	      *
	      * 1: lower bound
	      * 2: upper bound
	      * 3: size, (2)-(1)+1
	      * 4: new declaration flag
	      */
	     fprintf(file, "\n");
	     fprintf(file, "      RANGEA(%d, %d, 1) = %d\n", an, i, lb);
	     fprintf(file, "      RANGEA(%d, %d, 2) = %d\n", an, i, ub);
	     fprintf(file, "      RANGEA(%d, %d, 3) = %d\n", an, i, sz);
	     fprintf(file, "c\n");
	     fprintf(file, "      RANGEA(%d, %d, 4) = %d\n", an, i, decl);
	     
	     switch(decl)
	     {
	     case NO_NEW_DECLARATION:
		 break;
	     case ALPHA_NEW_DECLARATION:
		 /*
		  * 5: 1 - lower bound
		  */
		 fprintf(file, "      RANGEA(%d, %d, 5) = %d\n", 
			 an, i, (1-lb));
		 break;
	     case BETA_NEW_DECLARATION:
	     {
		 int tdim = alignment_templatedim(a);
		 int procdim = 0;
		 distribution
		     d = FindDistributionOfDim(distribute_distribution(di), 
					       tdim,
					       &procdim);
		 int param = HpfcExpressionToInt(distribution_parameter(d));
		 int rate;
		 int shift;
		 dimension 
		     dim = FindIthDimension(template, tdim);
		 
		 pips_assert("create_init_common_param", 
			     style_block_p(distribution_style(d)));
		 
		 rate = HpfcExpressionToInt(alignment_rate(a));
		 shift = (HpfcExpressionToInt(alignment_constant(a)) -
			  HpfcExpressionToInt(dimension_lower(dim)));
		 /*
		  * 5: distribution parameter n, 
		  * 6: alignment rate a,
		  * 7: alignment shift, b-t_{m}
		  */
		 fprintf(file, "      RANGEA(%d, %d, 5) = %d\n", an, i, param);
		 fprintf(file, "      RANGEA(%d, %d, 6) = %d\n", an, i, rate);
		 fprintf(file, "      RANGEA(%d, %d, 7) = %d\n", an, i, shift);
		 
		 break;
	     }
	     case GAMMA_NEW_DECLARATION:
	     {
		 int tdim = alignment_templatedim(a);
		 int procdim = 0;
		 distribution
		     d = FindDistributionOfDim(distribute_distribution(di), 
					       tdim,
					       &procdim);
		 int param = HpfcExpressionToInt(distribution_parameter(d));
		 int sc;
		 int no;
		 int shift;
		 dimension 
		     dim = FindIthDimension(template, tdim);
		 entity
		     proc = distribute_processors(di);
		 
		 pips_assert("create_init_common_param", 
			     style_cyclic_p(distribution_style(d)));
		 
		 sc = param*SizeOfIthDimension(proc, procdim);
		 shift = (HpfcExpressionToInt(alignment_constant(a)) -
			  HpfcExpressionToInt(dimension_lower(dim)));
		 no = (lb + shift) / sc ;
		 
		 /*
		  * 5: distribution parameter n,
		  * 6: cycle length n*p,
		  * 7: initial cycle number,
		  * 8: alignment shift, b-t_{m}
		  */
		 fprintf(file, "      RANGEA(%d, %d, 5) = %d\n", an, i, param);
		 fprintf(file, "      RANGEA(%d, %d, 6) = %d\n", an, i, sc);
		 fprintf(file, "      RANGEA(%d, %d, 7) = %d\n", an, i, no);
		 fprintf(file, "      RANGEA(%d, %d, 8) = %d\n", an, i, shift);
		 
		 break;
	     }
	     case DELTA_NEW_DECLARATION:
	     {
		 int tdim = alignment_templatedim(a);
		 int rate = HpfcExpressionToInt(alignment_rate(a));
		 int cst  = HpfcExpressionToInt(alignment_constant(a));
		 int param = HpfcExpressionToInt(distribution_parameter(d));
		 int procdim = 0;
		 int sc;
		 int no;
		 int shift;
		 int chck;		      
		 distribution
		     d = FindDistributionOfDim(distribute_distribution(di), 
					       tdim,
					       &procdim);
		 dimension 
		     templdim = FindIthDimension(template, tdim);
		 entity
		     proc = distribute_processors(di);
		 
		 pips_assert("create_init_common_param", 
			     style_cyclic_p(distribution_style(d)));
		 
		 sc = param*SizeOfIthDimension(proc, procdim);
		 shift = (cst - HpfcExpressionToInt(dimension_lower(templdim)));
		 no = (rate*lb + shift) / sc ;
		 chck = iceil(param, abs(rate));
		 
		 /*
		  *  5: distribution parameter n
		  *  6: cycle length n*p,
		  *  7: initial cycle number,
		  *  8: alignment shift, b-t_{m}
		  *  9: alignment rate a,
		  * 10: chunck size ceil(n,|a|)
		  */
		 fprintf(file, "      RANGEA(%d, %d, 5) = %d\n", an, i, param);
		 fprintf(file, "      RANGEA(%d, %d, 6) = %d\n", an, i, sc);
		 fprintf(file, "      RANGEA(%d, %d, 7) = %d\n", an, i, no);
		 fprintf(file, "      RANGEA(%d, %d, 8) = %d\n", an, i, shift);
		 fprintf(file, "      RANGEA(%d, %d, 9) = %d\n", an, i, rate);
		 fprintf(file, "      RANGEA(%d, %d, 10) = %d\n", an, i, chck);
		 
		 break;
	     }
	     default:
		 pips_error("create_init_common_param", 
			    "unexpected new declaration tag (%d)\n",
			    decl);
	     }
	 }

	 fprintf(file, "\n");

	 /*
	  * ALIGN
	  */

	 for(i=1 ; i<=NumberOfDimension(template) ; i++)
	 {
	     alignment
		 a = FindAlignmentOfTemplateDim(align_alignment(al), i);

	      if (a==alignment_undefined)
	      {
		  fprintf(file, "      ALIGN(%d, %d, 1) = INTFLAG\n", an, i);
	      }
	     else
	     {
		 int adim = alignment_arraydim(a);

		 fprintf(file, "      ALIGN(%d, %d, 1) = %d\n", an, i, adim);
		 
		 if (adim==0)
		     fprintf(file, "      ALIGN(%d, %d, 2) = 0\n", an, i);
		 else
		     fprintf(file, "      ALIGN(%d, %d, 2) = %d\n", an, i,
			     HpfcExpressionToInt(alignment_rate(a)));

		 fprintf(file, "      ALIGN(%d, %d, 3) = %d\n", an, i,
			 HpfcExpressionToInt(alignment_constant(a)));

	     }
	  }
     },
	 l);

    gen_free_list(l);
}

void create_init_common_param_for_templates(file)
FILE* file;
{
    fprintf(file, "c\nc Templates Initializations\nc\n");

    MAPL(ce,
     {
	 entity
	     template = ENTITY(CAR(ce));
	 int tn = load_entity_hpf_number(template);
	 int nd  = NumberOfDimension(template);
	 distribute
	     di = load_entity_distribute(template);
	 entity
	     proc = distribute_processors(di);
	 int pn = load_entity_hpf_number(proc);
	 int procdim = 1;
	 int tempdim = 1;
	 
	 fprintf(file, "c\nc initializing template %s, number %d\nc\n",
		 entity_local_name(template), tn);

	 /*
	  * NODIMT: Number Of  DIMensions of a Template
	  * TTOP: Template TO Processors arrangement
	  */
	 fprintf(file, "      NODIMT(%d) = %d\n", tn, nd);
	 fprintf(file, "      TTOP(%d) = %d\n", tn, pn);
	 
	 /*
	  * RANGET: lower, upper, size 
	  */
	 MAPL(cd,
	  {
	      dimension
		  d = DIMENSION(CAR(cd));
	      int lb = HpfcExpressionToInt(dimension_lower(d));
	      int ub = HpfcExpressionToInt(dimension_upper(d));
	      int sz = (ub-lb+1);

	      fprintf(file, "\n");
	      fprintf(file, "      RANGET(%d, %d, 1) = %d\n", tn, tempdim, lb);
	      fprintf(file, "      RANGET(%d, %d, 2) = %d\n", tn, tempdim, ub);
	      fprintf(file, "      RANGET(%d, %d, 3) = %d\n", tn, tempdim, sz);

	      tempdim++;
	  },
	      variable_dimensions(type_variable(entity_type(template))));

	 /*
	  * DIST
	  */
	 tempdim = 1;
	 fprintf(file, "\n");
	 MAPL(cd,
	  {
	      distribution 
		  d = DISTRIBUTION(CAR(cd));
	      int param;
	      bool 
		  block_case = FALSE;

	      switch(style_tag(distribution_style(d)))
	      {
	      case is_style_none:
		  break;
	      case is_style_block:
		  block_case = TRUE;
	      case is_style_cyclic:
		  param = HpfcExpressionToInt(distribution_parameter(d));
		  if (!block_case) param = -param;
		  fprintf(file, "      DIST(%d, %d, 1) = %d\n", 
			  tn, procdim, tempdim);
		  fprintf(file, "      DIST(%d, %d, 2) = %d\n", 
			  tn, procdim, param);
		  procdim++;
		  break;
	      default:
		  pips_error("create_init_common_param", 
			     "unexpected style tag\n");
		  break;
	      }

	      tempdim++;
	  },
	      distribute_distribution(di));
     },
	 list_of_templates());
}

void create_init_common_param_for_processors(file)
FILE* file;
{
    fprintf(file, "c\nc Processors Initializations\nc\n");

    MAPL(ce,
     {
	 entity
	     proc = ENTITY(CAR(ce));
	 int pn = load_entity_hpf_number(proc);
	 int nd  = NumberOfDimension(proc);
	 int procdim = 1;
	 
	 fprintf(file, "c\nc initializing processors %s, number %d\nc\n",
		 entity_local_name(proc), pn);

	 /*
	  * NODIMP: Number Of  DIMensions of a Processors arrangement
	  */
	 fprintf(file, "      NODIMP(%d) = %d\n", pn, nd);
	 
	 /*
	  * RANGEP: lower, upper, size 
	  */
	 MAPL(cd,
	  {
	      dimension
		  d = DIMENSION(CAR(cd));
	      int lb = HpfcExpressionToInt(dimension_lower(d));
	      int ub = HpfcExpressionToInt(dimension_upper(d));
	      int sz = (ub-lb+1);

	      fprintf(file, "\n");
	      fprintf(file, "      RANGEP(%d, %d, 1) = %d\n", pn, procdim, lb);
	      fprintf(file, "      RANGEP(%d, %d, 2) = %d\n", pn, procdim, ub);
	      fprintf(file, "      RANGEP(%d, %d, 3) = %d\n", pn, procdim, sz);
	      
	      procdim++;
	  },
	      variable_dimensions(type_variable(entity_type(proc))));
     },
	 list_of_processors());
}

/*   create_init_common_param (templates and modules)
 */
void create_init_common_param(file)
FILE* file;
{
    create_init_common_param_for_templates(file);
    create_init_common_param_for_processors(file);
}

