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

/* Name     : prgm_mapping.c
 * Package  : prgm_mapping
 * Author   : Alexis Platonoff
 * Date     : april 1993
 * Historic :
 * Documents: SOON
 * Comments : This file contains the functions for the computation of the
 * placement function. The main function is prgm_mapping().
 */

/* Ansi includes 	*/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <sys/times.h>          /* performance purpose */
#include <sys/time.h>           /* performance purpose */

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
#include "matrice.h"
#include "matrix.h"

/* Pips includes 	*/
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

/* Macro functions  	*/
#define DOT "."
#define BDT_STRING "bdt"
#define STMT_HT_SIZE 100	/* hash table max size */
#define DTF_HT_SIZE 200		/* hash table max size */
#define ENT_HT_SIZE 200		/* hash table max size */

/* Internal variables 	*/

/* Global variables */
plc pfunc;		/* The placement function */
graph the_dfg;		/* The data flow graph */
bdt the_bdt;		/* The timing function */
int nb_nodes,		/* The number of nodes in the DFG */
    nb_dfs;		/* The number of dataflows in the DFG */
hash_table DtfToSink;	/* Mapping from a dataflow to its sink statement */
hash_table DtfToDist;	/* Mapping from a dataflow to its distance */
hash_table DtfToWgh;	/* Mapping from a dataflow to its weight */
hash_table StmtToProto;	/* Mapping from a statement (int) to its prototype */
hash_table StmtToPdim;	/* Mapping from a statement to the dim of its plc func */
hash_table StmtToBdim;	/* Mapping from a statement to the dim of its bdt */
hash_table StmtToDim;	/* Mapping from a stmt to its iteration space dim */
hash_table StmtToLamb;	/* Mapping from a stmt to its lambda coeff */
hash_table StmtToMu;	/* Mapping from a stmt to its mu coeff */
hash_table UnkToFrenq;	/* Mapping from an entity to its frenq in the plc proto */
int count_mu_coeff;
list subs_l;
list prgm_parameter_l;

/* Local defines */
typedef dfg_vertex_label vertex_label;
typedef dfg_arc_label arc_label;

#define PLC_EXT ".plc_file"

/* ======================================================================== */
bool print_plc(module_name)
const char* module_name;
{
  char *localfilename;
  FILE        *fd;
  char        *filename;
  plc the_plc;

  debug_on( "PRINT_PLC_DEBUG_LEVEL" );

  if (get_debug_level() > 1)
      user_log("\n\n *** PRINTING PLC for %s\n", module_name);

  the_plc = (plc) db_get_memory_resource(DBR_PLC, module_name, true);

  localfilename = strdup(concatenate(module_name, PLC_EXT, NULL));
  filename = strdup(concatenate(db_get_current_workspace_directory(), 
				"/", localfilename, NULL));

  fd = safe_fopen(filename, "w");
  fprint_plc(fd, the_plc);
  safe_fclose(fd, filename);

  DB_PUT_FILE_RESOURCE(DBR_PLC_FILE, strdup(module_name), localfilename);
  
  free(filename);

  if(get_debug_level() > 0)
    fprintf(stderr, "\n\n *** PRINT_PLC DONE\n");

  debug_off();

  return(true);
}


/* ======================================================================== */
/*
 * list plc_make_proto():
 *
 * computes the prototype of placement function for each statement,
 * i.e. for each node of the DGF. Returns the list of unknowns
 * coefficients that have been created.
 *
 * For each node, this consists in building a polynome which will be
 * associated via the hash table "StmtToProto" to the corresponding
 * statement.
 *
 * The placement function is an integer linear function on the indices of
 * the englobing loops and structure parameters of the statement. We have
 * to determine the coefficients associated to each index. These unknowns
 * are represented by variables, so the prototype is a polynome of degree
 * two (2). Moreover there is an unknown constant term.
 *
 * For example, with a statement englobed by three (3) loops (I1, I2, I3)
 * and having one structure parameter (N), the prototype is :
 *
 *	COEFF1*I1 + COEFF2*I2 + COEFF3*I3 + COEFF$*N + COEFF5
 *
 * Note: The list "l_lamb" that contains all the coefficients is build
 * during by appending the sublists associated to each statement. Each
 * sublist is ordered as follow: you have first the coeff for the constant
 * term, the coeffs for the loop indices (from the outermost to the
 * innermost) and the coeffs of the structure parameters (the order of the
 * static_controlize). This order is used in the function
 * broadcast_conditions().*/
list plc_make_proto()
{
  extern graph the_dfg;		 /* the DFG */
  extern hash_table StmtToProto, /* Mapping from a statement to its
  	                          * prototype */
  	            StmtToLamb;  /* Mapping from a statement to its
 				  * lambda coeff */

 
  int count_coeff;
  list l, l_lamb = NIL, stmt_lamb;
  entity lamb;
  Ppolynome pp_proto, pp_ind, pp_par, pp_coeff, pp_new;

  /* We create the hash table that will contain the prototype */
  StmtToProto = hash_table_make(hash_int, nb_nodes+1);
  StmtToLamb = hash_table_make(hash_int, nb_nodes+1);

  /* We initialize the counter of coefficients in order to have all
   * unknowns represented by different entities.
   */
  count_coeff = 0;

  /* For each node of the data flow graph we compute a prototype. */
  for(l = graph_vertices(the_dfg); l != NIL; l = CDR(l)) {
    vertex v;
    int stmt;
    static_control stct;
    list ind_l, par_l, aux_par_l;

    v = VERTEX(CAR(l));
    stmt = dfg_vertex_label_statement((dfg_vertex_label) vertex_vertex_label(v));

    /* We get the "static_control" associated to the statement. */
    stct = get_stco_from_current_map(adg_number_to_statement(stmt));

    /* We get the list of indices of the englobing loops and structure
     * parameters.
     */
    ind_l = static_control_to_indices(stct);
    par_l = gen_append(prgm_parameter_l, NIL);

    stmt_lamb = NIL;

    /* First, we initialize the polynome in building the constant term. */
    lamb = make_coeff(CONST_COEFF, ++count_coeff);
    stmt_lamb = gen_nconc(stmt_lamb, CONS(ENTITY, lamb, NIL));
    pp_proto = make_polynome(1.0, (Variable) lamb, (Value) 1);

    /* Then, for each index (ind) we build the monome ind*COEFF#, and add it
     * to the polynome.
     */
    for( ; ind_l != NIL; ind_l = CDR(ind_l)) {
      entity ind = ENTITY(CAR(ind_l));

      lamb = make_coeff(INDEX_COEFF, ++count_coeff);
      stmt_lamb = gen_nconc(stmt_lamb, CONS(ENTITY, lamb, NIL));

      pp_ind = make_polynome(1.0, (Variable) ind, (Value) 1);
      pp_coeff = make_polynome(1.0, (Variable) lamb, (Value) 1);
      pp_new = polynome_mult(pp_ind, pp_coeff);

      polynome_add(&pp_proto, pp_new);
    }

    /* Then, for each parameter (par) we build the monome par*COEFF#, and add it
     * to the polynome.
     */
    for(aux_par_l = par_l ; !ENDP(aux_par_l); POP(aux_par_l)) {
      entity par = ENTITY(CAR(aux_par_l));

      lamb = make_coeff(PARAM_COEFF, ++count_coeff);
      stmt_lamb = gen_nconc(stmt_lamb, CONS(ENTITY, lamb, NIL));

      pp_par = make_polynome(1.0, (Variable) par, (Value) 1);
      pp_coeff = make_polynome(1.0, (Variable) lamb, (Value) 1);
      pp_new = polynome_mult(pp_par, pp_coeff);

      polynome_add(&pp_proto, pp_new);
    }

    /* We put the new prototype on the hash table. */
    hash_put(StmtToLamb, (char *) stmt , (char *) stmt_lamb);
    hash_put(StmtToProto, (char *) stmt , (char *) pp_proto);

    l_lamb = gen_append(stmt_lamb, l_lamb);
  }
  return(l_lamb);
}

/* ======================================================================== */
/*
 * void initialize_mu_list(int stmt, dim)
 */
void initialize_mu_list(stmt, dim)
int stmt, dim;
{
  extern int count_mu_coeff;
  extern hash_table StmtToMu;

  list mu_l = NIL;
  int i;

  for(i = 0; i < dim; i++) {
    entity mu = make_coeff(MU_COEFF, count_mu_coeff++);
    mu_l = gen_nconc(mu_l, CONS(ENTITY, mu, NIL));
  }
  hash_put(StmtToMu, (char *) stmt , (char *) mu_l);
}


/* ======================================================================== */
/*
 * int plc_make_min_dim():
 *
 * computes the minimum number of dimensions for the placement
 * function.
 *
 * It is equal to the dimension associated to the instruction
 * which is nested into the deepest sequential nest loop, i.e., which has
 * the schedule with the biggest number of dimensions. In all cases, the
 * placement has at least one dimension.
 */
int plc_make_min_dim()
{
  extern graph the_dfg;
  extern bdt the_bdt;
  extern hash_table StmtToPdim, StmtToBdim;

  int dmin = 1, bdmax = 1;
  list l;

  StmtToPdim = hash_table_make(hash_pointer, nb_nodes+1);
  StmtToBdim = hash_table_make(hash_pointer, nb_nodes+1);
  StmtToMu = hash_table_make(hash_pointer, nb_nodes+1);

  /* For each node of the data flow graph we compute its dimension. */
  for(l = graph_vertices(the_dfg); l != NIL; l = CDR(l)) {
    vertex v;
    int stmt, dim, b_dim, p_dim;
    static_control stct;
    list ind_l;

    v = VERTEX(CAR(l));
    stmt = vertex_int_stmt(v);
    stct = get_stco_from_current_map(adg_number_to_statement(stmt));
    ind_l = static_control_to_indices(stct);

    /* We compute the dimension of the current node */
    dim = gen_length(ind_l);

    /* We get the dimension of the timing and placement function of the
     * current node */
    b_dim = (int) hash_get(StmtToBdim, (char *) stmt);
    p_dim = (int) hash_get(StmtToPdim, (char *) stmt);

    if((bdmax <= b_dim) && (dmin < p_dim))
      dmin = p_dim;
  }
  return(dmin);
}


/* ======================================================================== */
/*
 * int plc_make_dim():
 *
 * computes the maximum possible dimension of the placement function
 * associated to each instruction and returns the greatest of these
 * dimensions.
 *
 * For an given instruction, the dimension of its placement function is the
 * substraction of the englobing space dimension and the timing function
 * dimension.
 */
int plc_make_dim()
{
  extern graph the_dfg;
  extern bdt the_bdt;
  extern hash_table StmtToPdim, StmtToBdim;

  int dmax = 0;
  list l;

  StmtToPdim = hash_table_make(hash_pointer, nb_nodes+1);
  StmtToBdim = hash_table_make(hash_pointer, nb_nodes+1);
  StmtToMu = hash_table_make(hash_pointer, nb_nodes+1);

  /* For each node of the data flow graph we compute its dimension. */
  for(l = graph_vertices(the_dfg); l != NIL; l = CDR(l)) {
    vertex v = VERTEX(CAR(l));
    int stmt = vertex_int_stmt(v);
    static_control stct = get_stco_from_current_map(adg_number_to_statement(stmt));
    list ind_l = static_control_to_indices(stct);
    int dim, b_dim = 0, sdim;

    /* We compute the dimension of the current node */
    dim = gen_length(ind_l);

    /* We compute the dimension of the timing function of the current node */
    if(the_bdt != bdt_undefined) {
      list bl;
      b_dim = 0;
      for(bl = bdt_schedules(the_bdt); bl != NIL; bl = CDR(bl)) {
	schedule sc = SCHEDULE(CAR(bl));
	if(schedule_statement(sc) == stmt) {
	  list dl;
	  sdim = 0;
	  for(dl = schedule_dims(sc); dl != NIL; dl = CDR(dl)) {
	    expression e = EXPRESSION(CAR(dl));
	    if(! expression_constant_p(e)) {
	      list l;
	      bool ind_not_null;
              Pvecteur pv;
  
              NORMALIZE_EXPRESSION(e);
              pv = (Pvecteur) normalized_linear(expression_normalized(e));

	      ind_not_null = false;
	      for(l = ind_l; (!ENDP(l)) && (!ind_not_null); POP(l)) {
	        entity var = ENTITY(CAR(l));
                if(vect_coeff((Variable) var, pv) != 0)
		  ind_not_null = true;
	      }
	      if(ind_not_null)
	        sdim++;
	    }
	  }
	  if(sdim > b_dim)
	    b_dim = sdim;
	}
      }
    }
    dim = dim - b_dim;
    if(dim > dmax)
      dmax = dim;
    hash_put(StmtToPdim, (char *) stmt, (char *) dim);
    hash_put(StmtToBdim, (char *) stmt, (char *) b_dim);

    initialize_mu_list(stmt, dim);
  }
  return(dmax);
}


/* ======================================================================== */
/*
 * void plc_make_distance():
 *
 * computes the distance equation associated to each dataflow of the DFG.
 *
 * The distance is : PI(sink, x) - PI(source, h(x))
 * where "sink" is the sink statement of the dataflow, "source" is the source
 * statement of the dataflow, "x" is the iteration vector of sink statement,
 * "h" is the transformation function of the dataflow and "PI" is the
 * placement function (or prototype).
 *
 * So, for one given dataflow, the computation needs both prototypes of sink
 * and source statements ("pp_sink" and "pp_source"), the transformations
 * ("trans_l") and the iteration vector of the source ("source_ind_l").
 *
 * Each distance is put in a hash table "DtfToDis".
 */
void plc_make_distance()
{
  extern graph the_dfg;
  extern hash_table StmtToProto, /* Mapping from a stmt to its prototype */
                    DtfToDist;	/* Mapping from a dataflow to its distance */

  list l, su_l, df_l, source_ind_l, trans_l, si_l;
  int source_stmt, sink_stmt;
  static_control source_stct;
  Ppolynome pp_source, pp_sink, pp_dist;

  /* We create the hash table. */
  DtfToDist = hash_table_make(hash_pointer, nb_dfs+1);

  for(l = graph_vertices(the_dfg); l != NIL; l = CDR(l)) {
    vertex v = VERTEX(CAR(l));

    source_stmt = vertex_int_stmt(v);
    source_stct = get_stco_from_current_map(adg_number_to_statement(source_stmt));

    /* Prototype of the source statement. */
    pp_source = (Ppolynome) hash_get(StmtToProto, (char *) source_stmt);

    /* Iteration vector of the source statement. */
    source_ind_l = static_control_to_indices(source_stct);

    su_l = vertex_successors(v);

    for( ; su_l != NIL; su_l = CDR(su_l)) {
      successor su = SUCCESSOR(CAR(su_l));
      vertex sink_v = successor_vertex(su);

      sink_stmt = vertex_int_stmt(sink_v);

      /* Prototype of the sink statement. */
      pp_sink = (Ppolynome) hash_get(StmtToProto, (char *) sink_stmt);

      df_l = dfg_arc_label_dataflows((dfg_arc_label) successor_arc_label(su));

      /* For each dataflow of the data flow graph we compute a distance. */
      for( ; df_l != NIL; df_l = CDR(df_l)) {
	Ppolynome aux_pp;
	predicate exec_domain, gov_pred;
	Psysteme impl_sc, elim_sc, sc_ed = SC_UNDEFINED,
	         sc_gp = SC_UNDEFINED, df_domain;
	list elim_vvs, impl_var, elim_var;

        dataflow df = DATAFLOW(CAR(df_l));

        /* Transformations of the dataflows. */
        trans_l = dataflow_transformation(df);

        /* There should be as much transformation expressions as source
	 * indices in the source iteration vector.
	 */
	pips_assert("plc_make_distance",
	     gen_length(trans_l) == gen_length(source_ind_l));

        if(get_debug_level() > 3) 
        {
          fprintf(stderr, "[plc_make_distance] \t for edge %d ->", source_stmt);
	  fprint_dataflow(stderr, sink_stmt, df);
          fprintf(stderr, "\n");
	}

        /* We now compute PI(source, h(x)). This is done by making the
	 * substitution, in the source prototype, of each index of the
	 * iteration vector of the source statement to the corresponding
	 * transformation expression.
	 *
	 * For this, we duplicate our polynome (source prototype) in
	 * "aux_pp". Then, for each index, we get its factor in "aux_pp",
	 * multiply it with the corresponding transformation, add it to
	 * "pp_dist" (initialized to POLYNOME_NUL) and eliminate the index
	 * in "aux_pp" (we substitute the index by POLYNOME_NUL). Then, we
	 * add "aux_pp" to "pp_dist", it contains the remnants of "pp_source"
	 * that are not factor of one of the indices.
         */
        pp_dist = POLYNOME_NUL;
	aux_pp = polynome_dup(pp_source);
	si_l = source_ind_l;
 if(get_debug_level() > 4) {
    fprintf(stderr, "[plc_make_distance] \t\tSource prototype:\n");
    polynome_fprint(stderr, aux_pp, pu_variable_name, pu_is_inferior_var);
    fprintf(stderr, "\n");
 }
        for( ; si_l != NIL; si_l = CDR(si_l), trans_l = CDR(trans_l)) {
	  entity var = ENTITY(CAR(si_l));
	  expression trans = EXPRESSION(CAR(trans_l));

          polynome_add(&pp_dist,
	    polynome_mult(expression_to_polynome(trans),
		       	  vecteur_to_polynome(prototype_factorize(aux_pp,
								  (Variable) var))));
/*
          polynome_add(&pp_dist,
	    polynome_mult(expression_to_polynome(trans),
		       	  polynome_factorize(aux_pp, (Variable) var, 1)));
*/

          aux_pp = prototype_var_subst(aux_pp, (Variable) var, POLYNOME_NUL);
/*
          di_polynome_var_subst_null(&aux_pp, var);
          aux_pp = polynome_var_subst(aux_pp, (Variable) var, POLYNOME_NUL);
*/

 if(get_debug_level() > 5) {
    fprintf(stderr, "[plc_make_distance] \t\t\tTransformation for index %s : %s\n",
           entity_local_name(var), words_to_string(words_expression(trans)));
    fprintf(stderr, "[plc_make_distance] \t\t\tCrt PI(source, h(x)):\n");
    polynome_fprint(stderr, pp_dist, pu_variable_name, pu_is_inferior_var);
    fprintf(stderr, "\n");
 }
	}
	polynome_add(&pp_dist, aux_pp);

 if(get_debug_level() > 5) {
    fprintf(stderr, "[plc_make_distance] \t\t\tPI(sink, x) :\n");
    polynome_fprint(stderr, pp_sink, pu_variable_name, pu_is_inferior_var);
    fprintf(stderr, "\n");
    fprintf(stderr, "[plc_make_distance] \t\t\tPI(source, h(x)) :\n");
    polynome_fprint(stderr, pp_dist, pu_variable_name, pu_is_inferior_var);
    fprintf(stderr, "\n");
 }

        /* We now compute the distance : D = PI(sink, x) - PI(source, h(x))
	 * Still, it is a polynome ("pp_dist").
	 */
	polynome_negate(&pp_dist);
	polynome_add(&pp_dist, pp_sink);

 if(get_debug_level() > 4) {
    fprintf(stderr, "[plc_make_distance] BEFORE IMPL ELIM \t\tDistance pp:\n");
    polynome_fprint(stderr, pp_dist, pu_variable_name, pu_is_inferior_var);
    fprintf(stderr, "\n");
 }

	/* We now eliminate variables in order to have free variables. This
         * is done using the implicit equations of the execution domain
	 * intersection the governing predicate.
         */
	exec_domain = VERTEX_DOMAIN(sink_v);
	gov_pred = dataflow_governing_pred(df);
        if(exec_domain == predicate_undefined) {
          if(gov_pred == predicate_undefined)
	    df_domain = sc_new();
          else
            df_domain = (Psysteme) predicate_system(gov_pred);
	}
        else {
          if(gov_pred == predicate_undefined)
	    df_domain = (Psysteme) predicate_system(exec_domain);
          else {
            df_domain = sc_new();
            sc_ed = (Psysteme) predicate_system(exec_domain);
	    sc_gp = (Psysteme) predicate_system(gov_pred);
	    df_domain = sc_intersection(df_domain, sc_ed, sc_gp);
	  }
        }

 if(get_debug_level() > 4) {
   fprintf(stderr, "[plc_make_distance] \t\t Exec domain: ");
   fprint_psysteme(stderr, sc_ed);
   fprintf(stderr, "[plc_make_distance] \t\t Gov pred   : ");
   fprint_psysteme(stderr, sc_gp);
   fprintf(stderr, "[plc_make_distance] \t\t Inter the 2: ");
   fprint_psysteme(stderr, df_domain);
 }

        impl_sc = find_implicit_equation(df_domain);

 if(get_debug_level() > 4) {
   fprintf(stderr, "[plc_make_distance] \t\t Impl sys   : ");
   fprint_psysteme(stderr, impl_sc);
 }

        if(impl_sc != NULL) {
	  impl_var = base_to_list(impl_sc->base);

	  elim_sc = elim_var_with_eg(impl_sc, &impl_var, &elim_var);
	  elim_vvs = make_vvs_from_sc(elim_sc, elim_var);

 if(get_debug_level() > 4) {
   fprintf(stderr, "[plc_make_distance] \t\t New subs  :\n");
   fprint_vvs(stderr, elim_vvs);
 }

	  pp_dist = vvs_on_polynome(elim_vvs, pp_dist);

	}
 if(get_debug_level() > 4) {
    fprintf(stderr, "[plc_make_distance] \t\tDistance pp:\n");
    polynome_fprint(stderr, pp_dist, pu_variable_name, pu_is_inferior_var);
    fprintf(stderr, "\n");
 }

        /* We put this polynome in the hash table. */
        hash_put(DtfToDist, (char *) df , (char *) pp_dist);

      }
    }
  }
}


/* ======================================================================== */
/*
 * Psysteme cutting_conditions(list df_l):
 *
 * returns a system of equations to be verified in order to zero out all
 * the distances associated to each dataflow of the list "df_l".
 *
 * The distance is taken from a hash table "DtfToDis".
 *
 * In fact, for each distance (i.e. dataflow), we create two systems: one that
 * nullified the factor of the loop indices (Mi) and one that nullified the
 * structure parameters and the constant term (Mp).
 *
 * For example, with I and J as indices, N as parameter, the distance
 *	I*(C1-C2) + J*(C3+C1) + N*(C2+C3) + C4-C1)
 * will leave the following systems :
 *	Mi = {C1-C2 = 0, C3+C1 = 0} 
 *      Mp = {C2+C3 = 0, C4-C1 = 0}
 * The C# are unknown coefficients created for the prototypes.
 *
 * The return system M is the union of this system so as to place first the
 * Mi systems (in decreasing weight order) and then the Mp systems (in 
 * decreasing weight order).
 */
Psysteme cutting_conditions(df_l)
list df_l;
{
  extern hash_table DtfToDist;	/* Mapping from a dataflow to its distance */
  extern list prgm_parameter_l;

  list l, sink_ind_l, sink_par_l;
  int sink_stmt;
  static_control sink_stct;
  Psysteme Mi, Mp, M;
  Ppolynome pp_dist;

  Mi = sc_new();
  Mp = sc_new();
  for(l = df_l; !ENDP(l); POP(l)) {
    dataflow df = DATAFLOW(CAR(l));
    Psysteme Mi_local, Mp_local;

    sink_stmt = (int) hash_get(DtfToSink, (char *) df);
    sink_stct = get_stco_from_current_map(adg_number_to_statement(sink_stmt));

    /* Iteration vector of the sink statement. */
    sink_ind_l = static_control_to_indices(sink_stct);

    /* Structure parameters. */
    /* sink_par_l = gen_append(static_control_params(sink_stct),
     * CONS(ENTITY, (entity) TCST, NIL)); */
    sink_par_l = gen_append(prgm_parameter_l,
			    CONS(ENTITY, (entity) TCST, NIL));

    pp_dist = polynome_dup((Ppolynome) hash_get(DtfToDist, (char *) df));

    /* We transforme this polynome ("pp_dist") into two systems of
     * equations, Mi and Mp (cf. above).
     */
    Mi_local = nullify_factors(&pp_dist, sink_ind_l, false);
    Mp_local = nullify_factors(&pp_dist, sink_par_l, true);

    polynome_rm(&pp_dist);

    if(get_debug_level() > 3) {
      fprintf(stderr, "[plc_make_distance] \tDistance Mi:\n");
      fprint_psysteme(stderr, Mi_local);
      fprintf(stderr, "[plc_make_distance] \tDistance Mp:\n");
      fprint_psysteme(stderr, Mp_local);
    }
    Mi = append_eg(Mi, Mi_local);
    Mp = append_eg(Mp, Mp_local);
  }
  M = append_eg(Mi, Mp);
  sc_normalize(M);
  return(M);
}


/* ======================================================================== */
/*
 * list sort_dfg_node(list l_nodes):
 *
 * returns the sorted list of the nodes of the list "l_nodes". The sorting
 * is based on the dimension of the nodes, in decreasing order.
 *
 * We need to compute for each node its dimension. This dimension is the
 * dimension of the iteration space of its associated statement, i.e. the
 * length of the list of the loop indices.
 */
list sort_dfg_node(l_nodes)
list l_nodes;
{
  extern hash_table StmtToDim;
  extern int nb_nodes;

  list l, new_l;

  StmtToDim = hash_table_make(hash_int, nb_nodes+1);

  /* For each node of the data flow graph we compute its dimension. */
  for(l = l_nodes; l != NIL; l = CDR(l)) {
    int stmt = vertex_int_stmt(VERTEX(CAR(l)));
    static_control stct = get_stco_from_current_map(adg_number_to_statement(stmt));
    list ind_l = static_control_to_indices(stct);

    hash_put(StmtToDim, (char *) stmt, (char *) gen_length(ind_l));
  }
  new_l = general_merge_sort(l_nodes, compare_nodes_dim);

  hash_table_free(StmtToDim);

  return(new_l);
}


/* ======================================================================== */
/*
 * void edge_weight():
 *
 * computes the weight of each  dataflows of DFG. If a
 * dataflow is a broadcast or a reduction, its weight is the dimension of
 * the space on which the data movement is done. Else it is the dimension
 * of its emitter set Ee = {y/ y = he(x), x in Pe}.
 * 
 * he: transformations
 * Pe: execution domain of the sink and governing predicate of the edge
 * y: indices of the sources
 * x: indices of the sink
 *
 * For the computation of the dimension of this emitter set, we eliminate x
 * by a combination of Gauss-Jordan and Fourier-Motzkin
 * algorithms. Ee is then defined by a set of inequalities for which we can
 * compute the set of implicit equations Eei. The dimension of Ee is then
 * equal to:
 *	Dim(y) - Card(Eei)
 */
void edge_weight()
{
  extern graph the_dfg;
  extern hash_table DtfToSink,
 		    DtfToWgh;

  list l, su_l, df_l;
  Psysteme sc_trans, sc_elim;
  predicate sink_domain;
  int source_stmt, sink_stmt, n_impl, poids;
  static_control sink_stct;

  /* We initialize the weight and sink statement hash tables */
  DtfToWgh = hash_table_make(hash_pointer, nb_dfs+1);
  DtfToSink = hash_table_make(hash_pointer, nb_dfs+1);

  /* For each dataflow of the data flow graph we compute its weight. */
  for(l = graph_vertices(the_dfg); l != NIL; l = CDR(l)) {
    vertex v = VERTEX(CAR(l));
    source_stmt = vertex_int_stmt(v);

    su_l = vertex_successors(v);

    for( ; su_l != NIL; su_l = CDR(su_l)) {
      successor su = SUCCESSOR(CAR(su_l));

      sink_stmt = vertex_int_stmt(successor_vertex(su));
      sink_stct = get_stco_from_current_map(adg_number_to_statement(sink_stmt));
      sink_domain = VERTEX_DOMAIN(successor_vertex(su));

      df_l = SUCC_DATAFLOWS(su);

      for(; df_l != NIL; df_l = CDR(df_l)) {
        dataflow df = DATAFLOW(CAR(df_l));

	if(is_broadcast_p(df) || is_reduction_p(df)) {
	  poids = communication_dim(df);
	}
	else {
	  predicate gov_pred;
	  list trans_l, si_l, aux_l;

	  gov_pred = dataflow_governing_pred(df);
	  trans_l = put_source_ind(dataflow_transformation(df));

          /* Transformation system of equations */
          sc_trans = make_expression_equalities(trans_l);

	  /* We append the execution domain of the sink. */
	  if(sink_domain != predicate_undefined)
	    sc_trans = sc_append(sc_trans, (Psysteme) predicate_system(sink_domain));

	  /* We append the governing predicate. "sc_trans" is what we called Ee
	   * (the emitter set, cf. above).
	   */
	  if(gov_pred != predicate_undefined)
	    sc_trans = sc_append(sc_trans, (Psysteme) predicate_system(gov_pred));

 if(get_debug_level() > 3) 
{
    fprintf(stderr, "[edge_weight] \tfor edge: %d ->", source_stmt);
    fprint_dataflow(stderr, sink_stmt, df);
    fprintf(stderr, "\n");
    fprintf(stderr, "[edge_weight] \ttrans system is:");
    fprint_psysteme(stderr, sc_trans);
 }

          /* Gauss-Jordan eliminations (with equalities). */
          si_l = static_control_to_indices(sink_stct);

          aux_l = NIL;

          sc_elim = elim_var_with_eg(sc_trans, &si_l, &aux_l);

 if(get_debug_level() > 5) {
    fprintf(stderr, "[edge_weight] \t\t\tElim equations are: ");
    fprint_psysteme(stderr, sc_elim);
    fprintf(stderr, "[edge_weight] \t\t\tAfter elim equations: ");
    fprint_psysteme(stderr, sc_trans);
 }
          /* Fourier-Motzkin eliminations (with inequalities). */
          for( ; si_l != NIL; si_l = CDR(si_l)) {
            entity var = ENTITY(CAR(si_l));
            sc_trans = sc_integer_projection_along_variable(sc_dup(sc_trans),
							    sc_trans,
							    (Variable) var);

            debug(7, "edge_weight", "\t\t\t\tElim ineq with %s\n",
		  entity_local_name(var));
          }

          /* We compute the number of implicit equations. */
	  n_impl = count_implicit_equation(sc_trans);

 if(get_debug_level() > 4) {
    fprintf(stderr, "[edge_weight] \t\tNumber of implicit equa is %d of: ", n_impl);
    fprint_psysteme(stderr, sc_trans);
 }
	  /* We compute the weight of the current dataflow. */
          poids = gen_length(trans_l) - n_impl;

        }

        debug(4, "edge_weight", "\tWeight of the edge: %d\n", poids);

        /* We update the hash tables keyed by the dataflow with its weight
	 * and its sink statement.
	 */
        hash_put(DtfToWgh, (char *) df , (char *) poids);
        hash_put(DtfToSink, (char *) df , (char *) sink_stmt);
      }
    }
  }
}


/* ======================================================================== */
/*
 * bool in_list_p(chunk *c, list l)
 *
 * returns true if "c" appears in "l".
 */
bool in_list_p(c, l)
chunk *c;
list l;
{
  list ll = l;
  bool not_found = true;

  for( ; !ENDP(ll) && (not_found); POP(ll) ) {
    if(c == CHUNK(CAR(ll)))
      not_found = false;
  }
  return(!not_found);
}

/* ======================================================================== */
/*
 * void add_to_list(chunk *c, list *l)
 *
 * Adds "c" in "l" if it does not appears in it yet.
 *
 * Note: "l" is a pointer to a list, the usage is add_to_list(c, &l);
 */
void add_to_list(c, l)
chunk *c;
list *l;
{
  list ll = *l;
  if(!in_list_p(c, ll))
    *l = CONS(CHUNK, c, ll);
}

/* ======================================================================== */
/*
 * int prototype_dimension(Ppolynome pp, list ind_l):
 *
 * returns the number of linearly independent vectors we can construct with
 * "pp" and "ind_l".
 *
 * Indeed, "pp" is a prototype, i.e. a 2-dimensional polynome without squared
 * variables. For instance (x.y + y.z) could be a prototype, but (x^2 + y.z)
 * could not.
 *
 * We distinguish three kinds of variables in this polynome: the indices (from
 * "ind_l"), the parameters and the coefficients. A monome of this polynome
 * can be of six kinds:
 *	_ a constant (integer value)
 *	_ a coefficient multiplied by a constant
 *	_ a parameter multiplied by a constant
 *	_ an index multiplied by a constant
 *	_ a parameter multiplied by a coefficient and a constant
 *	_ an index multiplied by a coefficient and a constant
 *
 * So "pp" may be represented as: pp = I.A.L + P.B.L + C.L + I.D + P.E + c,
 * where I and P are respectively row vectors of indices and of parameters,
 * A and B are 2-D integer matrices, L is a column vector of coefficients,
 * C, D and E are integer column vectors and c is an integer scalar.
 *
 * The goal of this function is to count how many independent vectors we can
 * have when giving values to the coefficients. These vectors represented as
 * linear combinations of the indices from "ind_l". So, only the monomes that
 * contain an index are useful for this computation: ppi = I.A.L + I.D
 *
 * So the number of independent vectors is also the number of independent
 * values of A.L, so it is rank(A).
 *
 * Note: this function take advantage of the special form of the prototype
 * polynome.
 */
int prototype_dimension(pp, ind_l)
Ppolynome pp;
list ind_l;
{
  int dim, n, m;
  Value det_q, det_p;
  matrice mat_A, mat_Z, mat_P, mat_Q, mat_H;
  list li;
  Pbase base_L;
  Psysteme ps_AL = sc_new();

if(get_debug_level() > 7) {
fprintf(stderr, "[prototype_dimension] Prototype : ");
polynome_fprint(stderr, pp, pu_variable_name, pu_is_inferior_var);
fprintf(stderr, "\n");
fprintf(stderr, "[prototype_dimension] Indices : ");
fprint_entity_list(stderr, ind_l);
fprintf(stderr, "\n");
}


  /* We construct A.L, in which each line correspond to the factor of one
   * index in "pp".
   */
  for(li = ind_l; !ENDP(li); POP(li)) {
    Pvecteur one_line = prototype_factorize(pp, (Variable) ENTITY(CAR(li)));
    sc_add_egalite(ps_AL, contrainte_make(one_line));
  }
  sc_creer_base(ps_AL);

  /* L is the list of variables contained in AL. */
  base_L = ps_AL->base;

if(get_debug_level() > 7) {
fprintf(stderr, "[prototype_dimension] ps_AL : ");
fprint_psysteme(stderr, ps_AL);
fprintf(stderr, "[prototype_dimension] base_L : ");
pu_vect_fprint(stderr, base_L);
fprintf(stderr, "\n");
}

  n = ps_AL->nb_eq;
  m = base_dimension(base_L);

  if( (n == 0) || (m == 0) )
    dim = 0;
  else {
    mat_A = matrice_new(n, m);
    mat_Z = matrice_new(n, 1);
    pu_contraintes_to_matrices(ps_AL->egalites, base_L, mat_A, mat_Z, n, m);
    mat_P = matrice_new(n, n);
    mat_Q = matrice_new(m, m);
    mat_H = matrice_new(n, m);
    matrice_hermite(mat_A, n, m, mat_P, mat_H, mat_Q, &det_p, &det_q);
    dim = dim_H(mat_H, n, m);

if(get_debug_level() > 7) {
fprintf(stderr, "[prototype_dimension] A of rank = %d : ", dim);
matrice_fprint(stderr, mat_A, n, m);
fprintf(stderr, "\n");
}

    matrice_free(mat_A);
    matrice_free(mat_Z);
    matrice_free(mat_P);
    matrice_free(mat_Q);
    matrice_free(mat_H);
  }

/* OLD VERSION */
/*
  for(ppp = pp; ppp != NULL; ppp = ppp->succ) {
    entity first = entity_undefined, second = entity_undefined;
    pv = (ppp->monome)->term;

    for(; (pv != NULL) && (second == entity_undefined); pv = pv->succ) {
      second = first;
      first = (entity) pv->var;
    }
    if(pv != NULL)
      pips_internal_error("Not a prototype polynome");

    if( (first != entity_undefined) && (second != entity_undefined) ) {
      if(in_list_p(first, ind_l)) {
	add_to_list(second, &ivf_l);
      }
      else if(in_list_p(second, ind_l)) {
	add_to_list(first, &ivf_l);
      }
    }
  }

  dim = gen_length(ivf_l);

  gen_free_list(ivf_l);
*/

  return(dim);
}


/* ======================================================================== */
/*
 * bool is_not_trivial_p(list vvs):
 *
 * returns true if all the prototypes are not trivial after applying "vvs".
 * Otherwise, returns FALSE.
 *
 * A prototype is not trivial if it depends on enough parameters that one may
 * construct the required number of linearly independent solutions.
 */
bool is_not_trivial_p(vvs)
list vvs;
{
  extern hash_table StmtToProto, /* Mapping from a statement to its prototype */
		    StmtToBdim;

  hash_table stp;

 list l, ind_l;
 int stmt, dim_plc, dim_bdt;
 Ppolynome pp;
 bool proto_not_triv = true;

  if(get_debug_level() > 5) {
    fprintf(stderr, "[is_not_trivial_p] \t\t\tSub is:\n");
    fprint_vvs(stderr, vvs);
    fprintf(stderr, "\n");
  }

  stp = hash_table_make(hash_int, nb_nodes+1);

  /* For each stmt in the dataflow graph we test if its proto is trivial */
  for(l = graph_vertices(the_dfg); (l != NIL) && proto_not_triv; l = CDR(l)) {
    vertex v = VERTEX(CAR(l));

    stmt = vertex_int_stmt(v);
    ind_l = static_control_to_indices(get_stco_from_current_map(adg_number_to_statement(stmt)));
    pp = polynome_dup((Ppolynome) hash_get(StmtToProto, (char *) stmt));

    dim_bdt = (int) hash_get(StmtToBdim, (char *) stmt);

    dim_plc = gen_length(ind_l) - dim_bdt;

  if(get_debug_level() > 6) {
    fprintf(stderr, "[is_not_trivial_p] \t\t\t\tProto is:");
    polynome_fprint(stderr, pp, pu_variable_name, pu_is_inferior_var);
    fprintf(stderr, "\n");
  }

    pp = vvs_on_polynome(vvs, pp);

  if(get_debug_level() > 5) {
    fprintf(stderr, "[is_not_trivial_p] must be of dim %d, After apply sub:", dim_plc);
    polynome_fprint(stderr, pp, pu_variable_name, pu_is_inferior_var);
    fprintf(stderr, "\n");
  }

    /* we carry on if the prototype is not trivial */
    if(prototype_dimension(pp, ind_l) < dim_plc)
      proto_not_triv = false;
    else
      hash_put(stp, (char *) stmt, (char *) pp);
  }

  if(get_debug_level() > 6) {
    fprintf(stderr, "[is_not_trivial_p] \t\t\t\tTriviality:");
    if(proto_not_triv)
       fprintf(stderr, "NON\n");
    else
       fprintf(stderr, "OUI\n");
  }

  if(proto_not_triv) {
    hash_table_free(StmtToProto);
    StmtToProto = stp;

    if(get_debug_level() > 3) {
      fprintf(stderr, "[is_not_trivial_p] New protos:\n");
      plc_fprint_proto(stderr, the_dfg, StmtToProto);
    }
  }
  else
    hash_table_free(stp);

 return(proto_not_triv);
}


/* ======================================================================== */
/*
 * list valuer(int dim, list xunks, list pcunks)
 *
 * valuation for the dimension "dim" of the placement function.
 *
 * "xunks" is the list of the INDEX coefficients not yet valuated: valuation
 * to 1 or 0.
 * "punks" is the list of the PARAM coefficients not yet valuated: valuation
 * using farkas lemma and PIP. These "punks" can be negative, so for this
 * resolution, we do the following: "punks = (p1, p2, p3) => p1 = a1 - a0,
 * p2 = a2 - a0, p3 = a3 - a0, with (a0, a1, a2, a3) positive. These a# are
 * called AUXIL coefficients, and "a0" is what we call "offset". Our list of
 * AUXIL coefficients is called "vl".
 */
list valuer(dim, xunks, pcunks)
int dim;
list xunks, pcunks;
{
  extern graph the_dfg;

  list l, ll, vl, var_l, sol, farkas_mult;
  int count_xc = 0, count_auxil_coeff = 0, stmt, count_farkas_mult;
  entity offset, nc;
  list new_vvs = NIL, vl_vvs;
  Psysteme pip_ps;
  Pcontrainte pc;
  Pvecteur pv_off;
  static_control stct;
  quast q_sol;
  quast_leaf ql;
  quast_value quv;

if(get_debug_level() > 3) {
fprintf(stderr, "\nPLC at dim %2d :\n===============\n", dim);
}

  offset = find_or_create_coeff(AUXIL_COEFF, count_auxil_coeff);
  pv_off = vect_new((Variable) offset, 1);
  vl = CONS(ENTITY, offset, NIL);

  for(l = xunks; l != NIL; l = CDR(l)) {
    list xus = CONSP(CAR(l));
    for(ll = xus; !ENDP(ll); POP(ll)) {
      entity e = ENTITY(CAR(ll));
      if(count_xc == dim)
        new_vvs = compose_vvs(new_vvs, make_vvs(e, 1, vect_new(TCST, 1)));
      else
	new_vvs = compose_vvs(new_vvs, make_vvs(e, 1, vect_new(TCST, 0)));
    }
    count_xc++;
  }

  for(l = pcunks; l != NIL; l = CDR(l)) {
    entity e = ENTITY(CAR(l));
    count_auxil_coeff++;
    nc = find_or_create_coeff(AUXIL_COEFF, count_auxil_coeff);
    vl = CONS(ENTITY, nc, vl);
    new_vvs = compose_vvs(new_vvs,
			  make_vvs(e, 1,
				   vect_cl2_ofl_ctrl(1,
						     vect_new((Variable) nc,1),
						     -1,pv_off,
						     NO_OFL_CTRL)));
  }

if(get_debug_level() > 3) {
fprintf(stderr, "\nNew subs:\n");
fprint_vvs(stderr, new_vvs);
fprintf(stderr, "\n");
}

  pip_ps = sc_new();
  farkas_mult = NIL;
  count_farkas_mult = 0;
  vl_vvs = NIL;
  for(l = graph_vertices(the_dfg); l != NIL; l = CDR(l)) {
    list vvs, elim_vvs, aux_vvs;
    Psysteme elim_ps, farkas_ps, sc_ed;
    Ppolynome farkas_pp, sub_pp, proto_pp;
    list init_l = NIL, elim_l = NIL, new_vl;
    vertex v = VERTEX(CAR(l));

    stmt = vertex_int_stmt(v);
    proto_pp = (Ppolynome) hash_get(StmtToProto, (char *) stmt);
    stct = get_stco_from_current_map(adg_number_to_statement(stmt));

    sub_pp = vvs_on_polynome(compose_vvs(vl_vvs, new_vvs),
			     polynome_dup(proto_pp));
    sc_ed = (Psysteme) predicate_system(VERTEX_DOMAIN(v));

if(get_debug_level() > 3) {
fprintf(stderr, "\nFarkas with %d : ", stmt);
polynome_fprint(stderr, sub_pp, pu_variable_name, pu_is_inferior_var);
fprint_psysteme(stderr, sc_ed);
fprint_entity_list(stderr, init_l);
fprintf(stderr, "\n");
}

    farkas_pp = apply_farkas(sub_pp, sc_ed, &init_l, &count_farkas_mult);

if(get_debug_level() > 3) {
fprintf(stderr, "\nFarkas DONE:\n");
polynome_fprint(stderr, farkas_pp, pu_variable_name, pu_is_inferior_var);
fprintf(stderr, "\nCoeff of Farkas: ");
fprint_entity_list(stderr, init_l);
fprintf(stderr, "\n");
}

    var_l = gen_concatenate(static_control_to_indices(stct),
			    prgm_parameter_l);

if(get_debug_level() > 3) {
fprintf(stderr, "\n List of Var to elim: ");
fprint_entity_list(stderr, var_l);
fprintf(stderr, "\n");
}

    farkas_ps = nullify_factors(&farkas_pp, var_l, true);

/*    (void) polynome_free(sub_pp);*/
    (void) polynome_free(farkas_pp);

if(get_debug_level() > 3) {
fprintf(stderr, "\nEquations :\n");
fprint_psysteme(stderr, farkas_ps);
fprintf(stderr, "\n");
}

    elim_ps = elim_var_with_eg(farkas_ps, &init_l, &elim_l);

if(get_debug_level() > 3) {
fprintf(stderr, "\nAfter ELIM LAMBDAS\n");
fprintf(stderr, "\nElim Coeff and System: ");
fprint_entity_list(stderr, elim_l);
fprint_psysteme(stderr, elim_ps);
fprintf(stderr, "Remaining Coeff and System :\n");
fprint_entity_list(stderr, init_l);
fprint_psysteme(stderr, farkas_ps);
fprintf(stderr, "\n");
}

    if(init_l != NIL)
      farkas_mult = gen_nconc(farkas_mult, init_l);

    aux_vvs = NIL;
    new_vl = gen_concatenate(vl, NIL);
    if(farkas_ps->nb_eq != 0) {
      list vl_elim = NIL;
      Psysteme vl_ps;

      vl_ps = elim_var_with_eg(farkas_ps, &new_vl, &vl_elim);

      aux_vvs = make_vvs_from_sc(vl_ps, vl_elim);

      vl_vvs = compose_vvs(vl_vvs, aux_vvs);

if(get_debug_level() > 3) {
fprintf(stderr, "\nAfter ELIM AUXIL\n");
fprintf(stderr, "\nElim Coeff and System: ");
fprint_entity_list(stderr, vl_elim);
fprint_psysteme(stderr, vl_ps);
fprintf(stderr, "Remaining Coeff and System :\n");
fprint_entity_list(stderr, new_vl);
fprint_psysteme(stderr, farkas_ps);
fprintf(stderr, "\n");
}
    }

    elim_vvs = make_vvs_from_sc(elim_ps, elim_l);

    for(vvs = elim_vvs ; vvs != NIL; vvs = CDR(vvs)) {
      var_val vv = VAR_VAL(CAR(vvs));
      Pvecteur pv = (Pvecteur) normalized_linear(expression_normalized(var_val_value(vv)));
      sc_add_inegalite(pip_ps, contrainte_make(vect_multiply(pv, -1)));
    }
    for(pc = farkas_ps->egalites; pc != NULL; pc = pc->succ) {
      sc_add_egalite(pip_ps, contrainte_dup(pc));
    }
    pip_ps = vvs_on_systeme(aux_vvs, pip_ps);
    sc_normalize(pip_ps);

if(get_debug_level() > 3) {
fprintf(stderr, "\nNEW PIP systeme :\n");
fprint_psysteme(stderr, pip_ps);
fprintf(stderr, "\nVL subs:\n");
fprint_vvs(stderr, vl_vvs);
fprintf(stderr, "\n");
}
  }

if(get_debug_level() > 3) {
fprintf(stderr, "\nNEW subs (before):\n");
fprint_vvs(stderr, new_vvs);
fprintf(stderr, "\n");
}

  new_vvs = vvs_on_vvs(vl_vvs, new_vvs);

if(get_debug_level() > 3) {
fprintf(stderr, "\nNEW subs:\n");
fprint_vvs(stderr, new_vvs);
fprintf(stderr, "\n");
}
  vect_rm(pip_ps->base);
  pip_ps->base = NULL;
  sc_creer_base(pip_ps);

  /* We sort the unknowns in order to have the auxiliary variables first */
  var_l = general_merge_sort(base_to_list(pip_ps->base), compare_coeff);
  pip_ps->base = list_to_base(var_l);

  q_sol = pip_integer_min(pip_ps, SC_EMPTY, pip_ps->base);

if(get_debug_level() > 3) {
fprintf(stderr, "\nSol to PIP sys :\n\tList of unks:");
fprint_entity_list(stderr, var_l);
fprintf(stderr, "\n");
imprime_quast(stderr, q_sol);
fprintf(stderr, "\n");
}

  if( (q_sol == quast_undefined) || (q_sol == NULL) )
    user_error("valuer", "Pip sol undefined\n");
  quv = quast_quast_value(q_sol);
  if( quv == quast_value_undefined )
    user_error("valuer", "Pip sol undefined\n");
  switch( quast_value_tag(quv)) {
    case is_quast_value_conditional:
      user_error("valuer", "Pip sol conditional\n");
    break;

    case is_quast_value_quast_leaf:
      ql = quast_value_quast_leaf( quv );
      sol = quast_leaf_solution(ql);
      for( ; sol != NIL; POP(sol), POP(var_l)) {
        expression exp = EXPRESSION(CAR(sol));
        entity var = ENTITY(CAR(var_l));
        Pvecteur pv;

        if(expression_constant_p(exp))
	  pv = vect_new(TCST, expression_to_int(exp));
        else {
          NORMALIZE_EXPRESSION(exp);
	  pv = (Pvecteur) expression_normalized(exp);
        }
        new_vvs = vvs_on_vvs(make_vvs(var, 1, pv), new_vvs);
      }
    break;
  }

if(get_debug_level() > 3) {
fprintf(stderr, "\nNEW subs (FINAL):\n");
fprint_vvs(stderr, new_vvs);
fprintf(stderr, "\n");
}

  return(new_vvs);
}



/* ======================================================================== */
/* void sort_unknowns(list *lambda, int dmax) :
 *
 * 
 */
void sort_unknowns(lambda, dmax)
list *lambda;
int dmax;
{
  extern hash_table StmtToProto;
  extern hash_table UnkToFrenq;

  list vl, xl = NIL, cl, newl = *lambda, pl = NIL;
  int nb_xl = 0, *frenq_tab, r;

  for(cl = newl; !ENDP(cl); POP(cl)) {
    entity u = ENTITY(CAR(cl));
    if(is_index_coeff_p(u)) {
      nb_xl++;
      xl = gen_nconc(xl, CONS(ENTITY, u, NIL));
    }
    else
      pl = CONS(ENTITY, u, pl);
  }
  newl = pl;
  if(nb_xl <= dmax) {
    for(; !ENDP(xl); POP(xl)) {
      entity e = ENTITY(CAR(xl));
      newl = CONS(ENTITY, e, newl);
    }
  }
  else {
    UnkToFrenq = hash_table_make(hash_pointer, ENT_HT_SIZE);
    frenq_tab = (int *) malloc(sizeof(int) * nb_xl);
    for(r = 0; r < nb_xl; r++) {frenq_tab[r] = 0;}

    for(vl = graph_vertices(the_dfg); vl != NIL; vl = CDR(vl)) {
      int stmt = vertex_int_stmt(VERTEX(CAR(vl)));
      Ppolynome pp = (Ppolynome) hash_get(StmtToProto, (char *) stmt);

      for(cl = xl, r = 0; cl != NIL; cl = CDR(cl), r++) {
        Pvecteur pv_fac = prototype_factorize(pp, (Variable) ENTITY(CAR(cl)));
        frenq_tab[r] += vect_size(pv_fac);
      }
    }
    for(cl = xl, r = 0 ; cl != NIL; cl = CDR(cl), r++) {
      entity ce = ENTITY(CAR(cl));
      hash_put(UnkToFrenq, (char *) ce, (char *) frenq_tab[r]);

      debug(5, "sort_unknowns", "Frenq of %s is %d\n",
            entity_local_name(ce), frenq_tab[r]);
    }
    newl = gen_nconc(general_merge_sort(xl, compare_unks_frenq), newl);

    free(frenq_tab);
    hash_table_free(UnkToFrenq);
  }
  *lambda = newl;
}


/* ======================================================================== */
list partition_unknowns(unks, dmax)
list *unks;
int dmax;
{
  extern hash_table StmtToProto;

  list l, xl = NIL, cl, xunks, pcunks, aux_xl, xul;
  int nb_xl = 0;

if(get_debug_level() > 4) {
fprintf(stderr, "[partition_unknowns] BEGIN with dmax = %d, and unks: ", dmax);
fprint_entity_list(stderr, *unks);
fprintf(stderr, "\n");
}

  pcunks = NIL;
  for(cl = *unks; !ENDP(cl); POP(cl)) {
    entity u = ENTITY(CAR(cl));
    if(is_index_coeff_p(u)) {
      nb_xl++;
      xl = gen_nconc(xl, CONS(ENTITY, u, NIL));
    }
    else
      pcunks = CONS(ENTITY, u, pcunks);
  }

if(get_debug_level() > 4) {
fprintf(stderr, "[partition_unknowns] pcunks: ");
fprint_entity_list(stderr, pcunks);
fprintf(stderr, "\n");
fprintf(stderr, "[partition_unknowns] %d in xl: ", nb_xl);
fprint_entity_list(stderr, xl);
fprintf(stderr, "\n");
}

  *unks = pcunks;
  xunks = NIL;
  if(nb_xl <= dmax) {
    for(; !ENDP(xl); POP(xl)) {
      entity e = ENTITY(CAR(xl));
      xunks = gen_nconc(xunks, CONS(CONSP, CONS(ENTITY, e, NIL), NIL));
    }
  }
  else {
    list rem_xl = gen_append(xl, NIL);
    for(l = graph_vertices(the_dfg); !ENDP(l) && !ENDP(rem_xl); POP(l)) {
      int stmt = vertex_int_stmt(VERTEX(CAR(l)));
      Ppolynome proto_pp = (Ppolynome) hash_get(StmtToProto, (char *) stmt);
      list lax, plax = NIL;

      aux_xl = gen_append(xl, NIL);

if(get_debug_level() > 4) {
fprintf(stderr, "[partition_unknowns] stmt %d, proto: ", stmt);
polynome_fprint(stderr, proto_pp, pu_variable_name, pu_is_inferior_var);
fprintf(stderr, "\n");
}

      for(cl = xl; cl != NIL; cl = CDR(cl)) {
        entity e = ENTITY(CAR(cl));
        Pvecteur pv_fac = prototype_factorize(proto_pp, (Variable) e);
        if(VECTEUR_NUL_P(pv_fac)) {
          gen_remove(&aux_xl, (chunk *) e);
	}
      }

if(get_debug_level() > 4) {
fprintf(stderr, "[partition_unknowns] xl of crt proto:");
fprint_entity_list(stderr, aux_xl);
fprintf(stderr, "\n");
}

      for(lax = xunks; !ENDP(lax) && !ENDP(rem_xl) && !ENDP(aux_xl); POP(lax)) {
        list crt_ax = CONSP(CAR(lax));
	bool not_found = true;
	entity e = entity_undefined;

	for(cl = aux_xl; !ENDP(cl) && not_found; cl = CDR(cl)) {
	  e = ENTITY(CAR(cl));
	  if(in_list_p((chunk *) e, crt_ax))
	    not_found = false;
	}
	if(not_found) {
	  for(cl = aux_xl; !ENDP(cl) && not_found; cl = CDR(cl)) {
	    e = ENTITY(CAR(cl));
	    if(in_list_p((chunk *) e, rem_xl))
	      not_found = false;
	  }
	  if(!not_found) {
	    CONSP(CAR(lax)) = gen_nconc(crt_ax, CONS(ENTITY, e, NIL));
            gen_remove(&rem_xl, (chunk *) e);
            gen_remove(&aux_xl, (chunk *) e);
	  }
	}
	else
          gen_remove(&aux_xl, (chunk *) e);
	plax = lax;
      }

if(get_debug_level() > 4) {
fprintf(stderr, "[partition_unknowns] xl of crt proto (again):");
fprint_entity_list(stderr, aux_xl);
fprintf(stderr, "\n");
fprintf(stderr, "[partition_unknowns] Remaining xl:");
fprint_entity_list(stderr, rem_xl);
fprintf(stderr, "\n");

fprintf(stderr, "[partition_unknowns] Crt unks (addition): ");
for(xul = xunks; !ENDP(xul); POP(xul)) {
  fprintf(stderr, "(");
  fprint_entity_list(stderr, CONSP(CAR(xul)));
  fprintf(stderr, ") ");
}
fprintf(stderr, "\n");
}

      if(!ENDP(rem_xl)) {
        for(cl = aux_xl; !ENDP(cl); cl = CDR(cl)) {
	  entity e = ENTITY(CAR(cl));
	  if(in_list_p((chunk *) e, rem_xl)) {
	    if(ENDP(plax)) {
	      xunks = CONS(CONSP, CONS(ENTITY, e, NIL), NIL);
	      plax = xunks;
	    }
	    else {
	      CDR(plax) = CONS(CONSP, CONS(ENTITY, e, NIL), NIL);
	      plax = CDR(plax);
	    }
	    gen_remove(&rem_xl, (chunk *) e);
	  }
        }

if(get_debug_level() > 4) {
fprintf(stderr, "[partition_unknowns] Remaining xl (again):");
fprint_entity_list(stderr, rem_xl);
fprintf(stderr, "\n");

fprintf(stderr, "[partition_unknowns] Crt unks (appendition): ");
for(xul = xunks; !ENDP(xul); POP(xul)) {
  fprintf(stderr, "(");
  fprint_entity_list(stderr, CONSP(CAR(xul)));
  fprintf(stderr, ") ");
}
fprintf(stderr, "\n");
}

      }
    }
  }
  return(xunks);
}


/* ======================================================================== */
/*
 * Psysteme system_inversion_restrict(Psysteme sys, list unks_l var_l par_l,
 *				      int nb_restrict, bool is_first):
 *
 * sys -> B.e, unks_l -> l, ps_res -> l.B^(-1), var_l -> e :
 *
 * a = B.e
 * m.a = l.e
 * =>
 * m = l.B^(-1)
 */
Psysteme system_inversion_restrict(sys, unks_l, var_l, par_l, nb_restrict, is_first)
Psysteme sys;
list unks_l, var_l, par_l;
int nb_restrict;
bool is_first;
{
  Psysteme full_ps;
  Pcontrainte new_pc;
  int n, m1, m2, r, d, i, j;
  matrice A, B, inv_A, Bz, R, Rt;

  full_ps = completer_base(sys, var_l, par_l);
  n = full_ps->nb_eq;
  m1 = gen_length(var_l);
  m2 = gen_length(par_l) + 1;

  A = matrice_new(n,n);
  B = matrice_new(n,m2);
  contraintes_with_sym_cst_to_matrices(full_ps->egalites,list_to_base(var_l),
				       list_to_base(par_l),A,B,n,n,m2);

  inv_A = matrice_new(n,n);
  matrice_general_inversion(A, inv_A, n);

  if(is_first) {
    r = nb_restrict;
    d = 0;
  }
  else {
    r = n - nb_restrict;
    d = nb_restrict;
  }

  R = matrice_new(n, r);
  for(i = 1; i <= n; i++)
    for(j = 1; j <= r; j++)
      ACCESS(R, n, i, j) = ACCESS(inv_A, n, i, j+d);
  R[0] = 1;

  Rt = matrice_new(r, n);
  matrice_transpose(R, Rt, n, r);

  Bz = matrice_new(r, 1);
  matrice_nulle(Bz, r, 1);

  pu_matrices_to_contraintes(&new_pc, list_to_base(unks_l), Rt, Bz, r, n);

  matrice_free(A);
  matrice_free(B);
  matrice_free(inv_A);
  matrice_free(Bz);
  matrice_free(R);
  matrice_free(Rt);

  return(sc_make(new_pc, NULL));
}


/* ======================================================================== */
bool solve_system_by_succ_elim(sys, sigma)
Psysteme sys;
list *sigma;
{
  bool result = true;
  list sig = *sigma, sigma_p;
  Pcontrainte leg;
  Pvecteur new_v;

  /* We walk through all the equations of M_ps. */
  for(leg = sys->egalites; leg != NULL; leg = leg->succ) {
    if(get_debug_level() > 3) {
      fprintf(stderr, "[solve_system_by_succ_elim] \tCrt equation:");
      pu_egalite_fprint(stderr, leg, pu_variable_name);
      fprintf(stderr, "\n");
    }

    /* We apply on this expression the substitution "sigma" */
    new_v = vvs_on_vecteur(sig, leg->vecteur);
    if(get_debug_level() > 1) {
      fprintf(stderr, "[solve_system_by_succ_elim] \t\tEqu after apply crt subs:");
      pu_vect_fprint(stderr, new_v);
      fprintf(stderr, "\n");
    }

    /* We create the elementary substitution with a variable not yet
     * eliminated.
     */
    sigma_p = plc_make_vvs_with_vector(new_v);
    if(get_debug_level() > 4) {
      fprintf(stderr, "[solve_system_by_succ_elim] \t\tSubs of non elim var:\n");
      fprint_vvs(stderr, sigma_p);
      fprintf(stderr, "\n");
    }
 
    if(sigma_p == NIL) {
      /*result = true;*/
    }
    /* We apply it on all prototypes, if none becomes trivial ... */
    else if(is_not_trivial_p(sigma_p)) {

      if(get_debug_level() > 3) {
        fprintf(stderr, "[solve_system_by_succ_elim] \tCrt local subs :\n");
        fprint_vvs(stderr, sigma_p);
        fprintf(stderr, "\n");
      }

      sig = compose_vvs(sigma_p, sig);
      /*result = true;*/
    }
    else
      result = false;
  }
  *sigma = sig;

  return(result);
}


/* ========================================================================= */
/*
 * bool constant_vecteur_p(Pvecteur pv)
 */
bool constant_vecteur_p(pv)
Pvecteur pv;
{
  if(pv == NULL)
    return(true);
  else
    return( (pv->succ == NULL) && (pv->var == TCST) );
}


/* ========================================================================= */
/* Psysteme broadcast_dimensions(placement pla, list mu_list)
 *
 */
Psysteme broadcast_dimensions(pla, mu_list)
placement pla;
list mu_list;
{
  list plc_dims, mu_l;
  Psysteme ps_bp;

  ps_bp = SC_EMPTY;
  mu_l = mu_list;
  plc_dims =  placement_dims(pla);
  if(plc_dims != NIL) {
    ps_bp = sc_new();

    for(; !ENDP(plc_dims); POP(plc_dims), POP(mu_l)) {
      Ppolynome crt_pp = (Ppolynome) CHUNK(CAR(plc_dims));
      Pvecteur pv;
      Variable crt_mu;

      crt_mu = (Variable) ENTITY(CAR(mu_l));
      pv = prototype_factorize(crt_pp, crt_mu);
      if(!constant_vecteur_p(pv))
        sc_add_egalite(ps_bp, contrainte_make(pv));
    }
  }
  return(ps_bp);
}


/* ======================================================================== */
/*
 * Psysteme completer_n_base(Psysteme sys dims_sys, list var_l par_l, int dim)
 *
 * idem as completer_base(), except that we add dimensions from dims_sys
 * until "sys" has "dim" vectors.
 */
Psysteme completer_n_base(sys, dims_sys, var_l, par_l, dim)
Psysteme sys, dims_sys;
list var_l, par_l;
int dim;
{
  Psysteme ps = sc_dup(sys), new_ps = sc_new();
  Pcontrainte pc;
  int crt_dim = sys->nb_eq;
  Pbase var_b, par_b;

  var_b = list_to_base(var_l);
  par_b = list_to_base(par_l);

  if(dim < crt_dim)
    pips_internal_error("There should not be so much dims");
  else if(dim == crt_dim)
    return(ps);

  for(pc = dims_sys->egalites; crt_dim < dim; pc = pc->succ) {
    Pvecteur pv = pc->vecteur;
    Psysteme aux_ps = sc_dup(ps);
    Psysteme aux_new_ps = sc_dup(new_ps);

    sc_add_egalite(aux_new_ps, contrainte_make(pv));
    aux_ps = append_eg(aux_ps, aux_new_ps);

    if(vecteurs_libres_p(aux_ps, var_b, par_b)) {
      new_ps = aux_new_ps;
      crt_dim++;
    }
    else
      sc_rm(aux_ps);
  }
  ps = append_eg(ps, new_ps);
  ps->base = NULL;
  sc_creer_base(ps);
  return(ps);
}


/* ========================================================================= */
/*
 * void pm_matrice_scalar_mult(int scal, matrice mat_M, int m n)
 */
void pm_matrice_scalar_mult(scal, mat_M, m, n)
int scal, m, n;
matrice mat_M;
{
  int i, j;
  for(i = 1; i <= m; i++) {
    for(j = 1; j <= n; j++) {
      ACCESS(mat_M, m, i, j) = scal * ACCESS(mat_M, m, i, j);
    }
  }
}


/* ========================================================================= */
/*
 * list partial_broadcast_coefficients(list var_l, list *used_mu)
 *
 * Takes into account the partial broadcast prototypes (contain in "pfunc") to
 * replace some of the coefficients in "var_l" by the "mu" coefficients
 * (used by these broadcast prototypes). The returned value the substitution
 * resulting of this computation. "used_mu" should be empty at the beginning
 * and is equal to the list of the mu coeff actually used.
 *
 * This computation is done on each statemwent.
 *
 * For a given statemwent s, we have to get the corresponding placement
 * object sp of the list contained in "pfunc" (a global variable). The
 * broadcast prototypes are contained in the field dims.
 * If this list is empty, we do not have anything to do. Else, this list
 * contained at most sd prototypes (sd is the dimension of the distribution
 * space for s). Each prototype has been associated to a Mu coefficient.
 * First, we have to construct the dimension not filled (if any). After this,
 * we will have sd prototypes as:
 *	for i in {1,...,sd}, bp_i = mu_i.eps_i
 * When summed:
 *	bp = J.M, where J is a row vector of broadcast directions and M is a
 * column vector of Mu coefficients.
 * These broadcast directions may be expressed with respect to the indices:
 *	J = I.P, where I is a row vector of indices and P is a 2-D matrice.
 *
 * Prototype of s may be represented as:
 *	pp = I.A.L + S.B.L + C.L + I.D + S.E + c, where S is a row vector of
 * parameters,  A and B are 2-D integer matrices, L is a column vector of
 * coefficients (those of "var_l"), C, D and E are integer column vectors
 * and c is an integer scalar.
 *
 * So, we have:
 *	I.A.L + I.D = J.M = I.P.M
 * i.e.:
 *	A.L + D = P.M
 *
 * So we have to construct the following system of equation:
 *	A.L + D - P.M == 0
 * From this system, we can express some of coefficients of L with respect to
 * the others (of L and M).
 *
 * 
 */
list partial_broadcast_coefficients(var_l, used_mu)
list var_l;
list *used_mu;
{
  extern plc pfunc;

  list plcs, new_vvs = NIL, uml = NIL;

  for(plcs = plc_placements(pfunc); !ENDP(plcs); POP(plcs)){
    placement crt_pla = PLACEMENT(CAR(plcs));
    list crt_dims = placement_dims(crt_pla);
    int crt_stmt = placement_statement(crt_pla);

if(get_debug_level() > 4) {
fprintf(stderr, "[partial_broadcast_coefficients] for stmt %d, with dims:\n", crt_stmt);
fprint_pla_pp_dims(stderr, crt_pla);
fprintf(stderr, "\n");
}

    if(crt_dims != NIL) {
      list mu_list, ind_l, par_l, vl, il, init_l, elim_l, vvs_L;
      Psysteme ps_bp, ps_pp_dims, new_ps, elim_ps;
      static_control stct;
      int i,j, sd, n, m, l;
      Pcontrainte new_pc;
      Ppolynome pp;
      matrice mat_Q, mat_Qc, mat_P, mat_Pc, mat_A;

      mu_list = (list) hash_get(StmtToMu, (char *) crt_stmt);
      uml = gen_append(mu_list, uml);
      stct = get_stco_from_current_map(adg_number_to_statement(crt_stmt));
      ind_l = static_control_to_indices(stct);
      par_l = gen_append(prgm_parameter_l, NIL);
      sd = (int) hash_get(StmtToPdim, (char *) crt_stmt);
      pp = (Ppolynome) hash_get(StmtToProto, (char *) crt_stmt);

      /* Construction of the directions that might be generated by the
       * prototype.
       */
      ps_pp_dims = sc_new();
      for(vl = var_l; !ENDP(vl); POP(vl)) {
        entity crt_var = ENTITY(CAR(vl));
        Pvecteur pv = prototype_factorize(pp, (Variable) crt_var);
	if(!VECTEUR_NUL_P(pv)) {
	  Psysteme aux_ps = sc_new();
          sc_add_egalite(aux_ps, contrainte_make(pv));
	  sc_creer_base(aux_ps);
	  ps_pp_dims = append_eg(ps_pp_dims, aux_ps);
	}
      }

if(get_debug_level() > 4) {
fprintf(stderr, "[partial_broadcast_coefficients] Prototype dir:\n");
fprint_psysteme(stderr, ps_pp_dims);
fprintf(stderr, "\n");
}

      /* Construction of -P.M */
      ps_bp = broadcast_dimensions(crt_pla, mu_list);

if(get_debug_level() > 4) {
fprintf(stderr, "[partial_broadcast_coefficients] Broadcast dir:\n");
fprint_psysteme(stderr, ps_bp);
fprintf(stderr, "\n");
}

      ps_bp = completer_n_base(ps_bp, ps_pp_dims, ind_l, par_l, sd);

if(get_debug_level() > 4) {
fprintf(stderr, "[partial_broadcast_coefficients] FULL Broadcast dir:\n");
fprint_psysteme(stderr, ps_bp);
fprintf(stderr, "\n");
}

      n = sd;
      m = gen_length(ind_l);
      mat_Q = matrice_new(n, m);
      mat_Qc = matrice_new(n, 1);
      pu_contraintes_to_matrices(ps_bp->egalites, list_to_base(ind_l),
				 mat_Q, mat_Qc, n, m);
      mat_P = matrice_new(m, n);
      mat_Pc = matrice_new(m, 1);
      matrice_nulle(mat_Pc, m, 1);
      matrice_transpose(mat_Q, mat_P, n, m);
      pm_matrice_scalar_mult(-1, mat_P, m, n);

if(get_debug_level() > 4) {
fprintf(stderr, "[partial_broadcast_coefficients] Matrix -P:\n");
matrice_fprint(stderr, mat_P, m, n);
fprintf(stderr, "\n");
}

      /* Construction of A.L + D */
      l = gen_length(var_l)+1;
      mat_A = matrice_new(m, l);
      for(il = ind_l, i=1; !ENDP(il); POP(il), i++) {
	entity crt_ind = ENTITY(CAR(il));
	Pvecteur pv = prototype_factorize(pp, (Variable) crt_ind);

        for(vl = var_l, j=1; !ENDP(vl); POP(vl), j++) {
	  entity crt_v = ENTITY(CAR(vl));
	  ACCESS(mat_A,m,i,j) = vect_coeff((Variable) crt_v, pv);
	}
	ACCESS(mat_A,m,i,l) = vect_coeff(TCST, pv);
      }
      DENOMINATOR(mat_A) = 1;

if(get_debug_level() > 4) {
fprintf(stderr, "[partial_broadcast_coefficients] Matrix A|D:\n");
matrice_fprint(stderr, mat_A, m, l);
fprintf(stderr, "\n");
}

      /* A.L + D - P.M */
      matrices_to_contraintes_with_sym_cst(&new_pc, list_to_base(mu_list),
					   list_to_base(var_l), mat_P, mat_A,
					   m, n, l);

      matrice_free(mat_Q);
      matrice_free(mat_Qc);
      matrice_free(mat_P);
      matrice_free(mat_Pc);
      matrice_free(mat_A);

      new_ps = sc_make(new_pc, NULL);

if(get_debug_level() > 4) {
fprintf(stderr, "[partial_broadcast_coefficients] A.L + D - P.M:\n");
fprint_psysteme(stderr, new_ps);
fprintf(stderr, "\t Var to eliminate : ");
fprint_entity_list(stderr, var_l);
fprintf(stderr, "\n");
}

      init_l = gen_append(var_l, NIL);
      elim_l= NIL;
      elim_ps = elim_var_with_eg(new_ps, &init_l, &elim_l);

if(get_debug_level() > 4) {
fprintf(stderr, "[partial_broadcast_coefficients] After ELIM LAMBDAs\n");
fprintf(stderr, "\tElim Coeff and System: ");
fprint_entity_list(stderr, elim_l);
fprint_psysteme(stderr, elim_ps);
fprintf(stderr, "\tRemaining Coeff and System : ");
fprint_entity_list(stderr, init_l);
fprint_psysteme(stderr, new_ps);
fprintf(stderr, "\n");
}

      vvs_L = make_vvs_from_sc(elim_ps, elim_l);

if(get_debug_level() > 4) {
fprintf(stderr, "[partial_broadcast_coefficients] Sub for elim lambdas:\n");
fprint_vvs(stderr, vvs_L);
fprintf(stderr, "\n");
}

      if(new_ps->nb_eq != 0) {
	Psysteme elim_ps2;
	list elim_l2, init_l2;
	list vvs_M;

	init_l2 = gen_append(mu_list, NIL);
	elim_l2 = NIL;
	elim_ps2 = elim_var_with_eg(new_ps, &init_l2, &elim_l2);

if(get_debug_level() > 4) {
fprintf(stderr, "[partial_broadcast_coefficients] After ELIM MUs\n");
fprintf(stderr, "\tElim Coeff and System: ");
fprint_entity_list(stderr, elim_l2);
fprint_psysteme(stderr, elim_ps2);
fprintf(stderr, "\tRemaining Coeff and System : ");
fprint_entity_list(stderr, init_l2);
fprint_psysteme(stderr, new_ps);
fprintf(stderr, "\n");
}

	vvs_M = make_vvs_from_sc(elim_ps2, elim_l2);

if(get_debug_level() > 4) {
fprintf(stderr, "[partial_broadcast_coefficients] Sub for elim mus:\n");
fprint_vvs(stderr, vvs_M);
fprintf(stderr, "\n");
}

	vvs_L = compose_vvs(vvs_L, vvs_M);

if(get_debug_level() > 4) {
fprintf(stderr, "[partial_broadcast_coefficients] Sub for MUs and LAMBDAs:\n");
fprint_vvs(stderr, vvs_L);
fprintf(stderr, "\n");
}
      }
      new_vvs = compose_vvs(new_vvs, vvs_L);

if(get_debug_level() > 4) {
fprintf(stderr, "[partial_broadcast_coefficients] Crt sub:\n");
fprint_vvs(stderr, new_vvs);
fprintf(stderr, "\n");
}
    }
  }
  *used_mu = uml;
  return(new_vvs);
}


/* ======================================================================== */
bool is_mu_coeff_p(e)
entity e;
{
  return(strncmp(entity_local_name(e), MU_COEFF, 4) == 0);
}


/* ========================================================================= */
/*
 * list get_mu_coeff(list sigma)
 */
list get_mu_coeff(sigma)
list sigma;
{
  list mu_list = NIL, vl;

  for(vl = sigma; !ENDP(vl); POP(vl)) {
    var_val vv = VAR_VAL(CAR(vl));
    entity c = var_val_variable(vv);

    if(is_mu_coeff_p(c))
      mu_list = CONS(ENTITY, c, mu_list);
  }
  return(mu_list);
}


/* ========================================================================= */
/*
 * void vvs_on_prototypes(list sigma)
 */
void vvs_on_prototypes(sigma)
list sigma;
{
  extern hash_table StmtToProto;
  extern graph the_dfg;

  list vl;

  for(vl = graph_vertices(the_dfg); vl != NIL; vl = CDR(vl)) {
    vertex v = VERTEX(CAR(vl));
    int stmt = vertex_int_stmt(v);
    Ppolynome pp = (Ppolynome) hash_get(StmtToProto, (char *) stmt);

    (void) hash_del(StmtToProto, (char *) stmt);
    hash_put(StmtToProto, (char *) stmt, (char *) vvs_on_polynome(sigma, pp));
  }
}


/* ========================================================================= */
/*
 * void prgm_mapping((char*) module_name):
 * 
 * It computes the placement function for all statement (i.e. nodes of "g").
 * This computation is done in three steps: initialization, edges treatment,
 * valuation.
 *
 * 1. The initialization consists in creating a prototype of placement
 * function for each statement (cf. plc_make_proto()), in computing the
 * placement function dimension and in computing the weight of the edges.
 *
 * 2. The edges treatment consists in determining some of the coefficients
 * of the prototypes using first the broadcast conditions and second the
 * distance conditions.
 *
 * 3. The valuation consists in determining the coefficients not yet
 * valuated and in building the dimensions of the placement function.
 */
bool prgm_mapping(module_name)
char*   module_name;
{
  extern plc pfunc;		/* The placement function */
  extern bdt the_bdt;		/* The timing function */
  extern graph the_dfg;
  extern int nb_nodes,		/* The number of nodes in the DFG */
	     nb_dfs;		/* The number of dataflows in the DFG */

  extern hash_table DtfToDist;	/* Mapping from a dataflow to its distance */
  extern hash_table StmtToProto;/* Mapping from a statement to its prototype */
  extern hash_table DtfToSink;
  extern hash_table DtfToWgh;

  extern list prgm_parameter_l;

  struct  tms             chrono1, chrono2; /* Perf.mesurement */
  statement_mapping STS;

  /* List of the unknowns coefficients */
  list lambda, xmu_lambda, mu_lambda, mu,
       sigma, sigma1, sigma2, sigma_p, *sigma3,
       su_l,
       sorted_df_l, l, remnants_df_l, df_l;
  Psysteme M_ps;
  int dmax, i;
  entity ent;
  static_control          stco;
  statement               mod_stat;
  char *md;

  /* Initialize debugging functions */
  debug_on("MAPPING_DEBUG_LEVEL");
  if(get_debug_level() > 0)
    fprintf(stderr, "\n\n *** COMPUTE MAPPING for %s\n", module_name);

  if(get_debug_level() > 1) {
    times(&chrono1);
  }

  /* We get the required data: module entity, code, static_control, dataflow
   * graph, timing function.
   */
  ent = local_name_to_top_level_entity( module_name );

  set_current_module_entity(ent);

  mod_stat = (statement) db_get_memory_resource(DBR_CODE, module_name, true);
  STS = (statement_mapping) db_get_memory_resource(DBR_STATIC_CONTROL,
						       module_name, true);
  stco     = (static_control) GET_STATEMENT_MAPPING(STS, mod_stat);
  if ( stco == static_control_undefined) {
    pips_internal_error("This is an undefined static control !");
  }
  if ( !static_control_yes( stco )) {
    pips_internal_error("This is not a static control program !");
  }

  set_current_stco_map(STS);

  prgm_parameter_l = static_control_params(stco);

  if(get_debug_level() > 2) {
    fprintf(stderr, "[prgm_mapping] Structure parameters of the program: ");
    fprint_entity_list(stderr, prgm_parameter_l);
    fprintf(stderr, "\n");
  }

  /* The DFG */
  the_dfg = adg_pure_dfg((graph) db_get_memory_resource(DBR_ADFG, module_name, true));

  /* the BDT */
  the_bdt = bdt_undefined;
  the_bdt = (bdt) db_get_memory_resource(DBR_BDT, module_name, true);

  if(get_debug_level() > 2) {
    fprint_dfg(stderr, the_dfg);
  }
  if(get_debug_level() > 0) {
    fprint_bdt(stderr, the_bdt);
  }
  /* First we count the number of nodes and dataflows */
  nb_nodes = 0;
  nb_dfs = 0;
  for(l = graph_vertices(the_dfg); !ENDP(l); POP(l)) {
    nb_nodes++;
    for(su_l = vertex_successors(VERTEX(CAR(l))); !ENDP(su_l); POP(su_l)) {
      for(df_l = SUCC_DATAFLOWS(SUCCESSOR(CAR(su_l))); df_l != NIL; df_l = CDR(df_l)) {
	nb_dfs++;
      }
    }
  }

  /* We look for the broadcasts */
  broadcast(the_dfg);

  /* We sort the nodes of "the_dfg" in decreasing dimension order. The
   * dimension of a node is the dimension of the iteration space of its
   * instruction.
   */
  graph_vertices(the_dfg) = sort_dfg_node(graph_vertices(the_dfg));

  if(get_debug_level() > 2)
{
    fprintf(stderr, "[prgm_mapping] Nodes order:");
    for(l = graph_vertices(the_dfg); ! ENDP(l); POP(l))
      fprintf(stderr, " %d,", vertex_int_stmt(VERTEX(CAR(l))));
    fprintf(stderr, "\n");
  }

/* INITIALIZATION */
  /* We create a prototype for each statement. Each prototype is mapped to
   * its statement in the hash table "StmtToProto". An other hash table
   * "StmtToProto" associates the unknown coefficients used in the
   * prototype and the statement. The returned value "lambda" gives all
   * the coefficients that have been created.
   */
  lambda = plc_make_proto();

  if(get_debug_level() > 2)
{
    fprintf(stderr, "[prgm_mapping] Nodes prototypes:\n");
    plc_fprint_proto(stderr, the_dfg, StmtToProto);
    fprintf(stderr, "[prgm_mapping] LAMBDAS: ");
    fprint_entity_list(stderr, lambda);
    fprintf(stderr, "\n");
  }

  /* plc_make_dim() has to initialize the Mu list*/

  /* We compute the dimension of the placement function of each instruction,
   * and the greatest one (dmax). This is based on the timing function, if it
   * exists.
   */
  count_mu_coeff = 1;
  dmax = plc_make_dim();

  /* The number of mapping dimensions can be computed as a minimum, see
   * plc_make_min_dim() */
  i = ((md = getenv("MINIMUM_DIMENSION")) != NULL) ? 1 : 0;
  if(i == 1) {
    int dmin;

    dmin = plc_make_min_dim();

    user_warning("prgm_mapping",
		 "Minimum number of dimensions: %d instead of %d\n",
		 dmin, dmax);

    dmax = dmin;
  }

  /* Mapping dimension can be fixed with the environment variable
   * MAPPING_DIMENSION */
  i = ((md = getenv("MAPPING_DIMENSION")) != NULL) ? atoi(md) : dmax;
  if(i != dmax) {
    user_warning("prgm_mapping",
		 "environment variable MAPPING_DIMENSION has set the mapping dimension to %d instead of %d\n",
		 i, dmax);

    dmax = i;
  }
  
  /* We initialize the prgm_mapping function. */
  pfunc = make_plc(NIL);
  for(l = graph_vertices(the_dfg); l != NIL; l = CDR(l)) {
    placement new_func;
    vertex v = VERTEX(CAR(l));
    int stmt = vertex_int_stmt(v);

    new_func = make_placement(stmt, NIL);
    plc_placements(pfunc) = gen_nconc(plc_placements(pfunc),
                                      CONS(PLACEMENT, new_func, NIL));
  }

  debug(3, "prgm_mapping", "DIM des fonctions de placement : %d\n", dmax);
  if(dmax == 0) {
    for(l = plc_placements(pfunc); !ENDP(l); POP(l)) {
      placement crt_func = PLACEMENT(CAR(l));
      placement_dims(crt_func) = CONS(EXPRESSION,
      				      int_to_expression(0),
				      NIL);
    }

    DB_PUT_MEMORY_RESOURCE(DBR_PLC, strdup(module_name), pfunc);
    reset_current_stco_map();
    reset_current_module_entity();
    debug_off();
    return(true);
  }

  /* Computation of the weight of each dataflow of the DFG. */
  edge_weight();

  /* We get all the dataflows of the graph */
  df_l = get_graph_dataflows(the_dfg);
  if(get_debug_level() > 5) {
    fprintf(stderr, "[prgm_mapping] Edges UNorder:\n");
    plc_fprint_dfs(stderr, df_l, DtfToSink, DtfToWgh);
  }

  /* We sort the dataflows in decreasing weight order */
  sorted_df_l = general_merge_sort(df_l, compare_dfs_weight);
  if(get_debug_level() > 2)
 {
    fprintf(stderr, "[prgm_mapping] Edges order:\n");
    plc_fprint_dfs(stderr, sorted_df_l, DtfToSink, DtfToWgh);
  }


/* EDGES TREATMENT */

/* BROADCAST CONDITIONS */
  /* We take into account the broadcast conditions */
  sigma = NIL;
  remnants_df_l = broadcast_conditions(lambda, sorted_df_l, &sigma);
  if(get_debug_level() > 2)
  {
    fprintf(stderr, "[prgm_mapping] Dif Red restriction:\n");
    fprint_vvs(stderr, sigma);
    fprintf(stderr, "[prgm_mapping] Remnants :\n");
    plc_fprint_dfs(stderr, remnants_df_l, DtfToSink, DtfToWgh);
  }

  for(sigma_p = sigma; !ENDP(sigma_p); POP(sigma_p))
    gen_remove(&lambda, (chunk *) var_val_variable(VAR_VAL(CAR(sigma_p))));

  if(get_debug_level() > 2)
  {
    fprintf(stderr, "[prgm_mapping] Prototypes after broadcast conditions:\n");
    plc_fprint_proto(stderr, the_dfg, StmtToProto);
    fprintf(stderr, "\n");
  }

  mu = NIL;
  sigma1 = partial_broadcast_coefficients(lambda, &mu);

  if(get_debug_level() > 3)
{
fprintf(stderr, "[prgm_mapping] ******* Partial broadcast sub:");
fprint_vvs(stderr, sigma1);
fprintf(stderr, "\n");
}

  vvs_on_prototypes(sigma1);

  if(get_debug_level() > 2) 
{
    fprintf(stderr, "[prgm_mapping] Prototypes after partial broadcast sub:\n");
    plc_fprint_proto(stderr, the_dfg, StmtToProto);
    fprintf(stderr, "\n");
}

  for(sigma_p = sigma1; !ENDP(sigma_p); POP(sigma_p)) {
    entity e = var_val_variable(VAR_VAL(CAR(sigma_p)));
    gen_remove(&lambda, (chunk *) e);
    gen_remove(&mu, (chunk *) e);
  }

if(get_debug_level() > 3) {
fprintf(stderr, "[prgm_mapping] ******* Remaining lambdas and Mus:\n");
fprintf(stderr, "\t LAMBDAs:");
fprint_entity_list(stderr, lambda);
fprintf(stderr, "\n");
fprintf(stderr, "\t MUs:");
fprint_entity_list(stderr, mu);
fprintf(stderr, "\n");
}

/*MOD : we could give "remnants_df_l" as an arg, in order to only compute the
useful distances. */

/* DISTANCE COMPUTATION */
  /* Computation of the distance of each dataflow. */
  plc_make_distance();
  if(get_debug_level() > 2) {
    fprintf(stderr, "[prgm_mapping] Edges distances:\n");
    plc_fprint_distance(stderr, the_dfg, DtfToDist);
  }

/* CUTTING CONDITIONS */
  /* We compute the list of equations that are to be nullified in order to zero
   * out all the distances.
   */
  M_ps = cutting_conditions(remnants_df_l);
  if(get_debug_level() > 2)
{
    fprintf(stderr, "[prgm_mapping] Matrix M:\n");
    fprint_psysteme(stderr, M_ps);
  }

  sigma2 = NIL;
  (void) solve_system_by_succ_elim(M_ps, &sigma2);

  if(get_debug_level() > 2) 
{
    fprintf(stderr, "Crt subs:\n");
    fprint_vvs(stderr, sigma2);
  }
    
if(get_debug_level() > 0) {
    fprintf(stderr, "[prgm_mapping] Prototypes after distance conditions:\n");
    plc_fprint_proto(stderr, the_dfg, StmtToProto);
    fprintf(stderr, "\n");
}

  /* We eliminate all the unknowns that are valuated by the substitution
   * computed above "sigma"; and then we cut it in two parts, one
   * containing the indices coefficients, one containing the parameters
   * coefficients, both sorted by decreasing frenquency in the
   * prototypes. Before cutting the list "lambda" into two parts, we take
   * into account the partial broadcast prototypes (contain in "pfunc") to
   * replace some of the "lambda" coeff by the "mu" coeff. The new sigma
   * is returned by this function.  */

  for(sigma_p = sigma2; !ENDP(sigma_p); POP(sigma_p)) {
    entity e = var_val_variable(VAR_VAL(CAR(sigma_p)));
    gen_remove(&lambda, (chunk *) e);
    gen_remove(&mu, (chunk *) e);
  }

if(get_debug_level() > 3) {
fprintf(stderr, "[prgm_mapping] ******* Remaining lambdas:");
fprint_entity_list(stderr, lambda);
fprintf(stderr, "\n");
fprintf(stderr, "[prgm_mapping] ******* Remaining mus:");
fprint_entity_list(stderr, mu);
fprintf(stderr, "\n");
}

  /* UNELIMINATED COEFF SORTING */
  sort_unknowns(&lambda, dmax);
  sort_unknowns(&mu, dmax);

if(get_debug_level() > 3) {
fprintf(stderr, "[prgm_mapping] ******* Sorted lambdas:");
fprint_entity_list(stderr, lambda);
fprintf(stderr, "\n");
fprintf(stderr, "[prgm_mapping] ******* Sorted mus:");
fprint_entity_list(stderr, mu);
fprintf(stderr, "\n");
}

  /* COEFF PARTITION */
  mu_lambda = gen_nconc(mu, lambda);
  xmu_lambda = partition_unknowns(&mu_lambda, dmax);

  if(get_debug_level() > 2) 
  {
    fprintf(stderr, "[prgm_mapping] \nRemaining unknowns\n");
    fprintf(stderr, "\tX COEFF: ");
    for(l = xmu_lambda; !ENDP(l); POP(l)) {
      fprintf(stderr, "(");
      fprint_entity_list(stderr, CONSP(CAR(l)));
      fprintf(stderr, ") ");
    }
    fprintf(stderr, "\tPC COEFF: ");
    fprint_entity_list(stderr, mu_lambda);
    fprintf(stderr, "\n");

fprint_plc_pp_dims(stderr, pfunc);
}

/* VALUATION */
  /* We valuate all the remaining unknowns by building successively each
   * dimension.
   */
  sigma3 = (list *) malloc(sizeof(list)*dmax);
  for(i = 0; i < dmax; i++) {
    list plcs;

    sigma3[i] = valuer(i, xmu_lambda, mu_lambda);

    if(get_debug_level() > 2) 
    {
      fprintf(stderr, "[prgm_mapping] Plc dim %d, new subs is\n", i);
      fprint_vvs(stderr, sigma3[i]);
      fprintf(stderr, "\n");
    }

    for(l = graph_vertices(the_dfg), plcs = plc_placements(pfunc); l != NIL;
        l = CDR(l), POP(plcs)) {
      placement crt_func = PLACEMENT(CAR(plcs));
      list dims;
      vertex v = VERTEX(CAR(l));
      int stmt = vertex_int_stmt(v);

      Ppolynome pp = polynome_dup((Ppolynome) hash_get(StmtToProto,
						       (char *) stmt));
      Ppolynome sub_pp = vvs_on_polynome(sigma3[i], pp);
      Pvecteur sub_vect = polynome_to_vecteur(sub_pp);
      expression exp = Pvecteur_to_expression(sub_vect);

      if(i == 0)
	dims = NIL;
      else
        dims = placement_dims(crt_func);

      if(exp == expression_undefined)
	exp = int_to_expression(0);

      placement_dims(crt_func) = gen_nconc(dims, CONS(EXPRESSION, exp, NIL));
    }
  }

  if(get_debug_level() > 0) {
    fprintf(stderr, "\n RESULT OF MAPPING:\n**************\n");
    df_l = get_graph_dataflows(the_dfg);
    for(i = 0; i < dmax; i++) {
      fprintf(stderr,
	      "Distance for dim %d\n=================================\n", i);
      
      for(l = df_l; !ENDP(l); POP(l)) {
	dataflow df = DATAFLOW(CAR(l));
	int stmt = (int) hash_get(DtfToSink, (char *) df);
	Ppolynome pp_dist = polynome_dup((Ppolynome) hash_get(DtfToDist,
							      (char *) df));

	pp_dist = vvs_on_polynome(sigma, pp_dist);
	pp_dist = vvs_on_polynome(sigma1, pp_dist);
	pp_dist = vvs_on_polynome(sigma2, pp_dist);
	pp_dist = vvs_on_polynome(sigma3[i], pp_dist);

	fprintf(stderr, "Dataflow ");
	fprint_dataflow(stderr, stmt, df);
	fprintf(stderr, "\tDist = ");
	polynome_fprint(stderr, pp_dist, pu_variable_name, pu_is_inferior_var);
	fprintf(stderr, "\n");
      }
    }
  }

  if(get_debug_level() > 1) {
    times(&chrono2);

    fprintf(stderr,
	    "\n*******\nTIMING:\n*******\n\tuser : %ld, system : %ld \n",
	    (long) chrono2.tms_utime - chrono1.tms_utime,
	    (long) chrono2.tms_stime - chrono1.tms_stime );
  }

  if(get_debug_level() > 0) {
    fprintf(stderr, "\n MAPPING:\n**************\n");
    fprint_plc(stderr, pfunc);
    fprintf(stderr, "\n\n *** MAPPING done\n");
  }

  DB_PUT_MEMORY_RESOURCE(DBR_PLC, strdup(module_name), pfunc);

  reset_current_stco_map();
  reset_current_module_entity();

  debug_off();

  return(true);
}
