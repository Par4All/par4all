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
/* Name     :   adg_prettyprint.c
 * Package  :   array_dfg
 * Author   :   Arnauld LESERVOT and Alexis PLATONOFF
 * Date     :   93/06/27
 * Modified :
 * Documents:   "Implementation du Data Flow Graph dans PIPS"
 * Comments :
 */

#define GRAPH_IS_DG
#include "local.h"

#define IS_INEG 0
#define IS_EG 1
#define IS_VEC 2

/*============================================================================*/
/* void fprint_dfg(FILE *fp, graph obj): prints in the file "fp" the Data
 * Flow Graph "obj".
 */
void fprint_dfg(fp, obj)
FILE *fp;
graph obj;
{
 list nodes_l, su_l, df_l;
 predicate exec_dom;
 int source_stmt, sink_stmt, source_nb, sink_nb;

 fprintf(fp,"\n Array Data Flow Graph:\n");
 fprintf(fp,"=======================\n");

 for(nodes_l = graph_vertices(obj); nodes_l != NIL; nodes_l = CDR(nodes_l))
   {
    vertex crt_v = VERTEX(CAR(nodes_l));

    source_stmt = vertex_int_stmt(crt_v);
    exec_dom = dfg_vertex_label_exec_domain((dfg_vertex_label) vertex_vertex_label(crt_v));

    if(source_stmt == ENTRY_ORDER) {
      source_nb = 0;
      fprintf(fp,"\nENTRY:\n******\n");
    }
    else {
      source_nb = source_stmt-BASE_NODE_NUMBER;
      fprintf(fp,"\nINS_%d:\n********\n", source_nb);
    }

    if(exec_dom != predicate_undefined) {
      fprintf(fp, " Execution Domain for %d:\n", source_nb);
      fprint_pred(fp, exec_dom);
    }
    else fprintf(fp, " Execution Domain for %d: Nil\n", source_nb);
    fprintf(fp, "\n");

    su_l = vertex_successors(crt_v);

    for( ; su_l != NIL; su_l = CDR(su_l))
      {
       successor su = SUCCESSOR(CAR(su_l));
       vertex su_ver = successor_vertex(su);

       sink_stmt = vertex_int_stmt(su_ver);
       sink_nb = sink_stmt-BASE_NODE_NUMBER;
       df_l = dfg_arc_label_dataflows((dfg_arc_label) successor_arc_label(su));

       for( ; df_l != NIL; df_l = CDR(df_l))
          fprint_dataflow(fp, sink_nb, DATAFLOW(CAR(df_l)));

       /* AP, Nov 28th 1995: Ronan asked me to add the execution domain of
          the destination. Only done if it is not equal to the source. */
       if(sink_nb != source_nb) {
	 exec_dom = dfg_vertex_label_exec_domain((dfg_vertex_label) vertex_vertex_label(su_ver));
	 if(exec_dom != predicate_undefined) {
	   fprintf(fp, "  Execution Domain for %d:\n", sink_nb);
	   fprint_pred(fp, exec_dom);
	 }
       }

       fprintf(fp, "\n");
      }
   }
}

#define ADFG_EXT ".adfg_file"

/*============================================================================*/
bool print_array_dfg( module_name )
const char* module_name;
{
  char *localfilename;
  FILE        *fd;
  char        *filename;
  graph the_dfg;

  debug_on( "PRINT_ARRAY_DFG__DEBUG_LEVEL" );

  if (get_debug_level() > 1)
    user_log("\n\n *** PRINTING ARRAY DFG for %s\n", module_name);

  the_dfg = (graph) db_get_memory_resource(DBR_ADFG, module_name, true);

  localfilename = strdup(concatenate(module_name, ADFG_EXT, NULL));
  filename = strdup(concatenate(db_get_current_workspace_directory(), 
				"/", localfilename, NULL));
  
  fd = safe_fopen(filename, "w");
  fprint_dfg(fd, the_dfg);
  safe_fclose(fd, filename);
  
  DB_PUT_FILE_RESOURCE(DBR_ADFG_FILE, strdup(module_name), localfilename);
  
  free(filename);
  
  debug_off();
  
  return(true);

}

/*============================================================================*/
/* void imprime_special_quast(fp, qu)				AL 04/11/93
 * Input  : An output file fp and a quast qu.
 * Output : Print in fp quast qu with real number of leal_labels.
 */
void imprime_special_quast(fp, qu)
FILE *fp;
quast qu;
{
  Psysteme        paux = NULL;
  predicate       pred_aux = predicate_undefined;
  conditional     cond_aux = NULL;
  quast_value     quv = NULL;
  quast_leaf      qul = NULL;
  leaf_label      ll = NULL;
  list            sol = NULL;

  if( (qu == quast_undefined) || (qu == NULL) ) {
    fprintf(fp, " quast_vide ");
    return;
  }
  quv = quast_quast_value(qu);
  if( quv == quast_value_undefined ) {
    fprintf(fp, " quast_value_undefined\n");
    return;
  }
  quv = quast_quast_value(qu);

  switch( quast_value_tag(quv)) {
    case is_quast_value_conditional:
      cond_aux = quast_value_conditional(quv);
      pred_aux = conditional_predicate(cond_aux);
      if (pred_aux != predicate_undefined)
          paux = (Psysteme)  predicate_system(pred_aux);

      fprintf(fp, "\nIF\n");
      fprint_psysteme(fp, paux);

      fprintf(fp, "\nTHEN\n");
      imprime_special_quast(fp, conditional_true_quast (cond_aux) );
      fprintf(fp, "\nELSE\n");
      imprime_special_quast(fp, conditional_false_quast (cond_aux) );
      fprintf(fp, "\nFI\n");
    break;

    case is_quast_value_quast_leaf:
      qul = quast_value_quast_leaf( quv );
      if (qul == quast_leaf_undefined) {
	fprintf(fp,"quast_leafundefined\n");
	break;
      }
      sol = quast_leaf_solution( qul );
      ll  = quast_leaf_leaf_label( qul );
      if (ll != leaf_label_undefined) {
        fprintf(fp, "Leaf label number : %d     ", leaf_label_statement(ll) );
        fprintf(fp, "Depth : %d\n", leaf_label_depth(ll) );
      }
      fprintf(fp, "Solutions : ");
      while (sol != NIL) {
	fprintf(fp, " %s, ", 
		    words_to_string( words_expression(EXPRESSION(CAR(sol)))));
        sol = CDR(sol);
      }
      fprintf(fp, "\n");
    break;
  }
}

/*============================================================================*/
/* static void adg_contrainte_fprint(FILE *fp, Pcontrainte c, int is_what,
 *				     char *(*variable_name)()):
 * 
 * prints in the file "fp" the constraint "c", of type equality, inequality or
 * vector according to the value of the integer argument "is_what", using the
 * function "variable_name" for the name of the variables.
 *
 * The function contrainte_fprint() exists, it is defined in contrainte.c (C3
 * library). We redefine this function because:
 *
 * 	1. we want the constant term in the left hand side
 * 	2. we want a third type of contrainte (vector) which does not print the
 *         inequality or equality symbol.
 *	3. we do not want a line feed at the end of the constraint
 *
 * We consider that CONTRAINTE_UNDEFINED => CONTRAINTE_NULLE
 *
 * Results for a constraint containing the following Pvecteur (2*I) (-J) (-4):
 *
 *   equality:		2 * I - J - 4 = 0
 *   inequality:	2 * I - J - 4 <= 0
 *   vector:		2 * I - J - 4
 *
 */
static void adg_contrainte_fprint(fp,c,is_what,variable_name)
FILE *fp;
Pcontrainte c;
int is_what;
char * (*variable_name)();
{
  Pvecteur v = NULL;
  short int debut = 1;
  Value constante = VALUE_ZERO;

  if (!CONTRAINTE_UNDEFINED_P(c)) v = contrainte_vecteur(c);
  else v = VECTEUR_NUL;

  if(!vect_check(v)) pips_internal_error("Non coherent vector");

  while (!VECTEUR_NUL_P(v)) {
    if (v->var!=TCST) {
      char signe = (char) NULL;
      Value coeff = v->val;

      if (value_notzero_p(coeff)) {
	if (value_pos_p(coeff)) signe = (debut) ? ' ' : '+';
	else { signe = '-'; value_oppose(coeff); };
	debut = 0;
	if (value_one_p(coeff)) 
	    fprintf(fp,"%c %s ", signe, variable_name(v->var));
	else 
	    { fprintf(fp,"%c ", signe);
	      fprint_Value(fp, coeff);
	      fprintf(fp, " %s ", variable_name(v->var)); }
      }
    }
    /* on admet plusieurs occurences du terme constant!?! */
    else value_addto(constante, v->val);
    
    v = v->succ;
  }
  
  /* sign */
  if (value_pos_p(constante))
      fprintf(fp, "+ ");
  else if (value_neg_p(constante))
      value_oppose(constante), fprintf(fp, "- ");

  /* value */
  if (value_notzero_p(constante))
      fprint_Value(fp, constante), fprintf(fp, " ");

  /* trail */
  if (is_what == IS_INEG)
      fprintf (fp,"<= 0 ,");
  else if(is_what == IS_EG) 
      fprintf (fp,"== 0 ,");
  else /* IS_VEC */ 
      fprintf (fp," ,");
}

/*============================================================================*/
/* void adg_inegalite_fprint(FILE *fp, Pcontraint ineg,
 *			     char *(*variable_name)()):
 * Redefinition of inegalite_fprint(). See adg_contrainte_fprint() for details.
 */
void adg_inegalite_fprint(fp,ineg,variable_name)
FILE *fp;
Pcontrainte ineg;
char * (*variable_name)();
{ adg_contrainte_fprint(fp,ineg,IS_INEG,variable_name);}

/*============================================================================*/
/* void adg_egalite_fprint(FILE *fp, Pcontraint eg, char *(*variable_name)()):
 * Redefinition of egalite_fprint(). See adg_contrainte_fprint() for details.
 */
void adg_egalite_fprint(fp,eg,variable_name)
FILE *fp;
Pcontrainte eg;
char * (*variable_name)();
{
 adg_contrainte_fprint(fp,eg,IS_EG,variable_name);
}

/*============================================================================*/
/* void adg_vecteur_fprint(FILE *fp, Pcontraint vec char *(*variable_name)()):
 * See adg_contrainte_fprint() for details.
 */
void adg_vecteur_fprint(fp,vec,variable_name)
FILE *fp;
Pcontrainte vec;
char * (*variable_name)();
{ adg_contrainte_fprint(fp,vec,IS_VEC,variable_name);}

/*============================================================================*/
/* void adg_fprint_dataflow(FILE *fp, int sink, dataflow df): prints in
 * the file "fp" the dataflow "df" with a sink statement "sink".
 */
void adg_fprint_dataflow(fp, sink, df)
FILE *fp;
int sink;
dataflow df;
{
  list trans_l = dataflow_transformation(df);
  reference ref = dataflow_reference(df);
  predicate gov_pred = dataflow_governing_pred(df);
  communication comm = dataflow_communication(df);

  fprintf(fp,"\t -> ins_%d:%s [", sink, words_to_string(words_reference(ref)));
  adg_fprint_list_of_exp(fp, trans_l);
  fprintf(fp,"]\n");
 
  fprintf(fp,"\t\t gov pred ");
  if(gov_pred != predicate_undefined) adg_fprint_pred(fp, gov_pred);
  else fprintf(fp, "{nil} \n");
  
  if(comm == communication_undefined) {/*fprintf(fp, "\t\t Communication general \n");*/}
  else {
    predicate pred = NULL;
    
    pred = communication_broadcast(comm);
    if(pred != predicate_undefined){
      fprintf(fp,"\t\t Vecteur(s) de diffusion :");
      adg_fprint_sc_pvecteur(fp, (Psysteme) predicate_system(pred));
    }

    pred = communication_reduction(comm);
    if(pred != predicate_undefined){
      fprintf(fp,"\t\t Vecteur(s) de reduction :");
      adg_fprint_sc_pvecteur(fp, (Psysteme) predicate_system(pred));
    }

    pred = communication_shift(comm);
    if(pred != predicate_undefined){
      fprintf(fp,"\t\t Vecteur(s) de shift :");
      adg_fprint_sc_pvecteur(fp, (Psysteme) predicate_system(pred));
    }
  }
}

/*============================================================================*/
/* void adg_fprint_dfg(FILE *fp, graph obj): prints in the file "fp" the Data
 * Flow Graph "obj".
 */
void adg_fprint_dfg(fp, obj)
FILE *fp;
graph obj;
{
  list nodes_l = NULL, su_l = NULL, df_l = NULL;
  predicate exec_dom = NULL;
  int source_stmt = (int) NULL, sink_stmt = (int) NULL;

  fprintf(fp,"\n Array Data Flow Graph:\n");
  fprintf(fp,"=======================\n");

  for(nodes_l = graph_vertices(obj); nodes_l != NIL; nodes_l = CDR(nodes_l)){
    vertex crt_v = VERTEX(CAR(nodes_l));
    
    source_stmt = vertex_int_stmt(crt_v);
    exec_dom = dfg_vertex_label_exec_domain((dfg_vertex_label) vertex_vertex_label(crt_v));

    fprintf(fp,"ins_%d: ", source_stmt);
    if(exec_dom != predicate_undefined) adg_fprint_pred(fp, exec_dom);
    else fprintf(fp, "{nil} \n");

    su_l = vertex_successors(crt_v);

    for( ; su_l != NIL; su_l = CDR(su_l)){
      successor su = SUCCESSOR(CAR(su_l));

      sink_stmt = vertex_int_stmt(successor_vertex(su));
      df_l = dfg_arc_label_dataflows((dfg_arc_label) successor_arc_label(su));

      for( ; df_l != NIL; df_l = CDR(df_l))
	adg_fprint_dataflow(fp, sink_stmt, DATAFLOW(CAR(df_l)));
    }
  }
}


/*============================================================================*/
/* void adg_fprint_list_of_exp(FILE *fp, list exp_l): prints in the file "fp"
 * the list of expression "exp_l". We separate the expressions with a colon
 * (","). We do not end the print with a line feed.
 */
void adg_fprint_list_of_exp(fp, exp_l)
FILE *fp;
list exp_l;
{
  list aux_l = NULL;
  expression exp = NULL;

  for(aux_l = exp_l; aux_l != NIL; aux_l = CDR(aux_l)) {
    exp = EXPRESSION(CAR(aux_l));
    fprintf(fp,"%s", words_to_string(words_expression(exp)));
    if(CDR(aux_l) != NIL)   fprintf(fp,",");
  }
}

/*============================================================================*/
/* void adg_fprint_pred(FILE *fp, predicate pred): prints in the file "fp" the
 * predicate "pred".
 */
void adg_fprint_pred(fp, pred)
FILE *fp;
predicate pred;
{
  Psysteme ps = NULL; 

  debug(9,"adg_fprint_pred", "begin\n");
  if ((pred != predicate_undefined) && (pred != (predicate) NIL)) {
    ps = (Psysteme) predicate_system(pred);
    adg_fprint_psysteme(fp, ps);
  }
  else fprintf(fp, "{nil} \n");
  debug(9,"adg_fprint_pred", "end\n");
}

/*============================================================================*/
/* void adg_fprint_psysteme(FILE *fp, Psysteme ps): prints in the file "fp" the
 * Psysteme "ps". Each constraint is printed either with adg_inegalite_fprint()
 * or adg_egalite_fprint(), both redefined above. See adg_contrainte_fprint()
 * for details.
 */
void adg_fprint_psysteme(fp, ps)
FILE *fp;
Psysteme ps;
{
 Pcontrainte peq;

 if (ps != NULL) {
   fprintf(fp,"{ ");
   for (peq = ps->inegalites; peq!=NULL;
	adg_inegalite_fprint(fp,peq,entity_local_name),peq=peq->succ);
   for (peq = ps->egalites; peq!=NULL;
	adg_egalite_fprint(fp,peq,entity_local_name),peq=peq->succ);
   fprintf(fp," } \n");
 }
 else fprintf(fp,"(nil)\n");
}

/*============================================================================*/
/* void adg_fprint_sc_pvecteur(FILE *fp, Psysteme ps): prints in the file "fp"
 * the Psysteme "ps" as a list of vectors. Each constraint is printed with
 * adg_vecteur_fprint() defined above. See adg_contrainte_fprint() for details.
 */
void adg_fprint_sc_pvecteur(fp, ps)
FILE *fp;
Psysteme ps;
{
  Pcontrainte peq = NULL;

  if (ps != NULL) {
    fprintf(fp,"{ ");
    for (peq = ps->inegalites; peq!=NULL;
       	 adg_vecteur_fprint(fp,peq,entity_local_name),peq=peq->succ);
    for (peq = ps->egalites; peq!=NULL;
	 adg_vecteur_fprint(fp,peq,entity_local_name),peq=peq->succ);
    fprintf(fp," } \n");
  }
  else fprintf(fp,"(nil)\n");
}


/*============================================================================*/
/* void  adg_fprint_predicate_list( file* fp, list sc_l )	AL 15/07/93
 * Input   : 	pointer on file fp, list of psysteme sc_l
 * Output  :  	A prettyprint of psystemes.
 */
void 	adg_fprint_predicate_list( fp, sc_l )
FILE* fp;
list  sc_l;
{
  predicate pred = NULL;

  debug(9, "adg_fprint_predicate_list", "begin\n");
  for(; !ENDP( sc_l ); POP( sc_l )) {
    pred = PREDICATE(CAR( sc_l ));
    adg_fprint_psysteme( fp, (Psysteme) predicate_system( pred ));
  }
  debug(9, "adg_fprint_predicate_list", "end\n");
}

/*============================================================================*/
