





/* Package  :   C3/union
 * Author   :   Arnauld LESERVOT (leservot(a)limeil.cea.fr)
 * Date     :   
 * Modified :   04 04 95
 * Documents:   UNION.tex : ``Extension de C3 aux unions de polyedres''
 * Comments :
 */
/* 
 *                  WARNING
 * 
 *   THOSE FUNCTIONS ARE AUTOMATICALLY DERIVED 
 *    
 *           FROM THE WEB SOURCES !
 */

/* Ansi includes        */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <setjmp.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>
jmp_buf overflow_error;

/* Linear includes      */
#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "sommet.h"
#include "polyedre.h"
#include "union.h" 




extern char* (*union_variable_name)();



/* Psysteme sc_supress_same_constraints( in_ps1, in_ps2 ) supress in 
 * in_ps2 constraints that are in in_ps1. Nothing is shared, nor modified.
 * Returned Psysteme have only inequalities.
 */
Psysteme sc_supress_same_constraints( in_ps1, in_ps2 )
Psysteme in_ps1, in_ps2;
{
  Psysteme        ret_ps = NULL;
  Pcontrainte     eq, ineq;
  
  if ( in_ps1 == SC_RN ) return sc_dup(in_ps2);
  
  C3_DEBUG("sc_supress_same_constraints", {
    fprintf(stderr, "\nInput systems, in_ps1, then in_ps2:\n");  
    sc_fprint( stderr, in_ps1, union_variable_name );
    sc_fprint( stderr, in_ps2, union_variable_name );
  });

  /* Compare with equalities a == 0   <=>   a <= 0 and -a <= 0 */
  for (eq = in_ps2->egalites; eq != NULL; eq = eq->succ) {
    Pcontrainte  co, eq2;
    Pvecteur     pv;
    boolean      eq_in_ineq, co_in_ineq;

    if (contrainte_in_liste(eq, in_ps1->egalites)) continue;
    
    pv = vect_dup(eq->vecteur); 
    vect_chg_sgn        ( pv );
    co = contrainte_make( pv );
    if (contrainte_in_liste(co, in_ps1->egalites ))
      { co = contrainte_free( co ); continue; }


    eq_in_ineq = contrainte_in_liste(eq, in_ps1->inegalites);
    co_in_ineq = contrainte_in_liste(co, in_ps1->inegalites);
    
    if (eq_in_ineq && co_in_ineq) { 
      co = contrainte_free( co ); 
    }
    else if (eq_in_ineq) { /* add co to returned inegs */
      if (ret_ps != NULL){ sc_add_inegalite( ret_ps, co ); }
      else ret_ps = sc_make( NULL, co );
    }
    else if (co_in_ineq) { /* add eq to returned inegs */
      eq2 = contrainte_dup(eq);
      if (ret_ps != NULL){ sc_add_inegalite( ret_ps, eq2 ); }
      else ret_ps = sc_make( NULL, eq2 );
      co = contrainte_free( co ); 
    }
    else { /* add co and eq to returned inegs */
      eq2 = contrainte_dup(eq);
      if (ret_ps != NULL){ sc_add_inegalite( ret_ps, eq2 ); }
      else ret_ps = sc_make( NULL, eq2 );
      sc_add_inegalite( ret_ps, co );
    }
  }

  /* Compare with inequalities */
  for (ineq = in_ps2->inegalites; ineq != NULL; ineq = ineq->succ) {
    Pcontrainte co;
    if (contrainte_in_liste(ineq, in_ps1->inegalites)) continue;
    if (contrainte_in_liste(ineq, in_ps1->egalites))   continue;
    co = contrainte_make(vect_dup(ineq->vecteur)) ;
    if (ret_ps != NULL){ sc_add_inegalite( ret_ps, co ); }
    else ret_ps = sc_make( NULL, co );
  }
  
  if (ret_ps != NULL) 
    { vect_rm(ret_ps->base); ret_ps->base = NULL; sc_creer_base( ret_ps );}
  
  ret_ps = sc_normalize( ret_ps );
  C3_RETURN( IS_SC, ret_ps );
}



/* Psysteme sc_elim_redund_with_first_ofl_ctrl( in_ps1, in_ps2, ofl_ctrl )      
 * Returns constraints of in_ps2 which cut in_ps1. AL 06 04 95
 * It is assumed that in_ps1 and in_ps2 are feasible !
 * Nothing is shared, nor modified.
 */
Psysteme sc_elim_redund_with_first_ofl_ctrl(in_ps1, in_ps2, ofl_ctrl)
Psysteme in_ps1, in_ps2;
int      ofl_ctrl;
{
  Psysteme    ps1; 
  Pcontrainte eq, tail = NULL;
  Pbase       pb;

  /* Return on special cases */
  if ( sc_full_p(in_ps1) )    return in_ps2;
  if ( in_ps1->nb_ineq == 0 ) return in_ps2;

  /* debuging */
  C3_DEBUG("sc_elim_redund_with_first", {
    fprintf(stderr, "\nInput systems, in_ps1, then in_ps2:\n");  
    sc_fprint( stderr, in_ps1, union_variable_name );
    sc_fprint( stderr, in_ps2, union_variable_name );
  });


  /* build in_ps1.and.in_ps2 with sharing on in_ps2
   * This also works if in_ps1 is full space */
  ps1 = sc_dup( in_ps1 );
  for (eq = ps1->inegalites; eq != NULL; tail = eq, eq = eq->succ) {}
  tail->succ = in_ps2->inegalites;

  /* debuging */
  C3_DEBUG("sc_elim_redund_with_first", {
    fprintf(stderr, "ps1 old: nb_eq= %d, nb_ineq= %d, dimension= %d, base= \n", 
            ps1->nb_eq, ps1->nb_ineq, ps1->dimension);  
    vect_fprint(stderr, ps1->base, union_variable_name);
    fprintf(stderr, "in_ps2: nb_eq= %d, nb_ineq= %d, dimension= %d, base= \n", 
            in_ps2->nb_eq, in_ps2->nb_ineq, in_ps2->dimension);  
    vect_fprint(stderr, in_ps2->base, union_variable_name);
  });

  /* update information on ps1 */
  ps1->nb_eq     = ps1->nb_eq   + in_ps2->nb_eq;
  ps1->nb_ineq   = ps1->nb_ineq + in_ps2->nb_ineq;
  pb             = ps1->base;
  ps1->base      = base_union( ps1->base, in_ps2->base );
  ps1->dimension = vect_size ( ps1->base );
  vect_rm( pb );

  /* debuging */
  C3_DEBUG("sc_elim_redund_with_first", {
    fprintf(stderr, "ps1: nb_eq= %d, nb_ineq= %d, dimension= %d, base= \n", 
            ps1->nb_eq, ps1->nb_ineq, ps1->dimension);
    vect_fprint(stderr, ps1->base, union_variable_name);
  });

  /* Normalize 2 inputs systems */
  for (eq = ps1->inegalites; eq != NULL; eq=eq->succ) 
    vect_normalize(eq->vecteur);
  
  /* returns if there is no intersection */
  if (!sc_rational_feasibility_ofl_ctrl(ps1, ofl_ctrl, TRUE)) { 
    tail->succ = NULL;  ps1 = sc_free(ps1); 
    in_ps2 = sc_free(in_ps2); in_ps2 = sc_empty(NULL);
    C3_RETURN( IS_SC, in_ps2 ); 
  }
    

  /* We run over in_ps2 constraints (shared by ps1) 
   * and detect redundance */
  for (eq = tail->succ; eq != NULL; eq = eq->succ) {
    contrainte_reverse(eq);     
    if (sc_rational_feasibility_ofl_ctrl(ps1, ofl_ctrl, TRUE))
        {  contrainte_reverse(eq); }
    else{  
        eq_set_vect_nul(eq);    
        sc_elim_empty_constraints(in_ps2, FALSE);
        tail->succ = in_ps2->inegalites;
    }
  }

  if ( in_ps2->inegalites == NULL ) 
    { in_ps2 = sc_free(in_ps2);  in_ps2 = sc_full(); }

  tail->succ = NULL; ps1 = sc_free( ps1 ); 
  C3_RETURN( IS_SC, in_ps2 );
}



/* Ppath pa_supress_same_constraints( (Ppath) in_pa )   
 * Supress from complements of in_pa same constraints than those in
 * positif Psystem in_pa->psys. Returned path have no more equalities. AL050795
 * No sharing, no modification of inputs.
 */
Ppath pa_supress_same_constraints( in_pa )
Ppath in_pa;
{
  Ppath        ret_pa = PA_UNDEFINED;
  Pcomplist    comp;
  Psysteme     positif;
  Psyslist     psl = NULL;

  /* Special cases */
  if ( PA_UNDEFINED_P( in_pa )) return PA_UNDEFINED;
  if ( pa_empty_p    ( in_pa )) return pa_empty();
  if ( pa_full_p     ( in_pa )) return pa_full ();

  /* debuging */
  C3_DEBUG( "pa_supress_same_constraints", {
    fprintf(stderr, "Input path:\n");
    pa_fprint_tab(stderr, in_pa, union_variable_name, 1);
  });

  /* General case */
  positif = in_pa->psys;
  if (!sc_faisabilite_ofl(positif)) return pa_empty();
  
  for( comp = in_pa->pcomp; comp != NULL; comp = comp->succ) {
    Psysteme ps = sc_supress_same_constraints( positif, comp->psys );
    if (ps == NULL) 
      {psl = sl_free(psl); ret_pa = pa_empty(); C3_RETURN(IS_PA, ret_pa);}
    else psl = sl_append_system( psl, ps );
  }

  positif = sc_dup(positif); sc_transform_eg_in_ineg( positif );
  ret_pa  = pa_make( positif, (Pcomplist) psl );
  C3_RETURN(IS_PA, ret_pa);
}



/* Pdisjunct pa_path_to_disjunct_rule4_ofl_ctrl( (Ppath) in_pa, int ofl_ctrl)   
 * Returns the corresponding disjunction according rule 4. AL 05/16/95
 * No sharing.
 */
Pdisjunct pa_path_to_disjunct_rule4_ofl_ctrl( in_pa, ofl_ctrl )
Ppath in_pa;
int   ofl_ctrl;
{
  Pcomplist   comp, lcomp = NULL;
  Pdisjunct   ret_dj  ; 
  Psysteme    systeme ; 
  Ppath       pa      ; 
  int         pa_clength1, pa_clength2; 

  if (in_pa == PA_UNDEFINED) return DJ_UNDEFINED;
  if (pa_empty_p(in_pa))     return dj_empty();

  C3_DEBUG( "pa_path_to_disjunct_rule4_ofl_ctrl", {
    fprintf(stderr, "\n\n Input path:\n\n");
    pa_fprint(stderr, in_pa, union_variable_name );
  });


  if (pa_max_constraints_nb(in_pa) > PATH_MAX_CONSTRAINTS) 
    C3_RETURN(IS_DJ, pa_path_to_disjunct_ofl_ctrl( in_pa, ofl_ctrl));
    
  systeme = in_pa->psys;
  if (in_pa->pcomp == NULL) 
    C3_RETURN(IS_DJ, sl_append_system(NULL,sc_dup(systeme)));

  for( comp = in_pa->pcomp; comp != NULL; comp = comp->succ ) {
    Psysteme ps = sc_dup(comp->psys);
    if (ps == SC_UNDEFINED) 
      { sl_free(lcomp); C3_RETURN( IS_DJ, DJ_UNDEFINED ); }

    ps = sc_elim_redund_with_first( systeme, ps );

    if (sc_empty_p( ps )) { ps = sc_free(ps); continue; }
    if (sc_full_p ( ps ))  
      { ps = sc_free(ps); C3_RETURN( IS_DJ, dj_empty() ); }

    lcomp = sl_append_system( lcomp, ps );
  }
 
  pa          = pa_make(sc_dup(in_pa->psys), lcomp); 
  pa_clength1 = sl_length( pa->pcomp );
  pa          = pa_reduce_simple_complement( pa );
  pa_clength2 = sl_length( pa->pcomp );
  systeme     = pa->psys;


  /* Returns according to different cases */
  if (pa_clength2 == 0) 
       { ret_dj = dj_append_system(NULL,sc_dup(systeme)); } 
  else if (pa_clength1 != pa_clength2)  /* we've modified P0 systeme */
       { ret_dj = pa_path_to_disjunct_rule4_ofl_ctrl( pa, ofl_ctrl); } 
  else { ret_dj = pa_path_to_disjunct_ofl_ctrl( pa, ofl_ctrl); }

  pa = pa_free( pa );

  C3_RETURN( IS_DJ, ret_dj );
}



/* 

   
   Pdisjunct    pa_path_to_few_disjunct_ofl_ctrl( (Ppath) in_pa, (int) ofl_ctrl )  
   Produces a Pdisjunct corresponding to the path Ppath and
   reduces the number of disjunctions.
   See "Extension de C3 aux Unions de Polyedres" Version 2,
   for a complete explanation about this function.
   in_pa is modified.              AL 23/03/95
   

 
*/
Pdisjunct pa_path_to_few_disjunct_ofl_ctrl( in_pa, ofl_ctrl )
Ppath   in_pa;
int     ofl_ctrl;
{
  

   Psysteme  systeme;  Pdisjunct ret_dj = DJ_UNDEFINED; 

          Ppath pa; Pcomplist lcomp;

          
  Pcontrainte common_cons = NULL, cons, cons_oppose = NULL; 
  Pvecteur    vect_1, cons_pv = NULL;
  Pcomplist   comp;
  

          
  Pcontrainte     common_cons_oppose;
  Psysteme        common_ps, common_ps_oppose;
  Ppath           pa1, pa2; 
  boolean         pa1_empty  = FALSE, pa2_empty  = FALSE;
  boolean         pa1_filled = FALSE, pa2_filled = FALSE;
  

          Pdisjunct  dj1 = NULL, dj2 = NULL; 



  

  
  C3_DEBUG( "pa_path_to_few_disjunct_ofl_ctrl", {
    fprintf(stderr, "\n\n Input path:\n\n");
    pa_fprint(stderr, in_pa, union_variable_name );
  });
  
  if (PA_UNDEFINED_P( in_pa ))   return DJ_UNDEFINED;
  if (pa_full_p     ( in_pa ))   return dj_full();
  if (pa_empty_p    ( in_pa ))   return dj_empty();
    
  /* If it's an empty path or if it has no complements : return */ 
  systeme = in_pa->psys ; 
  if (!sc_faisabilite_ofl( systeme )) return dj_empty();
  if (in_pa->pcomp == NULL) return (Pdisjunct) sl_append_system(NULL,sc_dup(systeme));
  


  

  
  pa      = pa_make(sc_dup(systeme), sl_dup(in_pa->pcomp)); 
  pa      = pa_reduce_simple_complement( pa );
  
  if (pa_empty_p(pa)) {pa = pa_free(pa); return dj_empty();}
  
  pa      = pa_transform_eg_in_ineg    ( pa ); 
  lcomp   = pa->pcomp ; 
  systeme = pa->psys  ;
  
  C3_DEBUG( "pa_path_to_few_disjunct_ofl_ctrl", {
    fprintf(stderr, "pa:\n");
    pa_fprint_tab(stderr, pa, union_variable_name, 1 );
  });
  
  if ( pa->pcomp == NULL ) { 
    pa     = pa_free1(pa); 
    ret_dj = (Pdisjunct) sl_append_system(NULL, systeme);
    
    C3_DEBUG( "pa_path_to_few_disjunct_ofl_ctrl", {
      fprintf(stderr, "No complement, returning:\n");
      dj_fprint_tab(stderr, ret_dj, union_variable_name, 1 );
    });
    
    return ret_dj;
  }
  
  
  /* We are looking for a common hyperplan */
  vect_1 = vect_new(TCST, 1); common_cons = NULL;
  
  for(cons = (lcomp->psys)->inegalites;
          (cons != NULL)&&(lcomp->succ != NULL);cons = cons->succ){
    boolean is_common = TRUE;
    cons_pv           = vect_dup( cons->vecteur ); vect_chg_sgn( cons_pv );
    cons_oppose       = contrainte_make(vect_add( cons_pv, vect_1 )); 
  
    for(comp = lcomp->succ;(comp != NULL) && is_common; comp = comp->succ){
      Pcontrainte ineg = (comp->psys)->inegalites;
      boolean     is_common1, is_common2;
  
      is_common1 = contrainte_in_liste( cons,        ineg );
      is_common2 = contrainte_in_liste( cons_oppose, ineg );
      is_common  = is_common1 || is_common2;
    }
    if (!is_common) { 
      /* removes cons_pv and vect_dup(vect_1) */
      cons_oppose = contrainte_free(cons_oppose);
      vect_rm( cons_pv ); cons_pv = (Pvecteur) NULL;
      continue; 
    }
    common_cons = cons;
    vect_chg_sgn( cons_pv );
    break;
  } 
  
  C3_DEBUG( "pa_path_to_few_disjunct_ofl_ctrl", {
    fprintf(stderr, "cons_pv: ");
    if (common_cons == NULL) fprintf(stderr, "NULL\n"); 
    else vect_fprint(stderr, cons_pv, union_variable_name); 
  });
  


  
   
  
  if( common_cons != NULL ) {
    

    
    common_ps          = sc_make( CONTRAINTE_UNDEFINED, contrainte_make(cons_pv) );
    cons_pv            = vect_dup( common_cons->vecteur ); vect_chg_sgn( cons_pv );
    common_cons_oppose = contrainte_make(vect_add(cons_pv,vect_1));
    common_ps_oppose   = sc_make( CONTRAINTE_UNDEFINED, common_cons_oppose );
    pa1 = pa_new(); pa2= pa_new();
    
    for(comp = lcomp; comp != NULL; comp = comp->succ){
      Psysteme     local_ps;
      Pcontrainte  co = comp->psys->inegalites;
      
      if (!pa1_empty && contrainte_in_liste(common_cons, co)) {
        local_ps = sc_supress_same_constraints( common_ps, comp->psys );
        if (local_ps == SC_EMPTY) { pa1 = pa_empty(); pa1_empty = TRUE; continue;}
        pa1->pcomp = sl_append_system( pa1->pcomp, local_ps ); pa1_filled = TRUE;
      }
      else if(!pa2_empty &&  contrainte_in_liste(common_cons_oppose, co)) {
        local_ps = sc_supress_same_constraints( common_ps_oppose, comp->psys );
        if (local_ps == SC_EMPTY) {pa2 = pa_empty(); pa2_empty = TRUE; continue;}
        pa2->pcomp = sl_append_system( pa2->pcomp, local_ps ); pa2_filled = TRUE;
      }
    }
    

  
    /* 21/11/95: added by BC - Patch from AL */
   dj1 = dj_empty();
   dj2 = dj_empty();
   /* end of patch */

    /* 21/11/95: removed by BC - Patch from AL */
    /* if (!pa1_filled && !pa2_filled) { dj1 = dj_empty(); dj2 = dj_empty(); } */
    
    if (pa1_filled) {
      /* take care of rule 2 */
      if (pa_full_p( pa2 )) pa1->psys = sc_dup( systeme );
      else pa1->psys = sc_append( sc_dup(common_ps), systeme );
    
      C3_DEBUG("pa_path_to_few_disjunct", {
        fprintf(stderr, "pa1:\n");  
        pa_fprint_tab( stderr, pa1, union_variable_name, 1 );
      });
    
      /* 21/11/95: changed by BC - Patch from AL */
    /* if (!pa_full_p(pa2)&&!sc_faisabilite_ofl(pa1->psys)) dj1 = dj_empty();
      else dj1 = pa_path_to_few_disjunct_ofl_ctrl(pa1, ofl_ctrl); */
      if (pa_full_p(pa2)||sc_faisabilite_ofl(pa1->psys)) 
      {dj_free(dj1);dj1 = pa_path_to_few_disjunct_ofl_ctrl(pa1, ofl_ctrl);}
    }
    
    if (pa2_filled) {
      /* take care of rule 2 */
      if (pa_full_p( pa1 )) pa2->psys = sc_dup( systeme );
      else pa2->psys = sc_append( sc_dup(common_ps_oppose), systeme );
    
      C3_DEBUG("pa_path_to_few_disjunct", {
        fprintf(stderr, "pa2:\n");  
        pa_fprint_tab( stderr, pa2, union_variable_name, 1 );
      });

      /* 21/11/95: changed by BC - Patch from AL */
      /* if (!pa_full_p(pa1)&&!sc_faisabilite_ofl(pa2->psys)) dj2 = dj_empty(); 
      else dj2 = pa_path_to_few_disjunct_ofl_ctrl( pa2, ofl_ctrl ); */
      if (pa_full_p(pa1)||sc_faisabilite_ofl(pa2->psys)) 
      {dj_free(dj2); dj2 = pa_path_to_few_disjunct_ofl_ctrl( pa2, ofl_ctrl );}
    }
    
    ret_dj = dj_union( dj1, dj2 ); 
    
    /* Manage memory, free:
     * cons_oppose, common_ps, common_ps_oppose, 
     * cons_pv, vect_1, pa1, pa2
     */
    cons_oppose      = contrainte_free( cons_oppose );
    common_ps        = sc_free( common_ps );
    common_ps_oppose = sc_free( common_ps_oppose );
    vect_rm(cons_pv); cons_pv = NULL;
    pa1 = pa_free(pa1);   pa2 = pa_free(pa2); 
    

  
  }
  else { 
    

     ret_dj = pa_path_to_disjunct_rule4_ofl_ctrl( pa, ofl_ctrl ); 
    

  
  }
  
  /* Manage memory */
  pa = pa_free(pa); vect_rm(vect_1); vect_1 = NULL;
  
  return ret_dj;
  


}



/* boolean pa_inclusion_p(Psysteme ps1, Psysteme ps2)   BA, AL 31/05/94
 * returns TRUE if ps1 represents a subset of ps2, false otherwise
 * Inspector (no sharing on memory).
 */
boolean pa_inclusion_p_ofl_ctrl(ps1, ps2, ofl_ctrl)
Psysteme ps1, ps2;
int ofl_ctrl;
{
  boolean   result;
  Ppath     chemin = pa_make(ps1, sl_append_system(NULL, ps2));
  
  if (setjmp(overflow_error)) {result = FALSE; }
  else {result = ! (pa_feasibility_ofl_ctrl(chemin, ofl_ctrl));}
  chemin = pa_free1(chemin); 
  return(result);
}



/* boolean pa_system_equal_p(Psysteme ps1, Psysteme ps2) BA
 */
boolean pa_system_equal_p_ofl_ctrl(ps1,ps2, ofl_ctrl)
Psysteme ps1, ps2;
int ofl_ctrl;
{
    return (  pa_inclusion_p_ofl_ctrl(ps1,ps2, ofl_ctrl) && 
              pa_inclusion_p_ofl_ctrl(ps2,ps1, ofl_ctrl) );
}



/* Pdisjunct pa_system_difference_ofl_ctrl(ps1, ps2)
 * input    : two Psystemes
 * output   : a disjunction representing ps1 - ps2
 * modifies : nothing
 * comment  : algorihtm : 
 *      chemin = ps1 inter complement of (ps2)
 *      ret_dj = dj_simple_inegs_to_eg( pa_path_to_few_disjunct(chemin) )
 */
Pdisjunct pa_system_difference_ofl_ctrl(ps1, ps2, ofl_ctrl)
Psysteme ps1,ps2;
int      ofl_ctrl;
{
  Ppath     chemin;
  Pdisjunct dj, ret_dj;
  
  if ((ps1 == SC_UNDEFINED)||(ps2 == SC_UNDEFINED)) return DJ_UNDEFINED; 
  if (sc_empty_p(ps2)) return sl_append_system(NULL,sc_dup(ps1));
  if (sc_empty_p(ps1)) return dj_empty();
  
  chemin  =  pa_make(ps1, sl_append_system(NULL,ps2));
  dj      =  pa_path_to_few_disjunct_ofl_ctrl(chemin, ofl_ctrl);
  chemin  =  pa_free1( chemin );
  ret_dj  =  dj_simple_inegs_to_eg( dj );
  dj      =  dj_free( dj );
  return ret_dj;
}



/* boolean pa_convex_hull_equals_union_p(conv_hull, ps1, ps2)
 * input    : two Psystems and their convex hull        AL,BC 23/03/95
 * output   : TRUE if ps1 U ps2 = convex_hull, FALSE otherwise
 * modifies : nothing
 * comment  : complexity = nb_constraints(ps1) * nb_constraints(ps2)    
 *            if ofl_ctrl = OFL_CTRL, conservatively returns ofl_ctrl 
 *            when an overflow error occurs
 */
boolean pa_convex_hull_equals_union_p_ofl_ctrl
            (conv_hull, ps1, ps2, ofl_ctrl, ofl_res)
Psysteme  conv_hull, ps1, ps2;
int       ofl_ctrl;
boolean   ofl_res;
{
  Ppath    chemin;
  boolean  result;
  int      local_ofl_ctrl = (ofl_ctrl == OFL_CTRL)?FWD_OFL_CTRL:ofl_ctrl;
  
  chemin = pa_make(conv_hull,sl_append_system(sl_append_system(NULL,ps1),ps2));
  
  if ((ofl_ctrl==OFL_CTRL)&&setjmp(overflow_error)) result = ofl_res;
  else result = !(pa_feasibility_ofl_ctrl(chemin, local_ofl_ctrl));
  chemin = pa_free1(chemin);
  return(result);
}



