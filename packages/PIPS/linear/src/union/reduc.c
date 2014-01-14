
#line 1366 "reduc.w"


#line 335 "UNION.w"

/* Package  :  C3/union
 * Author   :  Arnauld LESERVOT (leservot(a)limeil.cea.fr)
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
#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>

/* Linear includes      */
#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "sommet.h"
#include "polyedre.h"
#include "union.h" 


#line 1367 "reduc.w"


#line 148 "reduc.w"

static char* hspara_string[10] __attribute__ ((unused)) = 
{  
  "unpara",
  /**/                
    "sszero",   
    "ssplus",   
  /**/     
    /**/     
      /**/     
        "ssminus",    
      /**/            
        "opzero",        
        "opplus",    
    "keep",
    /**/          
      "opminus", 
    "empty",
  "full"
};

static enum  hspara_elem       
             hspara_jm[10][10] = { /* Lower left is join, upper right is meet */

 /*join\meet   unpara sszero ssplus ssminus opzero opplus keep opminus  empty full  */
 /* unpara  */ {  0,    0,     0,      0,     0,    0,      0,     0,     0,    0    },
 /* sszero  */ {  1,    1,     1,      0,     0,    0,      0,     0,     0,    1    },
 /* ssplus  */ {  2,    2,     2,      0,     0,    0,      0,     0,     0,    2    },
 /* ssminus */ {  3,    9,     9,      3,     0,    0,      3,     0,     3,    3    },
 /* opzero  */ {  4,    9,     9,      6,     4,    4,      4,     0,     4,    4    },
 /* opplus  */ {  5,    9,     9,      6,     5,    5,      5,     0,     5,    5    },
 /* keep    */ {  6,    9,     9,      6,     6,    6,      6,     0,     6,    6    },
 /* opminus */ {  7,    9,     9,      8,     8,    8,      8,     7,     7,    7    },
 /* empty   */ {  8,    9,     9,      8,     8,    8,      8,     8,     8,    8    },
 /* full    */ {  9,    9,     9,      9,     9,    9,      9,     9,     9,    9    }};


#define hspara_join(se1, se2)   (((se1) >= (se2))?hspara_jm[(se1)][(se2)]:hspara_jm[(se2)][(se1)])
#define hspara_meet(se1, se2)   (((se1) <= (se2))?hspara_jm[(se1)][(se2)]:hspara_jm[(se2)][(se1)])
#define hspara_to_string(se)    (char*) hspara_string[(int) (se)]   

#line 1368 "reduc.w"


#line 208 "reduc.w"

/* enum hspara_elem vect_parallel(Pvecteur in_v1, Pvecteur in_v2) AL950711
 * input:      2 Pvecteur in_v1 and in_v2 
 * output:     hspara_elem (element of the parallel half space lattice)
 * memory:     Inspector (nothing is shared, nor modified, output allocated).
 * complexity: length(in_v1) * length(in_v2)
 * comment:    in_v1 = a1 X + b1 represents a1 X+b1 <= 0 and in_v2 a2 X + b2 <=0.
 *             if      (a1!=a2) || (a1!=-a2), returns unpara 
 *             else if (a1==a2),  return sign(b2-b1)  in ss part of hspara
 *             else if (a1==-a2), return sign(-b2-b1) in op part of hspara.
 */
enum hspara_elem vect_parallel( in_v1, in_v2 )
Pvecteur in_v1, in_v2;
{
  Pvecteur            v1, v2;
  enum hspara_elem    ret_sle = unpara;
  bool             first     = true;
  bool             same_sign = false;
  Value                 gcd1, gcd2;   /* gcd of each vector                 */
  int                 l1, l2;   /* length of each vector without TCST */
  Value               b1, b2, diff; /* value of TCST and their diff       */

  if (!in_v1 || !in_v2) return unpara;

  /* debuging */
  /*
  C3_DEBUG("vect_parallel", {
    fprintf(stderr, "Input vectors, in_v1, then in_v2:\n");  
    vect_fprint( stderr, in_v1, union_variable_name );
    vect_fprint( stderr, in_v2, union_variable_name );
  });
  */


  /* get gcd of each vector and constant linked to TCST */

  l1 = 0; b1 = 0; gcd1 = value_abs(val_of(in_v1));
  for (v1 = in_v1; v1 != NULL; v1 = v1->succ) {
    gcd1 = pgcd( gcd1, value_abs(val_of(v1)) );
    if(var_of(v1)==TCST) b1 = val_of(v1); 
    else l1++;
  }

  l2 = 0; b2 = 0; gcd2 = value_abs(val_of(in_v2));
  for (v2 = in_v2; v2 != NULL; v2 = v2->succ) {
    gcd2 = pgcd( gcd2, value_abs(val_of(v2)) );
    if(var_of(v2)==TCST) b2 = val_of(v2);
    else l2++;
  }

  if (l1 != l2)    return unpara ;

  
  /* Determine what kind of parallel hyperplane we are in */
  for (v2 = in_v2; v2 != NULL; v2 = v2->succ) {
    Variable  var2  = var_of(v2);
    Value     val2  = val_of(v2);
    bool   found = false;

    if (var2 == TCST) continue;

    for (v1 = in_v1; v1 != NULL; v1 = v1->succ) {
      if (var_of(v1) == var2) {
   Value      i1 = value_mult(gcd2,val_of(v1));
   Value     i2 = value_mult(gcd1,val2);
   bool  ss = value_eq(i1,i2);
   bool  op = value_eq(i1,value_uminus(i2));
   
   if (!ss && !op) return unpara;
   if (first) {first = false; same_sign = (ss)?ss:op ;}
   if ((same_sign && op)||(!same_sign && ss)) return unpara; 
   found = true;
      }
    }

    /* coefficient value was 0 and was not represented */
    if(!found) return unpara;  
  }
   

  /* compute return value */
  {
      Value p1 = value_mult(gcd1,b2),
            p2 = value_uminus(value_mult(gcd2,b1));
      diff = (same_sign)? value_plus(p1,p2): value_minus(p2,p1);
  }
  if      (value_zero_p(diff)) ret_sle = (same_sign) ? sszero  : opzero  ;
  else if (value_pos_p(diff)) ret_sle = (same_sign) ? ssplus  : opplus  ;
  else if (value_neg_p(diff)) ret_sle = (same_sign) ? ssminus : opminus ;
  else ret_sle = unpara;

  /* debuging */
  /*
  C3_DEBUG("vect_parallel", 
     { fprintf(stderr, "Output hspara: %s\n", hspara_to_string(ret_sle));  });
  */

  return ret_sle;
}

#line 320 "reduc.w"

/* enum enum hspara_elem contrainte_parallel_in_liste( in_co, in_lc ) AL950711
 * input:      1 constraint in_co and a list of constraints in_lc 
 * output:     hspara_elem (element of the parallel half space lattice)
 * memory:     Inspector (nothing is shared, nor modified, output allocated).
 * complexity: length(in_lc) * comp(vect_parallel())
 * comment:    in_co represents a1 X+b1 <= 0 and in_lc aj X + bj <=0.
 *             Returns in_co/in_lc = join_j( vect_parallel( in_co, in_lc_j ) ) 
 *             between keep, empty and full. 
 */
enum hspara_elem contrainte_parallel_in_liste( in_co, in_lc )
Pcontrainte in_co, in_lc;
{
  Pcontrainte         c;
  Pvecteur            vpos;
  enum hspara_elem    ret_sle = keep;  

  assert(!CONTRAINTE_UNDEFINED_P(in_co));
  if (CONTRAINTE_NULLE_P(in_co)) return keep;
  
  /* debuging */
  C3_DEBUG("contrainte_parallel_in_list", {
    fprintf(stderr, "Input in_co:");  
    inegalite_fprint( stderr, in_co, union_variable_name ); 
    fprintf(stderr, "Input in_lc:\n"); 
    inegalites_fprint( stderr, in_lc, union_variable_name );
  });

  vpos = in_co->vecteur;
  
  for (c = in_lc; !CONTRAINTE_UNDEFINED_P(c) && (ret_sle != full); c=c->succ) {
    Pvecteur         cv   = c->vecteur;
    enum hspara_elem hs   = vect_parallel(vpos, cv);

    C3_DEBUG("contrainte_parallel_in_list", {
      fprintf(stderr, "ret_sle: %s ,  hs: %s\n", 
         hspara_to_string(ret_sle),  
         hspara_to_string( hs  )  ); 
    });
    
    ret_sle = hspara_join( ret_sle, hs);
  }


  /* debuging */
  C3_DEBUG("contrainte_parallel_in_list", 
    { fprintf(stderr, "Output hspara: %s\n", hspara_to_string(ret_sle)); });
  
  return ret_sle;
}

#line 387 "reduc.w"

/* Psysteme sc_supress_parallel_redund_constraints( in_ps1, in_ps2 )
 * input:    2 Psystemes in_ps1 and in_ps2
 * output:   in_ps1 / in_ps2   (cut operation on polyhedrons)
 * memory:   Inspector (nothing is shared, nor modified, output allocated).
 * comment:  Supress in dup(in_ps2) parallel constraints that are redundant 
 *           relatively to in_ps1.
 *           Returned Psysteme have only inequalities. 
 */
Psysteme sc_supress_parallel_redund_constraints( in_ps1, in_ps2 )
Psysteme in_ps1, in_ps2;
{
  Psysteme        ps1, ps2,  ret_ps = NULL;
  Pcontrainte     ineq1, ineqs2;
  bool         stop = false, dup1 = false, dup2 = false;
  
  if ( in_ps1 == SC_RN ) return sc_dup(in_ps2);
  
  /* debuging */
  C3_DEBUG("sc_supress_parallel_constraints", {
    fprintf(stderr, "Input systems, in_ps1, then in_ps2:\n");  
    sc_fprint( stderr, in_ps1, union_variable_name );
    sc_fprint( stderr, in_ps2, union_variable_name );
  });
  

  /* Transforms equalities in inequalities if necessary */
  if (in_ps1->nb_eq != 0) 
    { ps1 = sc_dup( in_ps1 ); sc_transform_eg_in_ineg( ps1 ); dup1 = true; }
  else ps1 = in_ps1;
 
  if (in_ps2->nb_eq != 0) 
    { ps2 = sc_dup( in_ps2 ); sc_transform_eg_in_ineg( ps2 ); dup2 = true; }
  else ps2 = in_ps2;


  /* Compare with inequalities */
  ineqs2 = ps2->inegalites;

  for (ineq1 = ps1->inegalites; ineq1 != NULL && !stop; ineq1 = ineq1->succ) {
    enum hspara_elem  sk = contrainte_parallel_in_liste( ineq1, ineqs2 );
    switch (sk) 
      {
      case keep:
   if (ret_ps != NULL){ sc_add_inegalite( ret_ps, contrainte_dup(ineq1) ); }
   else ret_ps = sc_make( NULL, contrainte_dup(ineq1) );
   break;
      case empty:
   ret_ps = sc_free(ret_ps);
   ret_ps = sc_empty(NULL);
   stop = true; 
   break;
      case full: continue; break;
      default:  
   {
     fprintf(stderr, "%s supress_kind == %d should not appear !",
        "[sc_supress_parallel_redund_constraints]", (int) sk ); 
     abort();
   } 
      }
      
  }

  /* update base and normalize */  
  if ((ret_ps != NULL) && !sc_empty_p(ret_ps))  { 
    vect_rm(ret_ps->base); 
    ret_ps->base = NULL; sc_creer_base( ret_ps );
    ret_ps = sc_normalize( ret_ps );
  }

  /* Manage memory and return */
  ps1 = (dup1)? sc_free(ps1) : ps1;
  ps2 = (dup2)? sc_free(ps2) : ps2;
  C3_RETURN( IS_SC, ret_ps );
}

#line 470 "reduc.w"

/* Psysteme sc_supress_same_constraints( in_ps1, in_ps2 ) supress in 
 * in_ps2 constraints that are in in_ps1. Nothing is shared, nor modified.
 * Returned Psysteme have only inequalities.
 * This function should be superseded by sc_supress_parallel_redund_contraints
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
    bool      eq_in_ineq, co_in_ineq;

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
    Pcontrainte io;
    if (contrainte_in_liste(ineq, in_ps1->inegalites)) continue;
    if (contrainte_in_liste(ineq, in_ps1->egalites))   continue;
    io = contrainte_dup( ineq ); contrainte_chg_sgn( io );
    if (contrainte_in_liste(io, in_ps1->egalites)) {
      io = contrainte_free(io);  
      continue;
    }
    
    if (ret_ps != NULL){ sc_add_inegalite( ret_ps, contrainte_dup(ineq) ); }
    else ret_ps = sc_make( NULL, contrainte_dup(ineq) );
    io = contrainte_free(io);  
  }
  
  if (ret_ps != NULL) 
    { vect_rm(ret_ps->base); ret_ps->base = NULL; sc_creer_base( ret_ps );}
  
  ret_ps = sc_normalize( ret_ps );
  C3_RETURN( IS_SC, ret_ps );
}

#line 570 "reduc.w"

/* Psysteme sc_elim_redund_with_first_ofl_ctrl( in_ps1, in_ps2, ofl_ctrl ) 
 * Returns constraints of in_ps2 which cut in_ps1. AL 06 04 95
 * It is assumed that in_ps1 and in_ps2 are feasible !
 * in_ps1 is not modified, in_ps2 is modified.
 */
Psysteme sc_elim_redund_with_first_ofl_ctrl(in_ps1, in_ps2, ofl_ctrl)
Psysteme in_ps1, in_ps2;
int      ofl_ctrl;
{
  Psysteme    ps1; 
  Pcontrainte prev_eq = NULL, eq, tail = NULL;
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
  if ( in_ps2->nb_eq != 0 ) sc_transform_eg_in_ineg( in_ps2 );
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
  {
      vect_normalize(eq->vecteur);
  }
  /* returns if there is no intersection */
  if (!sc_rational_feasibility_ofl_ctrl(ps1, ofl_ctrl, true)) { 
    tail->succ = NULL;  ps1 = sc_free(ps1); 
    in_ps2 = sc_free(in_ps2); in_ps2 = sc_empty(NULL);
    C3_RETURN( IS_SC, in_ps2 ); 
  }
    

  /* We run over in_ps2 constraints (shared by ps1) 
   * and detect redundance */
  assert(sc_weak_consistent_p(in_ps2));
  assert(sc_weak_consistent_p(ps1));
  for (eq = tail->succ, prev_eq = tail; eq != NULL; eq = eq->succ)
  {
      contrainte_reverse(eq); 
      assert(sc_weak_consistent_p(ps1));
      C3_DEBUG("sc_elim_redund_with_first", {
     fprintf(stderr, "\nps1:\n");  
     fprintf(stderr, "nb_eq= %d, nb_ineq= %d, dimension= %d\n", 
        ps1->nb_eq, ps1->nb_ineq, ps1->dimension);
     sc_fprint( stderr, ps1, union_variable_name );
      });

      if (sc_rational_feasibility_ofl_ctrl(ps1, ofl_ctrl, true))
      {
     contrainte_reverse(eq);
     prev_eq = prev_eq->succ;
      }
      else{  
     /* eliminate the constraint from in_ps2, and thus from ps1 */   
     eq_set_vect_nul(eq);  
     if (in_ps2->inegalites == eq)
         in_ps2->inegalites = eq->succ;
     prev_eq->succ = eq->succ;
     eq->succ = CONTRAINTE_UNDEFINED;
     eq = contrainte_free(eq);
     eq = prev_eq;
     in_ps2->nb_ineq--;
     ps1->nb_ineq--;
     assert(sc_weak_consistent_p(ps1));
     assert(sc_weak_consistent_p(in_ps2));
      }
  }


  if ( in_ps2->inegalites == NULL ) 
    { in_ps2 = sc_free(in_ps2);  in_ps2 = sc_full(); }

  tail->succ = NULL; ps1 = sc_free( ps1 ); 
  C3_RETURN( IS_SC, in_ps2 );
}

#line 832 "reduc.w"

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
 /*   Psysteme ps = sc_supress_same_constraints( positif, comp->psys ); */
    Psysteme ps = sc_supress_parallel_redund_constraints( comp->psys, positif );
    if (ps == NULL) 
      {psl = sl_free(psl); ret_pa = pa_empty(); C3_RETURN(IS_PA, ret_pa);}
    else psl = sl_append_system( psl, ps );
  }

  positif = sc_dup(positif); sc_transform_eg_in_ineg( positif );
  ret_pa  = pa_make( positif, (Pcomplist) psl );
  C3_RETURN(IS_PA, ret_pa);
}

#line 884 "reduc.w"

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
    Psysteme ps;
    if (comp->psys == SC_UNDEFINED) 
      { sl_free(lcomp); C3_RETURN( IS_DJ, DJ_UNDEFINED ); }

    ps = sc_dup(comp->psys);

    ps = sc_elim_redund_with_first_ofl_ctrl( systeme, ps, ofl_ctrl );

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
  if (pa_empty_p(pa)) 
       { ret_dj = dj_empty(); } 
  else if (pa_clength2 == 0) 
       { ret_dj = dj_append_system(NULL,sc_dup(systeme)); } 
  else if (pa_clength1 != pa_clength2)  /* we've modified P0 systeme */
       { ret_dj = pa_path_to_disjunct_rule4_ofl_ctrl( pa, ofl_ctrl); } 
  else { ret_dj = pa_path_to_disjunct_ofl_ctrl( pa, ofl_ctrl); }

  pa = pa_free( pa );

  C3_RETURN( IS_DJ, ret_dj );
}

#line 1207 "reduc.w"

/* 
#line 1197 "reduc.w"
   
   Pdisjunct    pa_path_to_few_disjunct_ofl_ctrl( (Ppath) in_pa, (int) ofl_ctrl )  
   Produces a Pdisjunct corresponding to the path Ppath and
   reduces the number of disjunctions.
   See "Extension de C3 aux Unions de Polyedres" Version 2,
   for a complete explanation about this function.
   in_pa is modified.      AL 23/03/95
   
#line 1208 "reduc.w"
 
*/
Pdisjunct pa_path_to_few_disjunct_ofl_ctrl( in_pa, ofl_ctrl )
Ppath   in_pa;
int     ofl_ctrl;
{
  
#line 980 "reduc.w"
   Psysteme  systeme;  Pdisjunct ret_dj = DJ_UNDEFINED; 
#line 1001 "reduc.w"
     Ppath pa; Pcomplist lcomp;
#line 1040 "reduc.w"
     
  Pcontrainte common_cons = NULL, cons, cons_oppose = NULL; 
  Pvecteur    vect_1, cons_pv = NULL;
  Pcomplist   comp;
  
#line 1111 "reduc.w"
     
  Pcontrainte common_cons_oppose;
  Psysteme    common_ps, common_ps_oppose;
  Ppath    pa1, pa2; 
  bool     pa1_empty  = false, pa2_empty  = false;
  bool     pa1_filled = false, pa2_filled = false;
  
#line 1144 "reduc.w"
     Pdisjunct  dj1 = dj_empty(), dj2 = dj_empty(); 
#line 1214 "reduc.w"


  
#line 981 "reduc.w"
  
  C3_DEBUG( "pa_path_to_few_disjunct_ofl_ctrl", {
    fprintf(stderr, "\n\n Input path:\n\n");
    pa_fprint(stderr, in_pa, union_variable_name );
  });
  
  if (PA_UNDEFINED_P( in_pa ))   C3_RETURN(IS_DJ, DJ_UNDEFINED);
  if (pa_full_p     ( in_pa ))   C3_RETURN(IS_DJ, dj_full());
  if (pa_empty_p    ( in_pa ))   C3_RETURN(IS_DJ, dj_empty());
    
  /* If it's an empty path or if it has no complements : return */ 
  systeme = in_pa->psys ; 
  if (!sc_faisabilite_ofl( systeme )) C3_RETURN(IS_DJ,dj_empty());
  if (in_pa->pcomp == NULL) C3_RETURN(IS_DJ,(Pdisjunct) sl_append_system(NULL,sc_dup(systeme)));
  
#line 1216 "reduc.w"

  
#line 1002 "reduc.w"
  
  pa      = pa_make(sc_dup(systeme), sl_dup(in_pa->pcomp)); 
  pa      = pa_reduce_simple_complement( pa );
  
  if (pa_empty_p(pa)) {pa = pa_free(pa); C3_RETURN(IS_DJ,dj_empty());}
  
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
  
#line 1217 "reduc.w"

  
#line 1045 "reduc.w"
  
  /* We are looking for a common hyperplan */
  vect_1 = vect_new(TCST, VALUE_ONE); common_cons = NULL;
  
  for(cons = (lcomp->psys)->inegalites;
     (cons != NULL)&&(lcomp->succ != NULL);cons = cons->succ){
    bool is_common = true;
    cons_pv           = vect_dup( cons->vecteur ); vect_chg_sgn( cons_pv );
    cons_oppose       = contrainte_make(vect_add( cons_pv, vect_1 )); 
  
    for(comp = lcomp->succ;(comp != NULL) && is_common; comp = comp->succ){
      Pcontrainte ineg = (comp->psys)->inegalites;
      bool     is_common1, is_common2;
  
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
  
#line 1218 "reduc.w"

  
#line 1086 "reduc.w"
  
  if( common_cons != NULL ) {
    
#line 1118 "reduc.w"
    
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
        if (local_ps == SC_EMPTY) { pa1 = pa_empty(); pa1_empty = true; continue;}
        pa1->pcomp = sl_append_system( pa1->pcomp, local_ps ); pa1_filled = true;
      }
      else if(!pa2_empty &&  contrainte_in_liste(common_cons_oppose, co)) {
        local_ps = sc_supress_same_constraints( common_ps_oppose, comp->psys );
        if (local_ps == SC_EMPTY) {pa2 = pa_empty(); pa2_empty = true; continue;}
        pa2->pcomp = sl_append_system( pa2->pcomp, local_ps ); pa2_filled = true;
      }
    }
    
#line 1088 "reduc.w"
  
    
#line 1145 "reduc.w"
    
    if (pa1_filled) {
      /* take care of rule 2 */
      if (pa_full_p( pa2 )) pa1->psys = sc_dup( systeme );
      else pa1->psys = sc_safe_append( sc_dup(common_ps), systeme );
    
      C3_DEBUG("pa_path_to_few_disjunct", {
        fprintf(stderr, "pa1:\n");  
        pa_fprint_tab( stderr, pa1, union_variable_name, 1 );
      });
    
      if (pa_full_p(pa2)||sc_faisabilite_ofl(pa1->psys)) 
       {dj_free(dj1);dj1 = pa_path_to_few_disjunct_ofl_ctrl(pa1, ofl_ctrl);}
    
    }
    
    if (pa2_filled) {
      /* take care of rule 2 */
      if (pa_full_p( pa1 )) pa2->psys = sc_dup( systeme );
      else pa2->psys = sc_safe_append( sc_dup(common_ps_oppose), systeme );
    
      C3_DEBUG("pa_path_to_few_disjunct", {
        fprintf(stderr, "pa2:\n");  
        pa_fprint_tab( stderr, pa2, union_variable_name, 1 );
      });
      if (pa_full_p(pa1)||sc_faisabilite_ofl(pa2->psys)) 
       {dj_free(dj2);dj2 = pa_path_to_few_disjunct_ofl_ctrl(pa2, ofl_ctrl);}
    
    
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
    
#line 1089 "reduc.w"
  
  }
  else { 
    
#line 1191 "reduc.w"
     ret_dj = pa_path_to_disjunct_rule4_ofl_ctrl( pa, ofl_ctrl ); 
    
#line 1092 "reduc.w"
  
  }
  
  /* Manage memory */
  pa = pa_free(pa); vect_rm(vect_1); vect_1 = NULL;
  
  C3_RETURN(IS_DJ, ret_dj);
  
#line 1219 "reduc.w"

}

#line 1249 "reduc.w"

/* bool pa_inclusion_p(Psysteme ps1, Psysteme ps2) BA, AL 31/05/94
 * returns true if ps1 represents a subset of ps2, false otherwise
 * Inspector (no sharing on memory).
 */
bool pa_inclusion_p_ofl_ctrl(ps1, ps2, ofl_ctrl)
Psysteme ps1, ps2;
int ofl_ctrl;
{
  bool   result;
  Ppath     chemin = pa_make(ps1, sl_append_system(NULL, ps2));
  
  CATCH(overflow_error) {
    result = false; 
  }
  TRY {
    result = ! (pa_feasibility_ofl_ctrl(chemin, ofl_ctrl));
    UNCATCH(overflow_error);
  }
  chemin = pa_free1(chemin); 
  return(result);
}

#line 1275 "reduc.w"

/* bool pa_system_equal_p(Psysteme ps1, Psysteme ps2) BA
 */
bool pa_system_equal_p_ofl_ctrl(ps1,ps2, ofl_ctrl)
Psysteme ps1, ps2;
int ofl_ctrl;
{
    return (  pa_inclusion_p_ofl_ctrl(ps1,ps2, ofl_ctrl) && 
         pa_inclusion_p_ofl_ctrl(ps2,ps1, ofl_ctrl) );
}

#line 1290 "reduc.w"

/* Pdisjunct pa_system_difference_ofl_ctrl(ps1, ps2)
 * input    : two Psystemes
 * output   : a disjunction representing ps1 - ps2
 * modifies : nothing
 * comment  : algorihtm : 
 *    chemin = ps1 inter complement of (ps2)
 *    ret_dj = dj_simple_inegs_to_eg( pa_path_to_few_disjunct(chemin) )
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

#line 1323 "reduc.w"

/* bool pa_convex_hull_equals_union_p(conv_hull, ps1, ps2)
 * input    : two Psystems and their convex hull   AL,BC 23/03/95
 * output   : true if ps1 U ps2 = convex_hull, false otherwise
 * modifies : nothing
 * comment  : complexity = nb_constraints(ps1) * nb_constraints(ps2) 
 *            if ofl_ctrl = OFL_CTRL, conservatively returns ofl_ctrl 
 *            when an overflow error occurs
 */
bool pa_convex_hull_equals_union_p_ofl_ctrl
            (conv_hull, ps1, ps2, ofl_ctrl, ofl_res)
Psysteme  conv_hull, ps1, ps2;
int       ofl_ctrl;
bool   ofl_res;
{
  volatile Ppath    chemin;
  bool  result;
  int      local_ofl_ctrl = (ofl_ctrl == OFL_CTRL)?FWD_OFL_CTRL:ofl_ctrl;
  
  chemin = pa_make(conv_hull,sl_append_system(sl_append_system(NULL,ps1),ps2));
  
  if (ofl_ctrl==OFL_CTRL) {
   CATCH(overflow_error) {
            result = ofl_res;
        }
        TRY {
            result = !(pa_feasibility_ofl_ctrl(chemin, local_ofl_ctrl));
            UNCATCH(overflow_error);
        }
  }
  else
      result = !(pa_feasibility_ofl_ctrl(chemin, local_ofl_ctrl));

  chemin = pa_free1(chemin);
  return(result);
}

#line 1369 "reduc.w"

