
#line 23 "sc_list.w"


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
#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

/* Ansi includes        */
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


#line 24 "sc_list.w"


#ifdef DEBUG_UNION_LINEAR
extern char* entity_local_name(Variable);
char* (*union_variable_name)(Variable) = entity_local_name;
#else
char* (*union_variable_name)(Variable) = variable_default_name;
#endif


#line 50 "sc_list.w"

/* Psysteme sc_full() similar to sc_new */
Psysteme sc_full() { return sc_new(); }

/* Psysteme sc_full_p( in_ps ) similar to sc_new */
bool sc_full_p( in_ps ) 
Psysteme in_ps;
{ return( (in_ps->nb_eq == 0) && (in_ps->nb_ineq == 0) && 
          (in_ps->egalites == NULL) && (in_ps->inegalites == NULL) ); }

#line 64 "sc_list.w"

/* Psysteme sc_dup1( in_ps )        AL 30/05/94
 * 1 depth copy of in_ps: no duplication of vectors (except for the base). 
 * Sharing !
 */
Psysteme sc_dup1(in_ps)
Psysteme in_ps;
{
  Psysteme cp = SC_UNDEFINED;
  Pcontrainte eq, eq_cp;
  
  if (!SC_UNDEFINED_P(in_ps)) {
    cp = sc_new();
    
    for (eq = in_ps->egalites; eq != NULL; eq = eq->succ) {
      eq_cp = contrainte_new();
      contrainte_vecteur(eq_cp) = contrainte_vecteur(eq);
      sc_add_egalite(cp, eq_cp);
    }
    
    for(eq=in_ps->inegalites;eq!=NULL;eq=eq->succ) {
      eq_cp = contrainte_new();
      contrainte_vecteur(eq_cp) = contrainte_vecteur(eq);
      sc_add_inegalite(cp, eq_cp);
    }

    if(in_ps->dimension==0) {
      assert(VECTEUR_UNDEFINED_P(in_ps->base));
      cp->dimension = 0;
      cp->base = VECTEUR_UNDEFINED;
    }
    else {
      assert(in_ps->dimension==vect_size(in_ps->base));
      cp->dimension = in_ps->dimension;
      cp->base = vect_dup(in_ps->base);    
    }
  }
  return(cp);
}

#line 106 "sc_list.w"

/* Psysteme sc_free( in_ps )        AL 30/05/94
 * Free of in_ps. Returns NULL to be used as in_ps = sc_free( in_ps );
 */
Psysteme sc_free( in_ps )
Psysteme in_ps;
{  sc_rm( in_ps ); return( (Psysteme) NULL ); }

#line 117 "sc_list.w"

/* Psysteme sc_free1( in_ps ) AL 30/05/94
 * Only pcontrainte of in_ps are freed.
 */
Psysteme sc_free1( in_ps )
Psysteme in_ps;
{
  Pcontrainte pc, pc2;

  if (in_ps != NULL) {
    for(pc = in_ps->inegalites; pc != NULL; ) {
      pc2 = pc;
      pc = pc->succ;
      free(pc2);
      pc2 = NULL;
    }
    for(pc = in_ps->egalites; pc != NULL; ) {
      pc2 = pc;
      pc = pc->succ;
      free(pc2);
      pc2 = NULL;
    }
    in_ps->nb_eq = 0;
    in_ps->nb_ineq = 0;
    in_ps->dimension = 0;
    vect_rm( in_ps->base );
    in_ps->base = NULL;
    
    free((char *) in_ps);
    in_ps = (Psysteme) NULL;
  }
  return( (Psysteme) NULL );
}

#line 155 "sc_list.w"

/* Psysteme sc_concatenate( in_s1, in_s2 )   AL 30/05/94
 * Append in_s2 to the end of in_s1 and returns in_s1. 
 * Freeable with sc_free1(). Sharing.
 */
Psysteme sc_concatenate(in_s1,in_s2)
Psysteme in_s1, in_s2;
{
  Pcontrainte  eq;
  Psysteme  s1, s2;

  s1 = sc_dup1( in_s1 );      s2 = sc_dup1( in_s2 );
  if (SC_UNDEFINED_P(in_s1))  {s1 = sc_free1(s1); return(s2);}
  if (SC_UNDEFINED_P(in_s2))  {s2 = sc_free1(s2); return(s1);}

  if (s1->nb_eq != 0) {
    for (eq = s1->egalites; eq->succ != (Pcontrainte)NULL; eq = eq->succ) ;
    eq->succ = s2->egalites; s1->nb_eq += s2->nb_eq;
  } 
  else { s1->egalites = s2->egalites; s1->nb_eq = s2->nb_eq; }
  
  if (s1->nb_ineq != 0) {
    for (eq = s1->inegalites;eq->succ != (Pcontrainte)NULL;eq = eq->succ) ;
    eq->succ = s2->inegalites;  s1->nb_ineq += s2->nb_ineq;
  } 
  else { s1->inegalites = s2->inegalites; s1->nb_ineq = s2->nb_ineq; }
  
  /* Memory management and returns */
  vect_rm( s1->base ); vect_rm( s2->base ); free( s2 ); s2 = NULL;
  s1->base = NULL; sc_creer_base( s1 );
  return(s1);
}

#line 33 "sc_list.w"


#line 202 "sc_list.w"

/* int sl_length( (Psyslist) in_sl )   AL 26/04/95
 * Returns length of in_sl.
 */
bool sl_length( in_sl )
Psyslist in_sl;
{
  int length; Psyslist sl = in_sl;  
  if (in_sl == NULL) return 0;
  for(length = 0; sl != NULL; sl = sl->succ, length++);
  return length; 
}

#line 220 "sc_list.w"

/* int sl_max_constraints_nb( (Psyslist) in_sl )
 * Give the maximum constraints nb among systems of in_sl. 
 */
int sl_max_constraints_nb( in_sl )
Psyslist in_sl;
{ 
  Psysteme   ps;
  int        ret_int = 0;
  
  if (in_sl == NULL) return 0;
  
  for(; in_sl != NULL; in_sl = in_sl->succ) {
    int loc;
    ps  = in_sl->psys; 
    if (sc_empty_p(ps)) continue;
    loc = 2*(ps->nb_eq) + ps->nb_ineq;
    if (loc > ret_int) ret_int = loc;
  }
  return ret_int; 
}

#line 246 "sc_list.w"

/* bool sl_is_system_p( (Psyslist) in_sl )   AL 16/11/93
 * Returns True if syslist in_sl has only one Psysteme in it.
 */
bool sl_is_system_p( in_sl )
Psyslist in_sl;
{ return ( sl_length(in_sl) == 1 ); } 
#line 256 "sc_list.w"

/* Psyslist sl_append_system( (Psyslist) in_sl, (Psysteme) in_ps )
 * Input  : A disjunct in_sl to wich in_ps will be added.   
 * Output : Disjunct in_sl with in_ps. => ! Sharing.
 * Comment: Nothing is checked on result in_sl.    AL 10/11/93
 */
Psyslist sl_append_system( in_sl, in_ps )
Psyslist in_sl;
Psysteme in_ps;
{
  Psyslist ret_sl;
  
  if (in_ps == NULL) return( in_sl );
  ret_sl = sl_new(); ret_sl->psys = in_ps; ret_sl->succ = in_sl;
  return( ret_sl );
}

#line 277 "sc_list.w"

/* Psyslist sl_append_system_first( in_sl, in_ps ) AL 23/03/95 
 * A new Psyslist with in_ps at the end of in_sl (sharing). 
 */
Psyslist sl_append_system_first( in_sl, in_ps )
Psyslist in_sl;
Psysteme in_ps;
{
  Psyslist new_sl = SL_NULL, sl = SL_NULL;
  
  if (in_ps == NULL) return( in_sl );
  new_sl = sl_append_system(NULL, in_ps);
  if (in_sl == SL_NULL) return new_sl; 
  if (in_sl->succ == NULL) { in_sl->succ = new_sl ; return in_sl; }
  for(sl = in_sl; (sl->succ != NULL); sl = sl->succ) {}
  sl->succ = new_sl;
  return( in_sl );
}

#line 298 "sc_list.w"

/* Psyslist sl_new() AL 26/10/93
 * Input  : Nothing.
 * Output : An empty syslist.
 */
Psyslist sl_new()
{
  Psyslist p;
  
  p = (Psyslist) malloc(sizeof(Ssyslist));
  if (p == NULL) {
    (void) fprintf(stderr,"sl_new: Out of memory space\n");
    exit(-1);
  }
  p->psys = (Psysteme ) NULL; p->succ = (Psyslist) NULL;
  return(p);
}

#line 318 "sc_list.w"

/* Psyslist sl_dup( (Psyslist) in_sl )    AL 15/11/93
 * w - 1 duplication : everything is duplicated, except entities.
 * Duplicates input syslist.
 */
Psyslist sl_dup( in_sl )
Psyslist in_sl;
{
  Psyslist  sl, ret_sl = SL_NULL;
  for( sl = in_sl; sl != NULL; sl = sl->succ ) {
    ret_sl = sl_append_system( ret_sl, sc_dup( sl->psys ) );
  }
  return ret_sl;
}

#line 336 "sc_list.w"

/* Psyslist sl_dup1( (Psyslist) in_sl )   AL 15/11/93
 * Duplicates input syslist. Sharing.
 */
Psyslist sl_dup1( in_sl )
Psyslist in_sl;
{
  Psyslist  sl, ret_sl = NULL;
  for( sl = in_sl; sl != NULL; sl = sl->succ ) {
    if ( sl->psys == SC_UNDEFINED ) continue;
    ret_sl = sl_append_system( ret_sl, sl->psys );
  }
  return ret_sl;
}

#line 354 "sc_list.w"

/* Psyslist sl_free(Psyslist psl) BA, AL 30/05/94
 * w - 1 depth free.
 */
Psyslist sl_free(psl)
Psyslist psl;
{
  if( psl != SL_NULL ) {
    psl->psys = sc_free( psl->psys );
    psl->succ = sl_free(psl->succ);
    free( psl ); psl = NULL;
  }
  return SL_NULL;
}

#line 373 "sc_list.w"

/* Psyslist sl_free1(Psyslist psl) AL 30/05/94
 * 1 depth free.
 */
Psyslist sl_free1(psl)
Psyslist psl;
{
  if( psl != SL_NULL ) {
    psl->psys = (Psysteme) NULL;
    psl->succ = sl_free1(psl->succ);
    free( psl );
  }
  return SL_NULL;
}

#line 470 "sl_io.w"

/* char* sl_set_variable_name( in_fu ) give the function to read variables */
void  sl_set_variable_name( in_fu )
char*  (*in_fu)();
{
  union_variable_name = in_fu;
}

#line 482 "sl_io.w"

/* char* sl_get_tab_string( in_tab ) returns a string of in_tab \t */
char*  sl_get_tab_string( in_tab )
int    in_tab;
{
  int            d;
  static  char   name[20];
#ifndef strdup
  extern  char*  strdup();
#endif

  if (in_tab == 0) return strdup("");
  assert( (in_tab > 0) && (in_tab < 20) );
  for(d = 0; d < in_tab; d++){ sprintf(&name[d],"\t"); }
  return strdup(name);
}

#line 505 "sl_io.w"

void  sl_fprint_tab( in_fi, in_sl, in_fu, in_tab )
FILE*       in_fi;
Psyslist    in_sl;
char        *(*in_fu)();
int         in_tab;
{
  Pcontrainte   peq  = NULL;
  Psyslist      sl   = NULL;
  Pbase         b    = NULL, b1;
  char*         tabs = sl_get_tab_string( in_tab );

  if (in_sl == SL_NULL) {
    fprintf( in_fi, "\n%sSL_NULL\n", tabs ); 
    free(tabs); return; 
  }
  
  /* Prints the VAR part */
  for(sl = in_sl; sl != NULL; sl = sl->succ) {
    if (sl->psys == NULL) continue;
    b1 = b;
    b  = base_union( b, (sl->psys)->base );
    if ( b != b1 ) { vect_rm( b1 ); b1 = (Pvecteur) NULL; } 
  }
  
  if (vect_size( b ) >= 1 ) {
    fprintf( in_fi,"%s", tabs);
    fprintf( in_fi,"VAR %s", (*in_fu)(vecteur_var(b)));
    for (b1=b->succ; !VECTEUR_NUL_P(b1); b1 = b1->succ)
      fprintf(in_fi,", %s",(*in_fu)(vecteur_var(b1)));
  }
  
  vect_rm( (Pvecteur) b ); b = (Pvecteur) NULL;

  /* Prints Psysteme list */
  for(sl = in_sl ; sl != NULL; sl = sl->succ) {
    Psysteme    ps = NULL;
    
    ps = sl->psys;
    
    /* Special cases */
    if ( SC_UNDEFINED_P(ps) ) 
      {fprintf( in_fi, "\n%sSC_UNDEFINED\n", tabs); continue; }
    if ( sc_full_p(ps) ) 
      {fprintf( in_fi, "\n%sSC_FULL\n", tabs); continue; }
    if ( sc_empty_p(ps) ) 
      {fprintf( in_fi, "\n%sSC_EMPTY\n", tabs); continue; }


    /* General Cases */
    fprintf(in_fi,"\n%s { \n", tabs);
    
    for (peq = ps->inegalites;peq!=NULL;
    fprintf(in_fi,"%s", tabs),
         inegalite_fprint(in_fi,peq,in_fu),peq=peq->succ);
    
    for (peq = ps->egalites;peq!=NULL;
    fprintf(in_fi,"%s", tabs),
         egalite_fprint(in_fi,peq,in_fu),peq=peq->succ);
    
    fprintf(in_fi,"%s } \n", tabs);
  }
  free( tabs );
}

void  sl_fprint( in_fi, in_sl, in_fu )
FILE*       in_fi       ;
Psyslist    in_sl       ;
char        *(*in_fu)() ;
{ sl_fprint_tab( in_fi, in_sl, in_fu, 0 );  }


 
extern  Psyslist  sl_yacc;  /* Psysteme construit par sl_gram.y */
extern  FILE*     slx_in;   /* fichier lu par sl_lex.l          */
extern void slx_parse();

/* void sl_read(FILE*) reads a Psyslist */
Psyslist  sl_read( nomfic )
char*     nomfic;
{
  if ((slx_in = fopen(nomfic, "r")) == NULL) {
    (void) fprintf(stderr, "Ouverture du fichier %s impossible\n",nomfic);
    exit(4);
  }
  sl_init_lex(); slx_parse(); fclose( slx_in );
  return( sl_yacc );
}

#line 677 "sl_io.w"

/* void un_fprint_tab(FILE*, Pdisjunct, function, type, tab) prints a union */
void un_fprint_tab( in_fi, in_un, in_fu, in_ty, in_tab )
FILE*   in_fi;
char*   in_un;
char    *(*in_fu)();
int     in_ty;
int     in_tab;
{ 
  switch( in_ty ) {
  
  case IS_SC: 
    fprintf  ( in_fi, "Systeme:\n");
    sc_fprint( in_fi, (Psysteme) in_un, in_fu );
    break;
   
  case IS_SL: 
    fprintf      ( in_fi, "%sSyslist:\n", sl_get_tab_string( in_tab ));
    sl_fprint_tab( in_fi, (Psyslist) in_un, in_fu, in_tab );
    break;
   
  case IS_DJ:
    dj_fprint( in_fi, (Pdisjunct) in_un, in_fu );
    break;
    
  case IS_PA:
    pa_fprint( in_fi, (Ppath) in_un, in_fu );
    break;
 
  default: {}
  }
}

#line 34 "sc_list.w"

