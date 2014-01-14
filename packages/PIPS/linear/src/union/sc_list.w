%%
%% $Id$
%%
%% Copyright 1989-2014 MINES ParisTech
%%
%% This file is part of Linear/C3 Library.
%%
%% Linear/C3 Library is free software: you can redistribute it and/or modify it
%% under the terms of the GNU Lesser General Public License as published by
%% the Free Software Foundation, either version 3 of the License, or
%% any later version.
%%
%% Linear/C3 Library is distributed in the hope that it will be useful, but WITHOUT ANY
%% WARRANTY; without even the implied warranty of MERCHANTABILITY or
%% FITNESS FOR A PARTICULAR PURPOSE.
%%
%% See the GNU Lesser General Public License for more details.
%%
%% You should have received a copy of the GNU Lesser General Public License
%% along with Linear/C3 Library.  If not, see <http://www.gnu.org/licenses/>.
%%

\section{Structure de donnée}
Rappelons qu'un Psysteme est un polyèdre convexe:
\[
P = \bigwedge_{1 \leq j \leq p} h_{j} \\
\mbox{ avec } h_{j} = a_{j}.x + b_{j} \leq 0
\]


Les disjonctions et les chemins se basent sur la structure
commune de liste de Psystemes : Psyslist.
@d Type @{
typedef struct Ssyslist  {
        Psysteme                psys;
        struct Ssyslist         *succ;
        } *Psyslist,Ssyslist;

#define SL_NULL      (Psyslist) NULL
@}


\section{Fonctions}
Les fonctions sont dans le fichier :
@O sc_list.c  -d @{
@< includes @>

#ifdef DEBUG_UNION_LINEAR
extern char* entity_local_name();
char* (*union_variable_name)(Variable) = entity_local_name;
#else
char* (*union_variable_name)(Variable) = variable_default_name;
#endif

@< fonctions Psysteme @>
@< fonctions sclist @>
@}

La variable statique {\tt union\_variable\_name} a été
rajoutée pour permettre aux fonctions d'impression de lire
correctement les variables.

Les fonctions qui suivent sont essentiellement introduites pour gérer
au mieux la mémoire. Nous avons introduit pour les polyèdres et
les Psyslists des fonctions utilisant le partage de sous-structures et
les fonctions associés pour libérer la mémoire.


\subsection{Quelques fonctions sur les polyèdres}
\paragraph{sc\_full} est le système représentant l'espace entier.\\
{\bf sc\_full\_p} est Vrai si le système représente l'espace entier.
@D fonctions Psysteme @{
/* Psysteme sc_full() similar to sc_new */
Psysteme sc_full() { return sc_new(); }

/* Psysteme sc_full_p( in_ps ) similar to sc_new */
boolean sc_full_p( in_ps ) 
Psysteme in_ps;
{ return( (in_ps->nb_eq == 0) && (in_ps->nb_ineq == 0) && 
          (in_ps->egalites == NULL) && (in_ps->inegalites == NULL) ); }
@| sc_full sc_full_p @}

\paragraph{sc\_dup1} duplique un Psysteme à la profondeur 1 : seuls
les vecteurs sous-jacent au Psysteme entré sont partagés.
sc\_free1(sc\_dup1( sc )) est neutre pour la mémoire.
@D fonctions Psysteme @{
/* Psysteme sc_dup1( in_ps )			AL 30/05/94
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
@| sc_dup1 @}

\paragraph{sc\_free} appelle sc\_rm et renvoie le Psysteme NULL.
@D fonctions Psysteme @{
/* Psysteme sc_free( in_ps )			AL 30/05/94
 * Free of in_ps. Returns NULL to be used as in_ps = sc_free( in_ps );
 */
Psysteme sc_free( in_ps )
Psysteme in_ps;
{  sc_rm( in_ps ); return( (Psysteme) NULL ); }
@| sc_free @}

\paragraph{sc\_free1} efface les structures qui contiennent le Psysteme d'entré
et ses contraintes. Les vecteurs sous-jacent sont sauvegardés. 
@D fonctions Psysteme @{
/* Psysteme sc_free1( in_ps )	AL 30/05/94
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
@| sc_free1 @}

\paragraph{sc\_concatenate} construit un nouveau Psysteme à partir de deux
autres en concatenant leurs contraintes. Il y a donc partage des contraintes. La
place mémoire du Psysteme généré est libérable par \verb+sc_free1+. 
@D fonctions Psysteme @{
/* Psysteme sc_concatenate( in_s1, in_s2 )	AL 30/05/94
 * Append in_s2 to the end of in_s1 and returns in_s1. 
 * Freeable with sc_free1(). Sharing.
 */
Psysteme sc_concatenate(in_s1,in_s2)
Psysteme in_s1, in_s2;
{
  Pcontrainte 	eq;
  Psysteme	s1, s2;

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
@|sc_concatenate @}












\subsection{Les listes de polyèdres}
\paragraph{sl\_length} renvoie la longeur de la liste.
@D fonctions sclist @{
/* int sl_length( (Psyslist) in_sl )	AL 26/04/95
 * Returns length of in_sl.
 */
boolean sl_length( in_sl )
Psyslist in_sl;
{
  int length; Psyslist sl = in_sl;	
  if (in_sl == NULL) return 0;
  for(length = 0; sl != NULL; sl = sl->succ, length++);
  return length; 
}
@| sl_length @}

\paragraph{sl\_max\_constraints\_nb} renvoie le nombre maximum de
contraintes des systèmes contenus dans la liste de systèmes 
{\tt in\_sl}. Une inégalité est comptée comme une contrainte
et une égalité comme deux contraintes.
@D fonctions sclist @{
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
@| sl_max_constraints_nb @}


\paragraph{sl\_is\_system\_p} indique si la liste de Psystemes d'entré n'est
composé que d'un seul Psysteme.
@D fonctions sclist @{
/* boolean sl_is_system_p( (Psyslist) in_sl )	AL 16/11/93
 * Returns True if syslist in_sl has only one Psysteme in it.
 */
boolean sl_is_system_p( in_sl )
Psyslist in_sl;
{ return ( sl_length(in_sl) == 1 ); } @| sl_is_system_p @}

\paragraph{sl\_append\_system} ajoute un Psysteme à la fin d'un Psyslist. Le
Psyslist résultant partage tous les éléments des entrées.
@D fonctions sclist @{
/* Psyslist sl_append_system( (Psyslist) in_sl, (Psysteme) in_ps )
 * Input  : A disjunct in_sl to wich in_ps will be added.	
 * Output : Disjunct in_sl with in_ps. => ! Sharing.
 * Comment: Nothing is checked on result in_sl.  	AL 10/11/93
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
@| sl_append_system @}

\paragraph{sl\_append\_system\_first} renvoie un Psyslist dont le premier
élément est le Psysteme entré, le reste de la liste étant
constitué du Psyslist entré. Toutes les données sont partagées.
@D fonctions sclist @{
/* Psyslist sl_append_system_first( in_sl, in_ps )	AL 23/03/95 
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
@| sl_append_system_first @}

\paragraph{sl\_new} alloue un Psyslist en mettant ses champs à NULL.
@D fonctions sclist @{
/* Psyslist sl_new()	AL 26/10/93
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
@| sl_new @}

\paragraph{sl\_dup} duplique un Psyslist et tout les Psystemes qui le constituent. 
@D fonctions sclist @{
/* Psyslist sl_dup( (Psyslist) in_sl )		AL 15/11/93
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
@| sl_dup @}

\paragraph{sl\_dup1} ne duplique que la liste pointant sur les Psystemes. Les
Psystemes de la Psyslist d'entré sont partagés.
@D fonctions sclist @{
/* Psyslist sl_dup1( (Psyslist) in_sl )	AL 15/11/93
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
@| sl_dup1 @}

\paragraph{sl\_free} efface un Psyslist et tous les Psystemes qui le constituent.
sl\_free(sl\_dup( sl )) est neutre pour la mémoire. 
@D fonctions sclist @{
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
@| sl_free @}


\paragraph{sl\_free1} efface un Psyslist sans toucher aux Psystemes qui le constituent.
sl\_free1(sl\_dup1( sl )) est neutre pour la mémoire. 
@D fonctions sclist @{
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
@| sl_free1 @}



