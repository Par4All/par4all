%%
%% $Id$
%%
%% Copyright 1989-2012 MINES ParisTech
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


\section{Analyse du fichier d'entré}
Les fonctions de cette partie réalisent une analyse lexicographique et
grammaticale d'un fichier pour y lire une liste de Psystemes. Les
variables des Psystemes doivent \^etre toutes déclarées en t\^ete
du fichier par le mot ``VAR''. Voici un exemple :
\begin{verbatim}
# Une ligne de commentaire precedee d'un "#"
VAR n, m, i, j
{
    n - 1 <= 0,
    i - n <= 0,
    j - m + 1 <= 0,
    -j +2 <= 0
}
{
    -m + j + 1 <= 0,
    i - n <= 0,
    -j + 2 <= 0
}
{
    -n + 1 <= 0,
    -n + i <= 0,
    j - 1 <= 0
}
{
    i - n <= 0,
    j - 1 <= 0
}
\end{verbatim}

Les fonctions développées ont un inter\^et pour les disjonctions
et les chemins mais peuvent aussi servir si l'on cherche à lire un
ensemble de Psystemes ayant des variables communes. La lecture
successive par sc\_read de fichiers différents contenant des
Psystemes ayant des variables communes n'impliquait pas que ces
variables avaient la m\^eme location mémoire, d'o\`u une
difficulté pour tester l'égalité entre les variables des
différents Psystemes.


\subsection{Analyse lexicographique}
L'analyse lexicographique est directement inspirée de
celle effectuée pour les Psystemes :
@O sl_lex.l @{
%START COMMENT TEXT
%{
/*
    Grammaire lex necessaire pour l'analyse lexicale d'une liste de systemes.
    Les tokens renvoyes sont commentes dans le fichier "sl_gram.y".
*/

#include <stdio.h>
#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"

#include "y.tab.h"
#ifdef SCAN_VIEW
#define RETURN(token,string) fprintf(stderr,"%s ", string) ; \
                             return(token)
#else
#define RETURN(token,string) return(token)
#endif
%}

%%
<TEXT>VAR                       { RETURN(VAR, "VAR "); }
<TEXT>[A-Za-z_]['A-Za-z0-9_:#]* { RETURN(IDENT, "IDENT "); }
<TEXT>[0-9]*                    { RETURN(CONSTANTE, "CONSTANTE "); }
<TEXT>"=="                      { RETURN(EGAL, "EGAL "); }
<TEXT>","                       { RETURN(VIRG, "VIRG "); }
<TEXT>"+"                       { RETURN(PLUS, "PLUS "); }
<TEXT>"-"                       { RETURN(MOINS, "MOINS "); }
<TEXT>">"                       { RETURN(SUP, "SUP "); }
<TEXT>">="                      { RETURN(SUPEGAL, "SUPEGAL "); }
<TEXT>"<"                       { RETURN(INF, "INF "); }
<TEXT>"<="                      { RETURN(INFEGAL, "INFEGAL "); }
<TEXT>"{"                       { RETURN(ACCOUVR, "ACCOUVR "); }
<TEXT>"}"                       { RETURN(ACCFERM, "ACCFERM "); }
<TEXT>[ \t\n]*          		;
<TEXT>"#"                       {BEGIN COMMENT;}
<COMMENT>\n                     {BEGIN TEXT;}
<COMMENT>[^\n]*         ;
%%

int yywrap() { return(-1); }
int sl_init_lex() { BEGIN TEXT; }
@}

\subsection{Analyse grammaticale}
L'analyse grammaticale est inspirée de celle effectuée
pour les Psystemes. Voici l'architecture du fichier :
@O sl_gram.y @{
/* explicit types: Value may be larger than a pointer (e.g. long long)
 */
%type <Value> const
%type <Variable> ident

%{
@< gram inludes @>
@< gram variables @>
@< gram define @>
%}
@< gram token @>
%%
@< regles de grammaire @>
%%
@< gestion des erreurs @>
@}

Avec pour fichier d'ent\^ete :
@D gram inludes @{
#include <stdio.h>
#include <string.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "malloc.h"
#include "union.h"
@}

Les variables utilisées :
@D gram variables @{
extern char yytext[]; /* dialogue avec l'analyseur lexical */
Psysteme ps_yacc;
boolean syntax_error;
Value valcst;
Value fac;        /* facteur multiplicatif suivant qu'on analyse un terme*/
                      /* introduit par un moins (-1) ou par un plus (1) */
int sens;             /* indique le sens de l'inegalite
                         sens = -1  ==> l'operateur est soit > ,soit >=,
                         sens = 1   ==> l'operateur est soit <, soit <=   */
short int cote;       /* booleen indiquant quel membre est en cours d'analyse*/
Value b1, b2;      /* element du vecteur colonne du systeme donne 
			 par l'analyse d'une contrainte */
Pcontrainte eq;       /* pointeur sur l'egalite ou l'inegalite courante */
Pvecteur cp ;         /* pointeur sur le membre courant */ 
short int operat;     /* dernier operateur rencontre */
@}

Pour pouvoir réutiliser les fonctions de \verb+sc.dir/read.c+,
nous devons déclarer certaines variables comme externes :
@D gram var... @{
extern	Pcontrainte p_eg_fin;
extern	Pcontrainte p_ineg_fin;
extern	Pvecteur p_pred;
extern	Pvecteur p_membre_courant;
extern	Pvecteur cp;
@}

Les opérateurs de comparaison :
@D gram define @{
/*code des operateurs de comparaison */
#define OPINF 1
#define OPINFEGAL 2
#define OPEGAL 3
#define OPSUPEGAL 4
#define OPSUP 5
#define DROIT 1
#define GAUCHE 2
/* #define NULL 0 */
@}

Les mots analysés :
@D gram token	@{
%token ACCFERM	        /* accolade fermante */ 1
%token ACCOUVR		/* accolade ouvrante */ 2
%token CONSTANTE	/* constante entiere sans signe 
			   a recuperer dans yytext */ 3
%token EGAL		/* signe == */ 4
%token IDENT		/* identificateur de variable 
			   a recuperer dans yytext */ 5
%token INF		/* signe < */ 6
%token INFEGAL		/* signe <= */ 7
%token MOINS		/* signe - */ 8
%token PLUS		/* signe + */ 9
%token SUP		/* signe > */ 10
%token SUPEGAL		/* signe >= */ 11
%token VAR		/* mot reserve VAR introduisant 
			   la liste de variables */ 12
%token VIRG		/* signe , */ 13

%union {
    Value Value;
    Variable Variable;
}

@}


Le fichier d'entré se présente sous forme d'une liste de systèmes s\_list :
@d gram variables @{Psyslist	sl_yacc; @}
@D regles de grammaire	@{
s_list	: inisl defvar l_sys endsl
	;

inisl	: 
	{ /* Initialisation de la liste des systemes */
	  sl_yacc      = NULL; 
	  syntax_error = FALSE;
	}
	;

endsl	: 
	{ /* Fin de la list des systemes */
	  vect_rm( (Pvecteur) ba_yacc ); ba_yacc = NULL;
	}
	;
@}

Les variables utilisées dans les systèmes sont lues et mises 
dans la base commune ba\_yacc :
@d gram variables @{Pbase	ba_yacc; @}
@D regles de grammaire	@{
defvar	: VAR l_var
	;

l_var	: newid
	| l_var VIRG newid
	;

newid	: IDENT
	{
   	  if(!base_contains_variable_p(ba_yacc, (Variable) yytext))
   	      ba_yacc = vect_add_variable(ba_yacc, (Variable) strdup(yytext));
	}
	;
@}

On lit la liste de systèmes :
@D regles de grammaire	@{
l_sys	: system
	| system l_sys
	;

system	: inisys ACCOUVR l_eq virg_opt ACCFERM endsys
	; 

inisys	:
	{ /* initialisation des parametres de la liste de systemes */
          /* et initialisation des variables */
	  ps_yacc = sc_new();
	  init_globals();
	}
	;

endsys	:
	{   
          /* on rajoute le systeme trouve a la liste */
	  if (ps_yacc != NULL) {
	    ps_yacc->base = NULL;
	    sc_creer_base( ps_yacc );
	  }
	  sl_yacc = sl_append_system_first( sl_yacc, ps_yacc );
	}
	;


l_eq	: eq
	| l_eq VIRG eq
	|
	;

eq	: debeq multi_membre op membre fin_mult_membre feq
	;

debeq	:
	{   
	  fac    = VALUE_ONE;
	  sens   = 1;
	  cote   = GAUCHE;
	  b1     = 0;
	  b2     = 0;
	  operat = 0;
	  cp     = NULL;
	  eq     = contrainte_new();
	}
	;

feq     :
	{ 
	  contrainte_free(eq); 
	}
	;

membre	: addop terme 
	| { fac = VALUE_ONE;} terme
	| membre addop terme
	;

terme	: const ident 
	{
	  if (cote==DROIT)
	    value_oppose(fac);

	  /* ajout du couple (ident,const) a la contrainte courante */
	  vect_add_elem(&(eq->vecteur), (Variable) $2,value_mult(fac,$1));
	  /* duplication du couple (ident,const) de la combinaison lineaire
	     traitee*/ 
	  if (operat) vect_add_elem(&cp,(Variable) $2,
                                    value_uminus(value_mult(fac,$1)));
	}
	| const
	{
	    Value v = value_mult(fac,$1);
            if (cote==DROIT)
            {
                value_addto(b1,v); value_substract(b2,v);
            }
            else
            {
                value_addto(b2,v); value_substract(b1,v);
            }
	}     
	| ident
	{
	    if (cote==DROIT) value_oppose(fac);

	  /* ajout du couple (ident,1) a la contrainte courante */
	  vect_add_elem (&(eq->vecteur),(Variable) $1,fac);
	  /* duplication du couple (ident,1) de la combinaison lineaire traitee */
	  if (operat) vect_add_elem(&cp,(Variable) $1,value_uminus(fac));
	}
	;
@}

Les variables courantes dans les systèmes sont mises
dans va\_yacc, et l'on vérifie si elle ne sont pas déja présentes.
@d gram variables @{Variable va_yacc; @}
@D regles de grammaire @{
ident	: IDENT
	{
	  va_yacc = base_find_variable_name(ba_yacc, 
					    (Variable) yytext, 
					    variable_default_name);
	  if(VARIABLE_UNDEFINED_P(va_yacc)) {
	    (void) fprintf(stderr, 
			   "Variable %s not declared. Add it to the VAR list!\n",
			   variable_default_name(yytext));
	    exit(1);
	  }
	  $$ = va_yacc;
	}
	;
@}

Règles de base utiles pour la lecture des contraintes :
@D regles de grammaire @{
const	: CONSTANTE
	{ sscan_Value(yytext, &valcst);
	  $$ = (Value) valcst;
	}
	;

op	: INF
	{ 
	  cote   = DROIT; 
	  sens   = 1;
	  operat = OPINF;
	  cp     = NULL;
	  b2     = 0; 
	}
	| INFEGAL
	{ 
	  cote   = DROIT; 
	  sens   = 1;
	  operat = OPINFEGAL;
	  cp     = NULL;
	  b2     = 0;
	}
	| EGAL
	{ 
	  cote   = DROIT; 
	  sens   = 1;
	  operat = OPEGAL; 
	  cp     = NULL;
	  b2     = 0;
	}
	| SUP
	{ 	
	  cote   = DROIT; 
	  sens   = -1;
	  operat = OPSUP; 
	  cp     = NULL;
	  b2     = 0;
	}
	| SUPEGAL
	{ 
	  cote   = DROIT;	
	  sens   = -1;
	  operat = OPSUPEGAL;
	  cp     = NULL;
	  b2     = 0;
	}
	;
addop	: PLUS
	{ fac = VALUE_ONE; }
	| MOINS
	{ fac = VALUE_MONE; }
	;

multi_membre : membre
	| multi_membre op membre fin_mult_membre
	;

fin_mult_membre :
	{
	  vect_add_elem(&(eq->vecteur),TCST,value_uminus(b1));
	  switch (operat) {
	  case OPINF:
	    creer_ineg(ps_yacc,eq,sens);
	    vect_add_elem(&(eq->vecteur),TCST,VALUE_ONE);
	    break;
	  case OPINFEGAL:
	    creer_ineg(ps_yacc,eq,sens);
	    break;
	  case OPSUPEGAL:
	    creer_ineg(ps_yacc,eq,sens);
	    break;
	  case OPSUP:
	    creer_ineg(ps_yacc,eq,sens);
	    vect_add_elem (&(eq->vecteur),TCST,VALUE_ONE);
	    break;
	  case OPEGAL:
	    creer_eg(ps_yacc,eq);
	    break;
	  }

	  eq = contrainte_new();
	  eq->vecteur = cp;
	  b1 = b2;
	}
	;

virg_opt : VIRG
	|
	; 
@}

@D gestion des erreurs @{
int yyerror(s)
char *s;
{
  /* procedure minimum de recouvrement d'erreurs */
  int c;
  (void) fprintf(stderr,"%s near %s\n",s,yytext);
  while ((c = getchar()) != EOF) putchar(c);
  syntax_error = TRUE;
  return 0;
}
@}


\section{Fonctions de lecture-écriture}
Des fonctions de lecture de liste de systèmes,
de disjonction et de chemin ont été écrites, ainsi
que des fonctions d'écriture. Ces fonctions d'écriture
écrivent dans un format compatible avec les fonctions de
lecture.

\subsection{Fonctions communes}
\paragraph{} Par défaut, le nom des variables sont fournies par la fonctions 
variable\_default\_name, mais {\bf sl\_set\_variable\_name} permet de changer
cette fonction d'accés à volonté. C'est utile lorsque les variables sont 
des entités de Linear/C3 Library.
@D fonctions sclist @{
/* char* sl_set_variable_name( in_fu ) give the function to read variables */
void  sl_set_variable_name( in_fu )
char*  (*in_fu)();
{
  union_variable_name = in_fu;
}
@| sl_set_variable_name @}

\paragraph{} Les fonctions d'impression étant tabifiées, 
{\bf sl\_get\_tab\_string} renvoie une chaine de caractère 
contenant {\tt in\_tab} caractères de tabulation.
@D fonctions sclist @{
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
@| sl_get_tab_string @}

\paragraph{}
{\bf sl\_fprint\_tab} imprime la liste de toute les variables
utilisées dans les Psystemes puis la liste de systèmes. 
{\bf sl\_read} utilise les fonctions d'analyse lexicales et
grammaticale pour lire les fichiers d'entrée. 
@D fonctions sclist @{
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
extern  int sl_init_lex();

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
@| sl_fprint_tab sl_fprint sl_read @}


\subsection{Lecture-écriture de disjonctions}
{\bf dj\_fprint\_tab} et {\bf dj\_read} sont de simples appels aux fonctions communes.
@D fonctions Pdisjunct @{
/* void dj_fprint_tab(FILE*, Pdisjunct, function, int) prints a Pdisjunct */
void    dj_fprint_tab( in_fi, in_dj, in_fu, in_tab )
FILE*       in_fi;
Pdisjunct   in_dj;
char        *(*in_fu)();
int         in_tab;
{
  char*  tabs = sl_get_tab_string( in_tab );

  if (dj_full_p(in_dj))    { fprintf(in_fi, "%sDJ_FULL\n",      tabs); return; }
  if DJ_UNDEFINED_P(in_dj) { fprintf(in_fi, "%sDJ_UNDEFINED\n", tabs); return; }

  fprintf      ( in_fi, "\n%s# -----DJ BEGIN-----\n", tabs   );
  sl_fprint_tab( in_fi, (Psyslist) in_dj, in_fu,      in_tab ); 
  fprintf      ( in_fi, "\n%s# -----DJ END-----\n",   tabs   );
}


/* void dj_read(FILE*) reads a Pdisjunct */
Pdisjunct dj_read( nomfic )
char* nomfic;
{ return ( (Pdisjunct) sl_read(nomfic) ); }
@| dj_fprint_tab dj_read @}


\subsection{Lecture-écriture de chemins}
Pour les chemins, le premier Psysteme écrit, m\^eme s'il est vide,
représente le système $\cal P_0$ et les autres la liste des
complémenataires. 

{\bf pa\_fprint\_tab} construit le Psyslist aproprié et appelle sl\_fprint\_tab.
{\bf pa\_read} associe le premier système lu à $\cal P_0$ et le reste
aux complémentaires. 
@D fonctions Ppath @{
/* void pa_fprint_tab(FILE*, Pdisjunct, function, tab) prints a Ppath */
void pa_fprint_tab( in_fi, in_pa, in_fu, in_tab )
FILE*   in_fi;
Ppath   in_pa;
char    *(*in_fu)();
int     in_tab;
{
  Psyslist    sl;
  char*       tabs = sl_get_tab_string( in_tab );
  
  if (pa_full_p(in_pa))    { 
    fprintf(in_fi, "%sPA_FULL\n", tabs); 
    free(tabs); return;
  }
  if PA_UNDEFINED_P(in_pa) { 
    fprintf(in_fi, "%sPA_UNDEFINED\n", tabs); 
    free(tabs); return;
  }

  sl = sl_new(); sl->succ = in_pa->pcomp; sl->psys = in_pa->psys;
  fprintf      ( in_fi, "\n%s# --------PA BEGIN------\n", tabs);
  sl_fprint_tab( in_fi, sl, in_fu, in_tab );
  fprintf      ( in_fi, "\n%s# --------PA END--------\n", tabs);
  free( sl ); free( tabs ); return;
}

/* void pa_read(FILE*) reads a Ppath */
Ppath pa_read( nomfic )
char* nomfic;
{
  Ppath       ret_pa;
  Psyslist    sl;

  sl = sl_read(nomfic);
  if (sl == SL_NULL) return PA_UNDEFINED;
  ret_pa = pa_make(sl->psys, (Pcomplist) sl->succ);
  free( sl );
  return ret_pa;
}
@| pa_fprint_tab pa_read @}


\subsection{Lecture-écriture générale}
{\bf un\_fprint\_tab} écrit un système, une disjonction ou un chemin, en
fonction du type {\tt in\_ty} entré.
@D fonctions sclist @{
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
@| un_fprint_tab @}
