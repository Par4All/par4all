
/* explicit types: Value may be larger than a pointer (e.g. long long)
 */
%type <Value> const
%type <Variable> ident

%{

#include <stdio.h>
#include <string.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "malloc.h"
#include "union.h"


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

extern   Pcontrainte p_eg_fin;
extern   Pcontrainte p_ineg_fin;
extern   Pvecteur p_pred;
extern   Pvecteur p_membre_courant;
extern   Pvecteur cp;
Psyslist sl_yacc; Pbase ba_yacc; Variable va_yacc; 

/*code des operateurs de comparaison */
#define OPINF 1
#define OPINFEGAL 2
#define OPEGAL 3
#define OPSUPEGAL 4
#define OPSUP 5
#define DROIT 1
#define GAUCHE 2
/* #define NULL 0 */

%}

%token ACCFERM         /* accolade fermante */ 1
%token ACCOUVR    /* accolade ouvrante */ 2
%token CONSTANTE  /* constante entiere sans signe 
            a recuperer dans yytext */ 3
%token EGAL    /* signe == */ 4
%token IDENT      /* identificateur de variable 
            a recuperer dans yytext */ 5
%token INF     /* signe < */ 6
%token INFEGAL    /* signe <= */ 7
%token MOINS      /* signe - */ 8
%token PLUS    /* signe + */ 9
%token SUP     /* signe > */ 10
%token SUPEGAL    /* signe >= */ 11
%token VAR     /* mot reserve VAR introduisant 
            la liste de variables */ 12
%token VIRG    /* signe , */ 13

%union {
    Value Value;
    Variable Variable;
}


%%

s_list   : inisl defvar l_sys endsl
   ;

inisl : 
   { /* Initialisation de la liste des systemes */
     sl_yacc      = NULL; 
     syntax_error = FALSE;
   }
   ;

endsl : 
   { /* Fin de la list des systemes */
     vect_rm( (Pvecteur) ba_yacc ); ba_yacc = NULL;
   }
   ;

defvar   : VAR l_var
   ;

l_var : newid
   | l_var VIRG newid
   ;

newid : IDENT
   {
        if(!base_contains_variable_p(ba_yacc, (Variable) yytext))
            ba_yacc = vect_add_variable(ba_yacc, (Variable) strdup(yytext));
   }
   ;

l_sys : system
   | system l_sys
   ;

system   : inisys ACCOUVR l_eq virg_opt ACCFERM endsys
   ; 

inisys   :
   { /* initialisation des parametres de la liste de systemes */
          /* et initialisation des variables */
     ps_yacc = sc_new();
     init_globals();
   }
   ;

endsys   :
   {   
          /* on rajoute le systeme trouve a la liste */
     if (ps_yacc != NULL) {
       ps_yacc->base = NULL;
       sc_creer_base( ps_yacc );
     }
     sl_yacc = sl_append_system_first( sl_yacc, ps_yacc );
   }
   ;


l_eq  : eq
   | l_eq VIRG eq
   |
   ;

eq : debeq multi_membre op membre fin_mult_membre feq
   ;

debeq :
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

membre   : addop terme 
   | { fac = VALUE_ONE;} terme
   | membre addop terme
   ;

terme : const ident 
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

ident : IDENT
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

const : CONSTANTE
   { sscan_Value(yytext, &valcst);
     $$ = (Value) valcst;
   }
   ;

op : INF
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
addop : PLUS
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

%%

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

