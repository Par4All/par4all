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
/* Name     : pip_interface.c
 * Package  : paf-util
 * Author   : A. Leservot
 * Date     : 01 12 1993
 * Historic :
 * Documents:
 * Comments : Functions to call directly PIP from C3 and get results back in 
 *		a newgen data (quast).
 */

/* Ansi includes 	*/
#include <setjmp.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

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
#include "matrix.h"

/* Pips includes 	*/
#include "ri.h"
#include "constants.h"
#include "ri-util.h"
#include "misc.h"
#include "bootstrap.h"
#include "complexity_ri.h"
#include "database.h"
#include "graph.h"
#include "dg.h"
#include "paf_ri.h"
#include "parser_private.h"
#include "property.h"
#include "reduction.h"
#include "text.h"
#include "paf-util.h"
#include "static_controlize.h"
#include "pip.h"


/* Global variables 	*/
Pbase 	base_var_ref, base_ref, old_base, old_base_var;
int   	ind_min_max;
extern 	quast quast_act;	
extern 	expression expression_act;	
extern 	int ind_min_max;	/* Tag for MIN or MAX resolution */
extern 	Pbase base_var_ref,	/* Base of the unknowns */
		old_base_var,   	/* Base of the unknowns */
		base_ref,			/* Base of the parameters */
		old_base;			/* Base of the parameters */

struct S {
        int     flags;
        Entier  param1, param2;
};
extern struct S sol_space[];


/* Internal variables 	*/
#define Nil  1
#define If   2
#define List 3
#define Form 4
#define New  5
#define Div  6
#define Val  7
Pvecteur vect_for_sort;	/* Useful for the sorting of the variables in the
			 * system's Pvecteur.
			 */



/*===========================================================================*/
/* int integer_sol_edit((int) i)				AL 28/03/94
 * We will simulate a patern matching when the solution is produced.
 * Warning : we use here the Feautrier version of the pgcd (accept negative nb).
 */
int integer_sol_edit(i)
int i;
{
    int j, n, first_entier, second_entier = 0, longueur_liste;
    struct S *p;
    Entier N, D, d;
    p = sol_space + i;

    switch(p->flags) {
	/* We have a newparm */
    case New : n = p->param1;
	first_entier = n;
	i++; p++; /* call to Div case */
	/* Let's take the first vector : vecteur2 (case Form) */
	i++; p++; /* Call to Form */
	n = p->param1;
	/* Take all the coefficient2 */
	init_vecteur();
	for(j = 0; j<n; j++) {
	    i++; p++;
	    N = p->param1; D = p->param2;
	    d = sol_pgcd(N, D);

	    if(d == D){ 
		if(N/d < 0) ecrit_coeff_neg2( -N/d );
		else ecrit_coeff2( N/d );
	    }
	    /* Should not be called here */
	    else{ pips_internal_error("Division 1 in newparm");
	      }
	}
	i++; p++;

	/* Take the corresponding new parameter */
	N = p->param1; D = p->param2;
	d = sol_pgcd(N, D);
	if(d == D){
	    second_entier =  N/d;
	}
	/* Should not happen here */
	else{
	    pips_internal_error("Division 2 in newparm");
	  }
	i++; p++;

	ajoute_new_var( second_entier, first_entier );

	i = integer_sol_edit( i ); /* Look at the superquast */
	
	retire_par_de_pile();
	break;

	/* Our quast is a conditional */
    case If  : init_quast();
	/* Take vecteur1 part of the if */
	i++; p++; /* Call to case Form */
	
	creer_Psysteme();

	n = p->param1;
	for(j = 0; j<n; j++) {
	    i++; p++;
	    N = p->param1; D = p->param2;
	    d = sol_pgcd(N, D);
	    if(d == D){
		ecrit_coeff1( N/d );
	    }
	    /* Should not be called here */
	    else{ pips_internal_error("Division 3 in newparm");
	      }
	}

	creer_predicat();
	i++; p++;

	/* Take true super quast */
	i = integer_sol_edit(i);

	creer_true_quast();

	/* Take false super quast */
	i = integer_sol_edit(i);

	fait_quast_value();
	
	fait_quast();
	break;

	/* Quast is a list of solutions */
    case List: init_quast();
	longueur_liste = p->param1;
	if (longueur_liste > 0) init_liste_vecteur();

	i++; p++; /* call to the liste_vecteur */
	/* Take each vecteur (call to Form case) */
	while(longueur_liste--) {
	    init_vecteur();
	    n = p->param1;
	    for(j = 0; j<n; j++) {
		i++; p++;
		N = p->param1; D = p->param2;
		d = sol_pgcd(N, D);
		if(d == D){
		    if (N/d < 0) ecrit_une_var_neg(-N/d);
		    else ecrit_une_var( N/d );
		}
		/* Should not happen here */
		else{
		    pips_internal_error("Division 4 in newparm");
		  }
	    }
	    ecrit_liste_vecteur();
	    i++; p++;
	}

	creer_quast_value();
	fait_quast();
	break;

	/* We have an undefined quast */
    case Nil : init_quast();
	creer_quast_value();
	fait_quast();
	i++; break;

	/* This should not happen any more */
    case Form: pips_internal_error("Form case call");
	break;

	/* This case should not happen any more */
    case Div: 	pips_internal_error("Div case call");
	break;

	/* This case should not happen any more */
    case Val:  pips_internal_error("Val case call");
	break;

    default  : pips_internal_error("Undefined kind of quast ");
    }
	
    return(i);
}

/*===========================================================================*/
/* int rational_sol_edit((int) i)							AL 28/03/94
 * We will simulate a patern matching when the solution is produced.
 * Warning : we use here the Feautrier version of the pgcd (accept negative nb).
 */
int rational_sol_edit(i)
int i;
{int j, n, longueur_liste;
 struct S *p,*p_init;
 Entier N, D, d;
 int	lcm, i_init;


 p = sol_space + i;

 switch(p->flags) {




     /* We have a newparm */
     case New: 
		pips_internal_error("There is a new parameter for a rational compute !");
		break;


    /* Our quast is a conditional */
     case If  : init_quast();

		/* Take vecteur1 part of the if */
		i++; p++; /* Call to case Form */

		creer_Psysteme();

		n = p->param1;

		/* First scan to get the lcm (ppcm) of the denominator */
		i_init = i;
		p_init = p;
		lcm = 1;
		for(j = 0; j<n; j++) {
		    i++; p++;
		    N = p->param1; D = p->param2;
			d = sol_pgcd( N, D );
			lcm = sol_ppcm( lcm, D/d );
		}

		/* Then write the new predicate */
		i = i_init;
		p = p_init;
		for(j = 0; j<n; j++) {
		    i++; p++;
		    N = p->param1; D = p->param2;
		    ecrit_coeff1( (lcm*N)/D );
		}

		creer_predicat();
		i++; p++;


		/* Take true super quast */
        i = rational_sol_edit(i);


 		creer_true_quast();


		/* Take false super quast */
        i = rational_sol_edit(i);


		fait_quast_value();

		fait_quast();
        break;





     /* Quast is a list of solutions */
     case List: init_quast();
        longueur_liste = p->param1;
		if (longueur_liste > 0) init_liste_vecteur();


		i++; p++; /* call to the liste_vecteur */
		/* Take each vecteur (call to Form case) */
		while(longueur_liste--) {
			init_vecteur();
			n = p->param1;

			/* First scan to get the lcm (ppcm) of the denominator */
			i_init = i;
			p_init = p;
			lcm = 1;
			for(j = 0; j<n; j++) {
		    	i++; p++;
		    	N = p->param1; D = p->param2;
				d = sol_pgcd( N, D );
				lcm = sol_ppcm( lcm, D/d );
			}

			/* Then write the expression */
			i = i_init;
			p = p_init;
			for(j = 0; j<n; j++) {
				i++; p++;
				N = p->param1; D = p->param2;
				d = (lcm*N)/D;
				if (d < 0) ecrit_une_var_neg( -d );
				else ecrit_une_var( d );
		   	}


			expression_act = make_op_exp(DIVIDE_OPERATOR_NAME,
						     expression_act,
						     int_to_expression(lcm));
			ecrit_liste_vecteur();
			i++; p++;
		}

		creer_quast_value();
		fait_quast();
		break;




     /* We have an undefined quast */
     case Nil : init_quast();
		creer_quast_value();
		fait_quast();
		i++; break;




     /* This should not happen any more */
     case Form: pips_internal_error("Form case call");
		break;

     /* This case should not happen any more */
     case Div: 	pips_internal_error("Div case call");
		break;

     /* This case should not happen any more */
     case Val:  pips_internal_error("Val case call");
		break;


     default: pips_internal_error("Undefined kind of quast ");

	}
	
    return(i);
}

/*===========================================================================*/
/* int new_sol_edit((int) i)				AL 8/12/93
 * We will simulate a patern matching when the solution is produced.
 * Warning : we use here the Feautrier version of the pgcd (accept negative nb).
 *
 * Just keep for compatibility. Should be thrown away AL 28 03 94.
 */
int new_sol_edit(i)
int i;
{
 int j, n, first_entier, second_entier = 0, longueur_liste;
 struct S *p;
 Entier N, D, d;
 p = sol_space + i;

 switch(p->flags) {




     /* We have a newparm */
     case New : n = p->param1;
		first_entier = n;

		i++; p++; /* call to Div case */
		/* Let's take the first vector : vecteur2 (case Form) */
		i++; p++; /* Call to Form */
                n = p->param1;
		/* Take all the coefficient2 */
		init_vecteur();
                for(j = 0; j<n; j++) {
                    i++; p++;
                    N = p->param1; D = p->param2;
                    d = sol_pgcd(N, D);

                    if(d == D){ 
			if(N/d < 0) ecrit_coeff_neg2( -N/d );
			else ecrit_coeff2( N/d );
                    }
		    /* Should not be called here */
                    else{ pips_internal_error("Division 1 in newparm");
		    }
                }
                i++; p++;


		/* Take the corresponding new parameter */
     		N = p->param1; D = p->param2;
                d = sol_pgcd(N, D);
                if(d == D){ second_entier =  N/d;
                }
		/* Should not happen here */
                else{ pips_internal_error("Division 2 in newparm");
		}
                i++; p++;

		ajoute_new_var( second_entier, first_entier );


		i = new_sol_edit( i ); /* Look at the superquast */

		retire_par_de_pile();
		break;



    /* Our quast is a conditional */
     case If  : init_quast();

		/* Take vecteur1 part of the if */
		i++; p++; /* Call to case Form */

		creer_Psysteme();

                n = p->param1;
                for(j = 0; j<n; j++) {
		    i++; p++;
                    N = p->param1; D = p->param2;
                    d = sol_pgcd(N, D);
                    if(d == D){
			ecrit_coeff1( N/d );
		    }
		    /* Should not be called here */
                    else{ pips_internal_error("Division 3 in newparm");
		    }
                }

		creer_predicat();
		i++; p++;


		/* Take true super quast */
                i = new_sol_edit(i);


 		creer_true_quast();


		/* Take false super quast */
                i = new_sol_edit(i);


		fait_quast_value();

		fait_quast();
                break;





     /* Quast is a list of solutions */
     case List: init_quast();
                longueur_liste = p->param1;
		if (longueur_liste > 0) init_liste_vecteur();


                i++; p++; /* call to the liste_vecteur */
		/* Take each vecteur (call to Form case) */
                while(longueur_liste--) {
			init_vecteur();
                	n = p->param1;
                	for(j = 0; j<n; j++) {
				i++; p++;
                    		N = p->param1; D = p->param2;
                    		d = sol_pgcd(N, D);
                    		if(d == D){
					if (N/d < 0) ecrit_une_var_neg(-N/d);
					else ecrit_une_var( N/d );
                       		}
				/* Should not happen here */
                    		else{ pips_internal_error("Division 4 in newparm");
		    		}
                   	}
			ecrit_liste_vecteur();
                	i++; p++;
		}

		creer_quast_value();
		fait_quast();
                break;




     /* We have an undefined quast */
     case Nil : init_quast();
		creer_quast_value();
		fait_quast();
                i++; break;




     /* This should not happen any more */
     case Form: pips_internal_error("Form case call");
                break;

     /* This case should not happen any more */
     case Div: 	pips_internal_error("Div case call");
                break;

     /* This case should not happen any more */
     case Val:  pips_internal_error("Val case call");
                break;


     default  : pips_internal_error("Undefined kind of quast ");
    }

	
    return(i);
}



/*===========================================================================*/
/* void new_ecrit_ligne(p_vect, p_sys_base, nb_var, in_val)		AL 6/12/93
 */
void new_ecrit_ligne(p_vect, p_sys_base, nb_var, in_val)
Entier		*in_val;
Pvecteur        p_vect;
Pbase           p_sys_base;
int             nb_var;
{
	Pvecteur        base = (Pvecteur) p_sys_base;
	int             aux = 0;      /* Compteur de variables deja vues */


	/* We run over the base and fill in in_val */
	/* First, we put variables, then constant terms and then parameters */
	for(; (base != NULL) || (aux <= nb_var); aux++) {
		int val;

		if(aux == nb_var) val = (int) vect_coeff(TCST, p_vect);
		else  {
			val = (int) vect_coeff(base->var, p_vect);
			base = base->succ;
		}

		*(in_val+aux) = -val;
	}
}

/*===========================================================================*/
/* Tableau* sc_to_tableau( (Psysteme) in_ps, (int) nb_var )		AL 6/12/93 
 * Allocates a new Tableau and fill it with in_ps.
 * nb_var represents the number of variables in the systeme in_ps.
 * The input systeme base should be ordered: 
 *			nb_var first, constant term, then the parameters.
 * If nb_var = 0, there is no variables : the order is then 
 *			parameters first, then constant.
 */
Tableau * sc_to_tableau( in_ps, nb_var )
Psysteme	in_ps;
int		nb_var;
{
	Tableau 	*p;
	int 		h, w, n;
	int 		i;
        Pcontrainte     cont;

	debug(7, "sc_to_tableau", "Input Psysteme :\n");
	if (get_debug_level()>7) fprint_psysteme(stderr, in_ps);

	/* Let's define h, w, and n according to tab_get in Pip/tab.c*/
	h = 2*in_ps->nb_eq + in_ps->nb_ineq;
	w = in_ps->dimension + 1;
	n = nb_var;


	p = tab_alloc(h, w, n);
	if (in_ps == NULL) return NULL;

	/* If nb_var = 0, put parameter before constant */
	if (nb_var == 0) nb_var = vect_size(in_ps->base);

	i = n;
	for( cont = in_ps->egalites; cont != NULL; cont = cont->succ) {
		p->row[i].flags = Unknown;
                new_ecrit_ligne(cont->vecteur, in_ps->base, nb_var, 
					p->row[i].objet.val);
		i++;


		p->row[i].flags = Unknown;
                new_ecrit_ligne(vect_multiply(vect_dup(cont->vecteur),
					      VALUE_MONE), 
				in_ps->base, nb_var, p->row[i].objet.val);
		i++;
        }
	for( cont = in_ps->inegalites; cont != NULL; cont = cont->succ) {
		p->row[i].flags = Unknown;
               	new_ecrit_ligne(cont->vecteur, in_ps->base, nb_var,
                	           p->row[i].objet.val);
		i++;
        }

	debug(7, "sc_to_tableau", "Output Tableau :\n");
	if (get_debug_level()>6) tab_display(p, stderr);
	return((Tableau *) p);
}

