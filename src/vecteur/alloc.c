 /*
  * CREATION, COPIE ET DESTRUCTION D'UN VECTEUR
  */

/*LINTLIBRARY*/

#include <stdio.h>
#include <malloc.h>
#include <varargs.h>

#include "boolean.h"
#include "vecteur.h"

#define MALLOC(s,t,f) malloc(s)
#define FREE(p,t,f) free(p)

/* Pvecteur vect_dup(Pvecteur v_in): duplication du vecteur v_in; allocation de
 * et copie dans v_out;
 *
 * allocate v_out;
 * v_out := v_in;
 */
Pvecteur vect_dup(v_in)
Pvecteur v_in;
{
    Pvecteur v_out;
    Pvecteur v;

    v_out = NULL;
    for(v=v_in; v!=NULL; v=v->succ) {
	v_out = vect_chain(v_out,var_of(v),val_of(v));
    }
    return(v_out);
}

/* void vect_rm(Pvecteur v): desallocation des couples de v;
 *
 * Attention! La procedure appelante doit penser a modifier la
 * valeur de v apres l'appel pour ne pas pointer dans le vide:
 *   vect_rm(v);
 *   v = NULL;
 *
 * Il vaudrait mieux que vect_rm retourne NULL et qu'on puisse
 * ecrire:
 *  v = vect_rm(v);
 * ou que vect_rm prenne un Pvecteur * en argument:
 *  vect_rm(&v);
 */
void vect_rm(v)
Pvecteur v;
{
    while (v != NULL) {
	Pvecteur nv = v->succ;
	free((char *) v);
	v = nv;
    }
}

/* Pvecteur vect_new(Variable var,Value coeff): 
 * allocation d'un vecteur colineaire
 * au vecteur de base var et de coefficient coeff (i.e. creation
 * d'un nouveau vecteur ne comportant qu'un seul couple (var,coeff))
 *
 *       --> 
 * coeff var
 * 
 * Pourrait etre remplace par un vect_chain(NULL,,)
 * 
 * Modifications:
 *  - a 0 coeff generates a null vector; Francois Irigoin, 26 March 1991
 */
Pvecteur vect_new(var,coeff)
Variable var;
Value coeff;
{
    Pvecteur v;

    if(coeff!=0) {
	v = (Pvecteur) MALLOC(sizeof(Svecteur),VECTEUR,"vect_new");
	if (v == NULL) {
	    (void) fprintf(stderr,"vect_new: Out of memory space\n");
	    exit(-1);
	}
	var_of(v) = var;
	val_of(v) = coeff;
	v->succ = NULL;
    
    }
    else
	v = NULL;

    return (v);
}

/* void dbg_vect_rm(Pvecteur v, char * f): desallocation d'un vecteur
 * avec marquage de la fonction provoquant la desallocation
 *
 * Apparemment obsolete.
 */
/*ARGSUSED*/
void dbg_vect_rm(v,f)
Pvecteur v;
char *f;
{
    Pvecteur v1,v2;
    v1 = v;
    while (v1!=NULL) {
	v2 = v1->succ;
	FREE((char *)v1,VECTEUR,f);
	v1 = v2;
    }
}

/* Pvecteur vect_make(v, [var, val,]* 0, val)
 * Pvecteur v;
 * Variable var;
 * Value val;
 *
 * builds a vector from the list of arguments, by successive additions.
 * ends when a 0 Variable is encountered (that is TCST if any!)
 * caution: because of the var val order, this function cannot be
 * called directly with a va_list, but (va_list, 0) should be used,
 * since the val argument is expected and read.
 *
 * CAUTION: the initial vector is modified by the process!
 */
Pvecteur vect_make(va_alist)
va_dcl
{
    va_list ap = NULL;
    Variable var = (Variable) NULL;
    Value val = (Value) 0;
    Pvecteur v = VECTEUR_UNDEFINED;

    va_start (ap);
    v = va_arg(ap, Pvecteur);
    do
    {
	var = va_arg(ap, Variable);
	val = va_arg(ap, Value);

	vect_add_elem(&v, var, val);
    }
    while (var != (Variable)0)
    va_end (ap);

    return(v);
}


Pbase base_dup( b )
Pbase b;
{
  Pvecteur v1, v2;

  v2 = vect_dup((Pvecteur) b);
  v1 = base_reversal(v2);
  vect_rm(v2);                
   return(v1);
}
