 /* NewGen interface with C3 type Pvecteur for PIPS project 
  *
  * Remi Triolet
  *
  * Bugs:
  *  - the NewGen interface makes it very cumbersome; function f()
  *    prevents a very simple fscanf() with format %d and %s and
  *    implies a copy in a temporary buffer (Francois Irigoin)
  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "ri-util.h"
#include "misc.h"

#define TCST_NAME "TERME_CONSTANT"

void vect_gen_write(fd, v)
FILE *fd;
Pvecteur v;
{
    Pvecteur p;

    fprintf(fd, "(");

    for (p = v; p != NULL; p = p->succ) {
	fprint_Value(fd, val_of(p));
	(void) fprintf(fd," %s ", 
		       (p->var == (Variable) 0) ? TCST_NAME : 
		       entity_name((entity) p->var));
    }

    (void) fprintf(fd, ")");
}

Pvecteur vect_gen_read(fd, f)
FILE *fd; /* ignored */
int (*f)();
{
/* VBUFSIZE was increased from 1024 to 2048 for cgg:opmkrnl; FI, 13 March 92 */
/* VBUFSIZE was increased from 2048 to 4096 for mat:zebulon5; FI, 3 Jan 97 */
/* VBUFSIZE was increased from 4096 to 8192 for mat:zebulon5; FI, 3 Jan 97 */
/* VBUFSIZE was increased from 8192 to 16384 for mat:zebulon5; FI, 3 Jan 97 */
/* VBUFSIZE was increased from 16384 to 32768 for mat:zebulon5; FI, 3 Jan 97 */
#define VBUFSIZE 32768
    Value val;
    Variable var;
    char *varname;
    char buffer[VBUFSIZE], *pbuffer;
    int ibuffer = 0;
    Pvecteur p = NULL;
    int c;

    if ((c = f()) != '(') {
	fprintf(stderr, "[vect_gen_read] missing '('\n");
	while ((c = f()) != EOF) {
	    (void) fprintf(stderr, "%c", c);
	}	    
	abort();
    }

    while ((c = f()) != ')') {
	if (ibuffer >= VBUFSIZE-1) {
	  while ((c = f()) != ')')
	    ibuffer++;
	    (void) fprintf(stderr,
			   "[vect_gen_read] buffer[%d] too small for %d bytes\n",
			   VBUFSIZE-1, ibuffer);
	    abort();
	}
	buffer[ibuffer++] = c;
    }
    buffer[ibuffer++] = '\0';

    pbuffer = strtok(buffer, " ");
    while (pbuffer != NULL) {
	sscan_Value(pbuffer, &val);
	varname = strtok(NULL, " ");
	if (strcmp(varname, TCST_NAME) == 0) {
	    var = (Variable) 0;
	}
	else {
	    var = (Variable) gen_find_tabulated(varname, entity_domain);
	    if (var == (Variable) entity_undefined) {
		fprintf(stderr, "[vect_gen_read] bad entity name: %s\n",
			varname);
		abort();
	    }
	}

	vect_add_elem(&p, var, val);

	pbuffer = strtok(NULL, " ");
    }

    p = vect_reversal(p);
    return(p);
}

void vect_gen_free(v)
Pvecteur v;
{
    vect_rm(v);
}

Pvecteur vect_gen_copy_tree(v)
Pvecteur v;
{
    return(vect_dup(v));
}

void vect_debug(v)
Pvecteur v;
{
    vect_fprint(stderr, v, entity_local_name);
}

/* comparison function for Pvecteur in pips
 */
int compare_Pvecteur(pv1, pv2)
Pvecteur *pv1, *pv2;
{
    return(compare_entities((entity*)&var_of(*pv1),
			    (entity*)&var_of(*pv2)));
}

int 
vect_gen_allocated_memory(
    Pvecteur v)
{
    int result = 0;
    for (; v; v=v->succ)
	result += sizeof(Svecteur);
    return result;
}

/*   That is all
 */
