/* package conversion
*/

#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"

#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "matrice.h"
#include "conversion.h"

/* void change_of_base_index(Pbase base_oldindex, Pbase *base_newindex)
 * change of variable index from  base_oldindex to
 * base_newindex 
*/
void change_of_base_index(base_oldindex, base_newindex)
Pbase base_oldindex;
Pbase *base_newindex;
{
    Pbase pb;
    entity new_name;

    for (pb=base_oldindex; pb!=NULL; pb=pb->succ)
    {
	new_name = make_index_entity((entity) pb->var);
	base_add_dimension(base_newindex, (Variable) new_name);
    }
    *base_newindex = base_reversal(*base_newindex);
}



/*entity make_index_entity(old_index)
 *create a new entity for a new index 
 */
entity make_index_entity(old_index)
entity old_index;
{
    entity new_index;
    string old_name;
    char *new_name = (char*) malloc(16);

    old_name = entity_name(old_index);
    for (sprintf(new_name, "%s%s", old_name, "p");
         gen_find_tabulated(new_name, entity_domain)!=entity_undefined; 

         old_name = new_name)
        sprintf(new_name, "%s%s", old_name, "p");
    new_index=make_entity(new_name,entity_type(old_index),
			  entity_storage(old_index),entity_initial(old_index));
    return(new_index);
}

/* Psysteme sc_change_baseindex(Psysteme sc, Pbase base_old, Pbase base_new)
 * le changement de base d'indice pour sc
 */
Psysteme sc_change_baseindex(sc, base_old, base_new)
Psysteme sc;
Pbase base_old;
Pbase base_new;
{
    Pbase pb1, pb2;

    for(pb1=base_old,pb2=base_new;pb1!=NULL;pb1=pb1->succ,pb2=pb2->succ)
	sc_variable_rename(sc,pb1->var,pb2->var);
    return(sc);
}

			    



