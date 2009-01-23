/* 
 * $Id$
 */
#include "genC.h"    

/* cette structure contient une pile. La tete de cette pile contient le statement courant */
/* depth represente  la profondeur des nids */ 

typedef struct {
  hash_table contenu;
  hash_table depth;
  stack statement_stack;
} * context_p, context_t;


