 /* main for transformer package
  *
  * Tests link-editing, no more
  *
  * Francois Irigoin, 21 April 1990
  */

#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "transformer.h"

main()
{
    transformer t1 = transformer_undefined;
    transformer t2 = transformer_undefined;
    transformer t3;
    
    t3 = transformer_convex_hull(t1, t2);
    (void) print_transformer(t3);
}
