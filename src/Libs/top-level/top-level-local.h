/* Top-level declares a extern jmp_buf pips_top_level :
 */
#include <setjmp.h>

/* the following use to be "constants.h" alone in Include.
 * I put it there not to lose it someday. FC.
 */

#include "specs.h"

/* And now, a nice set of (minor) memory leak sources...
 */
 
/* Auxiliary data files
 */

#define PIPSMAKE_RC "pipsmake.rc"
#define DEFAULT_PIPSMAKE_RC \
  (strdup(concatenate(getenv("PIPS_ROOT"), "/Share/", PIPSMAKE_RC, NULL)))
#define WPIPS_RC \
  (strdup(concatenate(getenv("PIPS_ROOT"), "/Share/", "wpips.rc", NULL)))
#define BOOTSTRAP_FILE \
  (strdup(concatenate(getenv("PIPS_ROOT"), "/Share/", "BOOT-STRAP.entities", NULL)))
#define XV_HELP_FILE \
  (strdup(concatenate(getenv("PIPS_ROOT"), "/Share/", "pips_help.txt", NULL)))
 
#define PROPERTIES_FILE "properties.rc"
#define PROPERTIES_LIB_FILE \
  (strdup(concatenate(getenv("PIPS_ROOT"), "/Share/", PROPERTIES_FILE, NULL)))
 
#define MODEL_RC "model.rc"
#define DEFAULT_MODEL_RC \
  (strdup(concatenate(getenv("PIPS_ROOT"), "/Share/", MODEL_RC, NULL)))
 
/* filename extensions
 */
#define SEQUENTIAL_CODE_EXT ".code"
#define PARALLEL_CODE_EXT ".parcode"
 
#define SEQUENTIAL_FORTRAN_EXT ".f"
#define PARALLEL_FORTRAN_EXT ".parf"
#define PREDICAT_FORTRAN_EXT ".pref"
#define PRETTYPRINT_FORTRAN_EXT ".ppf"
 
#define WP65_BANK_EXT ".bank"
#define WP65_COMPUTE_EXT ".wp65"
 
#define ENTITIES_EXT ".entities"
 
#define EMACS_FILE_EXT "-emacs"
 
#define GRAPH_FILE_EXT "-graph"


/* Some directory names... */

/* Where is the output of HPFC in the workspace: */
#define HPFC_COMPILED_FILE_DIR "hpfc"
   
/* say that's all
 */
