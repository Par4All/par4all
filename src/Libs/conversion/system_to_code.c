/*
 * HPFC module by Fabien COELHO
 *
 * SCCS stuff:
 * $RCSfile: system_to_code.c,v $ ($Date: 1994/03/16 12:04:56 $, ) version $Revision$, got on %D%, %T%
 * $Id$
 */

/*
 * Standard includes
 */
 
#include <stdio.h>
#include <string.h> 
extern fprintf();

/*
 * Psystems stuff
 */

#include "types.h"
#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

/*
 * Newgen stuff
 */

#include "genC.h"

#include "ri.h" 
#include "hpf.h" 
#include "hpf_private.h"

/*
 * PIPS stuff
 */

#include "ri-util.h" 
#include "misc.h" 
#include "control.h"
#include "regions.h"
#include "semantics.h"
#include "effects.h"

/* 
 * my own local includes
 */

#include "hpfc.h"
#include "defines-local.h"

/*
 * ONE ARRAY REFERENCES MODIFICATIONS
 */


/*
 * that's all
 */
