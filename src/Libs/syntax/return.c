#include <stdio.h>

#include "genC.h"
#include "parser_private.h"
#include "ri.h"
#include "ri-util.h"

#include "syntax.h"

LOCAL entity end_label = entity_undefined;
LOCAL char *end_label_local_name = RETURN_LABEL_NAME;



/* This function creates a goto instruction to label end_label. this is
 * done to eliminate return statements.
 *
 * Note: I was afraid the mouse trap would not work to analyze
 * multiple procedures but there is no problem. I guess that MakeGotoInst()
 * generates the proper label entity regardless of end_label. FI.
 */

instruction MakeReturn()
{
    if (end_label == entity_undefined) {
	end_label = MakeLabel(end_label_local_name);
    }

    return(MakeGotoInst(end_label_local_name));
}

void GenerateReturn()
{
    strcpy(lab_I, end_label_local_name);
    LinkInstToCurrentBlock(MakeZeroOrOneArgCallInst("RETURN", 
						    expression_undefined));
}
