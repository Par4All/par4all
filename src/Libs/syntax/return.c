#include <stdio.h>

#include "genC.h"
#include "parser_private.h"
#include "ri.h"

#include "syntax.h"

LOCAL entity end_label = entity_undefined;
LOCAL char *end_label_local_name = "00000";



/* 
this function creates a goto instruction to label end_label. this is
done to eliminate return statements.
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
