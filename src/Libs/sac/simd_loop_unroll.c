/*

  $Id$

  Copyright 1989-2009 MINES ParisTech

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

#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "resources.h"

#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"

#include "semantics.h"
#include "effects-generic.h"
#include "transformations.h"

#include "control.h"

#include "dg.h"

typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;

#include "graph.h"

#include "sac.h"

#include "properties.h"

#include <limits.h>
#include "preprocessor.h"

#include "text-util.h"


static void addSimdCommentToStat(statement s, int num)
{

    char comment[sizeof("c " SIMD_COMMENT) + 8*sizeof(int) ];

    sprintf(comment, "c " SIMD_COMMENT "%d\n", num);

    insert_comments_to_statement(s, comment);
}

static void addSimdCommentToStats(statement s)
{
    if(instruction_sequence_p(statement_instruction(s)))
    {
        sequence seq = instruction_sequence(statement_instruction(s));
        int num = 0;
        FOREACH(STATEMENT, curStat, sequence_statements(seq) )
        {
            addSimdCommentToStat(curStat, num++);
        }
    }
    else
    {
        addSimdCommentToStat(s, 0);
    }
}

void simd_loop_unroll(statement loop_statement, int rate)
{
    do_loop_unroll(loop_statement,rate,addSimdCommentToStats);
}
