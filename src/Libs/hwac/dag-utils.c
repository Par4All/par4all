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

#include <stdint.h>
#include <stdlib.h>

#include "genC.h"
#include "misc.h"
#include "freia_spoc.h"

#include "linear.h"
#include "pipsdbm.h"

#include "ri.h"
#include "ri-util.h"
#include "properties.h"

#include "freia_spoc_private.h"
#include "hwac.h"

#define cat concatenate

static void entity_list_dump(FILE * out, string what, list l)
{
  fprintf(out, "%s: ", what);
  FOREACH(entity, e, l)
    fprintf(out, " %s,", safe_entity_name(e));
  fprintf(out, "\n");
}

_int dagvtx_number(const dagvtx v)
{
  if (v==NULL) return 0;
  return statement_number
    (pstatement_statement(vtxcontent_source(dagvtx_content(v))));
}

string dagvtx_operation(const dagvtx v)
{
  if (v==NULL) return "null";
  int index = vtxcontent_opid(dagvtx_content(v));
  const freia_api_t * api = get_freia_api(index);
  return api->function_name + strlen(AIPO);
}

/* return (last) producer vertex or NULL if none found.
 * this is one of the two predecessors of sink.
 */
static dagvtx get_producer(dag d, dagvtx sink, entity e)
{
  FOREACH(dagvtx, v, dag_vertices(d))
  {
    vtxcontent c = dagvtx_content(v);
    // the image may kept within a pipe
    if (vtxcontent_out(c)==e &&
	(sink==NULL || gen_in_list_p(sink, dagvtx_succs(v))))
      return v;
  }
  return NULL; // it is an external parameter?
}

void dagvtx_nb_dump(FILE * out, string what, list l)
{
  fprintf(out, "%s: ", what);
  FOREACH(dagvtx, v, l)
    fprintf(out, " %" _intFMT ",", dagvtx_number(v));
  fprintf(out, "\n");
}

/* for dag debug.
 */
void dagvtx_dump(FILE * out, string name, dagvtx v)
{
  if (!v) {
    fprintf(out, "vertex %s is NULL\n", name? name: "");
    return;
  }
  fprintf(out, "vertex %s %" _intFMT " %s (%p)\n",
	  name? name: "", dagvtx_number(v), dagvtx_operation(v), v);
  dagvtx_nb_dump(out, "  succs", dagvtx_succs(v));
  vtxcontent c = dagvtx_content(v);
  statement s = pstatement_statement(vtxcontent_source(c));
  fprintf(out,
	  "  optype: %s\n"
	  "  opid: %" _intFMT "\n"
	  "  source: %" _intFMT "/%" _intFMT "\n",
	  what_operation(vtxcontent_optype(c)),
	  vtxcontent_opid(c),
	  statement_number(s),
	  statement_ordering(s));
  entity_list_dump(out, "  inputs", vtxcontent_inputs(c));
  fprintf(out, "  output: %s\n", safe_entity_name(vtxcontent_out(c)));
  // to be continued...
}

/* for dag debug
 */
void dag_dump(FILE * out, string what, dag d)
{
  fprintf(out, "dag '%s' (%p):\n", what, d);

  entity_list_dump(out, "inputs", dag_inputs(d));
  entity_list_dump(out, "outputs", dag_outputs(d));

  FOREACH(dagvtx, vx, dag_vertices(d)) {
    dagvtx_dump(out, NULL, vx);
    fprintf(out, "\n");
  }

  fprintf(out, "\n");
}

static void dagvtx_dot(FILE * out, dagvtx vtx)
{
  // TODO: set the shape/color depending on the operation?

  fprintf(out, "  \"%" _intFMT " %s\" [shape=circle];\n",
	  dagvtx_number(vtx), dagvtx_operation(vtx));

  FOREACH(dagvtx, succ, dagvtx_succs(vtx))
    fprintf(out, "  \"%" _intFMT " %s\" -> \"%" _intFMT " %s\";\n",
	    dagvtx_number(vtx), dagvtx_operation(vtx),
	    dagvtx_number(succ), dagvtx_operation(succ));
}

static void entity_list_dot(FILE * out, string comment, list l)
{
  if (comment) fprintf(out, "  // %s\n", comment);
  FOREACH(entity, e, l)
    fprintf(out, "  \"%s\" [shape=box];\n", entity_local_name(e));
  fprintf(out, "\n");
}

/* dag debug with dot format.
 */
void dag_dot(FILE * out, string what, dag d)
{
  fprintf(out, "digraph \"%s\" {\n", what);

  entity_list_dot(out, "inputs", dag_inputs(d));
  entity_list_dot(out, "outputs", dag_outputs(d));

  fprintf(out, "  // computation vertices\n");
  FOREACH(dagvtx, vx, dag_vertices(d))
  {
    dagvtx_dot(out, vx);
    vtxcontent c = dagvtx_content(vx);

    // show inputs
    FOREACH(entity, i, vtxcontent_inputs(c))
      if (gen_in_list_p(i, dag_inputs(d)) && get_producer(d, vx, i)==NULL)
	fprintf(out, "  \"%s\" -> \"%" _intFMT " %s\";\n",
		entity_local_name(i), dagvtx_number(vx), dagvtx_operation(vx));

    // and outputs
    entity o = vtxcontent_out(c);
    if (gen_in_list_p(o, dag_outputs(d)) && get_producer(d, NULL, o)==vx)
      // no!
      fprintf(out, "  \"%" _intFMT " %s\" -> \"%s\";\n",
	      dagvtx_number(vx), dagvtx_operation(vx), entity_local_name(o));
  }

  fprintf(out, "}\n");
}

#define DOT_SUFFIX ".dot"

/* generate a "dot" format from a dag to a file.
 */
void dag_dot_dump(string module, string name, dag d)
{
  // build file name
  string src_dir = db_get_directory_name_for_module(module);
  string fn = strdup(cat(src_dir, "/", name, DOT_SUFFIX, NULL));
  free(src_dir);

  FILE * out = safe_fopen(fn, "w");
  fprintf(out, "// graph for dag \"%s\" of module \"%s\" in dot format\n",
	  name, module);
  dag_dot(out, name, d);
  safe_fclose(out, fn);
  free(fn);
}

// debug help
static void check_removed(dagvtx v, dagvtx removed)
{ pips_assert("not removed vertex", v!=removed); }

/* remove vertex v from dag d.
 */
void dag_remove_vertex(dag d, dagvtx v)
{
  pips_assert("vertex is in dag", gen_in_list_p(v, dag_vertices(d)));

  // remove from vertex list
  gen_remove(&dag_vertices(d), v);

  // remove from successors of any...
  FOREACH(dagvtx, dv, dag_vertices(d))
    gen_remove(&dagvtx_succs(dv), v);

  // unlink vertex itself
  gen_free_list(dagvtx_succs(v)), dagvtx_succs(v) = NIL;

  ifdebug(8) gen_context_recurse(d, v, dagvtx_domain, gen_true, check_removed);

  // what about updating ins & outs?
}

/* append new vertex nv to dag d.
 */
void dag_append_vertex(dag d, dagvtx nv)
{
  pips_assert("not in dag", !gen_in_list_p(nv, dag_vertices(d)));
  pips_assert("no successors", dagvtx_succs(nv) == NIL);

  vtxcontent c = dagvtx_content(nv);
  list ins = vtxcontent_inputs(c);

  FOREACH(entity, e, ins)
  {
    dagvtx pv = get_producer(d, NULL, e);
    if (pv) {
      dagvtx_succs(pv) = gen_once(nv, dagvtx_succs(pv));
    }
    // global dag inputs are computed later.
    // else dag_inputs(d) = gen_once(e, dag_inputs(d));
  }
  dag_vertices(d) = CONS(dagvtx, nv, dag_vertices(d));
  // ??? what about scalar deps?
}

/* replace list items specified by hash_table
 */
static void replace_in_list(hash_table replace, list l)
{
  while (l) {
    void * x = CHUNKP(CAR(l));
    if (hash_defined_p(replace, x))
      CHUNKP(CAR(l)) = hash_get(replace, x);
    l = CDR(l);
  }
}

/* remove AIPO copies detected as useless.
 */
void dag_remove_useless_copies(dag d)
{
  hash_table replace = hash_table_make(hash_pointer, 10);
  dagvtx remove;

  do
  {
    hash_table_clear(replace);
    remove = NULL;

    // look for one replacement at a time...
    FOREACH(dagvtx, v, dag_vertices(d))
    {
      vtxcontent c = dagvtx_content(v);
      entity target = vtxcontent_out(c);
      const freia_api_t * api = get_freia_api(vtxcontent_opid(c));
      // freia_aipo_copy(out, in) where out is not used...
      if (same_string_p(AIPO "copy", api->function_name) &&
	  vtxcontent_out(c)!=entity_undefined &&
	  !gen_in_list_p(vtxcontent_out(c), dag_outputs(d)))
      {
	entity source = ENTITY(CAR(vtxcontent_inputs(c)));
	remove = v;
	hash_put(replace, target, source);
	dagvtx prod = get_producer(d, v, source);
	// if there is no producer, it must be an input image...
	if (prod) hash_put(replace, v, prod);
	break;
      }
    }

    if (remove)
    {
      // perform subsitutions in dag? statements??
      FOREACH(dagvtx, v, dag_vertices(d))
      {
	if (v!=remove)
	{
	  vtxcontent c = dagvtx_content(v);
	  replace_in_list(replace, vtxcontent_inputs(c));
	  replace_in_list(replace, dagvtx_succs(v));
	}
      }

      vtxcontent c = dagvtx_content(remove);
      hwac_kill_statement(pstatement_statement(vtxcontent_source(c)));
      dag_remove_vertex(d, remove);

      ifdebug(8)
	gen_context_recurse(d, remove, dagvtx_domain, gen_true, check_removed);

      free_dagvtx(remove);
    }
  }
  while (remove);

  hash_table_free(replace);
}

/* (re)compute the list of *GLOBAL* input & output images for this dag
 * ??? BUG the output is rather an approximation
 * should rely on used defs or out effects for the underlying sequence.
 */
void dag_set_inputs_outputs(dag d)
{
  list ins = NIL, outs = NIL, lasts = NIL;
  // vertices are in reverse computation order...
  FOREACH(dagvtx, v, dag_vertices(d))
  {
    vtxcontent c = dagvtx_content(v);
    entity out = vtxcontent_out(c);

    if (out!=entity_undefined)
    {
      // all out variables
      outs = gen_once(out, outs);

      // if it is not used, then it is a production...
      if (!gen_in_list_p(out, ins))
	lasts = gen_once(out, lasts);

      // remove inputs that are produced locally
      gen_remove(&ins, out);
    }

    FOREACH(entity, i, vtxcontent_inputs(c))
      ins = gen_once(i, ins);
  }

  // keep argument order...
  ins = gen_nreverse(ins);

  ifdebug(9)
  {
    dag_dump(stderr, "debug dag_set_inputs_outputs", d);
    entity_list_dump(stderr, "computed ins", ins);
    entity_list_dump(stderr, "computed outs", outs);
    entity_list_dump(stderr, "computed lasts", lasts);
  }

  gen_free_list(dag_inputs(d));
  dag_inputs(d) = ins;

  gen_free_list(dag_outputs(d));
  dag_outputs(d) = lasts;

  gen_free_list(outs);
}

/* ??? I'm unsure about what happens to dead code in the pipeline...
 */

/* ??? BUG the code may not work properly when image variable are
 * reused within a pipe. The underlying issues are subtle and
 * would need careful thinking: the initial code is correct, the dag
 * representation is correct, but the generated code may reorder
 * dag vertices so that reused variables are made to interact one with
 * the other. Maybe I should recreate output variables in the generated code
 * for every pipeline. this would imply a cleanup phase to removed
 * unused images at the end. I would really need an SSA form on
 * images? this function checks the assumption before proceeding
 * further.
 */
bool single_image_assignement_p(dag d)
{
  set outs = set_make(set_pointer);
  FOREACH(dagvtx, v, dag_vertices(d))
  {
    vtxcontent c = dagvtx_content(v);
    entity out = vtxcontent_out(c);
    if (out!=entity_undefined)
    {
      if (set_belong_p(outs, out)) {
	set_free(outs);
	return false;
      }
      set_add_element(outs, outs, out);
    }
  }
  set_free(outs);
  return true;
}
