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
  pips_assert("some image", e!=entity_undefined);
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
  string attribute =
    what_operation_shape(vtxcontent_optype(dagvtx_content(vtx)));

  fprintf(out, "  \"%" _intFMT " %s\" [%s];\n",
	  dagvtx_number(vtx), dagvtx_operation(vtx), attribute);

  FOREACH(dagvtx, succ, dagvtx_succs(vtx))
    fprintf(out, "  \"%" _intFMT " %s\" -> \"%" _intFMT " %s\";\n",
	    dagvtx_number(vtx), dagvtx_operation(vtx),
	    dagvtx_number(succ), dagvtx_operation(succ));
}

static void entity_list_dot(FILE * out, string comment, list l)
{
  if (comment) fprintf(out, "  // %s\n", comment);
  FOREACH(entity, e, l)
    fprintf(out, "  \"%s\" [shape=circle];\n", entity_local_name(e));
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

/* return target predecessors as a list.
 */
list dag_vertex_preds(dag d, dagvtx target)
{
  list preds = NIL;
  FOREACH(dagvtx, v, dag_vertices(d))
    if (v!=target && gen_in_list_p(target, dagvtx_succs(v)))
      preds = CONS(dagvtx, v, preds);
  return preds;
}

/* remove AIPO copies detected as useless.
 */
void dag_remove_useless_copies(dag d)
{
  set remove = set_make(set_pointer);

  ifdebug(4) {
    pips_debug(4, "considering dag:\n");
    dag_dump(stderr, "input", d);
  }

  // one pass is needed because we're going backwards
  FOREACH(dagvtx, v, dag_vertices(d))
  {
    vtxcontent c = dagvtx_content(v);
    const freia_api_t * api = get_freia_api(vtxcontent_opid(c));
    // freia_aipo_copy(out, in) where out is not used...
    if (same_string_p(AIPO "copy", api->function_name))
    {
      entity target = vtxcontent_out(c);
      pips_assert("one output and one input to copy",
	  target!=entity_undefined && gen_length(vtxcontent_inputs(c))==1);

      // replace by its source everywhere it is used
      entity source = ENTITY(CAR(vtxcontent_inputs(c)));
      // may be NULL if source is an input
      dagvtx prod = get_producer(d, v, source);

      // add v successors as successors of prod
      // v is kept as a successor in case it is not removed
      if (prod)
      {
	FOREACH(dagvtx, vs, dagvtx_succs(v))
	  dagvtx_succs(prod) = gen_once(vs, dagvtx_succs(prod));
      }

      // replace target image by source image in all v successors
      FOREACH(dagvtx, succ, dagvtx_succs(v))
      {
	vtxcontent sc = dagvtx_content(succ);
	gen_list_patch(vtxcontent_inputs(sc), target, source);
      }

      // v has no more successors
      gen_free_list(dagvtx_succs(v));
      dagvtx_succs(v) = NIL;

      // whether to actually remove v
      if (gen_in_list_p(target, dag_outputs(d))) {
	if (get_producer(d, NULL, target)!=v) {
	  set_add_element(remove, remove, v);
	}
	// else it is kept
      }
      else { // not needed at all
	set_add_element(remove, remove, v);
      }
    }
  }

  SET_FOREACH(dagvtx, r, remove)
  {
    pips_debug(5, "removing vertex %" _intFMT "\n", dagvtx_number(r));

    vtxcontent c = dagvtx_content(r);
    hwac_kill_statement(pstatement_statement(vtxcontent_source(c)));
    dag_remove_vertex(d, r);

    ifdebug(8)
      gen_context_recurse(d, r, dagvtx_domain, gen_true, check_removed);

    free_dagvtx(r);
  }

  set_free(remove);
}

/* (re)compute the list of *GLOBAL* input & output images for this dag
 * ??? BUG the output is rather an approximation
 * should rely on used defs or out effects for the underlying sequence.
 */
void dag_set_inputs_outputs(dag d)
{
  set ins = set_make(set_pointer), outs = set_make(set_pointer);

  FOREACH(dagvtx, v, dag_vertices(d))
  {
    vtxcontent c = dagvtx_content(v);
    entity out = vtxcontent_out(c);

    if (out!=entity_undefined)
    {
      // keep computations results that are not used afterwards?
      if (!set_belong_p(ins, out))
	set_add_element(outs, outs, out);
      // or that are formal parameters (external image reuse)
      else if (formal_parameter_p(out))
	set_add_element(outs, outs, out);

      // remove from current inputs as produced locally
      set_del_element(ins, ins, out);
    }

    FOREACH(entity, i, vtxcontent_inputs(c))
      set_add_element(ins, ins, i);
  }

  ifdebug(9)
  {
    dag_dump(stderr, "debug dag_set_inputs_outputs", d);
    set_fprint(stderr, "new computed ins", ins,
	       (gen_string_func_t) entity_local_name);
    set_fprint(stderr, "new computed outs", outs,
	       (gen_string_func_t) entity_local_name);
  }

  // update dag
  gen_free_list(dag_inputs(d));
  dag_inputs(d) = set_to_sorted_list(ins, (gen_cmp_func_t) compare_entities);

  gen_free_list(dag_outputs(d));
  dag_outputs(d) = set_to_sorted_list(outs, (gen_cmp_func_t) compare_entities);

  // cleanup
  set_free(ins);
  set_free(outs);
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
