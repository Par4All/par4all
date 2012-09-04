/*

  $Id$

  Copyright 1989-2011 MINES ParisTech

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

#ifdef HAVE_CONFIG_H
#include "pips_config.h"
#endif

#include <stdint.h>
#include <stdlib.h>

#include "genC.h"
#include "misc.h"
#include "freia.h"

#include "linear.h"
#include "pipsdbm.h"

#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "properties.h"

#include "freia_spoc_private.h"
#include "hwac.h"

// dirty debug helper...
string dagvtx_to_string(const dagvtx v)
{
  return itoa(dagvtx_number(v));
}

/* return statement if any, or NULL (for input nodes).
 */
statement dagvtx_statement(const dagvtx v)
{
  pstatement ps = vtxcontent_source(dagvtx_content(v));
  return pstatement_statement_p(ps)? pstatement_statement(ps): NULL;
}

/* @brief build the set of actual statements in d
 */
void dag_statements(set stats, const dag d)
{
  set_clear(stats);
  FOREACH(dagvtx, v, dag_vertices(d))
  {
    statement s = dagvtx_statement(v);
    if (s) set_add_element(stats, stats, s);
  }
}

/* a vertex with a non AIPO or image related statement.
 */
bool dagvtx_other_stuff_p(const dagvtx v)
{
  return vtxcontent_optype(dagvtx_content(v))==spoc_type_oth;
}

/* return the produced image or NULL */
entity dagvtx_image(const dagvtx v)
{
  vtxcontent c = dagvtx_content(v);
  return (vtxcontent_out(c) != entity_undefined)? vtxcontent_out(c): NULL;
}

static void entity_list_dump(FILE * out, const string what, const list l)
{
  fprintf(out, "%s: ", what);
  FOREACH(entity, e, l)
    fprintf(out, " %s,", safe_entity_name(e));
  fprintf(out, "\n");
}

/* returns the vertex number, i.e. the underlying statement number.
 */
_int dagvtx_number(const dagvtx v)
{
  if (v==NULL) return 0;
  pstatement source = vtxcontent_source(dagvtx_content(v));
  if (pstatement_statement_p(source))
  {
    statement s = pstatement_statement(source);
    return s? statement_number(s): 0;
  }
  else
    return 0;
}

string dagvtx_number_str(const dagvtx v)
{
  return itoa(dagvtx_number(v));
}

_int dagvtx_optype(const dagvtx v)
{
  return vtxcontent_optype(dagvtx_content(v));
}

_int dagvtx_opid(const dagvtx v)
{
  return vtxcontent_opid(dagvtx_content(v));
}

string dagvtx_function_name(const dagvtx v)
{
  if (v==NULL) return "null";
  int index = vtxcontent_opid(dagvtx_content(v));
  const freia_api_t * api = get_freia_api(index);
  return api->function_name;
}

string dagvtx_operation(const dagvtx v)
{
  string op = dagvtx_function_name(v);
  return strncmp(op, AIPO, strlen(AIPO))==0? op + strlen(AIPO): op;
}

string dagvtx_compact_operation(const dagvtx v)
{
  if (v==NULL) return "null";
  int index = vtxcontent_opid(dagvtx_content(v));
  const freia_api_t * api = get_freia_api(index);
  return api->compact_name;
}

int dagvtx_ordering(const dagvtx * v1, const dagvtx * v2)
{
  return dagvtx_number(*v1) - dagvtx_number(*v2);
}

/* return (last) producer of image e for vertex sink, or NULL if none found.
 * this is one of the two predecessors of sink.
 */
dagvtx dagvtx_get_producer(const dag d, const dagvtx sink,
                           const entity e, _int before_number)
{
  pips_assert("some image", e!=entity_undefined);
  FOREACH(dagvtx, v, dag_vertices(d))
  {
    vtxcontent c = dagvtx_content(v);
    pips_debug(8, "%"_intFMT" pred of %"_intFMT"?\n",
               dagvtx_number(v), dagvtx_number(sink));
    // they may have been reversed ordered when used to append a vertex
    if (before_number>0 && dagvtx_number(v)>0 && dagvtx_number(v)>before_number)
      continue;
    // the image may be kept within a pipe
    if (vtxcontent_out(c)==e &&
        (sink==NULL || gen_in_list_p(sink, dagvtx_succs(v))))
      return v;
  }
  return NULL; // it is an external parameter?
}

void dagvtx_nb_dump(FILE * out, const string what, const list l)
{
  fprintf(out, "%s: ", what);
  FOREACH(dagvtx, v, l)
    fprintf(out, " %" _intFMT ",", dagvtx_number(v));
  fprintf(out, "\n");
}

/* for dag debug.
 */
void dagvtx_dump(FILE * out, const string name, const dagvtx v)
{
  if (!v) {
    fprintf(out, "vertex %s is NULL\n", name? name: "");
    return;
  }
  fprintf(out, "vertex %s %" _intFMT " %s (%p)\n",
    name? name: "", dagvtx_number(v), dagvtx_operation(v), v);
  dagvtx_nb_dump(out, "  succs", dagvtx_succs(v));
  vtxcontent c = dagvtx_content(v);
  statement s = dagvtx_statement(v);
  fprintf(out,
    "  optype: %s\n"
    "  opid: %" _intFMT "\n"
    "  source: %" _intFMT "/%" _intFMT "\n",
    what_operation(vtxcontent_optype(c)),
    vtxcontent_opid(c),
    s? statement_number(s): 0,
    s? statement_ordering(s): 0);
  entity_list_dump(out, "  inputs", vtxcontent_inputs(c));
  fprintf(out, "  output: %s\n", safe_entity_name(vtxcontent_out(c)));
  // to be continued...
}

/* for dag debug
 */
void dag_dump(FILE * out, const string what, const dag d)
{
  fprintf(out, "dag '%s' (%p) %"_intFMT" vertices:\n",
          what, d, gen_length(dag_vertices(d)));

  dagvtx_nb_dump(out, "inputs", dag_inputs(d));
  dagvtx_nb_dump(out, "outputs", dag_outputs(d));

  FOREACH(dagvtx, vx, dag_vertices(d)) {
    dagvtx_dump(out, NULL, vx);
    fprintf(out, "\n");
  }

  fprintf(out, "\n");
}

// #define IMG_DEP " [arrowhead=normal]"
#define IMG_DEP ""
#define SCL_DEP "arrowhead=empty"

static const char* entity_dot_name(entity e)
{
  return entity_user_name(e);
}

static void dagvtx_dot_node(FILE * out, const string prefix, const dagvtx v)
{
  fprintf(out, "%s\"%"_intFMT" %s\"", prefix? prefix: "",
          dagvtx_number(v), dagvtx_compact_operation(v));
}

static void
dagvtx_dot_node_sb(string_buffer sb, const string prefix, const dagvtx v)
{
  sb_cat(sb, prefix? prefix: "");
  sb_cat(sb, "\"", itoa((int) dagvtx_number(v)),
         " ", dagvtx_compact_operation(v),  "\"");
}

static void dagvtx_dot(FILE * out, const dag d, const dagvtx vtx)
{
  // collect prettyprint properties
  bool label_nodes = get_bool_property("FREIA_DAG_LABEL_NODES");
  bool show_arcs = get_bool_property("FREIA_DAG_LABEL_ARCS");
  bool filter_nodes = get_bool_property("FREIA_DAG_FILTER_NODES");

  vtxcontent co = dagvtx_content(vtx);
  const char* vname = NULL;
  if (vtxcontent_out(co)!=entity_undefined)
    vname = entity_dot_name(vtxcontent_out(co));

  if (dagvtx_number(vtx)==0)
  {
    // this is an input image, only image dependencies
    FOREACH(dagvtx, succ, dagvtx_succs(vtx))
    {
      fprintf(out, "  \"%s\"", vname);
      dagvtx_dot_node(out, " -> ", succ);
      fprintf(out, ";\n");
    }
  }
  else
  {
    // we are dealing with a computation...
    string attribute = what_operation_shape(vtxcontent_optype(co));
    string_buffer sa = string_buffer_make(true);
    bool some_arcs = false;

    // first check for image dependencies
    FOREACH(dagvtx, succ, dagvtx_succs(vtx))
    {
      dagvtx_dot_node_sb(sa, "  ", vtx);
      dagvtx_dot_node_sb(sa, " -> ", succ);
      sb_cat(sa, show_arcs? " [label=\"": "",
             show_arcs? vname: "", show_arcs? "\"]": "", ";\n");
      some_arcs = true;
    }

    // then scalar dependencies anywhere... hmmm...
    FOREACH(dagvtx, v, dag_vertices(d))
    {
      list vars = NIL;
      if (// another vertex
          vtx!=v &&
          // with dependencies
          freia_scalar_rw_dep(dagvtx_statement(vtx),
                              dagvtx_statement(v), &vars))
      {
        // show scalar dep in sequence order
        if (dagvtx_number(v)>dagvtx_number(vtx))
        {
          dagvtx_dot_node_sb(sa, "  ", vtx);
          dagvtx_dot_node_sb(sa, " -> ", v);
        }
        else
        {
          dagvtx_dot_node_sb(sa, "  ", v);
          dagvtx_dot_node_sb(sa, " -> ", vtx);
        }

        sb_cat(sa, " [" SCL_DEP);

        if (vars && show_arcs)
        {
          int count = 0;
          sb_cat(sa, ",label=\"");
          FOREACH(entity, var, vars)
            sb_cat(sa, count++? " ": "", entity_dot_name(var));
          sb_cat(sa, "\"");
        }
        sb_cat(sa, "];\n");

        some_arcs = true;
      }
      gen_free_list(vars), vars = NIL;
    }

    bool some_image_stuff = !dagvtx_other_stuff_p(vtx);

    if (!filter_nodes || some_image_stuff || (filter_nodes && some_arcs))
    {
      // appends the node description
      dagvtx_dot_node(out, "  ", vtx);
      fprintf(out, " [%s", attribute);
      if (!label_nodes)
        // show a short label with only the operation
        fprintf(out, ",label=\"%s\"", dagvtx_compact_operation(vtx));
      fprintf(out, "];\n");
      // now appends arcs...
      string_buffer_to_file(sa, out);
    }

    // cleanup
    string_buffer_free(&sa);
  }
}

static void dagvtx_copy_list_dot(FILE * out, const list ls, const set inputs)
{
  FOREACH(statement, s, ls)
  {
    call c = freia_statement_to_call(s);
    if (c && same_string_p(entity_local_name(call_function(c)), AIPO "copy"))
    {
      list args = call_arguments(c);
      pips_assert("two args to aipo copy", gen_length(args)==2);
      entity dst = expression_to_entity(EXPRESSION(CAR(args)));
      entity src = expression_to_entity(EXPRESSION(CAR(CDR(args))));
      fprintf(out,
              "  \"%s%s\" [shape=circle];\n"
              "  \"%s =\" [shape=circle,label=\"=\",style=\"dashed\"]\n"
              "  \"%s%s\" -> \"%s =\";\n"
              "  \"%s =\" -> \"%s%s\";\n",
              entity_dot_name(dst), set_belong_p(inputs, dst)? "'": "",
              entity_dot_name(dst),
              entity_dot_name(src), set_belong_p(inputs, src)? "'": "",
              entity_dot_name(dst), entity_dot_name(dst),
              entity_dot_name(dst), set_belong_p(inputs, dst)? "'": "");
    }
    // should not be a else?
  }
}

static void dagvtx_list_dot(
  FILE * out, const string comment, const list l, const set used)
{
  if (comment) fprintf(out, "  // %s\n", comment);
  FOREACH(dagvtx, v, l)
  {
    entity img = vtxcontent_out(dagvtx_content(v));
    fprintf(out, "  \"%s%s\" [shape=circle];\n",
            entity_dot_name(img),
            used? (set_belong_p(used, img)? "'": ""): "");
  }
  fprintf(out, "\n");
}

static bool dagvtx_is_operator_p(const dagvtx v, const string opname)
{
  vtxcontent c = dagvtx_content(v);
  const freia_api_t * api = get_freia_api(vtxcontent_opid(c));
  return same_string_p(cat(AIPO, opname), api->function_name);
}

/* returns whether the vertex is an image copy operation.
 */
static bool dagvtx_is_copy_p(const dagvtx v)
{
  return dagvtx_is_operator_p(v, "copy");
}

/* @brief output dag in dot format, for debug or show
 * @param out, append to this file
 * @param what, name of dag
 * @param d, dag to output
 */
void dag_dot(FILE * out, const string what, const dag d,
             const list lb, const list la)
{
  // compute and show dag statistics
  // count image, copy, scalar operations
  int niops = 0, ncops = 0, nsops = 0;
  FOREACH(dagvtx, op, dag_vertices(d))
  {
    if (dagvtx_is_copy_p(op)) ncops++;
    else if (!gen_in_list_p(op, dag_inputs(d))) {
      if (dagvtx_other_stuff_p(op)) nsops++;
      else niops++;
    }
  }

  fprintf(out, "// DAG \"%s\": "
          // input, output, computation, scalar vertices
          "#i=%d #o=%d #c=%d #s=%d "
          // copies: internal, before, after
          "#I=%d #B=%d #A=%d\n",
          what,
          (int) gen_length(dag_inputs(d)), (int) gen_length(dag_outputs(d)),
          niops, nsops, ncops, (int) gen_length(lb), (int) gen_length(la));

  // start graph
  fprintf(out, "digraph \"%s\" {\n", what);

  // collect set of input images
  set inputs = set_make(hash_pointer);
  FOREACH(dagvtx, i, dag_inputs(d))
  {
    entity img = dagvtx_image(i);
    if (img && img!=entity_undefined)
      set_add_element(inputs, inputs, img);
  }

  // first, declare inputs and outputs as circles
  dagvtx_list_dot(out, "inputs", dag_inputs(d), NULL);
  dagvtx_list_dot(out, "outputs", dag_outputs(d), inputs);

  // second, show computations
  fprintf(out, "  // computation vertices\n");
  FOREACH(dagvtx, vx, dag_vertices(d))
  {
    dagvtx_dot(out, d, vx);
    vtxcontent c = dagvtx_content(vx);

    // outputs arcs for vx when the result is extracted
    if (gen_in_list_p(vx, dag_outputs(d)))
    {
      entity img = vtxcontent_out(c);
      dagvtx_dot_node(out, "  ", vx);
      fprintf(out, " -> \"%s%s\";\n",
              entity_dot_name(img), set_belong_p(inputs, img)? "'": "");
    }
  }

  // handle external copies after the computation
  if (lb)
  {
    fprintf(out, "\n  // external before copies: %d\n", (int) gen_length(lb));
    dagvtx_copy_list_dot(out, lb, inputs);
  }

  if (la)
  {
    fprintf(out, "\n  // external after copies: %d\n", (int) gen_length(la));
    dagvtx_copy_list_dot(out, la, inputs);
  }

  fprintf(out, "}\n");

  set_free(inputs);
}

#define DOT_SUFFIX ".dot"

/* generate a "dot" format from a dag to a file.
 */
void dag_dot_dump(const string module, const string name,
                  const dag d, const list lb, const list la)
{
  // build file name
  string src_dir = db_get_directory_name_for_module(module);
  string fn = strdup(cat(src_dir, "/", name, DOT_SUFFIX, NULL));
  free(src_dir);

  FILE * out = safe_fopen(fn, "w");
  fprintf(out, "// graph for dag \"%s\" of module \"%s\" in dot format\n",
    name, module);
  dag_dot(out, name, d, lb, la);
  safe_fclose(out, fn);
  free(fn);
}

void dag_dot_dump_prefix(const string module, const string prefix, int number,
                         const dag d, const list lb, const list la)
{
  string name = strdup(cat(prefix, itoa(number), NULL));
  dag_dot_dump(module, name, d, lb, la);
  free(name);
}

// debug help
static void check_removed(const dagvtx v, const dagvtx removed)
{
  pips_assert("not removed vertex", v!=removed);
}

static int dagvtx_cmp_entity(const dagvtx * v1, const dagvtx * v2)
{
  return compare_entities(&vtxcontent_out(dagvtx_content(*v1)),
        &vtxcontent_out(dagvtx_content(*v2)));
}

static void vertex_list_sorted_by_entities(list l)
{
  gen_sort_list(l, (gen_cmp_func_t) dagvtx_cmp_entity);
}

/* do some consistency checking...
 */
void dag_consistency_asserts(dag d)
{
  FOREACH(dagvtx, v, dag_inputs(d))
    pips_assert("vertex once in inputs", gen_occurences(v, dag_inputs(d))==1);

  set variable_seen = set_make(set_pointer);
  FOREACH(dagvtx, v, dag_inputs(d))
  {
    entity image = dagvtx_image(v);
    // fprintf(stdout, " - image %s\n", entity_name(image));
    pips_assert("image is defined", image!=entity_undefined);
    pips_assert("variable only once in inputs",
                !set_belong_p(variable_seen, image));
    set_add_element(variable_seen, variable_seen, image);
  }
  set_free(variable_seen);
}

/* remove unused inputs
 */
static void dag_remove_unused_inputs(dag d)
{
  list nl = NIL;
  FOREACH(dagvtx, v, dag_inputs(d))
  {
    if (dagvtx_succs(v)!=NIL) // do we keep it?
      nl = CONS(dagvtx, v, nl);
    else
      gen_remove(&dag_vertices(d), v);
      // ??? memory leak? there may be "pointers" somewhere to these?
      // free_dagvtx(v);
  }
  if (dag_inputs(d)) gen_free_list(dag_inputs(d));
  dag_inputs(d) = gen_nreverse(nl);
}

/* remove vertex v from dag d.
 * if v isx a used computation vertex, it is substituted by an input vertex.
 */
void dag_remove_vertex(dag d, const dagvtx v)
{
  // hmmm...
  // pips_assert("vertex is in dag", gen_in_list_p(v, dag_vertices(d)));
  if (!gen_in_list_p(v, dag_vertices(d)))
    // already done???
    return;

  if (dagvtx_succs(v))
  {
    // the ouput of the removed operation becomes an input to the dag
    entity var = vtxcontent_out(dagvtx_content(v));
    pips_assert("some variable", var!=entity_undefined);
    dagvtx input =
      make_dagvtx(make_vtxcontent(0, 0, make_pstatement_empty(), NIL, var),
                  gen_copy_seq(dagvtx_succs(v)));
    dag_inputs(d) = CONS(dagvtx, input, dag_inputs(d));
    dag_vertices(d) = CONS(dagvtx, input, dag_vertices(d));
    vertex_list_sorted_by_entities(dag_inputs(d));
  }

  // remove from all vertex lists
  gen_remove(&dag_vertices(d), v);
  gen_remove(&dag_inputs(d), v);
  gen_remove(&dag_outputs(d), v);

  // remove from successors of any ???
  FOREACH(dagvtx, dv, dag_vertices(d))
    gen_remove(&dagvtx_succs(dv), v);

  // cleanup input list
  dag_remove_unused_inputs(d);

  // unlink vertex itself
  gen_free_list(dagvtx_succs(v)), dagvtx_succs(v) = NIL;

  ifdebug(8) gen_context_recurse(d, v, dagvtx_domain, gen_true, check_removed);
}

/* copy a vertex, but without its successors.
 */
dagvtx copy_dagvtx_norec(dagvtx v)
{
  list lsave = dagvtx_succs(v);
  // temporary cut costs
  dagvtx_succs(v) = NIL;
  dagvtx copy = copy_dagvtx(v);
  dagvtx_succs(v) = lsave;
  return copy;
}

/* returns whether the vertex is an image measurement operation.
 */
bool dagvtx_is_measurement_p(const dagvtx v)
{
  vtxcontent c = dagvtx_content(v);
  const freia_api_t * api = get_freia_api(vtxcontent_opid(c));
  return strncmp(api->function_name, AIPO "global_", strlen(AIPO "global_"))==0;
}

/* append new vertex nv to dag d.
 */
void dag_append_vertex(dag d, dagvtx nv)
{
  pips_assert("not in dag", !gen_in_list_p(nv, dag_vertices(d)));
  pips_assert("no successors", dagvtx_succs(nv) == NIL);

  // pips_assert("dag d ok 1", dag_consistent_p(d));
  // pips_assert("nv is ok", dagvtx_consistent_p(nv));

  FOREACH(entity, e, vtxcontent_inputs(dagvtx_content(nv)))
  {
    pips_assert("e is defined", e!=entity_undefined);
    dagvtx pv = dagvtx_get_producer(d, NULL, e, dagvtx_number(nv));
    if (!pv)
    {
      // side effect, create an input node of type 0 (not a computation)
      pv = make_dagvtx
        (make_vtxcontent(0, 0, make_pstatement_empty(), NIL, e), NIL);

      dag_inputs(d) = CONS(dagvtx, pv, dag_inputs(d));
      vertex_list_sorted_by_entities(dag_inputs(d));
      dag_vertices(d) = CONS(dagvtx, pv, dag_vertices(d));
    }
    // a vertex may have several time the same successor: b = a + a
    dagvtx_succs(pv) = CONS(dagvtx, nv, dagvtx_succs(pv));
  }
  dag_vertices(d) = CONS(dagvtx, nv, dag_vertices(d));

  // ??? what about scalar deps?
}

/* return the number of actual operations in dag d.
 * may be zero if only input vertices remain in the dag after optimizations.
 */
int dag_computation_count(const dag d)
{
  int count = 0;
  FOREACH(dagvtx, v, dag_vertices(d))
    if (dagvtx_number(v)!=0)
      count++;
  return count;
}

/* return target predecessor vertices as a list.
 * the same predecessor appears twice in b = a+a
 * build them in call order!
 * ??? maybe I should build a cache for performance,
 * but I would have to keep track of when dags are modified...
 */
list dag_vertex_preds(const dag d, const dagvtx target)
{
  pips_debug(8, "building predecessors of %"_intFMT"\n", dagvtx_number(target));
  list inputs = vtxcontent_inputs(dagvtx_content(target));
  int nins = (int) gen_length(inputs);
  entity first_img = NULL, second_img = NULL;
  if (nins>=1) first_img = ENTITY(CAR(inputs));
  if (nins==2) second_img = ENTITY(CAR(CDR(inputs)));
  dagvtx first_v = NULL, second_v = NULL;

  FOREACH(dagvtx, v, dag_vertices(d))
  {
    if (v!=target && gen_in_list_p(target, dagvtx_succs(v)))
    {
      if (dagvtx_image(v)==first_img)
        first_v = v;
      if (dagvtx_image(v)==second_img)
        second_v = v;
    }
  }

  list preds = NIL;
  if (second_v) preds = CONS(dagvtx, second_v, NIL);
  if (first_v) preds = CONS(dagvtx, first_v, preds);
  return preds;
}

static bool gen_list_equals_p(const list l1, const list l2)
{
  bool equal = true;
  list p1 = (list) l1, p2 = (list) l2;
  while (equal && p1 && p2) {
    if (CHUNK(CAR(p1))!=CHUNK(CAR(p2)))
      equal = false;
    p1 = CDR(p1), p2 = CDR(p2);
  }
  equal &= (!p1 && !p2);
  return equal;
}

/* replace target measure to a copy of source result...
 * @param target to be removed/replaced because redundant
 * @param source to be kept
 * @return whether the statement is to be actually removed
 */
static bool switch_vertex_to_assign(dagvtx target, dagvtx source)
{
  pips_debug(5, "replacing measure %"_intFMT" by %"_intFMT" result\n",
             dagvtx_number(target), dagvtx_number(source));

  // update target vertex
  vtxcontent cot = dagvtx_content(target);
  vtxcontent_optype(cot) = spoc_type_oth;
  vtxcontent_opid(cot) = hwac_freia_api_index("freia_aipo_scalar_copy");
  // no more image inputs
  gen_free_list(vtxcontent_inputs(cot));
  vtxcontent_inputs(cot) = NIL;

  // ??? source -> target scalar dependency?

  // update actual statement.
  call cref = statement_call(dagvtx_statement(source));
  expression eref = EXPRESSION(CAR(CDR(call_arguments(cref))));
  call cnew = statement_call(dagvtx_statement(target));
  expression enew = EXPRESSION(CAR(CDR(call_arguments(cnew))));

  if (expression_equal_p(eref, enew))
    // just remove the fully redundant measure
    return true;

  // replace by assign
  call_function(cnew) = local_name_to_top_level_entity("=");

  // ??? memory leak or core dump
  // gen_full_free_list(call_arguments(cnew));
  // possibly due to effects which are loaded?
  call_arguments(cnew) =
    gen_make_list(expression_domain,
                  dereference_expression(enew),
                  dereference_expression(eref),
                  NIL);

  return false;
}

/* subtitute produced or used image in the statement of vertex v.
 * @param v vertex
 * @param source image variable to replace
 * @param target new image variable to use
 * @param used whether to replace used image (input, forward propagation)
 *             or procuded image (output, backward propagation)
 */
static void substitute_image_in_statement
(dagvtx v, entity source, entity target, bool used)
{
  pips_debug(8, "image %s -> %s in %"_intFMT"\n",
             entity_name(source), entity_name(target), dagvtx_number(v));

  int nsubs=0;
  vtxcontent vc = dagvtx_content(v);
  pstatement ps = vtxcontent_source(vc);
  pips_assert("is a statement", pstatement_statement_p(ps));

  // get call
  call c = freia_statement_to_call(pstatement_statement(ps));
  list args = call_arguments(c);

  const freia_api_t * api = dagvtx_freia_api(v);

  // how many argument to skip, how many to replace
  unsigned int skip, subs;

  if (used)
    skip = api->arg_img_out, subs = api->arg_img_in;
  else
    skip = 0, subs = api->arg_img_out;

  pips_assert("call length is okay", gen_length(args)>=skip+subs);

  while (skip--)
    args = CDR(args);

  while (subs--)
  {
    expression e = EXPRESSION(CAR(args));
    // I'm not sure what to do if an image is not a simple reference...
    pips_assert("image argument is a reference", expression_reference_p(e));
    reference r = expression_reference(e);

    if (reference_variable(r)==source)
      nsubs++, reference_variable(r) = target;
    args = CDR(args);
  }

  // ??? Hmmm... this happens in freia_3muls, not sure why.
  // pips_assert("some image substitutions", nsubs>0);
}

/* replace target vertex by a copy of source results...
 * @param target to be removed
 * @param source does perform the same computation
 * @param tpreds target predecessors to be added to source
 */
static void switch_vertex_to_a_copy(dagvtx target, dagvtx source, list tpreds)
{
  pips_debug(5, "replacing %"_intFMT" by %"_intFMT"\n",
             dagvtx_number(target), dagvtx_number(source));

  ifdebug(9) {
    dagvtx_dump(stderr, "in source", source);
    dagvtx_dump(stderr, "in target", target);
  }

  entity src_img = dagvtx_image(source);
  // fix contents
  vtxcontent cot = dagvtx_content(target);
  vtxcontent_optype(cot) = spoc_type_nop;
  vtxcontent_opid(cot) = hwac_freia_api_index(AIPO "copy");
  gen_free_list(vtxcontent_inputs(cot));
  vtxcontent_inputs(cot) = CONS(entity, src_img, NIL);

  // fix vertices
  FOREACH(dagvtx, v, tpreds)
    gen_remove(&dagvtx_succs(v), target);

  entity trg_img = dagvtx_image(target);
  FOREACH(dagvtx, s, dagvtx_succs(target))
  {
    gen_list_patch(vtxcontent_inputs(dagvtx_content(s)), trg_img, src_img);
    // also fix subsequent statements, for AIPO output
    substitute_image_in_statement(s, trg_img, src_img, true);
  }
  dagvtx_succs(source) = gen_once(target, dagvtx_succs(source));
  dagvtx_succs(source) = gen_nconc(dagvtx_succs(target), dagvtx_succs(source));
  dagvtx_succs(target) = NIL;

  // should I kill the statement? no, done by the copy removal stuff

  ifdebug(9) {
    dagvtx_dump(stderr, "out source", source);
    dagvtx_dump(stderr, "out target", target);
  }
}

/* @return whether two vertices are the same operation
 */
static bool same_operation_p(const dagvtx v1, const dagvtx v2)
{
  return
    dagvtx_optype(v1) == dagvtx_optype(v2) &&
    dagvtx_opid(v1) == dagvtx_opid(v2);
}

/* @return whether two vertices are commutors
 */
static bool commutative_operation_p(const dagvtx v1, const dagvtx v2)
{
  if (dagvtx_optype(v1) == dagvtx_optype(v2))
  {
    int n1 = (int) dagvtx_opid(v1), n2 = (int) dagvtx_opid(v2);
    const freia_api_t * f1 = get_freia_api(n1);
    return f1->commutator && hwac_freia_api_index(f1->commutator)==n2;
  }
  else return false;
}

/* @return whether two lists are commuted
 */
static bool list_commuted_p(const list l1, const list l2)
{
  pips_assert("length 2", gen_length(l1)==2 && gen_length(l2)==2);
  return CHUNKP(CAR(CDR(l1)))==CHUNKP(CAR(l2)) &&
    CHUNKP(CAR(l1))==CHUNKP(CAR(CDR(l2)));
}

/* "copy" copies "source" image in dag "d".
 * remove it properly (forward copy propagation)
 */
static void unlink_copy_vertex(dag d, const entity source, dagvtx copy)
{
  pips_debug(8, "unlinking %"_intFMT"\n", dagvtx_number(copy));

  entity target = vtxcontent_out(dagvtx_content(copy));
  // may be NULL if source is an input
  dagvtx prod = dagvtx_get_producer(d, copy, source, 0);

  // add copy successors as successors of prod
  // it is kept as a successor in case it is not removed
  if (prod)
  {
    FOREACH(dagvtx, vs, dagvtx_succs(copy))
      dagvtx_succs(prod) = gen_once(vs, dagvtx_succs(prod));
  }

  // replace use of target image by source image in all v successors
  FOREACH(dagvtx, succ, dagvtx_succs(copy))
  {
    vtxcontent sc = dagvtx_content(succ);
    gen_list_patch(vtxcontent_inputs(sc), target, source);

    // also in the statement inputs... (needed for AIPO target)
    substitute_image_in_statement(succ, target, source, true);
  }

  // copy has no more successors
  gen_free_list(dagvtx_succs(copy));
  dagvtx_succs(copy) = NIL;
}

/* @return whether all vertices in list "lv" are copies.
 */
static bool all_vertices_are_copies_or_measures_p(const list lv)
{
  FOREACH(dagvtx, v, lv)
    if (!(dagvtx_is_copy_p(v) || dagvtx_is_measurement_p(v)))
      return false;
  return true;
}

/* @return the number of copies in the vertex list
 */
static int number_of_copies(list /* of dagvtx */ l)
{
  int n=0;
  FOREACH(dagvtx, v, l)
    if (dagvtx_is_copy_p(v))
      n++;
  return n;
}

// tell whether this is this (short name) aipo function (static test)?
#define aipo_op_p(a, name)                      \
  same_string_p(AIPO name, a->function_name)

/* @brief compute a constant expression for FREIA
 * ??? TODO partial eval if values are constant
 */
static expression compute_constant(string op, expression val1, expression val2)
{
  // rough! could/should do some partial evaluation here
  entity eop = local_name_to_top_level_entity(op);
  pips_assert("operator function found", eop!=entity_undefined);
  call c = make_call(eop,
                     CONS(expression, copy_expression(val1),
                          CONS(expression, copy_expression(val2), NIL)));
  return call_to_expression(c);
}

/* switch vertex statement to an aipo call
 * @param v vertex to modify
 * @param name function short name to switch to, must be an AIPO function
 * @param img possible image argument, may be entity_undefined or NULL
 * @param val possible scalar argument, may be expression undefined or NULL
 */
static void set_aipo_call(dagvtx v, string name, entity img, expression val)
{
  vtxcontent c = dagvtx_content(v);
  pstatement ps = vtxcontent_source(c);
  pips_assert("some statement!", pstatement_statement_p(ps));

  // build function name
  string fname = strdup(cat(AIPO, name));

  // remove previous image inputs
  gen_free_list(vtxcontent_inputs(c));
  vtxcontent_inputs(c) =
    (img && img!=entity_undefined)? CONS(entity, img, NIL): NIL;

  // set id & type
  vtxcontent_opid(c) = hwac_freia_api_index(fname);
  pips_assert("function index found", vtxcontent_opid(c)!=-1);
  vtxcontent_optype(c) =
    same_string_p(name, "copy")? spoc_type_nop: spoc_type_alu;

  // set call
  call cf = freia_statement_to_call(pstatement_statement(ps));
  entity func = local_name_to_top_level_entity(fname);
  pips_assert("AIPO function found", func!=entity_undefined);
  call_function(cf) = func;

  // build expression list in reverse order
  list args = NIL;
  if (val && val!=expression_undefined)
    args = CONS(expression, copy_expression(val), args);
  if (img && img!=entity_undefined)
    args = CONS(expression, entity_to_expression(img), args);
  args = CONS(expression, entity_to_expression(vtxcontent_out(c)), args);
  // with a bold memory leak (effects may use refs?)
  call_arguments(cf) = args;

  // cleanup
  free(fname);
}

/* @brief switch vertex to a copy of an image
 */
static void set_aipo_copy(dagvtx v, entity e)
{
  set_aipo_call(v, "copy", e, NULL);
}

/************************************************************** DAG SIMPLIFY */

// forward declarations for recursion
static bool propagate_constant_image(dagvtx, entity, expression);
static void set_aipo_constant(dagvtx, expression);
static expression constant_image_p(dagvtx);
static entity copy_image_p(dagvtx);

/* @brief propagate constant image to vertex successors
 */
static void propagate_constant_image_to_succs(dagvtx v, expression val)
{
  vtxcontent c = dagvtx_content(v);
  list nsuccs = NIL;
  // i=? -> i+i => to identical successors, but forward once
  set seen = set_make(hash_pointer);
  FOREACH(dagvtx, s, dagvtx_succs(v))
  {
    if (!set_belong_p(seen, s) &&
        !propagate_constant_image(s, vtxcontent_out(c), val))
      nsuccs = CONS(dagvtx, s, nsuccs);
    set_add_element(seen, seen, s);
  }
  set_free(seen);
  dagvtx_succs(v) = gen_nreverse(nsuccs);
}

/* @brief recursively propagate a constant image on a vertex
 * @param v starting vertex
 * @param image which is constant
 * @param val actual value of all image pixels
 * @return whether v can be unlinked
 * ??? small memory leaks
 */
static bool propagate_constant_image(dagvtx v, entity image, expression val)
{
  vtxcontent c = dagvtx_content(v);
  _int op = vtxcontent_opid(c);
  expression newval = expression_undefined;
  if (op<=0) return false;
  // op>0
  const freia_api_t * api = get_freia_api(op);
  int noccs = gen_occurences(image, vtxcontent_inputs(c));
  pips_assert("image is used at least once", noccs>0);
  int nargs = gen_length(vtxcontent_inputs(c));
  bool image_is_first = ENTITY(CAR(vtxcontent_inputs(c)))==image;

  if (copy_image_p(v)!=NULL)
  {
    propagate_constant_image_to_succs(v, val);
    set_aipo_copy(v, image);
    return false;
  }

  if (noccs==1 && nargs==1) // Cst(n)<op>.m => Cst(n<op>m)
  {
    string op = NULL;
    if (aipo_op_p(api, "add_const"))      op = "+";
    else if (aipo_op_p(api, "sub_const")) op = "-";
    else if (aipo_op_p(api, "mul_const")) op = "*";
    else if (aipo_op_p(api, "div_const")) op = "/";
    else if (aipo_op_p(api, "and_const")) op = BITWISE_AND_OPERATOR_NAME;
    else if (aipo_op_p(api, "or_const"))  op = "|";
    else if (aipo_op_p(api, "xor_const")) op = BITWISE_XOR_OPERATOR_NAME;
    // ??? TODO addsat, subsat, inf, sup, not, log2...
    // but I need the corresponding scalar functions
    // some of which depends on the actual pixel type
    else if (aipo_op_p(api, "erode_6c") || aipo_op_p(api, "dilate_6c") ||
             aipo_op_p(api, "erode_8c") || aipo_op_p(api, "dilate_8c"))
      newval = val;
    // ??? TODO threshold
    // ??? TODO global_min/global_max can be switch to a simple assign
    // ??? TODO global_vol can be switched to n_pixels*val

    if (op)
      newval = compute_constant(op, val, freia_get_nth_scalar_param(v, 1));
    if (newval!=expression_undefined)
    {
      set_aipo_constant(v, newval);
      return true;
    }
  }
  else if (noccs==1 && nargs==2)
  {
    string op = NULL;
    entity other = image_is_first?
      ENTITY(CAR(CDR(vtxcontent_inputs(c)))): ENTITY(CAR(vtxcontent_inputs(c)));

    if (aipo_op_p(api, "add"))      op = "add_const";
    else if (aipo_op_p(api, "mul")) op = "mul_const";
    else if (aipo_op_p(api, "and")) op = "add_const";
    else if (aipo_op_p(api, "or"))  op = "or_const";
    else if (aipo_op_p(api, "xor")) op = "xor_const";
    else if (aipo_op_p(api, "inf")) op = "inf_const";
    else if (aipo_op_p(api, "sup")) op = "sup_const";
    else if (aipo_op_p(api, "addsat")) op = "addsat_const";
    else if (aipo_op_p(api, "subsat")) op = "subsat_const";
    else if (aipo_op_p(api, "absdiff")) op = "absdiff_const";
    else if (aipo_op_p(api, "sub"))
      op = image_is_first? "const_sub": "sub_const";
    else if (aipo_op_p(api, "div"))
      op = image_is_first? "const_div": "div_const";

    if (op)
    {
      set_aipo_call(v, op, other, val);
      return true;
    }
  }
  else if (noccs==2)
  {
    // special case for (image <op> image)
    if (aipo_op_p(api, "add"))
    {
      set_aipo_constant(v, compute_constant("*", val, int_to_expression(2)));
      return true;
    }
    else if (aipo_op_p(api, "sub"))
    {
      set_aipo_constant(v, int_to_expression(0));
      return true;
    }
  }
  return false;
}

/* @brief set vertex as a constant image and propagate to successors
 */
static void set_aipo_constant(dagvtx v, expression val)
{
  set_aipo_call(v, "set_constant", entity_undefined, val);
  propagate_constant_image_to_succs(v, val);
}

/* @brief whether vertex generates a constant image
 * @param v vertex to consider
 * @return constant allocated expression or NULL
 */
static expression constant_image_p(dagvtx v)
{
  vtxcontent c = dagvtx_content(v);
  _int op = vtxcontent_opid(c);
  const freia_api_t * api = get_freia_api(op);
  if (api->arg_img_in==2)
  {
    pips_assert("2 input images", gen_length(vtxcontent_inputs(c))==2);
    entity e1 = ENTITY(CAR(vtxcontent_inputs(c))),
      e2 = ENTITY(CAR(CDR(vtxcontent_inputs(c))));
    if (e1==e2 && (aipo_op_p(api, "sub") || aipo_op_p(api, "xor")))
      return int_to_expression(0);
    if (e1==e2 && aipo_op_p(api, "div"))
      return int_to_expression(1);
  }
  else if (api->arg_img_in==1)
  {
    bool
      erosion = aipo_op_p(api, "erode_8c") || aipo_op_p(api, "erode_6c"),
      dilatation = aipo_op_p(api, "dilate_8c") || aipo_op_p(api, "dilate_6c");
    // aipo_op_p(api, "convolution") 3x3?
    if (erosion || dilatation)
    {
      _int i00, i10, i20, i01, i11, i21, i02, i12, i22;
      if (freia_extract_kernel_vtx(v, true, &i00, &i10, &i20,
                                   &i01, &i11, &i21, &i02, &i12, &i22))
      {
        // zero boundary
        if (i00==0 && i10==0 && i20==0 && i01==0 &&
            i21==0 && i02==0 && i12==0 && i22==0)
        {
          if (i11==0)
          {
            if (erosion)
              return int_to_expression(0);
            else if (dilatation)
              return int_to_expression(freia_max_pixel_value());
          }
        }
      }
    }
    else if (api->arg_misc_in==1) // arith cst op
    {
      expression e1 = freia_get_nth_scalar_param(v, 1);
      _int val;
      if (expression_integer_value(e1, &val))
      {
        if ((aipo_op_p(api, "mul_const") && val==0) ||
            (aipo_op_p(api, "and_const") && val==0) ||
            (aipo_op_p(api, "inf_const") && val==0) ||
            (aipo_op_p(api, "const_div") && val==0))
          return int_to_expression(val);

        int maxval = freia_max_pixel_value();
        if ((aipo_op_p(api, "sup_const") && val==maxval) ||
            (aipo_op_p(api, "or_const") && val==maxval) ||
            (aipo_op_p(api, "addsat_const") && val==maxval))
          return int_to_expression(val);

        if ((aipo_op_p(api, "subsat_const") && val==maxval))
          return int_to_expression(0);
      }
    }
  }
  else if (api->arg_img_in==0)
  {
    if (aipo_op_p(api, "set_constant"))
      return copy_expression(freia_get_nth_scalar_param(v, 1));
  }
  return NULL;
}

/* @brief tell whether vertex can be assimilated to a copy
 * e.g.  i+.0  i*.1  i/.1  i|.0  i-.0 i&i i|i i>i i<i
 * @return image to copy, or NULL
 */
static entity copy_image_p(dagvtx v)
{
  vtxcontent c = dagvtx_content(v);
  _int op = vtxcontent_opid(c);
  const freia_api_t * api = get_freia_api(op);
  if (api->arg_img_in==2)
  {
    pips_assert("2 input images", gen_length(vtxcontent_inputs(c))==2);
    entity e1 = ENTITY(CAR(vtxcontent_inputs(c))),
      e2 = ENTITY(CAR(CDR(vtxcontent_inputs(c))));
    if (e1==e2 && (aipo_op_p(api, "and") ||
                   aipo_op_p(api, "or") ||
                   aipo_op_p(api, "sup") ||
                   aipo_op_p(api, "inf")))
      return e1;
  }
  else if (api->arg_img_in==1)
  {
    if (aipo_op_p(api, "erode_8c") ||
        aipo_op_p(api, "erode_6c") ||
        aipo_op_p(api, "dilate_8c") ||
        aipo_op_p(api, "dilate_6c"))
        // aipo_op_p(api, "convolution") 3x3?
    {
      _int i00, i10, i20, i01, i11, i21, i02, i12, i22;
      if (freia_extract_kernel_vtx(v, true, &i00, &i10, &i20,
                                   &i01, &i11, &i21, &i02, &i12, &i22))
      {
        // zero boundary
        if (i00==0 && i10==0 && i20==0 && i01==0 &&
            i21==0 && i02==0 && i12==0 && i22==0)
        {
          if (i11==1) return ENTITY(CAR(vtxcontent_inputs(c)));
        }
      }
    }
    else if (api->arg_misc_in==1)
    {
      expression e1 = freia_get_nth_scalar_param(v, 1);
      _int val;
      if (expression_integer_value(e1, &val))
      {
        if ((aipo_op_p(api, "mul_const") && val==1) ||
            (aipo_op_p(api, "div_const") && val==1) ||
            (aipo_op_p(api, "or_const") && val==0) ||
            (aipo_op_p(api, "add_const") && val==0) ||
            (aipo_op_p(api, "sub_const") && val==0) ||
            (aipo_op_p(api, "sup_const") && val==0))
          return ENTITY(CAR(vtxcontent_inputs(c)));

        int maxval = freia_max_pixel_value();
        if ((aipo_op_p(api, "inf_const") && val==maxval) ||
            (aipo_op_p(api, "addsat") && val==maxval))
          return ENTITY(CAR(vtxcontent_inputs(c)));
      }
    }
    else if (api->arg_misc_in==0)
    {
      if (aipo_op_p(api, "copy"))
        return ENTITY(CAR(vtxcontent_inputs(c)));
    }
  }
  return NULL;
}

/* @brief apply basic algebraic simplification to dag
 */
static bool dag_simplify(dag d)
{
  bool changed = false;

  // propagate constants and detect copies
  FOREACH(dagvtx, v, dag_vertices(d))
  {
    expression val;
    entity img;
    if ((val=constant_image_p(v))!=NULL)
    {
      changed = true;

      // fix successors of predecessors of new "constant" operation
      list preds = dag_vertex_preds(d, v);
      FOREACH(dagvtx, p, preds)
        gen_remove(&dagvtx_succs(p), v);
      gen_free_list(preds);

      // swich operation to constant
      set_aipo_constant(v, val);
      free_expression(val);
    }
    else if ((img=copy_image_p(v))!=NULL)
    {
      changed = true;
      set_aipo_copy(v, img);
    }
    // another one just for fun: I+I -> 2*I
    // this is good for SPoC
    else if (dagvtx_is_operator_p(v, "add"))
    {
      list preds = dag_vertex_preds(d, v);
      pips_assert("two args to add", gen_length(preds)==2);
      if (DAGVTX(CAR(preds))==DAGVTX(CAR(CDR(preds))))
      {
        changed = true;
        set_aipo_call(v, "mul_const",
                      dagvtx_image(DAGVTX(CAR(preds))), int_to_expression(2));
      }
    }
  }

  return changed;
}

/* normalize, that is use less image operators
 * only transformation is: sub_const(i, v) -> add_const(i, -v)
 */
static bool dag_normalize(dag d)
{
  bool changed = false;
  FOREACH(dagvtx, v, dag_vertices(d))
  {
    vtxcontent ct = dagvtx_content(v);
    statement s = dagvtx_statement(v);
    call c = s? freia_statement_to_call(s): NULL;
    if (c)
    {
      list args = call_arguments(c);
      entity func = call_function(c);
      if (same_string_p(entity_local_name(func), AIPO "sub_const"))
      {
        changed = true;
        // sub_const -> add_const(opposite)
        // does it always make sense for unsigned pixels?
        vtxcontent_opid(ct) = hwac_freia_api_index(AIPO "add_const");
        call_function(c) = local_name_to_top_level_entity(AIPO "add_const");
        list l3 = CDR(CDR(args));
        _int v;
        if (expression_integer_value(EXPRESSION(CAR(l3)), &v))
        {
          // partial eval
          free_expression(EXPRESSION(CAR(l3)));
          EXPRESSION_(CAR(l3)) = int_to_expression(-v);
        }
        else
        {
          // symbolic
          EXPRESSION_(CAR(l3)) =
            MakeUnaryCall(
              local_name_to_top_level_entity(UNARY_MINUS_OPERATOR_NAME),
              EXPRESSION(CAR(l3)));
        }
      }
      // what else?
    }
  }
  return changed;
}

/* @return 0 keep both, 1 keep first, 2 keep second
 */
static int compatible_reduction_operation(const dagvtx v1, const dagvtx v2)
{
  const string
    o1 = dagvtx_compact_operation(v1),
    o2 = dagvtx_compact_operation(v2);
  if ((same_string_p(o1, "max!") && same_string_p(o2, "max")) ||
      (same_string_p(o1, "min!") && same_string_p(o2, "min")))
    return 1;
  else if ((same_string_p(o1, "max") && same_string_p(o2, "max!")) ||
           (same_string_p(o1, "min") && same_string_p(o2, "min!")))
    return 2;
  else
    return 0;
}

/* remove dead image operations.
 * remove AIPO copies detected as useless.
 * remove identical operations.
 * return list of statements to be managed outside (external copies)...
 * ??? maybe there should be a transitive closure...
 */
void freia_dag_optimize(
  dag d, hash_table exchanges,
  list * lbefore, list * lafter)
{
  set remove = set_make(set_pointer);
  size_t dag_output_count = gen_length(dag_outputs(d));

  ifdebug(6) {
    pips_debug(6, "considering dag:\n");
    dag_dump(stderr, "input", d);
  }

  if (get_bool_property("FREIA_NORMALIZE_OPERATIONS"))
  {
    dag_normalize(d);

    ifdebug(8) {
      pips_debug(4, "after FREIA_NORMALIZE_OPERATIONS:\n");
      dag_dump(stderr, "normalized", d);
    }
  }

  // algebraic simplifications
  // currently constant images are detected and propagated and constant pixels
  if (get_bool_property("FREIA_SIMPLIFY_OPERATIONS"))
  {
    dag_simplify(d);

    ifdebug(8) {
      pips_debug(4, "after FREIA_SIMPLIFY_OPERATIONS (1):\n");
      dag_dump(stderr, "simplified_1", d);
      // dag_dot_dump_prefix("main", "simplified", 0, d);
    }
  }

  // look for identical image operations (same inputs, same params)
  // (that produce image, we do not care about measures for SPOC,
  // but we should for terapix!)
  // the second one is replaced by a copy.
  // also handle commutations.
  if (get_bool_property("FREIA_REMOVE_DUPLICATE_OPERATIONS"))
  {
    pips_debug(6, "removing duplicate operations\n");

    // all vertices in dependence order
    list vertices = gen_nreverse(gen_copy_seq(dag_vertices(d)));

    // for all already processed vertices, the list of their predecessors
    hash_table previous = hash_table_make(hash_pointer, 10);

    // subset of vertices to be investigated
    set candidates = set_make(hash_pointer);

    // subset of vertices already encountered which do not have predecessors
    set sources = set_make(hash_pointer);

    // already processed vertices
    set processed = set_make(hash_pointer);

    // potential n^2 loop, optimized by comparing only to the already processed
    // successors of the vertex predecessors . See "candidates" computation.
    FOREACH(dagvtx, vr, vertices)
    {
      int op = (int) vtxcontent_optype(dagvtx_content(vr));
      pips_debug(7, "at vertex %"_intFMT" (op=%d)\n", dagvtx_number(vr), op);

      // skip no-operations
      if (op<spoc_type_poc || op>spoc_type_mes) continue;

      // skip already removed ops
      if (set_belong_p(remove, vr)) continue;

      // pips_debug(8, "investigating...\n");
      list preds = dag_vertex_preds(d, vr);

      // build only "interesting" vertices, which shared inputs
      set_clear(candidates);

      // successors of predecessors
      FOREACH(dagvtx, p, preds)
        set_append_list(candidates, dagvtx_succs(p));

      // keep only already processed vertices
      set_intersection(candidates, candidates, processed);

      // possibly add those without predecessors (is that only set_const?)
      if (!preds) set_union(candidates, candidates, sources);

      // whether the vertex was changed
      bool switched = false;

      // I do not need to sort them, they are all different, only one can match
      SET_FOREACH(dagvtx, p, candidates)
      {
        pips_debug(8, "comparing %"_intFMT" and %"_intFMT"\n",
                   dagvtx_number(vr), dagvtx_number(p));

        list lp = hash_get(previous, p);

        // ??? maybe I should not remove all duplicates, because
        // recomputing them may be cheaper?
        if (dagvtx_is_measurement_p(vr) || dagvtx_is_measurement_p(p))
        {
          // special handling for measures, esp for terapix
          if (gen_list_equals_p(preds, lp))
          {
            if (same_operation_p(vr, p))
            {
              // min(A, px), min(A, py) => min(A, px) && *py = *px
              if (switch_vertex_to_assign(vr, p))
                set_add_element(remove, remove, vr);
              // only one can match!
              switched = true;
              break;
            }
            else
            {
              int n = compatible_reduction_operation(p, vr);
              if (n==2)
              {
                // exchange role of vr & p
                set_del_element(processed, processed, p);
                set_add_element(processed, processed, vr);
                hash_del(previous, p);
                hash_put(previous, vr, lp);
                dagvtx t = p; p = vr; vr = t;
                // also in the list of statements to fix dependencies!
                if (exchanges) hash_put(exchanges, p, vr);
              }
              if (n==1 || n==2)
              {
                // we keep p which is a !, and vr is a simple reduction
                if (switch_vertex_to_assign(vr, p))
                  set_add_element(remove, remove, vr);

                // done
                switched = true;
                break;
              }
            }
          }
        }
        else if (same_operation_p(vr, p) &&
                 gen_list_equals_p(preds, lp) &&
                 same_constant_parameters(vr, p))
        {
          switch_vertex_to_a_copy(vr, p, preds);
          // only one can match!
          switched = true;
          break;
        }
        else if (commutative_operation_p(vr, p) &&
                 list_commuted_p(preds, (list) lp))
        {
          switch_vertex_to_a_copy(vr, p, preds);
          // only one can match!
          switched = true;
          break;
        }
      }

      if (switched)
        gen_free_list(preds);
      else
      {
        // update map & sets
        hash_put(previous, vr, preds);
        set_add_element(processed, processed, vr);
        if (!preds) set_add_element(sources, sources, vr);
      }
    }

    // cleanup
    HASH_MAP(k, v, if (v) gen_free_list(v), previous);
    hash_table_free(previous), previous = NULL;
    gen_free_list(vertices), vertices = NULL;
    set_free(candidates);
    set_free(sources);
    set_free(processed);

    ifdebug(8) {
      pips_debug(4, "after FREIA_REMOVE_DUPLICATE_OPERATIONS:\n");
      dag_dump(stderr, "remove duplicate", d);
      // dag_dot_dump_prefix("main", "remove_duplicates", 0, d);
    }
  }

  // algebraic simplifications *AGAIN*
  // some duplicate operation removal may have enabled more simplifications
  // for instance -(a,b) & a=~b => -(a,a) => cst(0)
  // we may have a convergence loop on both duplicate/simplify
  if (get_bool_property("FREIA_SIMPLIFY_OPERATIONS"))
  {
    dag_simplify(d);

    ifdebug(8) {
      pips_debug(4, "after FREIA_SIMPLIFY_OPERATIONS (2):\n");
      dag_dump(stderr, "simplified_2", d);
      // dag_dot_dump_prefix("main", "simplified", 0, d);
    }
  }

  // remove dead image operations
  // ??? hmmm... measures are kept because of the implicit scalar dependency?
  if (get_bool_property("FREIA_REMOVE_DEAD_OPERATIONS"))
  {
    pips_debug(6, "removing dead code\n");
    list vertices = gen_copy_seq(dag_vertices(d));
    FOREACH(dagvtx, v, vertices)
    {
      // skip non-image operations
      if (!vtxcontent_inputs(dagvtx_content(v)) &&
          (vtxcontent_out(dagvtx_content(v))==entity_undefined ||
           vtxcontent_out(dagvtx_content(v))==NULL))
        continue;

      if (// no successors or they are all removed
          (!dagvtx_succs(v) || list_in_set_p(dagvtx_succs(v), remove)) &&
          // but we keep output nodes...
          !gen_in_list_p(v, dag_outputs(d)) &&
          // and measures...
          !dagvtx_is_measurement_p(v))
      {
        pips_debug(7, "vertex %"_intFMT" is dead\n", dagvtx_number(v));
        set_add_element(remove, remove, v);
      }
      else
        pips_debug(8, "vertex %"_intFMT" is alive\n", dagvtx_number(v));
    }
    gen_free_list(vertices), vertices = NULL;

    ifdebug(8) {
      pips_debug(4, "after FREIA_REMOVE_DEAD_OPERATIONS:\n");
      dag_dump(stderr, "remove dead", d);
      // dag_dot_dump_prefix("main", "remove_dead", 0, d);
    }
  }

  if (get_bool_property("FREIA_REMOVE_USELESS_COPIES"))
  {
    set forwards = set_make(set_pointer);
    bool changed = true;

    // I -copy-> X -> ... where I is an input is moved forward

    // we iterate the hard way, it could be little more subtle
    // this is necessary, see copy_02
    while (changed)
    {
      changed = false;

      // first propagate input images through copies
      FOREACH(dagvtx, v, dag_inputs(d))
      {
        list append = NIL;
        entity in = vtxcontent_out(dagvtx_content(v));

        FOREACH(dagvtx, s, dagvtx_succs(v))
        {
          entity copy = vtxcontent_out(dagvtx_content(s));
          if (dagvtx_is_copy_p(s))
          {
            // forward propagation in dag & statements
            FOREACH(dagvtx, s2, dagvtx_succs(s))
            {
              substitute_image_in_statement(s2, copy, in, true);
              gen_replace_in_list(vtxcontent_inputs(dagvtx_content(s2)),
                                copy, in);
            }
            // update succs
            append = gen_nconc(dagvtx_succs(s), append);
            dagvtx_succs(s) = NIL;

            set_add_element(forwards, forwards, s);
          }
        }
        if (append)
        {
          FOREACH(dagvtx, a, append)
          {
            if (!gen_in_list_p(a, dagvtx_succs(v)))
            {
              dagvtx_succs(v) = CONS(dagvtx, a, dagvtx_succs(v));
              changed = true;
            }
          }
        }
      }
    }

    // op -> X -copy-> A where A is an output is moved backwards
    FOREACH(dagvtx, v, dag_vertices(d))
    {
      // skip special input nodes
      if (dagvtx_number(v)==0) continue;

      // skip already removed ops
      if (set_belong_p(remove, v)) continue;

      // skip forward propagated inputs
      if (set_belong_p(forwards, v)) continue;

      if (dagvtx_is_copy_p(v))
      {
        list preds = dag_vertex_preds(d, v);
        vtxcontent c = dagvtx_content(v);
        entity res = vtxcontent_out(c);
        pips_assert("one output and one input to copy",
                    res!=entity_undefined &&
                    gen_length(preds)==1 &&
                    gen_length(vtxcontent_inputs(c))==1);

        // first check for A->copy->A really useless copies, which are skipped
        entity inimg = ENTITY(CAR(vtxcontent_inputs(c)));
        if (inimg==res)
        {
          set_add_element(remove, remove, v);
          dagvtx pred = DAGVTX(CAR(preds));
          // update predecessor's successors
          gen_remove(&dagvtx_succs(pred), v);
          dagvtx_succs(pred) = gen_nconc(dagvtx_succs(pred), dagvtx_succs(v));
          // fix global output if necessary
          if (gen_in_list_p(v, dag_outputs(d)))
            gen_replace_in_list(dag_outputs(d), v, pred);
        }
        // check for internal-t -one-copy-and-others-> A (output)
        // could be improved by dealing with the first copy only?
        else if (gen_in_list_p(v, dag_outputs(d)))
        {
          pips_assert("one predecessor to used copy", gen_length(preds)==1);
          dagvtx pred = DAGVTX(CAR(preds));

          if (number_of_copies(dagvtx_succs(pred))==1 &&
              !gen_in_list_p(pred, dag_outputs(d)) &&
              dagvtx_number(pred)!=0)
          {
            // BACKWARD COPY PROPAGATION
            // that is we want to produce the result directly

            vtxcontent pc = dagvtx_content(pred);
            entity old = vtxcontent_out(pc);

            // fix statement, needed for AIPO target
            substitute_image_in_statement(pred, old, res, false);

            // fix vertex
            vtxcontent_out(pc) = res;
            gen_remove(& dagvtx_succs(pred), v);
            bool done = gen_replace_in_list(dag_outputs(d), v, pred);
            pips_assert("output node was replaced", done);
            set_add_element(remove, remove, v);

            // BACK-FORWARD COPY PROPAGATION
            FOREACH(dagvtx, s, dagvtx_succs(pred))
            {
              substitute_image_in_statement(s, old, res, true);
              gen_replace_in_list(vtxcontent_inputs(dagvtx_content(s)),
                                  old, res);
            }
          }
          gen_free_list(preds);
        }
      }
    }

    // only one pass is needed because we're going backwards?
    // op-> X -copy-> Y images copies are replaced by op-> X & Y
    FOREACH(dagvtx, v, dag_vertices(d))
    {
      // skip special input nodes
      if (dagvtx_number(v)==0) continue;

      // skip already removed ops
      if (set_belong_p(remove, v)) continue;

      if (dagvtx_is_copy_p(v))
      {
        vtxcontent c = dagvtx_content(v);
        entity target = vtxcontent_out(c);
        pips_assert("one output and one input to copy",
              target!=entity_undefined && gen_length(vtxcontent_inputs(c))==1);

        // FORWARD COPY PROPAGATION
        // replace by its source everywhere it is used
        entity source = ENTITY(CAR(vtxcontent_inputs(c)));

        // remove!
        unlink_copy_vertex(d, source, v);

        // whether to actually remove v
        if (!gen_in_list_p(v, dag_outputs(d)))
          set_add_element(remove, remove, v);
      }
    }

    set_free(forwards);

    ifdebug(8) {
      pips_debug(4, "after FREIA_REMOVE_USELESS_COPIES:\n");
      dag_dump(stderr, "remove useless copies", d);
      // dag_dot_dump_prefix("main", "useless_copies", 0, d);
    }
  }

  if (get_bool_property("FREIA_MOVE_DIRECT_COPIES"))
  {
    // A-copy->B where A is an input is removed from the dag and managed outside
    // idem A-copy->B where A is an output

    // if A-copy->X and A-copy->Y where A is not an input, the second copy
    // is replaced by an external X-copy->Y

    // ??? BUG: it should be moved after the computation

    // what copies are kept in the dag
    hash_table intra_pipe_copies = hash_table_make(hash_pointer, 10);

    FOREACH(dagvtx, w, dag_vertices(d))
    {
      // skip already to-remove nodes
      if (set_belong_p(remove, w))
        continue;

      if (dagvtx_is_copy_p(w))
      {
        vtxcontent c = dagvtx_content(w);
        entity target = vtxcontent_out(c);
        pips_assert("one output and one input to copy",
              target!=entity_undefined && gen_length(vtxcontent_inputs(c))==1);

        entity source = ENTITY(CAR(vtxcontent_inputs(c)));
        dagvtx prod = dagvtx_get_producer(d, w, source, 0);

        if (source==target)
        {
          // A->copy->A??? this should not happen?
          set_add_element(remove, remove, w);
        }
        else if (dagvtx_number(prod)==0)
        {
          // producer is an input
          // fprintf(stderr, "COPY 1 removing %"_intFMT"\n", dagvtx_number(w));
          unlink_copy_vertex(d, source, w);
          set_add_element(remove, remove, w);
          *lbefore =
            CONS(statement, freia_copy_image(source, target), *lbefore);
        }
        else if (gen_in_list_p(prod, dag_outputs(d)))
        {
          // the producer is an output, which can be copied outside...
          // well, it may happen that the copy could be performed by the
          // accelerator for free, eg for SPoC one output link would happen
          // to be available... so we may really add an operation when
          // extracting it. However, this should not be an accelerator call?
          unlink_copy_vertex(d, source, w);
          set_add_element(remove, remove, w);
          *lafter = CONS(statement, freia_copy_image(source, target), *lafter);
        }
        else // source not an input, but the result of an internal computation
        {
          if (all_vertices_are_copies_or_measures_p(dagvtx_succs(prod)))
          {
            // ??? hmmm... there is an implicit assumption here that the
            // source of the copy will be an output...
            unlink_copy_vertex(d, source, w);
            set_add_element(remove, remove, w);
            *lafter =
              CONS(statement, freia_copy_image(source, target), *lafter);
          }
          else if (hash_defined_p(intra_pipe_copies, source))
          {
            // the source is already copied
            unlink_copy_vertex(d, source, w);
            set_add_element(remove, remove, w);
            *lafter = CONS(statement,
                 freia_copy_image((entity) hash_get(intra_pipe_copies, source),
                                  target), *lafter);
          }
          else // keep first copy
            hash_put(intra_pipe_copies, source, target);
        }
      }
    }
    hash_table_free(intra_pipe_copies);

    ifdebug(8) {
      pips_debug(4, "after FREIA_MOVE_DIRECT_COPIES:\n");
      dag_dump(stderr, "move direct copies", d);
      // dag_dot_dump_prefix("main", "direct_copies", 0, d);
    }
  }

  // cleanup dag (argh, beware that the order is not deterministic...)
  SET_FOREACH(dagvtx, r, remove)
  {
    pips_debug(7, "removing vertex %" _intFMT "\n", dagvtx_number(r));

    vtxcontent c = dagvtx_content(r);
    if (pstatement_statement_p(vtxcontent_source(c)))
      hwac_kill_statement(pstatement_statement(vtxcontent_source(c)));
    dag_remove_vertex(d, r);

    ifdebug(8)
      gen_context_recurse(d, r, dagvtx_domain, gen_true, check_removed);

    free_dagvtx(r);
  }

  set_free(remove);

  // further check for unused input nodes
  // this seems needed because some non determinism in the above cleanup.
  list vertices = gen_copy_seq(dag_vertices(d));
  FOREACH(dagvtx, v, vertices)
  {
    if (dagvtx_number(v)==0 && !dagvtx_succs(v))
    {
      dag_remove_vertex(d, v);
      free_dagvtx(v);
    }
  }
  gen_free_list(vertices);

  // show result
  ifdebug(6) {
    pips_debug(4, "resulting dag:\n");
    dag_dump(stderr, "cleaned", d);
    // dag_dot_dump_prefix("main", "cleaned", 0, d);
  }

  // former output images are either still computed or copies of computed
  size_t recount = gen_length(dag_outputs(d)) +
    gen_length(*lbefore) + gen_length(*lafter);
  pips_assert("right output count after dag optimizations",
              dag_output_count==recount);
}

/* return whether all vertices in list are mesures...
 */
static bool all_mesures_p(list lv)
{
  bool only_mes = true;
  FOREACH(dagvtx, v, lv)
  {
    if (dagvtx_optype(v)!=spoc_type_mes)
    {
      only_mes = false;
      break;
    }
  }
  return only_mes;
}

/* @return the set of all statements in dag
 */
static set dag_stats(dag d)
{
  set stats = set_make(set_pointer);
  statement s;
  FOREACH(dagvtx, v, dag_vertices(d))
    if ((s = dagvtx_statement(v)))
      set_add_element(stats, stats, s);
  return stats;
}

#define starts_with(s1, s2) (strncmp(s1, s2, strlen(s2))==0) /* a la java */

/* hmmm... this is poor, should rather rely on use-def chains.
 */
static bool any_use_statement(set stats)
{
  bool used = false;
  SET_FOREACH(statement, s, stats)
  {
    call c = freia_statement_to_call(s);
    if (c) {
      const char* name = entity_local_name(call_function(c));
      pips_debug(9, "call to %s\n", name);
      // some freia utils are considered harmless,
      // others imply an actual "use"
      if (!same_string_p(name, FREIA_FREE) &&
          !same_string_p(name, FREIA_ALLOC) &&
          !same_string_p(name, "freia_common_rx_image"))
        used = true;
    }
  }
  return used;
}

/* @return whether there is a significant use of e outside of stats
 */
static bool other_significant_uses
(entity e, const hash_table occs, const set stats)
{
  if (!occs) return true; // safe
  set all_stats = (set) hash_get(occs, e);
  pips_assert("some statement set", all_stats);
  set others = set_make(set_pointer);
  set_difference(others, all_stats, stats);
  // is there something significant in others?
  bool used = any_use_statement(others);
  set_free(others);
  pips_debug(9, "%s is %susefull\n", entity_name(e), used? "": "not ");
  return used;
}

/* hmmm...
 */
static bool variable_used_as_later_input(entity img, list ld)
{
  bool used = false;
  FOREACH(dag, d, ld)
  {
    FOREACH(dagvtx, v, dag_inputs(d))
    {
      if (dagvtx_image(v)==img)
      {
        used = true;
        break;
      }
    }
    if (used) break;
  }
  pips_debug(8, "%s: %s\n", entity_name(img), bool_to_string(used));
  return used;
}

static bool dag_image_is_an_input(dag d, entity img)
{
  bool is_input = false;
  FOREACH(dagvtx, v, dag_inputs(d))
  {
    if (dagvtx_image(v)==img)
    {
      is_input = true;
      break;
    }
  }

  pips_debug(4, "image %s input: %s\n",
             entity_name(img), bool_to_string(is_input));

  return is_input;
}

/* (re)compute the list of *GLOBAL* input & output images for this dag
 * ??? BUG the output is rather an approximation
 * should rely on used defs or out effects for the underlying
 * sequence. however, the status of chains and effects on C does not
 * allow it at the time. again after a look at DG (FC 08/08/2011)
 * @param d dag to consider
 * @param occs statement image occurences, may be NULL
 * @param output_images images that are output, may be NULL
 * @param ld list of some other dags, possibly NIL
 */
void dag_compute_outputs(
  dag d,
  const hash_table occs, const set output_images, const list ld, bool inloop)
{
  pips_debug(4, "inloop=%s\n", bool_to_string(inloop));

  set outvars = set_make(set_pointer);
  set outs = set_make(set_pointer);
  set toremove = set_make(set_pointer);
  set stats = dag_stats(d);

  FOREACH(dagvtx, v, dag_vertices(d))
  {
    pips_debug(8, "considering vertex %" _intFMT "\n", dagvtx_number(v));
    vtxcontent c = dagvtx_content(v);
    // skip special input nodes...
    if (dagvtx_number(v)!=0)
    {
      // get entity produced by vertex
      entity out = vtxcontent_out(c);

      pips_debug(8, "entity is %s\n", safe_entity_name(out));

      if (// we have an image
          out!=entity_undefined &&
          // it is not already an output
          !set_belong_p(outvars, out) &&
          // and either...
          (// this image is used as output (typically a tx call)
           (output_images && set_belong_p(output_images, out)) ||
           // no successors to this vertex BUT it is used somewhere (else?)
           (!dagvtx_succs(v) && other_significant_uses(out, occs, stats)) ||
           // all non-empty successors are measures?!
           (dagvtx_succs(v) && all_mesures_p(dagvtx_succs(v))) ||
           // new function parameter not yet an output
           (formal_parameter_p(out) && !set_belong_p(outvars, out)) ||
           // hmmm... yet another hack for freia_61
           (variable_used_as_later_input(out, ld)) ||
           // an output image is only reused by this dag within a loop?
           (inloop && dag_image_is_an_input(d, out))))
      {
        pips_debug(7, "appending %" _intFMT "\n", dagvtx_number(v));
        set_add_element(outvars, outvars, out);
        set_add_element(outs, outs, v);
      }
    }
    else
    {
      // ??? this aborts with terapix...
      // pips_assert("is an input vertex", gen_in_list_p(v, dag_inputs(d)));
      if (!dagvtx_succs(v))
        set_add_element(toremove, toremove, v);
    }
  }

  ifdebug(8)
  {
    dag_dump(stderr, "dag_compute_outputs", d);
    set_fprint(stderr, "new outs", outs, (gen_string_func_t) dagvtx_number_str);
  }

  // cleanup unused node inputs
  SET_FOREACH(dagvtx, vr, toremove)
    dag_remove_vertex(d, vr);

  gen_free_list(dag_outputs(d));
  dag_outputs(d) = set_to_sorted_list(outs, (gen_cmp_func_t) dagvtx_ordering);

  // cleanup
  set_free(stats);
  set_free(outs);
  set_free(outvars);
  set_free(toremove);
}

static dagvtx find_twin_vertex(dag d, dagvtx target)
{
  // pips_debug(9, "target is %"_intFMT"\n", dagvtx_number(target));
  FOREACH(dagvtx, v, dag_vertices(d))
    if (dagvtx_number(target)==dagvtx_number(v))
      return v;
  pips_internal_error("twin vertex not found for %" _intFMT "\n",
                      dagvtx_number(target));
  return NULL;
}

/* catch some cases of missing outs between splits...
 * for "freia_scalar_03"...
 * I'm not that sure about the algorithm.
 * This should rely be based on the CHAINS/DG, but the status still seems
 * hopeless, as too many arcs currently kept (FC 08/08/2011)
 * @param dfull full dag
 */
void freia_hack_fix_global_ins_outs(dag dfull, dag d)
{
  // cleanup inputs?
  // cleanup outputs
  FOREACH(dagvtx, v, dag_vertices(d))
  {
    // skip input nodes
    if (dagvtx_number(v)==0)
      continue;

    dagvtx twin = find_twin_vertex(dfull, v);
    if (// the vertex was an output node in the full dag
        gen_in_list_p(twin, dag_outputs(dfull)) ||
        // OR there were more successors in the full dag
        gen_length(dagvtx_succs(v))!=gen_length(dagvtx_succs(twin)))
    {
      pips_debug(4, "adding %" _intFMT " as output\n", dagvtx_number(v));
      dag_outputs(d) = gen_once(v, dag_outputs(d));
    }
  }
}

/* remove unneeded statements?
 * you must know they are really un-needed!
 */
void dag_cleanup_other_statements(dag d)
{
  set toremove = set_make(set_pointer);

  FOREACH(dagvtx, v, dag_vertices(d))
    if (dagvtx_other_stuff_p(v))
      set_add_element(toremove, toremove, v);

  SET_FOREACH(dagvtx, vr, toremove)
    dag_remove_vertex(d, vr);
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

/* returns whether there is a scalar RW dependency from any vs to v
 */
static bool any_scalar_dep(dagvtx v, set vs)
{
  bool dep = false;
  statement target = dagvtx_statement(v);

  // hmmm... may be called on a input node
  if (!target) return dep;

  SET_FOREACH(dagvtx, source, vs)
  {
    if (source==v)
      continue;

    pips_debug(9, "%"_intFMT" rw dep to %"_intFMT"?\n",
               dagvtx_number(source), statement_number(target));
    if (freia_scalar_rw_dep(dagvtx_statement(source), target, NULL))
    {
      dep = true;
      break;
    }
  }

  ifdebug(8) {
    string svs = set_to_string("vs", vs, (gen_string_func_t) dagvtx_number_str);
    pips_debug(8, "scalar rw dep on %d for (%s): %s\n",
             (int) dagvtx_number(v), svs, bool_to_string(dep));
    free(svs);
  }

  return dep;
}

/* check scalar dependency from computed to v.
 */
static bool
all_previous_stats_with_deps_are_computed(dag d, const set computed, dagvtx v)
{
  bool okay = true;

  // scan in statement order... does it matter?
  list lv = gen_nreverse(gen_copy_seq(dag_vertices(d)));
  FOREACH(dagvtx, pv, lv)
  {
    // all previous have been scanned
    if (pv==v) break;
    if (freia_scalar_rw_dep(dagvtx_statement(pv), dagvtx_statement(v), NULL) &&
        !set_belong_p(computed, pv))
    {
      okay = false;
      break;
    }
  }
  gen_free_list(lv);

  pips_debug(8, "all %d deps are okay: %s\n",
             (int) dagvtx_number(v), bool_to_string(okay));
  return okay;
}

/* return the vertices which may be computed from the list of
 * available images, excluding vertices in exclude.
 * return a list for determinism.
 * @param d is the considered full dag
 * @param computed holds all previously computed vertices
 * @param currents holds those in the current pipeline
 * @params maybe holds vertices with live images
 */
list /* of dagvtx */ dag_computable_vertices
  (dag d, const set computed, const set maybe, const set currents)
{
  list computable = NIL;
  set local_currents = set_make(set_pointer);
  set not_computed = set_make(set_pointer);
  set not_computed_before = set_make(set_pointer);

  set_assign(local_currents, currents);

  FOREACH(dagvtx, v, dag_vertices(d))
    if (!set_belong_p(computed, v))
      set_add_element(not_computed, not_computed, v);

  // hmmm... should reverse the list to handle implicit dependencies?
  // where, there is an assert() to check that it does not happen.
  list lv = gen_nreverse(gen_copy_seq(dag_vertices(d)));

  FOREACH(dagvtx, v, lv)
  {
    if (set_belong_p(computed, v))
      continue;

    if (dagvtx_other_stuff_p(v))
    {
      // a vertex with other stuff is assimilated to the pipeline
      // as soon as its dependences are fullfilled.
      // I have a problem here... I really need use_defs?
      if (all_previous_stats_with_deps_are_computed(d, computed, v))
      {
        computable = CONS(dagvtx, v, computable);
        set_add_element(local_currents, local_currents, v);
        set_del_element(not_computed, not_computed, v);
      }
    }
    else // we have an image computation
    {
      list preds = dag_vertex_preds(d, v);
      pips_debug(8, "%d predecessors to %" _intFMT "\n",
                 (int) gen_length(preds), dagvtx_number(v));

      // build subset of "not computed" which should occur before v
      // from the initial sequence point of view
      set_clear(not_computed_before);
      SET_FOREACH(dagvtx, vnc, not_computed)
      {
        if (dagvtx_number(vnc)<dagvtx_number(v))
          set_add_element(not_computed_before, not_computed_before, vnc);
      }

      if(// no scalar dependencies in the current pipeline
        !any_scalar_dep(v, local_currents) &&
        // or in the future of the graph
        !any_scalar_dep(v, not_computed_before) &&
        // and image dependencies are fulfilled.
        list_in_set_p(preds, maybe))
      {
        computable = CONS(dagvtx, v, computable);
        // we do not want deps with other currents considered!
        set_add_element(local_currents, local_currents, v);
        set_del_element(not_computed, not_computed, v);
      }

      gen_free_list(preds), preds = NIL;
    }

    // update availables: not needed under assert for no img reuse.
    // if (vtxcontent_out(c)!=entity_undefined)
    //  set_del_element(avails, avails, vtxcontent_out(c));
  }

  // cleanup
  set_free(local_currents);
  set_free(not_computed);
  set_free(not_computed_before);
  gen_free_list(lv);
  return computable;
}

void set_append_vertex_statements(set s, list lv)
{
  FOREACH(dagvtx, v, lv)
  {
    pstatement ps = vtxcontent_source(dagvtx_content(v));
    if (pstatement_statement_p(ps))
      set_add_element(s, s, pstatement_statement(ps));
  }
}

/* convert the first n items in list args to entities.
 */
static list
fs_expression_list_to_entity_list(list /* of expression */ args, int nargs)
{
  list /* of entity */ lent = NIL;
  int n=0;
  FOREACH(expression, ex, args)
  {
    syntax s = expression_syntax(ex);
    pips_assert("is a ref", syntax_reference_p(s));
    reference r = syntax_reference(s);
    pips_assert("simple ref", reference_indices(r)==NIL);
    lent = CONS(entity, reference_variable(r), lent);
    if (++n==nargs) break;
  }
  lent = gen_nreverse(lent);
  return lent;
}

/* extract first entity item from list.
 */
static entity extract_fist_item(list * lp)
{
  list l = gen_list_head(lp, 1);
  entity e = ENTITY(CAR(l));
  gen_free_list(l);
  return e;
}

/* append statement s to dag d
 */
static void dag_append_freia_call(dag d, statement s)
{
  pips_debug(5, "adding statement %" _intFMT "\n", statement_number(s));

  call c = freia_statement_to_call(s);

  if (c && entity_freia_api_p(call_function(c)))
  {
    entity called = call_function(c);
    const freia_api_t * api = hwac_freia_api(entity_local_name(called));
    pips_assert("some api", api!=NULL);
    list /* of entity */ args =
      fs_expression_list_to_entity_list(call_arguments(c),
          api->arg_img_in+api->arg_img_out);

    // extract arguments
    entity out = entity_undefined;
    pips_assert("one out image max for an AIPO", api->arg_img_out<=1);
    if (api->arg_img_out==1)
      out = extract_fist_item(&args);
    list ins = gen_list_head(&args, api->arg_img_in);
    pips_assert("no more arguments", gen_length(args)==0);

    vtxcontent cont =
      make_vtxcontent(-1, 0, make_pstatement_statement(s), ins, out);
    freia_spoc_set_operation
      (api, &vtxcontent_optype(cont), &vtxcontent_opid(cont));

    dagvtx nv = make_dagvtx(cont, NIL);
    dag_append_vertex(d, nv);
  }
  else // some other kind of statement that we may keep in the DAG
  {
    dagvtx nv =
      make_dagvtx(make_vtxcontent(spoc_type_oth, 0,
    make_pstatement_statement(s), NIL, entity_undefined), NIL);
    dag_vertices(d) = CONS(dagvtx, nv, dag_vertices(d));
  }
}

/* build a full dag from list of statements ls.
 * @param module
 * @param list of statements in sequence
 * @param number dag identifier in function
 * @param occurrences entity -> set of statements where they appear
 * @param output_images set of images that are output
 * @param ld list of other dags... (???)
 * @param inloop whether we might be in a loop
 */
dag freia_build_dag(
  string module, list ls, int number,
  const hash_table occurrences, const set output_images, const list ld,
  bool inloop)
{
  // build full dag
  dag fulld = make_dag(NIL, NIL, NIL);

  FOREACH(statement, s, ls)
    dag_append_freia_call(fulld, s);

  dag_compute_outputs(fulld, occurrences, output_images, ld, inloop);

  ifdebug(3) dag_dump(stderr, "fulld", fulld);

  // dump resulting dag
  dag_dot_dump_prefix(module, "dag_", number, fulld, NIL, NIL);

  return fulld;
}

/* tell whether we have something to do with images
   ??? hmmm... what about vol(cst_img()) ?
 */
bool dag_no_image_operation(dag d)
{
  return !dag_inputs(d) && !dag_outputs(d);
}

/**************************************************** NEW INTERMEDIATE IMAGE */

// in phrase
extern entity clone_variable_with_new_name(entity, const char*, const char*);
// in pipsmake
extern string compilation_unit_of_module(const char *);

// ??? could not find this anywhere
static entity freia_data2d_field(string field)
{
  string cu =
    compilation_unit_of_module(entity_local_name(get_current_module_entity()));
  entity fi = FindOrCreateEntity(cu, TYPEDEF_PREFIX FREIA_IMAGE_TYPE);
  pips_assert("freia_data2d type found", fi!=entity_undefined);
  type ut = ultimate_type(entity_type(fi));

  list fields =
    type_struct(entity_type(basic_derived(variable_basic(type_variable(ut)))));
  pips_assert("some fields", fields);
  FOREACH(entity, f, fields)
    if (same_string_p(field,
                      entity_local_name(f)+strlen(DUMMY_STRUCT_PREFIX)+2))
      return f;
  // should not get there
  pips_internal_error("cannot find freia_data2d field %s\n", field);
  return NULL;
}

static expression do_point_to(entity var, string field_name)
{
  // entity f2d = local_name_to_top_level_entity(FREIA_IMAGE_TYPE);
  expression evar = entity_to_expression(var);
  return call_to_expression(make_call(
    local_name_to_top_level_entity(POINT_TO_OPERATOR_NAME),
    gen_make_list(expression_domain, evar,
                  entity_to_expression(freia_data2d_field(field_name)),
                  NULL)));
}

/************************************* FIND FIRST REFERENCED IMAGE SOMEWHERE */
// hmmm... only works if initialized at the declaration level

static bool image_ref_flt(reference r, entity * image)
{
  entity var = reference_variable(r);
  if (freia_image_variable_p(var))
  {
    *image = var;
    gen_recurse_stop(NULL);
  }
  // do not go in array indexes
  return false;
}

// ??? l'arrache (tm)
static entity get_upper_model(entity model, const hash_table occs)
{
  entity image = NULL;
  // first look in declarations
  gen_context_recurse(entity_initial(model), &image,
                      reference_domain, image_ref_flt, gen_null);
  // then look for statements...
  if (!image && hash_defined_p(occs, model))
  {
    SET_FOREACH(statement, s, (set) hash_get(occs, model))
    {
      if (is_freia_alloc(s))
        gen_context_recurse(freia_statement_to_call(s), &image,
                            reference_domain, image_ref_flt, gen_null);
      if (image && image!=model) break;
      image = NULL;
    }
  }
  // go upwards!
  if (image && image!=model) image = get_upper_model(image, occs);
  return image? image: model;
}

/* @return statement where image is allocated/deallocated
 * maybe I should build another hash_table for this purpose?
 */
statement freia_memory_management_statement
(entity image, const hash_table occs, bool alloc)
{
  pips_debug(8, "look for %s statement for %s\n",
             alloc? "allocation": "deallocation", entity_name(image));

  if (hash_defined_p(occs, image))
  {
    SET_FOREACH(statement, s, (set) hash_get(occs, image))
    {
      // it may already have been switched to a sequence by a prior insert...
      // ??? it is unclear why I encounter nested sequences
      while (statement_sequence_p(s))
        s = STATEMENT(CAR(sequence_statements(statement_sequence(s))));

      // fprintf(stderr, "statement %"_intFMT"?\n", statement_number(s));
      if (alloc && is_freia_alloc(s))
      {
        call c = statement_call(s);
        pips_assert("must be an assignment", ENTITY_ASSIGN_P(call_function(c)));
        entity var;
        gen_context_recurse(EXPRESSION(CAR(call_arguments(c))), &var,
                            reference_domain, image_ref_flt, gen_null);
        if (var==image) return s;
      }
      else if (!alloc && is_freia_dealloc(s))
      {
        call c = statement_call(s);
        entity var;
        gen_context_recurse(EXPRESSION(CAR(call_arguments(c))), &var,
                            reference_domain, image_ref_flt, gen_null);
        if (var==image) return s;
      }
    }
  }
  return NULL;
}

/* @return a new image from the model
 */
static entity new_local_image_variable(entity model, const hash_table occs)
{
  entity img = NULL;
  int index = 1;
  do
  {
    string varname;
    int n = asprintf(&varname, "%s_%d", entity_local_name(model), index);
    pips_assert("asprintf succeeded", n!=-1);
    img = clone_variable_with_new_name
      (model, varname, module_local_name(get_current_module_entity()));
    free(varname);
    index++;
    if (img)
    {
      // use the reference of the model instead of the model itself, if found.
      entity ref = get_upper_model(model, occs);

      call alloc = make_call(local_name_to_top_level_entity(FREIA_ALLOC),
                             gen_make_list(expression_domain,
                                           do_point_to(ref, "bpp"),
                                           do_point_to(ref, "widthWa"),
                                           do_point_to(ref, "heightWa"),
                                           NULL));

      // handle allocation as the declaration initial value, if possible
      if (formal_parameter_p(ref))
      {
        free_value(entity_initial(img));
        entity_initial(img) = make_value_expression(call_to_expression(alloc));
      }
      else
      {
        statement sa = freia_memory_management_statement(ref, occs, true);

        if (sa)
        {
          // insert statement just after the reference declaration
          free_value(entity_initial(img));
          entity_initial(img) = make_value_expression(int_to_expression(0));

          statement na = make_assign_statement(entity_to_expression(img),
                                               call_to_expression(alloc));
          insert_statement(sa, na, false);
        }
        else
        {
          // this may happen if the "ref" is allocated at its declaration point,
          // as this statement is not currently found by our search

          // ??? try with a direct declaration
          free_value(entity_initial(img));
          entity_initial(img) =
            make_value_expression(call_to_expression(alloc));
        }
      }

      // should not be a parameter! overwrite storage if so...
      if (formal_parameter_p(img))
      {
        free_storage(entity_storage(img));
        entity_storage(img) = make_storage_rom();
        // make_storage_ram(make_ram(get_current_module_entity(),
        // entity_undefined, UNKNOWN_RAM_OFFSET, NIL));
      }
    }
  }
  while (img==NULL);
  return img;
}

/************************************************** FREIA IMAGE SUBSTITUTION */

typedef struct { entity old, new; bool write; } swis_ctx;

static void swis_ref_rwt(reference r, swis_ctx * ctx)
{
  if (reference_variable(r)==ctx->old)
    reference_variable(r) = ctx->new;
}

static bool swis_call_flt(call c, swis_ctx * ctx)
{
  const freia_api_t * api = hwac_freia_api(entity_local_name(call_function(c)));
  if (api)
  {
    list args = call_arguments(c);
    pips_assert("freia api call length is ok",
                gen_length(args)== (api->arg_img_out + api->arg_img_in +
                                    api->arg_misc_out + api->arg_misc_in));
    unsigned int i;
    // output arguments
    for(i=0; i<api->arg_img_out; i++)
    {
      if (!ctx->write)
        gen_recurse_stop(EXPRESSION(CAR(args)));
      args = CDR(args);
    }
    // input arguments
    for(i=0; i<api->arg_img_in; i++)
    {
      if (ctx->write)
        gen_recurse_stop(EXPRESSION(CAR(args)));
      args = CDR(args);
    }
    // skip remainder anyway
    while (args)
    {
      gen_recurse_stop(EXPRESSION(CAR(args)));
      args = CDR(args);
    }
  }
  // else this is not an AIPO call... we substitute everywhere
  return true;
}

/* switch read or written image in statement
 * if this is an AIPO call, only substitute output or input depending on write
 * otherwise all occurrences are substituted
 */
void freia_switch_image_in_statement(
  statement s, entity old, entity img, bool write)
{
  swis_ctx ctx = { old, img, write };
  gen_context_multi_recurse(s, &ctx,
                            call_domain, swis_call_flt, gen_null,
                            reference_domain, gen_true, swis_ref_rwt,
                            NULL);
}

/* switch to new image variable in v & its direct successors
 */
static void switch_image_variable(dagvtx v, entity old, entity img)
{
  vtxcontent c = dagvtx_content(v);
  pips_assert("switch image output", vtxcontent_out(c)==old);
  vtxcontent_out(c) = img;
  statement st = dagvtx_statement(v);
  pips_assert("some production statement", st!=NULL);
  freia_switch_image_in_statement(st, old, img, true);
  FOREACH(dagvtx, s, dagvtx_succs(v))
  {
    vtxcontent cs = dagvtx_content(s);
    gen_replace_in_list(vtxcontent_inputs(cs), old, img);
    statement st = dagvtx_statement(s);
    pips_assert("some producer statement", st!=NULL);
    freia_switch_image_in_statement(st, old, img, false);
  }
}

/* fix intermediate image reuse in dag
 * @return new image entities to be added to declarations
 */
list dag_fix_image_reuse(dag d, hash_table init, const hash_table occs)
{
  list images = NIL;
  set seen = set_make(hash_pointer);

  // dag vertices are assumed to be in reverse statement order,
  // so that last write is seen first and is kept by default
  FOREACH(dagvtx, v, dag_vertices(d))
  {
    entity img = dagvtx_image(v);
    if (!img || img==entity_undefined)
      continue;

    if (set_belong_p(seen, img) &&
        // but take care of image reuse, we must keep input images!
        !gen_in_list_p(v, dag_inputs(d)))
    {
      entity new_img = new_local_image_variable(img, occs);
      switch_image_variable(v, img, new_img);
      images = CONS(entity, new_img, images);
      // keep track of new->old image mapping, to reverse it latter eventually
      hash_put(init, new_img, img);
    }
    set_add_element(seen, seen, img);
  }

  set_free(seen);
  return images;
}

/************************************************************* DAG SPLITTING */

/* @brief split a dag on scalar dependencies only, with a greedy heuristics.
 * @param initial dag to split
 * @param alone_only whether to keep it alone (for non implemented cases)
 * @param choose_vertex chose a vertex from computable ones, may be NULL
 * @param priority how to prioritize computable vertices
 * @param priority_update stuff to help prioritization
 * @return a list of sub-dags, some of which may contain no image operations
 * For terapix, this pass also decides the schedule of image operations,
 * with the aim or reducing the number of necessary imagelets, so as to
 * maximise imagelet size.
 */
list // of dags
dag_split_on_scalars(const dag initial,
                     bool (*alone_only)(const dagvtx),
                     dagvtx (*choose_vertex)(const list, bool),
                     gen_cmp_func_t priority,
                     void (*priority_update)(const dag),
                     const set output_images)
{
  if (!single_image_assignement_p(initial))
    // well, it should work most of the time, so only a warning
    pips_user_warning("still some image reuse...\n");

  // ifdebug(1) pips_assert("initial dag ok", dag_consistent_p(initial));
  // if everything was removed by optimizations, there is nothing to do.
  if (dag_computation_count(initial)==0) return NIL;

  // work on a copy of the dag.
  dag dall = copy_dag(initial);
  list ld = NIL;
  set
    // current set of vertices to group
    current = set_make(set_pointer),
    // all vertices which are considered computed
    computed = set_make(set_pointer);

  do
  {
    list lcurrent = NIL, computables;
    set_clear(current);
    set_clear(computed);
    set_assign_list(computed, dag_inputs(dall));

    // GLOBAL for sorting
    if (priority_update) priority_update(dall);

    // transitive closure
    while ((computables =
            dag_computable_vertices(dall, computed, computed, current)))
    {
      ifdebug(7) {
        FOREACH(dagvtx, v, computables)
          dagvtx_dump(stderr, "computable", v);
      }

      // sort, mostly for determinism
      gen_sort_list(computables, priority);

      // choose one while building the schedule?
      dagvtx choice = choose_vertex?
        choose_vertex(computables, lcurrent? true: false):
        DAGVTX(CAR(computables));
      gen_free_list(computables), computables = NIL;

      ifdebug(7)
        dagvtx_dump(stderr, "choice", choice);

      // do not add if not alone
      if (alone_only && alone_only(choice) && lcurrent)
        break;

      set_add_element(current, current, choice);
      set_add_element(computed, computed, choice);
      lcurrent = gen_nconc(lcurrent, CONS(dagvtx, choice, NIL));

      // getout if was alone
      if (alone_only && alone_only(choice))
        break;
    }

    // is there something?
    if (lcurrent)
    {
      // build extracted dag
      dag nd = make_dag(NIL, NIL, NIL);
      FOREACH(dagvtx, v, lcurrent)
      {
        pips_debug(7, "extracting node %" _intFMT "\n", dagvtx_number(v));
        dag_append_vertex(nd, copy_dagvtx_norec(v));
      }
      dag_compute_outputs(nd, NULL, output_images, NIL, false);
      dag_cleanup_other_statements(nd);

      ifdebug(7) {
        // dag_dump(stderr, "updated dall", dall);
        dag_dump(stderr, "pushed dag", nd);
      }

      // ??? hmmm... should not be needed?
      freia_hack_fix_global_ins_outs(initial, nd);

      // update global list of dags to return.
      ld = CONS(dag, nd, ld);

      // cleanup full dag for next round
      FOREACH(dagvtx, w, lcurrent)
        dag_remove_vertex(dall, w);

      gen_free_list(lcurrent), lcurrent = NIL;
    }
  }
  while (set_size(current));

  free_dag(dall);
  return gen_nreverse(ld);
}

/*********************************************** BUILD A CONNECTED COMPOTENT */

// debug
static string dagvtx_nb(const void * s) {
  return itoa((int) dagvtx_number((dagvtx) s));
}

/* extract a sublist of lv which is a connected component.
 * update lv so that the extracted vertices are not there.
 * the extracted list must keep the order of the initial list!
 */
list dag_connected_component(
  dag d, list *plv, bool (*compat)(const dagvtx, const set, const dag))
{
  ifdebug(7) {
    pips_debug(7, "entering\n");
    gen_fprint(stderr, "initial list", *plv, dagvtx_nb);
  }

  set connected = set_make(set_pointer);
  set extracted = set_make(set_pointer);

  // boostrap.
  dagvtx first = DAGVTX(CAR(*plv));
  set_add_element(extracted, extracted, first);
  set_add_element(connected, connected, first);
  set_append_list(connected, dagvtx_succs(first));

  bool stable = false;
  while (!stable)
  {
    stable = true;

    // expand thru common dag inputs
    FOREACH(dagvtx, i, dag_inputs(d))
    {
      bool merge = false;
      FOREACH(dagvtx, v, dagvtx_succs(i)) {
        if (set_belong_p(extracted, v)) {
          merge = true;
          break;
        }
      }
      if (merge)
        set_append_list(connected, dagvtx_succs(i));
    }

    // expand with new vertices from the list
    FOREACH(dagvtx, v, *plv)
    {
      if (!set_belong_p(extracted, v) && (!compat || compat(v, extracted, d)))
      {
        // do we need to extract v?
        bool connect =  set_belong_p(connected, v);
        if (!connect) {
          FOREACH(dagvtx, s, dagvtx_succs(v))
            if (set_belong_p(extracted, s)) {
              connect = true;
              break;
            }
        }
        if (connect) {
          stable = false;
          set_add_element(extracted, extracted, v);
          set_add_element(connected, connected, v);
          set_append_list(connected, dagvtx_succs(v));
        }
      }
    }
  }

  // split initial list
  list lnew = NIL, lold = NIL;
  FOREACH(dagvtx, v, *plv)
  {
    if (set_belong_p(extracted, v))
      lnew = CONS(dagvtx, v, lnew);
    else
      lold = CONS(dagvtx, v, lold);
  }
  lnew = gen_nreverse(lnew);
  gen_free_list(*plv);
  *plv = gen_nreverse(lold);

  set_free(connected);
  set_free(extracted);

  ifdebug(7) {
    pips_debug(7, "exiting\n");
    gen_fprint(stderr, "final list", *plv, dagvtx_nb);
    gen_fprint(stderr, "returned", lnew, dagvtx_nb);
  }

  return lnew;
}
