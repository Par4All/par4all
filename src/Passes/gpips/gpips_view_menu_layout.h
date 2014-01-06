/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

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
/* The following description of the ``View'' menu is automatically generated from ``pipsmake-rc.tex'' through \verb|./generate_all_menu_documentation|. */

{ "Sequential View", "PRINTED_FILE", gpips_display_plain_file, "sequential" },
{ "User View", "PARSED_PRINTED_FILE", gpips_display_plain_file, "user" },
{ "Alias View", "ALIAS_FILE", gpips_display_plain_file, "-" },
{ "Control Graph Sequential View", "GRAPH_PRINTED_FILE", gpips_display_graph_file_display, "-" },
{ "", "", NULL, "" },
{ "Dependence Graph View", "DG_FILE", gpips_display_plain_file, "DG" },
{ "", "", NULL, "" },
{ "Callgraph View", "CALLGRAPH_FILE", gpips_display_plain_file, "callgraph" },
{ "Graphical Call Graph", "DVCG_FILE", gpips_display_graph_file_display, "callgraph" },
{ "ICFG View", "ICFG_FILE", gpips_display_plain_file, "ICFG" },
{ "", "", NULL, "" },
{ "Parallel View", "PARALLELPRINTED_FILE", gpips_display_plain_file, "parallel" },
{ "", "", NULL, "" },
{ "Flint View", "FLINTED_FILE", gpips_display_plain_file, "-" },
