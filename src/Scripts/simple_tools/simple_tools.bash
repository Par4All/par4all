# $Id$
#
# Copyright 1989-2014 MINES ParisTech
#
# This file is part of PIPS.
#
# PIPS is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIPS.  If not, see <http://www.gnu.org/licenses/>.
#

# Base functions to build simple shell scripts around tpips.

[ $# -eq 0 -o "$1" = "-h" -o "$1" = "--help" ] && display_usage_and_exit

workspace=`basename $0$$`

function create_workspace () {
    cat <<EOF
echo Create the workspace from the files to parallelize
create $workspace $*

EOF
}


function display_end () {
    # Display this message on stderr to avoid blinking the validation
    echo "The parallelized code is in $workspace.database/Src" 1>&2
}


function select_c_language () {
    cat <<EOF
echo Select some options to deal with the C Language:
echo Do not display original number lines as comment:
setproperty PRETTYPRINT_STATEMENT_NUMBER FALSE
echo If possible, transform simple for-loops into do-loop à la Fortran, simpler to analyze:
apply FOR_LOOP_TO_DO_LOOP[%ALLFUNC]
echo Desugaring other for-loops into plain while-loops for the time we improve semantics ameliorations in PIPS:
apply FOR_LOOP_TO_WHILE_LOOP[%ALLFUNC]

EOF
}


function select_most_precise_analysis () {
    cat <<EOF
echo Select the most precise analysis:
activate MUST_REGIONS
activate TRANSFORMERS_INTER_FULL
activate INTERPROCEDURAL_SUMMARY_PRECONDITION
activate PRECONDITIONS_INTER_FULL
activate REGION_CHAINS
echo Compute the intraprocedural preconditions at the same time as
echo   transformers and use them to improve the accuracy of expression
echo   and statement transformers:
setproperty SEMANTICS_COMPUTE_TRANSFORMERS_IN_CONTEXT TRUE
# Use the more precise fix point operator to cope with while loops:
setproperty SEMANTICS_FIX_POINT_OPERATOR "derivative"
echo Try to restructure the code for more precision:
setproperty UNSPAGHETTIFY_TEST_RESTRUCTURING=TRUE
setproperty UNSPAGHETTIFY_RECURSIVE_DECOMPOSITION=TRUE
echo
echo "Warning: assume that there is no aliasing between IO streams (FILE * variables)"
setproperty ALIASING_ACROSS_IO_STREAMS FALSE
echo "Warning: this is a work in progress. Assume no weird aliasing"
setproperty CONSTANT_PATH_EFFECTS FALSE
EOF
}


function privatize_scalar_variables () {
    cat <<EOF
echo Privatize scalar variables on all the modules of the program:
apply PRIVATIZE_MODULE[%ALLFUNC]

EOF
}


function openmp_parallelization_rice () {
    cat <<EOF
echo Ask for some statistics about the job to be done:
setproperty PARALLELIZATION_STATISTICS=TRUE
echo Ask for the parallelization of all the modules of the program with OpenMP output:
activate RICE_ALL_DEPENDENCE
activate PRINT_PARALLELIZEDOMP_CODE
display PARALLELPRINTED_FILE[%ALLFUNC]

EOF
}


function openmp_parallelization_coarse_grain () {
    cat <<EOF
echo Ask for some statistics about the job to be done:
setproperty PARALLELIZATION_STATISTICS=TRUE
echo Ask for the parallelization of all the modules of the program with OpenMP output:
apply COARSE_GRAIN_PARALLELIZATION[%ALLFUNC]
display PRINTED_FILE[%ALLFUNC]

EOF
}


function internalize_parallel_code () {
    cat <<EOF
echo Consider the generated parallel as the sequential code now:
# Since INTERNALIZE_PARALLEL_CODE modify the sequentiel code,
# applying it on the parallel code of another module that may depend
# of the previous module can lead to reapplying the parallelization each time...
# So use cappy insted of apply here:
capply INTERNALIZE_PARALLEL_CODE[%ALLFUNC]

EOF
}


function select_omp_sequential_output () {
    cat <<EOF
# Force the code regeneration with OpenMP parallel syntax
# instead of sequential output:
setproperty PRETTYPRINT_SEQUENTIAL_STYLE    "omp"

EOF
}


function regenerate_source () {
    cat <<EOF
echo Regenerate the sources from the PIPS transformed code:
apply UNSPLIT[%PROGRAM]
#display PRINTED_FILE[%ALL]

EOF
}
