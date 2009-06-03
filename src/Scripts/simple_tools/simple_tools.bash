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
echo Parse as C:
activate C_PARSER
echo Prettyprint the source as C, of course
setproperty PRETTYPRINT_C_CODE TRUE
echo Do not display original number lines as comment:
setproperty PRETTYPRINT_STATEMENT_NUMBER FALSE
echo If possible, transform simple for-loops into do-loop à la Fortran, simpler to analyze:
setproperty FOR_TO_DO_LOOP_IN_CONTROLIZER   TRUE
echo Desugaring other for-loops into plain while-loops fot the time we improve semantics ameliorations in PIPS:
setproperty FOR_TO_WHILE_LOOP_IN_CONTROLIZER   TRUE

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
activate COARSE_GRAIN_PARALLELIZATION
activate PRINT_PARALLELIZEDOMP_CODE
display PARALLELPRINTED_FILE[%ALLFUNC]

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
