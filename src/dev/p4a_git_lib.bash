#! /bin/bash

# Some function definitions to be used by other scripts

# Script to deal with Par4All repositories

# The PIPS modules:
PIPS_MODULES="linear newgen nlpmake pips validation"
#PIPS_MODULES=nlpmake
P4A_PACKAGES="$PIPS_MODULES polylib"

# All the suffix to have a standard branch infrastructure:
P4A_BRANCH_SUFFIX="$P4A_PACKAGES packages own"

# Set variables pointing to various Par4All parts if not already set:
# Where to get the git-svn instances from:
P4A_CRI_GIT_SVN=${P4A_CRI_GIT_SVN:-$P4A_TOP/CRI-git-svn}
# Can be overridden with the --root option:
P4A_ROOT=${P4A_ROOT:-$P4A_TOP/par4all}
P4A_PRIV_ROOT=${P4A_PRIV_ROOT:-$P4A_TOP/par4all-priv}

script=${0/*\//}

function stop_on_error() {
    # Stop on error, since going on after errors may lead to havoc for
    # example by working on some incorrect branch and so on
    set -o errexit
}


# verb level 'message to be shown...'
verb=0
function verb() {
  local level=$1 msg=$2
  if (( $verb >= $level )); then
      while (( level-- )) ; do echo -n '#' >&2 ; done
      echo " $msg" >&2
  fi
}


# Get into $current_branch the current git branch we are in:
function get_current_git_branch() {
    verb 1 "Entering get_current_git_branch"
    branch=`git branch | grep '^*' | sed 's/* //'`
    if [[ "$branch" == "(no branch)" ]] ; then
	echo "We are not in a branch, exiting..."
	exit
    fi
    current_branch=$branch
}


# Revert to current branch:
function revert_current_git_branch() {
    verb 1 "Reverting to the current branch"
    git checkout $branch
}


# Check if a branch exist:
function check_branch() {
    local branch=$1
    local find=`git branch | cut -c 3- | grep '^'$branch'$'`
    # The return status:
    [[ $branch = $find ]]
}


# Create a branch if it does not already exist:
function create_branch_if_needed() {
    local branch=$1
    local parent=$2
    if ! check_branch $branch; then
	# Create the branch if it does not exist:
	git branch $branch $parent
    fi
}


function do_update_CRI_git_svn() {
    verb 1 "Entering do_update_CRI_git_svn"
  enforce_P4A_TOP
  cd $P4A_CRI_GIT_SVN
  # Execute again this script to do a recursive git svn rebase:
  $0 --recursive-git-svn
}


# Apply a git action on all the git copy found inside the current directory:
function do_recursive_git() {
    verb 1 "Entering do_recursive_git"
    # If no argument is given, act as git svn rebase
    if [[ ! $@ ]] ; then
	actions="svn rebase"
    else
	actions=$@
    fi

    # Use -prune since for each .git encountered it is useless to dig into:
    git_dirs=`find . -name .git -prune`

    for g in $git_dirs; do
	d=`dirname $g`
	(
	    echo Entering directory $d:
	    cd $d
	    echo "    Action: git $actions"
	    git $actions
	)
	echo
    done
}


# Fetch into the par4all git all the parts from the remote git associated
# and merge into the tracking branches:
function do_fetch_remote_git() {
    verb 1 "Entering do_fetch_remote_git"
    enforce_P4A_TOP
    stop_on_error
    (
	cd $P4A_ROOT
	echo Assume the correct remotes are set in this par4all git copy:
	for i in $PIPS_MODULES; do
	    echo Fetching CRI/$i...
	    git fetch CRI/$i
	done
	echo Fetching ICPS/polylib...
	git fetch ICPS/polylib
    )
}


# Pull one module into Par4All:
function pull_remote_1_git() {
    MODULE=$1
    TRACKING_BRANCH=$2
    REMOTE_NAME=$3

    # First update the tracking branch:
    git checkout $TRACKING_BRANCH
    git pull $REMOTE_NAME master
    # Then jump into the branch of this name that will receive the
    # pull.  The idea is to have the possibility for a central place
    # to be able to change things here against remote inputs.
    git checkout $MODULE
    # Pull the master branch of the remote git. Use a subtree merge
    # strategy since the root directory of the remote git is relocated
    # in a subdirectory of the Par4All git:
    git merge --log --strategy=subtree $TRACKING_BRANCH
}


# Pull into the par4all git all the parts from the remote git associated:
function do_pull_remote_git() {
    verb 1 "Entering do_pull_remote_git"
    enforce_P4A_TOP
    stop_on_error
    (
	cd $P4A_ROOT
        # Get the current branch to come back into later:
	get_current_git_branch

	echo Assuming the correct remotes are set in this par4all git copy...
	for i in $PIPS_MODULES; do
	    pull_remote_1_git p4a-$i CRI-$i CRI/$i
	done
        # Same for the polylib:
        pull_remote_1_git p4a-polylib ICPS-polylib ICPS/polylib

	echo Update the global p4a hierarchy to keep in sync:
	do_merge_remote_branches p4a

        # Revert back into the branch we were at the beginning:
        git checkout $current_branch
    )
}


# Pull into the given branch hierarchy all the parts from the p4a remote
# git hierarchy:
function do_merge_remote_branches() {
    verb 1 "Entering do_merge_remote_git"
    enforce_P4A_TOP
    stop_on_error
    local merge_to_prefix_branches=$1
    local merge_origin_branch_prefix=$2
    if [[ -z $merge_origin_branch_prefix ]]; then
	# If we do not have $merge_origin_branch_prefix defined,
	# we use the standard reference branch:
	merge_origin_branch_prefix=p4a
    fi
    # Create target branches if needed:
    create_branch_if_needed $merge_to_prefix_branches $merge_origin_branch_prefix
    create_branch_if_needed $merge_to_prefix_branches-packages $merge_origin_branch_prefix-packages
    (
	cd $P4A_ROOT
        # Funny loop distribution to factorize out the checkout of the
        # package branch :-)
	for i in $P4A_PACKAGES; do
	    # Start the branch from the origin one:
	    create_branch_if_needed $merge_to_prefix_branches-$i $merge_origin_branch_prefix-$i
	    git checkout $merge_to_prefix_branches-$i
	    # Merge into the current branch the branch that buffers the remote
	    # PIPS git svn gateway that should have been populated by a
	    # previous do_pull_remote_git:
	    git merge --log p4a-$i
	done
	git checkout $merge_to_prefix_branches-packages
	for i in $P4A_PACKAGES; do
	    # The merge into packages branch:
	    git merge --log $merge_to_prefix_branches-$i
	done
	# And finish with the own branch:
        create_branch_if_needed $merge_to_prefix_branches-own $merge_origin_branch_prefix-own
	git checkout $merge_to_prefix_branches
	# Then merge into main branch the 2 subsidiary branches:
	git merge --log $merge_to_prefix_branches-packages
	git merge --log $merge_to_prefix_branches-own
    )
}


# Pull into the par4all git all the parts from the remote git associated:
function do_branch_action() {
    verb 1 "Entering do_branch_action"
    enforce_P4A_TOP
    stop_on_error
    action="$@"
    for b in $P4A_BRANCH_SUFFIX; do
	suffix=-$b
	eval $action
    done
    suffix=
    eval $action
}


# Pull into the par4all git all the parts from the remote git associated:
function do_add_remotes() {
    verb 1 "Entering add_svn_remote_path"
    enforce_P4A_TOP
    stop_on_error
    (
	cd $P4A_ROOT
	for i in $PIPS_MODULES; do
	    TRACKING_BRANCH=CRI-$i
	    REMOTE_NAME=CRI/$i
	    REMOTE_GIT_URL=$P4A_CRI_GIT_SVN/$i
	    # Add the git svn as remote git repository:
	    git remote add $REMOTE_NAME $REMOTE_GIT_URL
            # Fetch the history:
	    git fetch $REMOTE_NAME
	    # Create the tracking branch:
	    git branch $TRACKING_BRANCH $REMOTE_NAME/master
	done
	git remote add ICPS/polylib git://repo.or.cz/polylib.git packages/polylib
	git fetch ICPS/polylib
	git branch ICPS-polylib ICPS/polylib/master
    )
}

# Some Emacs stuff:
### Local Variables:
### mode: sh
### mode: flyspell
### ispell-local-dictionary: "american"
### End:
