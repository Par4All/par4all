#! /bin/bash
#
# $Id: svn_branch.sh 367 2006-01-04 10:34:49Z coelho $
#
# $URL: file:///users/cri/coelho/SVN/svn/svn_branch.sh $
#
#
# What is a branch?
#
# From the "svn" point of view, it is a directory copy. Fine, but it lacks
# clear administrativa data such as where the branch comes from, what
# merges have been performed, and so on.
#
#
# This script is inspired by the "svnmerge" set of scripts proposed in svn 
# contrib area.  However its philosophy is different. 
#
# "svnmerge" targets a development model where the trunk holds a moving
# version which may be a little unstable, and branches are used for 
# stableizing the software and picking patches to be integrated in the 
# release. It emphasizes the management of a list of patches merged
# from the source to the branch.
#
# trunk/_________+________*_____+____+_____*__+__
#   |                     |                |
#   | initial copy        | some merges....
#   V                     V                V
# branches/1.2.x/____x____*____x____*______*____x____
#
#
# This "svnbranch" script aims at having a more stable 'production' trunk 
# and manage development in less stable branches, which are expected to hold
# long standing developments. The branches can get some trunk changes,
# and are to be pushed back to the trunk when the development is over.
# So mergings are to be managed in both directions. Also, less commands
# are required wrt "svnmerge".
#
# trunk/____________*__+_____*__+_____X__
#   |               |        |        A
#   | initial copy  | some pulls...   | push (merge back)
#   V               V        V        |
# branches/dev/__x__*_____x__*_____x__/
#
#
# Portability: 
#  - this script is *pure* bash (and svn, obviously)
#  - the intention behind this development is to be a specification and 
#    testbed tool, but the functionnalities should be included in some 
#    future svn release.
#
# TODO:
#  - option for giving a source wc path to push
#  - option to include logs on push? log pointer is enough??
#  - add file option to put the suggested commit message
#  - store pulled revisions in compact form

# keep command name
command=${0/*\//}

# keep revision number
cmd_rev='$Rev: 367 $'
cmd_rev=${cmd_rev/*: /}
cmd_rev=${cmd_rev/ \$/}

# how verbose to be
# 0: silent
# 1: normal messages
# 2,3: more verbose
# 4 and more: debug
verbosity=1

# executable to be used to issue svn commands
svn='svn'

# directory for temporary data
tmp='/tmp'

# help string
usage="usage: $command action [options...] arguments
  revision of $command is $cmd_rev

  Possible actions with their expected arguments:
  * help
    . show usage help about this command.
  * version
    . show version of this command.
  * create [options] source_url destination_url [checkout-local-path]
    . create a branch managed by this command.
    --url url: source and destination can be relative to this url.
  * diff [options] wc_path...
    . show differences between branches and their sources.
  * pull [options] wc_path...
    . merge into the branches from their sources.
  * push [options] wc_path...
    . merge back from the branches into their sources.
    --remove: automatically remove branch after push.
  * avail path...
    . revisions available from source for merging
  * log branch_path
    . log of revisions of the branch source 
  * relocate source_url branch
    . change location of branch source.
  * test dir
    . tell whether dir is a branch

  Other common options:
    --verbose: be more verbose (show issued svn commands)
    --debug: be really more verbose (you don't want that)
    --quiet: be quiet (not advisable)
    --tmp tmp: directory for temporary files or directories, default is /tmp
    --commit: try to perform all commits (not really advisable)
    --revision rev: revision to consider (for create, pull, push...)
    --dry-run: just pretend"

# useful for pattern matching
newline="
"

##################################################################### FUNCTIONS

function verb()
{
    local level=$1 ; shift
    [[ $verbosity -ge $level ]] && echo "[$command] $@" >&2
}

function error()
{
    local code=$1 ; shift
    echo "error: $@" >&2
    exit $code
}

function usage()
{
    echo "$usage"
    [ $1 -ne 0 ] && error "$@"
    exit 0
}

# does it look like an svn url?
function is_svn_url()
{
    # obscure error: # [[ $1 == @(http:*|https:*|svn:*|file:*|svn+ssh:*) ]]
    case $1 in http:*|https:*|svn:*|file:*|svn+ssh:*) return 0 ;; esac
    return 1
}

function is_action()
{
    # idem
    case $1 in 
	create|diff|pull|push|relocate|test|info|avail|log|help|version) 
	    return 0 ;;
    esac
    return 1
}

# is_svn_working_url tested-url
# returns whether the url exists and is working
function is_svn_working_url()
{
    is_svn_url $1 && svn proplist $1 > /dev/null 2>&1
}

function is_svn_wcpath()
{
    test -d $1 && test -d $1/.svn
}

function is_svn_branch()
{
    local version=$(do_svn propget 'svnbranch:version' $1)
    [ "$version" ]
}

function is_svn_branch_wcpath()
{
    is_svn_wcpath $1 || return 1
    is_svn_branch $1
}

# perform a svn command but quit on error.
function safe_svn()
{
    verb 2 $svn $svn_options "$@"
    $nodo $svn $svn_options "$@" || error 2 "svn failed: $svn $@"
}

# same as above, but to it anyway even under dry-run
function do_svn()
{
    verb 2 $svn $svn_options "$@"
    $svn $svn_options "$@" || error 2 "svn failed: $svn $@"
}

function get_info()
{
    local url=$1 begin=$2 ; shift 2
    local info=$(do_svn info $url) 
    info=${info/*$newline$begin/}
    info=${info/$newline*/}
    echo $info
}

# get_last_revision url
function get_last_revision()
{
    get_info $1 'Last Changed Rev: '
}

function get_url()
{
    get_info $1 'URL: '
}

# reset property $pname on $path to its value $pval
function reset_property()
{
    local pname=$1 pval=$2 path=$3 ; shift 3

    if [ "$pval" ] ; then
	safe_svn propset $pname "$pval" $path
    else
	safe_svn propdel $pname $path
    fi
}

# check_status allowed dir
# return whether some non-allowed status are present in dir
function check_status()
{
    local allowed=$1 dir=$2 ; shift 2
    local answer=$(
	do_svn status --ignore-externals $dir |
	while read line ; do
	    [[ $line != [$allowed]* ]] && { echo 1 ; break ; }
	done
    )
    return $answer
}

# is everything committed in dir?
function all_is_committed()
{
    check_status 'X?' $1
}

function dirname()
{
    echo ${1%\/*}
}

function basename()
{
    echo ${1##*\/}
}

function get_repository_root_depth()
{
    local rep=$1 ; shift
    local next=$(dirname $rep) depth=0
    while is_svn_working_url $next ; do
	let depth++ 
	rep=$next 
	next=$(dirname $next)
    done
    echo $rep $depth
}

function get_repository_root()
{
    local root_depth=$(get_repository_root_depth $1)
    echo ${root_depth/ */}
}

function get_repository_depth()
{
    local root_depth=$(get_repository_root_depth $1)
    echo ${root_depth/* /}
}

# perform_merge revision dir ref_url src_url
function perform_merge()
{
    local revs=$1 dir=$2 ref_url=$3 src_url=$4 ; shift 4

    # some checks
    is_svn_url $ref_url || error 16 "$ref_url not an svn url"
    is_svn_url $src_url || error 17 "$src_url not an svn url"

    if ! is_svn_wcpath $dir ; then
	# get a working copy
	test -d $dir && error 17 "$dir already exists and is not working copy"
	safe_svn checkout $ref_url $dir
    else
	# check the working copy
	local url=$(get_url $dir)
	[ $url = $ref_url ] || error 18 "$dir not a working copy of $ref_url"
	safe_svn update $dir
    fi

    # save svnbranch:* props
    local save_src_ver=$(safe_svn propget 'svnbranch:version'    $dir)
    local save_src_url=$(safe_svn propget 'svnbranch:source-url' $dir)
    local save_src_rev=$(safe_svn propget 'svnbranch:source-rev' $dir)
    local save_pll_rev=$(safe_svn propget 'svnbranch:pulled-rev' $dir)
    local save_psh_rev=$(safe_svn propget 'svnbranch:pushed-rev' $dir)

    # apply differences
    # obscure issue with 1.2.3: svn: Move failed...
    ##safe_svn merge --revision $revs $src_url $dir
    pushd $dir
    safe_svn merge --revision $revs $src_url .
    popd

    # restore saved svnbranch:* props
    reset_property 'svnbranch:version'    "$save_src_ver" $dir
    reset_property 'svnbranch:source-url' "$save_src_url" $dir
    reset_property 'svnbranch:source-rev' "$save_src_rev" $dir
    reset_property 'svnbranch:pulled-rev' "$save_pll_rev" $dir
    reset_property 'svnbranch:pushed-rev' "$save_psh_rev" $dir

    # ???
    safe_svn update $dir

    verb 1 "status of $dir"
    safe_svn status --ignore-externals $dir
}

# safe_svn_commit message target command...
function safe_svn_commit()
{
    local message=$1 target=$2 ; shift 2
    if [[ $do_commit ]] ; then
	safe_svn commit --message "$message" $target
	[[ $@ ]] && "$@"
    else
	echo "please commit $target if you agree"
	echo "suggested message:"
	echo "$message"
    fi
}

################################################################# REVISION LIST
# revision lists come in two flavors.
# expanded: " 1 2 3 7 9 10 11 20 21"
# compact: " 1:3 7 9:11 20:21"

# get all revisions in log...
function get_all_revisions()
{
    local revs=$1 target=$2 ; shift 2
    do_svn log --quiet --revision $revs $target |
    while read rev line ; do
	case $rev in r*) echo -n " ${rev/r/}" ;; esac
    done
}

# substract from expanded list $1 list $2
function list_sub()
{
    local l1="${1# } " l2="${2# } " ; shift 2
    local h1 h2
    while [[ $l1 && $l2 ]] ; do
	[[ $h1 ]] || h1=${l1%% *} l1=${l1#* }
	[[ $h2 ]] || h2=${l2%% *} l2=${l2#* }
	if [[ $h1 -eq $h2 ]] ; then
	    h1= h2=
	elif [[ $h1 -lt $h2 ]] ; then
	    echo -n " $h1"
	    h1=
	else
	    h2=
	fi
    done
    [[ $h1 ]] && echo -n " $h1"
    [[ $l1 ]] && echo -n " $l1"
}

# union of both expanded lists
function list_union()
{
    local l1="${1# } " l2="${2# } " ; shift 2
    local h1 h2
    while [[ $l1 && $l2 ]] ; do
	[[ $h1 ]] || h1=${l1%% *} l1=${l1#* }
	[[ $h2 ]] || h2=${l2%% *} l2=${l2#* }
	if [[ $h1 -eq $h2 ]] ; then
	    echo -n " $h1" 
	    h1= h2=
	elif [[ $h1 -lt $h2 ]] ; then
	    echo -n " $h1"
	    h1=
	else
	    echo -n " $h2"
	    h2=
	fi
    done
    [[ $h1 ]] && echo -n " $h1"
    [[ $h2 ]] && echo -n " $h2"
    [[ $l1 ]] && echo -n " $l1"
    [[ $l2 ]] && echo -n " $l2"
}

# build a compact form for a list of revisions
function compact()
{
    local prev=$1 expect=$1 start=$1
    for r ; do
	if [[ $r -ne $expect ]] ; then
	    echo -n " $start"
	    [[ $prev -ne $start ]] && echo -n ":$prev"
	    start=$r
	fi
	prev=$r expect=$r
	let expect++
    done
    echo -n " $start"
    [[ $prev -ne $start ]] && echo -n ":$prev"
}

# build an expanded list of revisions from a compact form
function expand()
{
    local i
    for i ; do
	if [[ $i == *:* ]] ; then
	    local n=${i/:*/} end=${i/*:/}
	    while true ; do
		echo -n " $n"
		[ $n -eq $end ] && break
		let n++
	    done
	else
	    echo -n " $i"
	fi
    done
}

################################################################### DO THE JOBS

# create source_url branche_url
# globals: tmp command revision do_commit
function create()
{
    local src=$1 bch=$2 ; shift 2
    local tmp_co=$tmp/$command.$$ 
    local parent=$(dirname $bch) branch=$(basename $bch)

    is_svn_working_url $src \
	|| error 3 "source url $src not found"

    is_svn_working_url $parent \
	|| error 4 "parent $parent of destination url not found"

    is_svn_working_url $bch \
	&& error 5 "destination url $bch already exists"

    test -d $tmp_co \
	&& error 6 "needed temporary directory $tmp_co already exists"

    local src_rev=$revision
    [[ ! $revision ]] && src_rev=$(get_last_revision $src)

    # this is to perform a URL to WCPATH copy
    safe_svn checkout --non-recursive $parent $tmp_co
    
    # get branch contents
    safe_svn copy --revision $src_rev $src $tmp_co/$branch

    # set administrativa data
    safe_svn propset 'svnbranch:version'           1 $tmp_co/$branch
    safe_svn propset 'svnbranch:source-url'     $src $tmp_co/$branch
    safe_svn propset 'svnbranch:source-rev' $src_rev $tmp_co/$branch
    safe_svn propset 'svnbranch:pulled-rev'       '' $tmp_co/$branch
    safe_svn propset 'svnbranch:pushed-rev'       '' $tmp_co/$branch
    #store "pushable" or "non-pushable" revisions somewhere?
    # the point is that when pushing differences and reusing the
    # logs, logs due to pull of the master source should not be included.
    # this could be managed with svn more easily?

    safe_svn_commit 'create new branch' $tmp_co	\
	$nodo rm -rf $tmp_co 
}

# pull wc_path
# globals: 
function pull()
{
    local dir=$1 ; shift
    test -d $dir || error 7 "no such directory $dir"
    is_svn_branch $dir || error 8 "$dir is not a branch"

    local src_url=$(do_svn propget 'svnbranch:source-url' $dir)
    local pulled_rev=$(do_svn propget 'svnbranch:pulled-rev' $dir)
    
    if [[ ! $revision ]] ; then
	local last_pulled_rev=${pulled_rev/* /}
	last_pulled_rev=${last_pulled_rev/*:/}
	[[ $last_pulled_rev ]] || \
	    last_pulled_rev=$(do_svn propget 'svnbranch:source-rev' $dir)
	src_rev=$(get_last_revision $src_url)
	if [ $src_rev -gt $last_pulled_rev ] ; then
	    revision=$last_pulled_rev:$src_rev
	else
	    revision=
	fi
    fi

    if [[ $revision ]] ; then
	# fix revision
	[[ $revision == *:* ]] || revision=$(( $revision - 1 )):$revision

	perform_merge $revision $dir $(get_url $dir) $src_url

	local pulled_stuff=$(get_all_revisions $revision $src_url)
	local new_pulled=$(list_union "$pulled_rev" "$pulled_stuff")

	# update merged status with respect to merge above.
	safe_svn propset 'svnbranch:pulled-rev' "$new_pulled" $dir

	# ??? this seems necessary for some obscure reasons.
	safe_svn update $dir

	local br_url=$(get_url $dir)
	local message=$(
	    echo "pulled revisions $revision"
	    echo "from master $src_url"
	    echo "to branch $br_url"
	)
	safe_svn_commit "$message" $dir
    else
	verb 1 "nothing to pull into $dir"
    fi
}

# -- merge back developments into source url
# push wc_path subname
# globals: revision do_remove
function push()
{
    local dir=$1 sub=$2 ; shift 2
    test -d $dir || error 9 "no such directory $dir" 
    is_svn_branch $dir || error 10 "$dir is not a branch"
    all_is_committed $dir || error 11 "$dir is not committed, cannot push"

    # update needed so that last revision is ok.
    safe_svn update $dir
    local src_url=$(do_svn propget 'svnbranch:source-url' $dir)

    local pushed_rev=$(do_svn propget 'svnbranch:pushed-rev' $dir)
    [[ ! $pushed_rev ]] && \
	pushed_rev=$(do_svn propget 'svnbranch:source-rev' $dir)

    local current_rev=$revision
    [[ ! $revision ]] && current_rev=$(get_last_revision $dir)

    #TODO: should check that all merged are already done? or not??

    if [ $current_rev -gt $pushed_rev ] ; then

	local tmp_co=$tmp/$command.$$.$sub
	test -d $tmp_co && error 5 "temporary directory $tmp_co already exists"
	local branch_url=$(get_url $dir)

	perform_merge $pushed_rev:$current_rev $tmp_co $src_url $branch_url

	if [[ $do_remove ]] ; then
	    # ???
	    safe_svn remove $dir
	else
	    # fix svnbranch pushed-rev status in $dir
	    safe_svn propset 'svnbranch:pushed-rev' $current_rev $dir
	fi

	verb 1 "status of $dir"
	safe_svn status --ignore-externals $dir

	local message=$(
	    echo "pushed revisions $pushed_rev:$current_rev"
	    echo "from branch $branch_url"
	    echo "to master $src_url"
	)

	safe_svn_commit "$message" $tmp_co \
	    $nodo rm -rf $tmp_co

        # remove the current directory while being inside it is no good
	[[ $do_remove && $dir = '.' ]] && \
	    verb 1 "your shell will block the cleanup of $dir"
	
	safe_svn_commit "$message" $dir
    else
	verb 1 "nothing to push into $src_url from $dir"
    fi
}

# info wc_path
# globals: revision
function info()
{
    local dir=$1 ; shift
    local version=$(do_svn propget 'svnbranch:version' $dir)
    local rev=
    [[ $revision ]] && rev="--revision $revision"
    safe_svn $rev info $dir
    if [ "$version" ] ; then
	echo "Branch Management: version $version"
	local src_url=$(do_svn $rev propget 'svnbranch:source-url' $dir)
	echo "Branch Source URL: $src_url"
	echo "Branch Source Initial Rev:" \
	    $(do_svn $rev propget 'svnbranch:source-rev' $dir)
	echo "Branch Source Current Rev:" $(get_last_revision $src_url)
	echo "Branch Pulled Rev:" \
	    $(do_svn $rev propget 'svnbranch:pulled-rev' $dir)
	echo "Branch Last Pushed Rev:" \
	    $(do_svn $rev propget 'svnbranch:pushed-rev' $dir)
    else
	echo "Branch Management: none"
    fi
}

# diff wc_path
function diff()
{
    if is_svn_branch $1 ; then
	local src_url=$(do_svn propget 'svnbranch:source-url' $1)
	local dir_url=$(get_url $1)
	safe_svn diff $src_url $dir_url
    else
	echo "Path $dir not under branch management."
    fi
}

# relocate new-source-url branch
function relocate()
{
    local new_src=$1 branch=$2 ; shift 2
    is_svn_branch $branch || error 21 "$branch is not a branch"

    [[ ! $force ]] && is_svn_working_url $new_src || \
	    error 22 "$new_src is not a working svn url"

    local old_src=$(safe_svn propget 'svnbranch:source-url' $branch)
    safe_svn propset 'svnbranch:source-url' $new_src $branch

    local message=$(
	echo "branch relocation"
	echo "from $old_src"
	echo "to $new_src"
    )

    safe_svn_commit "$message" $branch
}

function default_revisions()
{
    local pname=$1 dir=$2 ; shift 2
    [[ $revision ]] || \
    {
	local start=$(do_svn propget $pname $dir)
	let start++ # log is inclusive
	revision=$start:HEAD
    }
    echo $revision
}

function avail()
{
    local dir=$1 ; shift
    is_svn_branch $dir || error 33 "expecting a branch target"
    local src_url=$(do_svn propget 'svnbranch:source-url' $dir)
    local revisions=$(default_revisions 'svnbranch:source-rev' $dir)
    local all_src_revs=$(get_all_revisions $revisions $src_url)
    local pulled_revs=$(do_svn propget 'svnbranch:pulled-rev' $dir)
    echo $(list_sub "$all_src_revs" "$(expand $pulled_revs)")
}

function log()
{
    local dir=$1 ; shift
    local rev= src_url=$(do_svn propget 'svnbranch:source-url' $dir)
    for rev in $(avail $dir) ; do
	safe_svn log --revision $rev $src_url
    done
}

################################################################# PARSE OPTIONS

action=
do_commit=
do_remove=
url=
nodo=
revision=
svn_options=

while [[ $# -gt 0 ]] ; do
  arg=$1
  if [[ $arg == -* ]] ; then # is it an option?
      shift 
      verb 4 "handling argument $arg"
      case $arg in
	  # other options
	  -d|--debug)
	      let verbosity+=3
	      ;;
	  -v|--verbose) 
	      let verbosity++
	      ;;
	  -q|--no-verbose|--quiet)
	      verbosity=0
	      ;;
	  -h)
	      action='help'
	      ;;
	  -s|--svn)
	      svn=$1
	      shift
	      ;;
	  --svn=*)
	      svn=${arg/*=/}
	      ;;
	  -t|--tmp|--temporary)
	      tmp=$1
	      shift
	      ;;
	  --tmp=*|--temporary=*)
	      tmp=${arg/*=/}
	      ;;
	  # forward some options to svn
	  --username|--password|--config-dir)
	      svn_options="$svn_options $arg $1"
	      shift
	      ;;
	  --no-auth-cache|--non-interactive)
	      svn_options="$svn_options $arg"
	      ;;
	  # level of operations
	  --commit|--checkin|--ci)
	      # I'm not sure it is a good idea to propose this option...
	      do_commit=1
	      ;;
	  --remove|--rm)
	      # Idem
	      do_remove=1
	      ;;
	  -u|--url)
	      url=$1
	      shift
	      ;;
	  --url=*)
	      url=${arg/*=/}
	      ;;
	  -r|--revision)
	      revision=$1
	      shift
	      ;;
	  --revision=*)
	      revision=${arg/*=/}
	      ;;
	  -n|--dry-run)
	      nodo=:
	      ;;
	  -f|--force)
	      force=1
	      ;;
	  --)
	      break # manual end of options
	      ;;
	  # actions provided as options
	  --*)
	      [[ $action ]] && error 30 "got two actions: $action and $arg"
	      action=${arg/--/}
	      is_action $action || error 31 "unexpected option $arg"
	      ;;
	  # unexpected
	  *)
	      error 1 "unexpected option $arg"
	      ;;
      esac
  else
      if [[ ! $action ]] ; then
	  # first encountered bare word must be the action
	  action=$arg
	  shift
      else
	  break # end of options
      fi
  fi
done

# default is to show some help
[[ $action ]] || action='help'
is_action $action || error 31 "unexpected action $action"

verb 4 "action=$action args=$@"
verb 4 "svn=$svn tmp=$tmp url=$url nodo=$nodo"
verb 4 "do_commit=$do_commit do_remove=$do_remove"

test -d $tmp || error 10 "temporary directory $tmp not available"

################################################################ HANDLE ACTIONS

case $action in
    create)
	# get arguments
	[ $# -eq 2 -o $# -eq 3 ] || \
	    error 11 "create expect 2 or 3 arguments, got $# instead"
	src=$1 bch=$2 lwc= ; shift 2

	[ $# -eq 1 ] && {
	    # do_commit=1
	    lwc=$3
	    shift
	}

	# what are the urls
	src_url=$src
	bch_url=$bch

	# best effort to try to find the source and destination url
	if [[ $url ]] ; then
	    # fix source and destination target according to provided base url
	    ! is_svn_url $src && src_url=$url/$src
	    ! is_svn_url $bch && bch_url=$url/$bch
	else
	    # if it is a svn path, try its corresponding url
	    if ! is_svn_url $src && is_svn_wcpath $src ; then
	        # should not fail
		src_url=$(get_url $src)

		# do we also need the destination?
		if ! is_svn_url $bch ; then
		    is_svn_wcpath $bch && \
			error 12 "branch $bch is a valid wcpath"

		    bch_parent=$(dirname $bch)
		    [ $bch_parent = $bch ] && bch_parent='.'
		    
		    if is_svn_wcpath $bch_parent ; then
			bch_url=$(get_url $bch_parent)/$(basename $bch)
			lwc=$bch
		    else
		        # let us try to find the base of the repository
			url=$(get_repository_root $(dirname $src_url))
			bch_url=$url/$bch
		    fi
		fi
	    fi
	fi

	# various checks
	is_svn_url $src_url \
	    || error 12 "$src_url is not an svn url"

	src_url_depth=$(get_repository_depth $src_url)
	[ $src_url_depth -lt 1 -a ! "$force" ] && \
	    error 13 "invalid $src_url depth $src_url_depth"

	is_svn_url $bch_url || error 12 "$bch_url is not an svn url"

	bch_url_depth=$(get_repository_depth $bch_url)
	[ $bch_url_depth -lt 2 -a ! "$force" ] && \
	    error 13 "invalid $bch_url depth $bch_url_depth"

	# no subdir
	[[ $bch_url == $src_url* ]] && \
	    error 14 "$bch_url is a subdirectory of $src_url"
	
	# the new branch should not be within an existing branch

	verb 1 "source url: $src_url"
	verb 1 "branch url: $bch_url"
	verb 1 "local working copy: $lwc"

	# do the job
	create $src_url $bch_url

	[[ $do_commit && $lwc ]] && safe_svn checkout $bch_url $lwc
	;;
    pull)
	[ $# -eq 0 ] && set .
	for dir ; do
	    pull $dir
	done
	;;
    push)
	# does not provide current directory under the --remove option
	[[ $# -eq 0 && ! $do_remove ]] && set .
	number=1
	for dir ; do
	    push $dir $number
	    let number++
	done
	;;
    info)
	[ $# -eq 0 ] && set .
	for dir ; do
	    info $dir
	done
	;;
    diff)
	[ $# -eq 0 ] && set .
	for dir ; do
	    diff $dir
	done
	;;
    relocate)
	[ $# -eq 2 ] || usage 20 "expecting 2 arguments"
	new_src=$1 branch=$2 ; shift 2
	relocate $new_src $branch
	;;
    test)
	[ $# -eq 0 ] && set .
	[ $# -eq 1 ] || usage 21 "expecting 1 argument"
	dir=$1 ; shift
	is_svn_wcpath $dir || { verb 1 "$dir is not a wc" ; exit 1 ; }
	is_svn_branch $dir || { verb 1 "$dir is not a branch" ; exit 1 ; }
	verb 1 "$dir is a branch" ; exit 0
	;;
    avail)
	[ $# -eq 0 ] && set .
	[ $# -eq 1 ] || usage 21 "expecting 1 argument"
	for dir ; do
	    echo -n "$dir: " ; avail $dir
	done
	;;
    log)
	[ $# -eq 0 ] && set .
	[ $# -eq 1 ] || usage 21 "expecting 1 argument"
	dir=$1 ; shift
	log $dir
	;;
    help)
	usage 0
	;;
    version)
	echo "$command version is $cmd_rev"
	exit 0
	;;
    *)
	usage 1 "unexpected action '$action'"
	;;
esac

exit 0
