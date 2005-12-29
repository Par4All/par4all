#! /bin/bash
#
# $Id: svn_branch.sh 341 2005-12-29 15:30:49Z coelho $
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
# and are to be joined back to the trunk when the development is over.
# So mergings are to be managed in both directions. Also, less commands
# are required wrt "svnmerge".
#
# trunk/____________*__+_____*__+_____X__
#   |               |        |        A
#   | initial copy  | some merges...  | join (merge back)
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
#  - add svnmerge revision list management.
#  - option for giving a source wc path to join
#  - test/validate scenario
#

# keep command name
command=${0/*\//}

# keep revision number
cmd_rev='$Rev: 341 $'
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
  * merge [options] wc_path...
    . merge into the branches from their sources.
  * join [options] wc_path...
    . merge back from the branches into their sources.
    --remove: automatically remove branch after join.
  * relocate source_url branch
    . change location of branch source.

  Other common options:
    --verbose: be more verbose (show issued svn commands)
    --debug: be really more verbose (you don't want that)
    --quiet: be quiet (not advisable)
    --tmp tmp: directory for temporary files or directories, default is /tmp
    --commit: try to perform all commits (not really advisable)
    --revision rev: revision to consider (for create, merge, join...)
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

# is_svn_working_url tested-url
# returns whether the url exists and is working
function is_svn_working_url()
{
    is_svn_url $1 && svn proplist $1 > /dev/null 2>&1
}

function is_svn_wcpath()
{
    test -d $1 -a -d $1/.svn
}

# perform a svn command but quit on error.
function safe_svn()
{
    verb 2 $svn "$@"
    $nodo $svn "$@" || error 2 "svn failed: $svn $@"
}

# same as above, but to it anyway even under dry-run
function do_svn()
{
    verb 2 $svn "$@"
    $svn "$@" || error 2 "svn failed: $svn $@"
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
	safe_svn propset $pname $pval $path
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
    local save_mrg_rev=$(safe_svn propget 'svnbranch:merged-rev' $dir)
    local save_joi_rev=$(safe_svn propget 'svnbranch:joined-rev' $dir)

    # apply differences
    # obscure issue with 1.2.3: svn: Move failed...
    safe_svn merge --revision $revs $src_url $dir

    # restore saved svnbranch:* props
    reset_property 'svnbranch:version'    "$save_src_ver" $dir
    reset_property 'svnbranch:source-url' "$save_src_url" $dir
    reset_property 'svnbranch:source-rev' "$save_src_rev" $dir
    reset_property 'svnbranch:merged-rev' "$save_mrg_rev" $dir
    reset_property 'svnbranch:joined-rev' "$save_joi_rev" $dir

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
    safe_svn propset 'svnbranch:version'    1        $tmp_co/$branch
    safe_svn propset 'svnbranch:source-url' $src     $tmp_co/$branch
    safe_svn propset 'svnbranch:source-rev' $src_rev $tmp_co/$branch
    safe_svn propset 'svnbranch:merged-rev' $src_rev $tmp_co/$branch
    # the revision of the next commit would be better.
    safe_svn propset 'svnbranch:joined-rev' $src_rev $tmp_co/$branch

    safe_svn_commit 'create new branch' $tmp_co	\
	$nodo rm -rf $tmp_co 
}

# merge wc_path
# globals: 
function merge()
{
    local dir=$1 ; shift
    test -d $dir || error 7 "no such directory $dir"

    local version=$(do_svn propget 'svnbranch:version' $dir)
    [ "$version" ] || \
	error 8 "no svnbranch property in $dir, not a managed branch"

    local src_url=$(do_svn propget 'svnbranch:source-url' $dir)
    local merged_rev=$(do_svn propget 'svnbranch:merged-rev' $dir)

    local src_rev=$revision
    [[ ! $revision ]] && src_rev=$(get_last_revision $src_url)

    if [ $src_rev -gt $merged_rev ] ; then

	perform_merge $merged_rev:$src_rev $dir $(get_url $dir) $src_url 

	# update merged status with respect to merge above.
	safe_svn propset 'svnbranch:merged-rev' $src_rev $dir

	# ??? this seems necessary for some obscure reasons.
	safe_svn update $dir

	local br_url=$(get_url $dir)
	local message=$(
	    echo "merge revisions $merged_rev:$src_rev"
	    echo "from $src_url"
	    echo "to branch $br_url"
	)
	safe_svn_commit "$message" $dir
    else
	verb 1 "nothing to merge into $dir"
    fi
}

# -- merge back developments into source url
# merge wc_path subname
# globals: revision do_commit
function join()
{
    local dir=$1 sub=$2 ; shift 2
    test -d $dir || error 9 "no such directory $dir" 

    local version=$(do_svn propget 'svnbranch:version' $dir )
    [ "$version" ] || error 10 "$dir is not a managed branch"

    all_is_committed $dir || error 11 "$dir is not committed, cannot join"

    # update needed so that last revision is ok.
    safe_svn update $dir
    local src_url=$(do_svn propget 'svnbranch:source-url' $dir)
    local joined_rev=$(do_svn propget 'svnbranch:joined-rev' $dir)
    local current_rev=$revision
    [[ ! $revision ]] && current_rev=$(get_last_revision $dir)

    #TODO: should check that all merged are already done? or not??

    if [ $current_rev -gt $joined_rev ] ; then

	local tmp_co=$tmp/$command.$$.$sub
	test -d $tmp_co && error 5 "temporary directory $tmp_co already exists"
	local branch_url=$(get_url $dir)

	perform_merge $joined_rev:$current_rev $tmp_co $src_url $branch_url

	if [[ $do_remove ]] ; then
	    # ???
	    safe_svn remove $dir
	else
	    # fix svnbranch joined-rev status in $dir
	    safe_svn propset 'svnbranch:joined-rev' $current_rev $dir
	fi

	verb 1 "status of $dir"
	safe_svn status --ignore-externals $dir

	local message=$(
	    echo "join revisions $joined_rev:$current_rev"
	    echo "from branch $branch_url"
	    echo "to $src_url"
	)

	safe_svn_commit "$message" $tmp_co \
	    $nodo rm -rf $tmp_co

        # remove the current directory while being inside it is no good
	[[ $do_remove && $dir = '.' ]] && \
	    verb 1 "your shell will block the cleanup of $dir"
	
	safe_svn_commit "$message" $dir
    else
	verb 1 "nothing to join into $src_url from $dir"
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
	echo "Branch Source Rev:" \
	    $(do_svn $rev propget 'svnbranch:source-rev' $dir)
	echo "Branch Last Merged Rev:" \
	    $(do_svn $rev propget 'svnbranch:merged-rev' $dir)
	echo "Branch Last Joined Rev:" \
	    $(do_svn $rev propget 'svnbranch:joined-rev' $dir)
	echo "Branch Current Source Rev:" $(get_last_revision $src_url)
    else
	echo "Branch Management: none"
    fi
}

# diff wc_path
function diff()
{
    local dir=$1 ; shift
    local version=$(do_svn propget 'svnbranch:version' $dir)
    if [[ $version ]] ; then
	local src_url=$(do_svn propget 'svnbranch:source-url' $dir)
	local dir_url=$(get_url $dir)
	safe_svn diff $src_url $dir_url
    else
	echo "Path $dir not under branch management."
    fi
}

# relocate new-source-url branch
function relocate()
{
    local new_src=$1 branch=$2 ; shift 2
    local version=$(do_svn propget 'svnbranch:version' $branch)
    [[ $version ]] || error 21 "$branch is not under branch management"

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

#################################################################### GET ACTION

[ $# -eq 0 ] && usage 1
action=$1
shift

################################################################# PARSE OPTIONS

do_commit=
do_remove=
url=
nodo=
revision=

while true ; do
  opt=$1
  if [[ $opt == -* ]] ; then # is it an option?
      shift 
      verb 4 "handling option $opt"
      case $opt in
	  -d|--debug)
	      let verbosity+=3
	      ;;
	  -v|--verbose) 
	      let verbosity++
	      ;;
	  -q|--no-verbose|--quiet)
	      verbosity=0
	      ;;
	  -h|--help)
	      usage 0
	      ;;
	  -s|--svn)
	      svn=$1
	      shift
	      ;;
	  --svn=*)
	      svn=${opt/*=/}
	      ;;
	  -t|--tmp|--temporary)
	      tmp=$1
	      shift
	      ;;
	  --tmp=*|--temporary=*)
	      tmp=${opt/*=/}
	      ;;
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
	      url=${opt/*=/}
	      ;;
	  -r|--revision)
	      revision=$1
	      shift
	      ;;
	  --revision=*)
	      revision=${opt/*=/}
	      ;;
	  --dry-run)
	      nodo=:
	      ;;
	  --force)
	      force=1
	      ;;
	  --)
	      break # manual end of options
	      ;;
	  *)
	      error 1 "unexpected option $opt"
	      ;;
      esac
  else
      break # end of options
  fi
done

verb 4 "remaining arguments: $@"
verb 4 "svn=$svn tmp=$tmp url=$url nodo=$nodo"
verb 4 "do_commit=$do_commit do_remove=$do_remove"

test -d $tmp || error 10 "temporary directory $tmp not available"

################################################################ HANDLE ACTIONS

case $action in
    create|--create)
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

	# do the job
	create $src_url $bch_url

	[[ $do_commit ]] && safe_svn checkout $bch_url $lwc
	;;
    merge|--merge)
	[ $# -eq 0 ] && set .
	for dir ; do
	    merge $dir
	done
	;;
    join|--join)
	# does not provide current directory under the --remove option
	[[ $# -eq 0 && ! $do_remove ]] && set .
	number=1
	for dir ; do
	    join $dir $number
	    let number++
	done
	;;
    info|--info)
	[ $# -eq 0 ] && set .
	for dir ; do
	    info $dir
	done
	;;
    diff|--diff)
	[ $# -eq 0 ] && set .
	for dir ; do
	    diff $dir
	done
	;;
    relocate|--relocate)
	[ $# -eq 2 ] || usage 20 "expecting 2 arguments"
	new_src=$1 branch=$2 ; shift 2
	relocate $new_src $branch
	;;
    help|--help)
	usage 0
	;;
    version|--version)
	echo "$command version is $cmd_rev"
	;;
    *)
	usage 1 "unexpected action '$action'"
	;;
esac
