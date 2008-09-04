# $Id$
#
# subversion related targets
#

SVN =		svn
SVN_USERNAME =
SVN_FLAGS =

IS_SVN_WC =	test -d .svn

BRANCH = 	$(MAKE.d)/svn_branch.sh
BRANCH_FLAGS =
IS_BRANCH =	$(BRANCH) test --quiet

# fix command flags if username is provided
ifdef SVN_USERNAME
SVN_FLAGS	+= --username $(SVN_USERNAME)
BRANCH_FLAGS 	+= --username $(SVN_USERNAME)
endif

diff:
	-@$(IS_SVN_WC) && $(SVN) $(SVN_FLAGS) $@

status:
	-@$(IS_SVN_WC) && $(SVN) $(SVN_FLAGS) $@

info:
	-@$(IS_SVN_WC) && $(SVN) $(SVN_FLAGS) $@

commit:
	-@$(IS_SVN_WC) && $(SVN) $(SVN_FLAGS) $@
