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

# JPips is a java interface to tpips. It is based on swing.

F.java	=	\
		ObservableStream.java \
		StreamObserver.java \
		Console.java \
		Pawt.java \
		JPipsComponent.java \
		Displayer.java \
		TextDisplayer.java \
		GraphDisplayer.java \
		EmacsDisplayer.java \
		Resetable.java \
		Activatable.java \
		Stateable.java \
		Requestable.java \
		Listener.java \
		Watcher.java \
		TPips.java \
		Option.java \
		Parser.java \
		OptionParser.java \
		DirectoryManager.java \
		ModuleManager.java \
		WorkspaceManager.java \
		JPips.java 

F.menu	=	jpips_test.menu \
		jpips_basic.menu \
		jpips_corinne.menu \
		jpips_developer.menu

F.gif	=	DownArrow.gif

# should not install anything if java is not available?
INSTALL_EXE = 	jpips

INSTALL_SHR =	$(F.gif) $(F.menu)

# common stuff
ROOT    = ../../..
PROJECT = pips
include $(ROOT)/makes/main.mk

# local stuff
ifdef _HAS_JDK_

pkg	= fr.ensmp.cri.jpips

JAVAC	= javac
JAR	= jar
#JAVADOC	= javadoc
#JDFLAGS	= -author -d doc

CLASS.d	= classes

$(CLASS.d):
	test -d $@ || $(MKDIR) $@

manifest.mf:
	echo "Main-Class: $(pkg).JPips" > $@

INSTALL_SHR += jpips.jar

# main.mk INSTALL_SHR dependency uses the initial value.
# the dependency is fixed here.
.build_shr: jpips.jar

jpips.jar: manifest.mf $(F.java) $(CLASS.d) 
	$(JAVAC) -classpath $(CLASS.d) -d $(CLASS.d) $(F.java)
	# should put everything in the jar file: menus, images...
	cd $(CLASS.d) && \
	$(JAR) cmf ../manifest.mf ../$@ $(subst .,/,$(pkg))/*.class

endif

clean: local-clean

local-clean:
	$(RM) -r $(CLASS.d)
	$(RM) jpips.jar manifest.mf tpips_session.* *.class
