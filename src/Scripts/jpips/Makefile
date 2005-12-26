#
# $Id$
#
# JPips is a java interface to tpips. It is based on swing.
#

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

# do not install if java is not available?
INSTALL_EXE = 	jpips

INSTALL_SHR =	jpips.jar $(F.gif) $(F.menu)

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

PROJECT = pips
include $(ROOT)/makes/main.mk

# local stuff
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

ifdef _HAS_JAVA_

jpips.jar: $(CLASS.d) manifest.mf
	$(JAVAC) -classpath $(CLASS.d) -d $(CLASS.d) $(F.java)
	# should put everything in the jar file: menus, images...
	cd $(CLASS.d) && \
	$(JAR) cmf ../manifest.mf ../$@ $(subst .,/,$(pkg))/*.class

else

jpips.jar:
	touch $@

endif

clean: local-clean

local-clean:
	$(RM) -r $(CLASS.d)
	$(RM) jpips.jar manifest.mf
