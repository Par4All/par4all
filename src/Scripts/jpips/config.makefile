#
# $Id$
#
# JPips is a java interface to tpips. It is based on swing.
#
# $Log: config.makefile,v $
# Revision 1.3  1998/07/02 15:02:01  coelho
# also cleans jar and zip files.
#
# Revision 1.2  1998/07/02 14:57:40  coelho
# jpips.jar entry added.
#
# Revision 1.1  1998/07/01 17:08:48  coelho
# Initial revision
#
#

jdk_dir	=	/usr/local/jdk1.2beta3
jdk_bin	=	$(jdk_dir)/bin
jdk_lib	=	$(jdk_dir)/lib
pkg_dir	= 	classes
doc_dir	=	doc

#
# global definitions
#

J_FILES	=	\
		ObservableStream.java \
		StreamObserver.java \
		Console.java \
		Pawt.java \
		JPipsComponent.java \
		Displayer.java \
		TextDisplayer.java \
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

MENUS	=	jpips_menus.test

GIF	=	DownArrow.gif

SCRIPTS	=	jpips

SOURCES	=	$(J_FILES) $(SCRIPTS) $(MENUS) $(GIF)

JZIP	=	jpips.zip
JJAR	=	jpips.jar

INSTALL_SHR =	$(MENUS) $(GIF) $(JZIP)

#
# compilation and related stuff

d	=	JAVA
$(d):;	mkdir $(d)
$(pkg_dir):; mkdir $(pkg_dir)

all: $(d) $(pkg_dir) $(JZIP)

OJ_FILES=	$(addprefix $(d)/, $(J_FILES:.java=))

# local stuff

package	=	JPips

swing	=	/usr/local/swing/swingall.jar
jdk	=	$(jdk_lib)/classes.zip

JENV	=	CLASSPATH=.:classes:$(swing):$(jdk)
JAVAC	=	$(jdk_bin)/javac
JCFLAGS	=	-g -deprecation -d $(pkg_dir)

JAVA	=	$(jdk_bin)/java
JFLAGS	=	-Djava.compiler=sunwjit

JAR	=	jar
JARFLAGS=	cvf

ZIP	=	zip
ZIPFLAGS=	-9 -v

JDOC	=	$(jdk_bin)/javadoc
JDFLAGS	=	-author -d $(doc_dir)

# default rule for compiling java files.
$(d)/%: %.java; $(JENV) $(JAVAC) $(JCFLAGS) $< && touch $@

.doc:
	$(JDOC) $(JDFLAGS) $(J_FILES) && touch $@

# jpips.jar: $(OJ_FILES)
# 	$(JENV) $(JAR) $(JARFLAGS) $@ $(pkg_dir)/$(package)/*.class

$(JZIP): $(OJ_FILES)
	pwd=`pwd` ; cd $(pkg_dir); \
		$(ZIP) $(ZIPFLAGS) $$pwd/$@ $(package)/*.class

# jar files are a little bigger than simple zip files.
$(JJAR): $(OJ_FILES)
	pwd=`pwd` ; cd $(pkg_dir); \
		$(JAR) $(JARFLAGS) $$pwd/$@ $(package)/*.class

zrun: $(JZIP)
	CLASSPATH=$(JZIP) $(JAVA) $(JFLAGS) $(package).JPips

jrun: $(JJAR)
	CLASSPATH=$(JJAR) $(JAVA) $(JFLAGS) $(package).JPips

clean:local-clean
local-clean:; $(RM) $(OJ_FILES) $(JJAR) $(JZIP); $(RM) -rf $(pkg_dir) $(d)

#
# java module dependencies (set manually)
#

$(d)/Activatable:
$(d)/Stateable:
$(d)/Resetable:
$(d)/Requestable:
$(d)/StreamObserver:

$(d)/ObservableStream: $(d)/StreamObserver
$(d)/Console: $(d)/ObservableStream $(d)/StreamObserver

$(d)/Pawt: $(d)/Activatable
$(d)/JPipsComponent: $(d)/Activatable $(d)/Pawt $(d)/Stateable
$(d)/Displayer: $(d)/JPipsComponent
$(d)/TextDisplayer: $(d)/Pawt $(d)/Displayer 

$(d)/Listener: $(d)/Requestable
$(d)/Watcher: $(d)/Requestable
$(d)/TPips: $(d)/Listener $(d)/Watcher $(d)/Stateable $(d)/Requestable

$(d)/Option: $(d)/TPips $(d)/Stateable
$(d)/Parser: $(d)/Option
$(d)/OptionParser: $(d)/Parser $(d)/Option

$(d)/DirectoryManager: $(d)/TPips $(d)/Pawt $(d)/JPipsComponent
$(d)/ModuleManager: $(d)/TPips $(d)/Pawt $(d)/JPipsComponent
$(d)/WorkspaceManager: $(d)/TPips $(d)/Pawt $(d)/JPipsComponent

$(d)/JPips: $(d)/DirectoryManager $(d)/ModuleManager $(d)/WorkspaceManager \
		$(d)/TPips $(d)/TextDisplayer
