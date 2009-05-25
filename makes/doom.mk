#	$Id$

# Enable NewGen DOoM model to have XML backend and XPath functionalities.

ifdef USE_NEWGEN_DOOM

NEWGEN_DOOM_INCLUDES=$(shell pkg-config --cflags glib-2.0 libxml-2.0)
NEWGEN_DOOM_LIBS=$(shell pkg-config --libs glib-2.0 libxml-2.0)

else

NEWGEN_DOOM_INCLUDES=
NEWGEN_DOOM_LIBS=

endif
