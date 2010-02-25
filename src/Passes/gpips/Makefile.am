bin_PROGRAMS=gpips
BUILT_SOURCES=gpips.h
gpips_SOURCES=	directory_menu.c \
		gpips.c \
		gtk_compile.c \
		gtk_edit2.c \
		gtk_help.c \
		gtk_log.c \
		gtk_mchoose.c \
		gtk_props.c \
		gtk_query.c \
		gtk_quit.c \
		gtk_schoose2.c \
		gtk_select.c \
		gtk_status.c \
		gtk_transform.c \
		gtk_utils.c \
		gtk_windows.c \
		vararg.c\
		main_gpips.c
gpips_LDADD=../../Libs/libpipslibs.la
gpips_LDFLAGS=$(GTKp_2_0_LIBS)

AM_CPPFLAGS=\
	$(GTKp_2_0_CFLAGS)\
	$(NEWGENLIBS_CFLAGS) $(LINEARLIBS_CFLAGS)\
	-I$(top_srcdir)/src/Documentation/constants\
	-I../../Documentation/pipsmake\
	-I../../Documentation/newgen\
	-I../../Libs/newgen\
	-I../../Libs/misc\
	-I../../Libs/ri-util\
	-I../../Libs/pipsmake\
	-I../../Libs/top-level\
	-I../../Libs/properties\
	-I../../Libs/pipsdbm


TARGET	= gpips
nodist_noinst_HEADERS=$(TARGET).h
dist_noinst_HEADERS=gpips_view_menu_layout.h gtk_sizes.h gpips-labels.h
dist_sysconf_DATA=pips.icon
include $(top_srcdir)/makes/cproto.mk
