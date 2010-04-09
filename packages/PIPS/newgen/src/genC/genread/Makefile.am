noinst_LTLIBRARIES=libgenread.la
libgenread_la_SOURCES=genread.y genread_lex.l
AM_YFLAGS=-d -p genread_
AM_LFLAGS=-p genread_
AM_CPPFLAGS=-I$(srcdir)/.. -I..

# additionnal dep needed for // build
genread_lex.c:genread.h
