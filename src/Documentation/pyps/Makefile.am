
pyps_doc.tex: pyps_doc.nw
	$(NOWEAVE) -delay $< > $@

dist_noinst_DATA=pyps_doc.nw
doc_DATA=pyps_doc.pdf

CLEANFILES=$(doc_DATA) pyps_doc.tex
include $(top_srcdir)/makes/latex.mk

