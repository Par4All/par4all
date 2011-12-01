F_c =	ri.c \
        effects.c \
	dg.c \
	compsec.c \
	word_attachment.c \
	reductions_private.c \
	interval_graph.c \
	pipsdbm_private.c \
	eole_private.c \
	abc_private.c \
	alias_private.c \
	ubs_private.c \
	c_parser_private.c \
	complexity_ri.c \
	database.c \
	paf_ri.c \
	tiling.c \
	graph.c \
	parser_private.c \
	hpf_private.c \
	property.c \
	makefile.c \
	reduction.c \
	message.c \
	text.c \
	hpf.c \
	sac_private.c \
	ri_C.c \
	cloning.c \
	step_private.c \
	points_to_private.c \
	kernel_memory_mapping.c \
	freia_spoc_private.c \
	scalopes_private.c

F_newgen	= $(F_c:%.c=%.newgen)
F_pdf		= $(F_c:%.c=%.pdf)
F_h			= $(F_c:%.c=%.h)
F_tex		= $(F_c:%.c=%.tex)
F_spec		= $(F_c:%.c=%.spec)
