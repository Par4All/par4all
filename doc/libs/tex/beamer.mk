# OK, not really usable except by Ronan Keryell right now... :-)

ifndef FORMATION_ROOT
$(error FORMATION_ROOT environment variable should contain the formation git working copy)
endif

# point to where our local TeX stuff is installed:
TEX_ROOT=$(FORMATION_ROOT)/TeX

# Add this to the TeX path:
INSERT_TEXINPUTS=::$(TEX_ROOT)//:$(TEX_ROOT)/../Images//
#APPEND_TEXINPUTS=$(TEX_ROOT)//:$(TEX_ROOT)/../Images//
include $(TEX_ROOT)/Makefiles/beamer.mk
