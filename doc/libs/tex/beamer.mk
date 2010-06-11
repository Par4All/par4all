# OK, not really usable except by Ronan Keryell right now... :-)

# point to where our local TeX stuff is installed:
TEX_ROOT=/home/keryell/projets/Wild_Systems/svn/doc/TeX

# Add this to the TeX path:
INSERT_TEXINPUTS=::$(TEX_ROOT)//:$(TEX_ROOT)/../Images//
#APPEND_TEXINPUTS=$(TEX_ROOT)//:$(TEX_ROOT)/../Images//
include $(TEX_ROOT)/Makefiles/beamer.mk
