#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors:
# - Grégoire Péan <gregoire.pean@hpc-project.com>
#
import p4a_util
import os

'''
Par4All Environment .sh/.csh Files Writer
'''



actual_script = p4a_util.change_file_ext(os.path.realpath(os.path.abspath(__file__)), ".py", if_ext = ".pyc")
script_dir = os.path.split(actual_script)[0]

rc_sh_template_file = os.path.join(script_dir, "par4all-rc.sh.tpl")
rc_csh_template_file = os.path.join(script_dir, "par4all-rc.csh.tpl")

def p4a_write_rc(dir, subs_map):
    global rc_sh_template_file, rc_csh_template_file
    p4a_util.subs_template_file(rc_sh_template_file, subs_map, dir)
    p4a_util.subs_template_file(rc_csh_template_file, subs_map, dir)

if __name__ == "__main__":
    print(__doc__)
    print("This module is not directly executable")

# Some Emacs stuff:
### Local Variables:
### mode: python
### mode: flyspell
### ispell-local-dictionary: "american"
### tab-width: 4
### End:
