#! /usr/bin/env python3.1

# -*- coding: utf-8 -*-


from p4a_validate import *

p4a = ValidationClass().from_file('/home/keryell/projets/Wild_Systems/Par4All/par4all/packages/PIPS/validation/RESULTS/SUMMARY')

svn = ValidationClass().from_file('/home/keryell/projets/Wild_Systems/PIPS/git-svn-work/validation/RESULTS/SUMMARY')

cri = ValidationClass().from_file('/home/keryell/projets/Wild_Systems/Par4All/par4all/src/validation/valid.cri')
d = svn - cri
d = p4a - svn

#print(cri)
print (d)
d.show_diff_files()

#print(p.elements)
