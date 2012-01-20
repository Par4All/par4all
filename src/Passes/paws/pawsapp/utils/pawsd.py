import os, time, shutil
from datetime import datetime

#path1 = "/home/local/szymczak/MYPIPS/pips_dev/paas/pawsapp/pawsapp/public/res"
#path2 = "/home/local/szymczak/MYPIPS/pips_dev/paas/pawsapp/files"

def remove_files(path, sec):
    for f in os.listdir(path):
		if sec - os.stat(path + '/' + f).st_mtime > 7200:
			if os.path.isdir(path + '/' + f):
				shutil.rmtree(path + '/' + f)
			else:
				os.remove(path + '/' + f)


seconds = time.time()
remove_files(path1, seconds)
remove_files(path2, seconds)
