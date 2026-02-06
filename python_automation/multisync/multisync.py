#!/usr/bin/python3
import os, sys, re
import multiprocessing as mp

def _sync_file(str0):
	os.system("rsync -zrvh "+str0)
def _sync_dir(str0):
	os.system("rsync -zavh "+str0)

if __name__ == '__main__':
	src0 = sys.argv[1]
	dest0 = sys.argv[2]
	src = os.path.abspath(src0)
	dest = os.path.abspath(dest0)
	mp.set_start_method('spawn', force=bool)
	q = mp.Queue()
	for root, subdirs, files in os.walk(src):
		for subdir in subdirs:
			str0 = root+'/'+subdir+' '+dest+'/'+subdir
			print(str0)
			p = mp.Process(target=_sync_dir, args=(str0,))
			p.start()
			p.join()
		for filename in files:
			str0 = root+'/'+filename+' '+dest+'/'+filename
			p = mp.Process(target=_sync_file, args=(str0,))
			p.start()
			p.join()

