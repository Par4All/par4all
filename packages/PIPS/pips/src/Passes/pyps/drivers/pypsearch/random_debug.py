from __future__ import with_statement # to cope with python2.5
import random

"""Allows the user to save the calls to random to a file, and get the exact same back later"""

loaded_data = []
save_file = None

def start_saving(filename):
	global save_file
	finish_saving()
	save_file = open(filename, 'w')

def finish_saving():
	global save_file
	if save_file != None:
		save_file.close()
		save_file = None

def restore(filename):
	global loaded_data
	global save_file
	with open(filename, 'r') as f:
		for line in f:
			loaded_data.append(int(line))

def randint(low, up):
	global loaded_data
	global save_file
	if len(loaded_data) > 0:
		value = loaded_data[0]
		saved = value
		loaded_data = loaded_data[1:]
		if value < low or value > up:
			value = random.randint(low, up)
	else:
		value = random.randint(low, up)
		saved = value
	if save_file:
		save_file.write(str(saved) + '\n')
	return value
