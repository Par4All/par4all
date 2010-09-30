from pyps import *
with workspace(["detect_red06.c"]) as ws:
	m=ws.fun.main
	m.simd_atomizer()
	m.display()
	try: 
		while True:
			m.reduction_detection()
			m.display(With="PRINT_CODE_PROPER_REDUCTIONS")
	except:pass
	m.display()

