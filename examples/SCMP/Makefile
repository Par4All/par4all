# A GNU makefile to run a demo of Par4All on Jacobi application excerpt

LOCAL_TARGET= data_flow01

SOURCES= $(LOCAL_TARGET:=.c)


default:
	echo "This the content of the file README.txt:"
	# Use more and not less because when quitting, the displayed text
	# remains displayed...
	more README.txt

demo : $(LOCAL_TARGET)_scmp

%_scmp : %.c
	p4a --scmp -p $* $<

clean :
	rm -rf *~ *.database ./applis ./applis_processing

