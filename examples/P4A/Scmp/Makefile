# A GNU makefile to run a demo of Par4All on Jacobi application excerpt

TARGET= data_flow01

SOURCES= $(TARGET:=.c)


default:
	echo "This the content of the file README.txt:"
	# Use more and not less because when quitting, the displayed text
	# remains displayed...
	more README.txt

demo : $(TARGET)_scmp

%_scmp : %.c
	p4a --scmp -p $(TARGET) $<

clean :
	rm -rf *~ *.database ./applis ./applis_processing

