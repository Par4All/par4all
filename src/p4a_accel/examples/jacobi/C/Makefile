# Add source files here
TARGET = jacobi 
CFILES = jacobi.c
CFLAGS = -std=c99

$(TARGET): $(CFILES) 
	$(CC) $(CFLAGS) -o $@ $(CFILES)  $(LDLIBS)  

run: $(TARGET)
	time ./$< ../Logo_HPC-Project-512x247-crop.pgm 

display:
	gimp output.pgm

clean:
	rm $(TARGET) output.pgm
