# Add source files here
TARGET = hyantes
CFILES = hyantes.c
CFLAGS = -std=c99
LDLIBS = -lm

$(TARGET): $(CFILES) 
	$(CC) $(CFLAGS) -o $@ $(CFILES)  $(LDLIBS)  

run: $(TARGET)
	time ./$(TARGET) ../Rhone-alpesXYLongLat_pop.txt 1.1 32 0.01 40 > h.out

display:
	gnuplot -persist -e "set pm3d map; set size ratio 1; splot 'h.out'"

clean:
	rm $(TARGET) h.out
