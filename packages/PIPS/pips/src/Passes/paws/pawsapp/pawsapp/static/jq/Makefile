F.gif	= $(wildcard *.gif)
F.jpg	= $(wildcard *.jpg)
F.png	= $(wildcard *.png)

F.js	= $(wildcard *.js)
F.css	= $(wildcard *.css)

F.src-js	= $(filter-out %-min.js %.min.js,$(F.js)) 
F.src-css	= $(filter-out %-min.css %.min.css,$(F.css)) 

F.min	= $(F.src-js:%.js=%-min.js) $(F.src-css:%.css=%-min.css) 

#F.meta	= $(F.gif:%=%.metadata) $(F.jpg:%=%.metadata) $(F.png:%=%.metadata) $(F.js:%=%.metadata) $(F.css:%=%.metadata)

F.static = $(F.gif) $(F.jpg) $(F.png) $(F.js) $(F.css)

F.meta	= $(F.static:%=%.metadata)

min: $(F.min)

meta: $(F.meta)

all: min meta

test:
	@echo "files: " $(F.src-js) $(F.src-css)
	@echo "generated: " $(F.min)

test-meta:
	@echo "static: " $(F.static)
	@echo "metadata: " $(F.meta)

%-min.js: %.js
	yui-compressor $< -o $@


%-min.css: %.css
	yui-compressor $< -o $@

%.metadata: %
	cp __.metadata $@

clean:
	$(RM) $(F.min) *~
