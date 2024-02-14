CFLAGS = -Wall -pedantic -O2 -g3
CPPFLAGS = -D CL_TARGET_OPENCL_VERSION=100
LDLIBS = -lOpenCL

all: mandel.png

%.png: %.ppm
	convert $^ $@

mandel.ppm: mandel mandel.cl
	./mandel

.PHONY: clean
clean:
	$(RM) mandel mandel.ppm mandel.png
