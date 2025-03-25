CC = nvcc
CXXFLAGS = -code=sm_89 -o
%.out : %.cu
	$(CC) $(CXXFLAGS) $@ $<
	./$@

clean :
	rm -rf %.out
