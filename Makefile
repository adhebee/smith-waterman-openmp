CC       = gcc
CFLAGS   = -O3 -march=native -Wall -funroll-loops
VECFLAGS = -fopt-info-vec-optimized
OMPFLAGS = -fopenmp

all: sw_serial sw_parallel


sw_serial: sw_serial.c
	$(CC) $(CFLAGS) -o sw_serial sw_serial.c

sw_parallel: sw_parallel.c
	$(CC) $(CFLAGS) $(OMPFLAGS) -o sw_parallel sw_parallel.c

vecreport: sw_parallel.c
	$(CC) $(CFLAGS) $(OMPFLAGS) $(VECFLAGS) -o sw_parallel sw_parallel.c 2> vec_report.txt
	@echo "Vectorization report written to vec_report.txt"

clean:
	rm -f sw_serial sw_parallel *.o vec_report.txt
