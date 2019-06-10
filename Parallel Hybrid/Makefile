LD = mpicc
tasks = convolution

all:	${tasks}.out

${tasks}.out: ${tasks}.c
	${LD} ${tasks}.c -fopenmp -o ${tasks}

execute10:
	mpiexec -np 2 ./${tasks} ../images/im10.ppm ../kernel/kernel3x3_Edge.txt ../results/im10.ppm 16

execute5:
	mpiexec -np 2 ./${tasks} ../images/im05.ppm ../kernel/kernel3x3_Edge.txt ../results/im05.ppm 8

execute3:
	mpiexec -np 2 ./${tasks} ../images/im03.ppm ../kernel/kernel3x3_Edge.txt ../results/im03.ppm 1

clean:
	rm ${tasks}
