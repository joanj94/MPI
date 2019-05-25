//
//  convolution.c
//
//
//  Created by Josep Lluis Lerida on 11/03/15.
//
// This program calculates the convolution for PPM images.
// The program accepts an PPM image file, a text definition of the kernel matrix and the PPM file for storing the convolution results.
// The program allows to define image partitions for processing large images (>500MB)
// The 2D image is represented by 1D vector for chanel R, G and B. The convolution is applied to each chanel separately.

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <mpi.h>

// Structure to store image.
struct imagenppm{
    int altura;
    int ancho;
    char *comentario;
    int maxcolor;
    int P;
    int *R;
    int *G;
    int *B;
};
typedef struct imagenppm* ImagenData;

// Structure to store the kernel.
struct structkernel{
    int kernelX;
    int kernelY;
    float *vkern;
};
typedef struct structkernel* kernelData;

//Functions Definition
ImagenData initimage(char* nombre, FILE **fp, int partitions, int halo);
ImagenData duplicateImageData(ImagenData src, int partitions, int halo);

int readImage(ImagenData Img, FILE **fp, int dim, int halosize, long int *position);
int duplicateImageChunk(ImagenData src, ImagenData dst, int dim);
int initfilestore(ImagenData img, FILE **fp, char* nombre, long *position);
int savingChunk(ImagenData img, FILE **fp, int dim, int offset);
int convolve2D(int* inbuf, int* outbuf, int sizeX, int sizeY, float* kernel, int ksizeX, int ksizeY);
void freeImagestructure(ImagenData *src);

//MPI global vars
int rank, size;
int partitions, dataSizeXMPI, dataSizeYMPI, chunkMPI;
MPI_Op convolve2DOp;
kernelData kern=NULL;
ImagenData source=NULL, output=NULL;

//Open Image file and image struct initialization
ImagenData initimage(char* nombre, FILE **fp,int partitions, int halo){
    char c;
    char comentario[300];
    int i=0,chunk=0;
    ImagenData img=NULL;
    
    /*Opening ppm*/

    if ((*fp=fopen(nombre,"r"))==NULL){
        perror("Error: ");
    }
    else{
        //Memory allocation
        img=(ImagenData) malloc(sizeof(struct imagenppm));

        //Reading the first line: Magical Number "P3"
        fscanf(*fp,"%c%d ",&c,&(img->P));
        
        //Reading the image comment
        while((c=fgetc(*fp))!= '\n'){comentario[i]=c;i++;}
        comentario[i]='\0';
        //Allocating information for the image comment
        img->comentario = calloc(strlen(comentario),sizeof(char));
        strcpy(img->comentario,comentario);
        //Reading image dimensions and color resolution
        fscanf(*fp,"%d %d %d",&img->ancho,&img->altura,&img->maxcolor);
        chunk = img->ancho*img->altura / partitions;
        //We need to read an extra row.
        chunk = chunk + img->ancho * halo;
        if ((img->R=calloc(chunk,sizeof(int))) == NULL) {return NULL;}
        if ((img->G=calloc(chunk,sizeof(int))) == NULL) {return NULL;}
        if ((img->B=calloc(chunk,sizeof(int))) == NULL) {return NULL;}
    }
    return img;
}

//Duplicate the Image struct for the resulting image
ImagenData duplicateImageData(ImagenData src, int partitions, int halo){
    char c;
    char comentario[300];
    unsigned int imageX, imageY;
    int i=0, chunk=0;
    //Struct memory allocation
    ImagenData dst=(ImagenData) malloc(sizeof(struct imagenppm));

    //Copying the magic number
    dst->P=src->P;
    //Copying the string comment
    dst->comentario = calloc(strlen(src->comentario),sizeof(char));
    strcpy(dst->comentario,src->comentario);
    //Copying image dimensions and color resolution
    dst->ancho=src->ancho;
    dst->altura=src->altura;
    dst->maxcolor=src->maxcolor;
    chunk = dst->ancho*dst->altura / partitions;
    //We need to read an extra row.
    chunk = chunk + src->ancho * halo;
    if ((dst->R=calloc(chunk,sizeof(int))) == NULL) {return NULL;}
    if ((dst->G=calloc(chunk,sizeof(int))) == NULL) {return NULL;}
    if ((dst->B=calloc(chunk,sizeof(int))) == NULL) {return NULL;}
    chunkMPI = chunk; //Guardar el valor per enviar-lo després
    return dst;
}

//Read the corresponding chunk from the source Image
int readImage(ImagenData img, FILE **fp, int dim, int halosize, long *position){
    int i=0, k=0,haloposition=0;
    if (fseek(*fp,*position,SEEK_SET))
        perror("Error: ");
    haloposition = dim-(img->ancho*halosize*2);
    for(i=0;i<dim;i++) {
        // When start reading the halo store the position in the image file
        if (halosize != 0 && i == haloposition) *position=ftell(*fp);
        fscanf(*fp,"%d %d %d ",&img->R[i],&img->G[i],&img->B[i]);
        k++;
    };
    return 0;
}

//Duplication of the  just readed source chunk to the destiny image struct chunk
int duplicateImageChunk(ImagenData src, ImagenData dst, int dim){
    int i=0;
    
    for(i=0;i<dim;i++){
        dst->R[i] = src->R[i];
        dst->G[i] = src->G[i];
        dst->B[i] = src->B[i];
    }
    return 0;
}

// Open kernel file and reading kernel matrix. The kernel matrix 2D is stored in 1D format.
kernelData leerKernel(char* nombre){
    FILE *fp;
    int i=0;
    kernelData kern=NULL;
    
    /*Opening the kernel file*/
    fp=fopen(nombre,"r");
    if(!fp){
        perror("Error: ");
    }
    else{
        //Memory allocation
        kern=(kernelData) malloc(sizeof(struct structkernel));
        
        //Reading kernel matrix dimensions
        fscanf(fp,"%d,%d,", &kern->kernelX, &kern->kernelY);
        kern->vkern = (float *)malloc(kern->kernelX*kern->kernelY*sizeof(float));
        
        // Reading kernel matrix values
        for (i=0;i<(kern->kernelX*kern->kernelY)-1;i++){
            fscanf(fp,"%f,",&kern->vkern[i]);
        }
        fscanf(fp,"%f",&kern->vkern[i]);
        fclose(fp);
    }
    return kern;
}

// Open the image file with the convolution results
int initfilestore(ImagenData img, FILE **fp, char* nombre, long *position){
    /*Se crea el fichero con la imagen resultante*/
    if ( (*fp=fopen(nombre,"w")) == NULL ){
        perror("Error: ");
        return -1;
    }
    /*Writing Image Header*/
    fprintf(*fp,"P%d\n%s\n%d %d\n%d\n",img->P,img->comentario,img->ancho,img->altura,img->maxcolor);
    *position = ftell(*fp);
    return 0;
}

// Writing the image partition to the resulting file. dim is the exact size to write. offset is the displacement for avoid halos.
int savingChunk(ImagenData img, FILE **fp, int dim, int offset){
    int i,k=0;
    //Writing image partition
    for(i=offset;i<dim+offset;i++){
        fprintf(*fp,"%d %d %d ",img->R[i],img->G[i],img->B[i]);
        k++;
    }
    return 0;
}

// This function free the space allocated for the image structure.
void freeImagestructure(ImagenData *src){
    
    free((*src)->comentario);
    free((*src)->R);
    free((*src)->G);
    free((*src)->B);
    
    free(*src);
}

///////////////////////////////////////////////////////////////////////////////
// 2D convolution
// 2D data are usually stored in computer memory as contiguous 1D array.
// So, we are using 1D array for 2D data.
// 2D convolution assumes the kernel is center originated, which means, if
// kernel size 3 then, k[-1], k[0], k[1]. The middle of index is always 0.
// The following programming logics are somewhat complicated because of using
// pointer indexing in order to minimize the number of multiplications.
//
//
// signed integer (32bit) version:
///////////////////////////////////////////////////////////////////////////////

/*
- MPI_Reduce usant l'array in, guardant en out, kernel compartit. https://www.mpi-forum.org/docs/mpi-1.1/mpi-11-html/node80.html
Si no podem usar reduce creant una funció custom d'aplicar el kernel fer el següent:

- MPI_Bcast per enviar a tothom el kernel
- MPI_Scatter per dividir l'array in en porcions iguals  (en ordre)
- En comptes d'usar els punters in/out usarem un puinter privat per cada fil que serà del buffer rebut abans en el Scatter
- MPI_Gatther per rebre l'array dividit i guardar-lo en out (es retornen ordenats)
*/

int convolve2DMPI(int* in, int* out, int len){
	int kCenterX, kCenterY;
    
    kCenterX = (int)kern->kernelX / 2;
    kCenterY = (int)kern->kernelY / 2;

    int i, j, m, n;
    float *kPtr;
    int *inPtr, *inPtr2, *outPtr;
    int rowMin, rowMax;                             // to check boundary of input array
    int colMin, colMax;                             //
    float sum;                                      // temp accumulation buffer

    kPtr = kern->vkern;
    inPtr = inPtr2 = &in[dataSizeXMPI * kCenterY + kCenterX];  // note that  it is shifted (kCenterX, kCenterY),
    outPtr = out;

    for(i= 0; i < len; ++i)                   // number of rows
    {
        // compute the range of convolution, the current row of kernel should be between these
        rowMax = i + kCenterY;
        rowMin = i - dataSizeYMPI + kCenterY;
        
        for(j = 0; j < dataSizeXMPI; ++j)              // number of columns
        {
            // compute the range of convolution, the current column of kernel should be between these
            colMax = j + kCenterX;
            colMin = j - dataSizeXMPI + kCenterX;

            //if (i == len-1)printf("rowMax:%d rowMin:%d colMax:%d colMin:%d RANK:%d\n",rowMax, rowMin, colMax, colMin, rank );
            
            sum = 0;                                // set to 0 before accumulate
            // flip the kernel and traverse all the kernel values
            // multiply each kernel value with underlying input data
            for(m = 0; m < kern->kernelY; ++m)        // kernel rows
            {
                // check if the index is out of bound of input array
                if(m <= rowMax && m > rowMin)
                {
                    for(n = 0; n < kern->kernelX; ++n)
                    {
                        // check the boundary of array
                        if(n <= colMax && n > colMin)
                            sum += *(inPtr - n) * *kPtr;
                        
                        ++kPtr;                     // next kernel
                    }
                }
                else
                    kPtr += kern->kernelX;            // out of bound, move to next row of kernel
                
                inPtr -= dataSizeXMPI;                 // move input data 1 raw up
            }
            
            // convert integer number
            if(sum >= 0) *outPtr = (int)(sum + 0.5f);
            else *outPtr = (int)(sum - 0.5f);
            
            kPtr = kern->vkern;                          // reset kernel to (0,0)
            inPtr = ++inPtr2;                       // next input
            ++outPtr;                               // next output
        }
    }//for general
}//convolve2DMPI

void processFrontiers(int row, int kCenter){
	int buffSize = dataSizeXMPI*kern->kernelX;
	int *tmpRBuffIn = (int *)calloc(buffSize, sizeof(int)); //Array de tantes linies com necessita per processar el kernel
	int *tmpRBuffOut = (int *)calloc(buffSize, sizeof(int)); //Array de tantes linies com necessita per processar el kernel
	int *tmpGBuffIn = (int *)calloc(buffSize, sizeof(int)); //Array de tantes linies com necessita per processar el kernel
	int *tmpGBuffOut = (int *)calloc(buffSize, sizeof(int)); //Array de tantes linies com necessita per processar el kernel
	int *tmpBBuffIn = (int *)calloc(buffSize, sizeof(int)); //Array de tantes linies com necessita per processar el kernel
	int *tmpBBuffOut = (int *)calloc(buffSize, sizeof(int)); //Array de tantes linies com necessita per processar el kernel

	//printf("row: %d\n ",row);

	for(int i=0;i<buffSize;i++){
		tmpRBuffIn[i] = source->R[(row*dataSizeXMPI)+i];
		tmpGBuffIn[i] = source->G[(row*dataSizeXMPI)+i];
		tmpBBuffIn[i] = source->B[(row*dataSizeXMPI)+i];
	}//copiar les dades als buffers temporals

	convolve2DMPI(tmpRBuffIn, tmpRBuffOut, (int)kern->kernelX); 
	convolve2DMPI(tmpGBuffIn, tmpGBuffOut, (int)kern->kernelX); 
	convolve2DMPI(tmpBBuffIn, tmpBBuffOut, (int)kern->kernelX); 

	for(int i=(kCenter*dataSizeXMPI);i<((kCenter+1)*dataSizeXMPI);i++){ 
		output->R[(row*dataSizeXMPI)+(i - (kCenter*dataSizeXMPI))] = tmpRBuffOut[i];
		output->G[(row*dataSizeXMPI)+(i - (kCenter*dataSizeXMPI))] = tmpGBuffOut[i];
		output->B[(row*dataSizeXMPI)+(i - (kCenter*dataSizeXMPI))] = tmpBBuffOut[i];
	}//copiar solament la linea resultant que ens interessa, estarà al mig del buffer de sortida
}//processFrontiers

//////////////////////////////////////////////////////////////////////////////////////////////////
// MAIN FUNCTION
//////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	//Init
	MPI_Init( &argc, &argv );
	MPI_Comm_rank( MPI_COMM_WORLD, &rank );
	MPI_Comm_size( MPI_COMM_WORLD, &size );

	//Functions
    MPI_Op_create((MPI_User_function *)convolve2DMPI, 1, &convolve2DOp); //Crear l'assignació per a l'operació 

    int i=0,j=0,k=0;
	struct timeval tim;
	double start, tstart=0, tend=0, tread=0, tcopy=0, tconv=0, tstore=0, treadk=0;
    int imagesize, partsize, chunksize, halosize, halo;
	long position=0;
	FILE *fpsrc=NULL,*fpdst=NULL;

    if(rank == 0){
        if(argc != 5)
	    {
	        printf("Usage: %s <image-file> <kernel-file> <result-file> <partitions>\n", argv[0]);
	        
	        printf("\n\nError, Missing parameters:\n");
	        printf("format: ./serialconvolution image_file kernel_file result_file\n");
	        printf("- image_file : source image path (*.ppm)\n");
	        printf("- kernel_file: kernel path (text file with 1D kernel matrix)\n");
	        printf("- result_file: result image path (*.ppm)\n");
	        printf("- partitions : Image partitions\n\n");
	        return -1;
	    }
	    //////////////////////////////////////////////////////////////////////////////////////////////////
	    // READING IMAGE HEADERS, KERNEL Matrix, DUPLICATE IMAGE DATA, OPEN RESULTING IMAGE FILE
	    //////////////////////////////////////////////////////////////////////////////////////////////////

		// Store number of partitions
	    partitions = atoi(argv[4]);
	    ////////////////////////////////////////
	    //Reading kernel matrix
	    //gettimeofday(&tim, NULL);
	    //start = tim.tv_sec+(tim.tv_usec/1000000.0);
	    start = MPI_Wtime();
	    tstart = start;
	    if ( (kern = leerKernel(argv[2]))==NULL) {
	        //        free(source);
	        //        free(output);
	        return -1;
	    }

	    //The matrix kernel define the halo size to use with the image. The halo is zero when the image is not partitioned.
	    if (partitions==1) halo=0;
	    else halo = (kern->kernelY/2)*2;
	    //gettimeofday(&tim, NULL);
	    //treadk = treadk + (tim.tv_sec+(tim.tv_usec/1000000.0) - start);
	    treadk = treadk + (MPI_Wtime() - start);

	    ////////////////////////////////////////
	    //Reading Image Header. Image properties: Magical number, comment, size and color resolution.
	    //gettimeofday(&tim, NULL);
	    //start = tim.tv_sec+(tim.tv_usec/1000000.0);
	    start = MPI_Wtime();
	    //Memory allocation based on number of partitions and halo size.
	    if ( (source = initimage(argv[1], &fpsrc, partitions, halo)) == NULL) {
	        return -1;
	    }
	    //gettimeofday(&tim, NULL);
	    //tread = tread + (tim.tv_sec+(tim.tv_usec/1000000.0) - start);
	    tread = tread + (MPI_Wtime() - start);
	    
	    //Duplicate the image struct.
	    //gettimeofday(&tim, NULL);
	    //start = tim.tv_sec+(tim.tv_usec/1000000.0);
	    start = MPI_Wtime();
	    if ( (output = duplicateImageData(source, partitions, halo)) == NULL) {
	        return -1;
	    }
	    //gettimeofday(&tim, NULL);
	    //tcopy = tcopy + (tim.tv_sec+(tim.tv_usec/1000000.0) - start);
	    tcopy = tcopy + (MPI_Wtime() - start);
	    
	    ////////////////////////////////////////
	    //Initialize Image Storing file. Open the file and store the image header.
	    //gettimeofday(&tim, NULL);
	    //start = tim.tv_sec+(tim.tv_usec/1000000.0);
	    start = MPI_Wtime();
	    if (initfilestore(output, &fpdst, argv[3], &position)!=0) {
	        perror("Error: ");
	        //        free(source);
	        //        free(output);
	        return -1;
	    }
	    //gettimeofday(&tim, NULL);
	    //tstore = tstore + (tim.tv_sec+(tim.tv_usec/1000000.0) - start);	
	    tstore = tstore + (MPI_Wtime() - start);

	    //////////////////////////////////////////////////////////////////////////////////////////////////
	    // CHUNK READING
	    //////////////////////////////////////////////////////////////////////////////////////////////////
	    
	    imagesize = source->altura*source->ancho;
	    partsize  = (source->altura*source->ancho)/partitions;
	}//RANK 0  

	if(rank != 0) {
		kern = (kernelData) malloc(sizeof(struct structkernel));
	}//Crear l'estructura del kernel als fils
	MPI_Barrier(MPI_COMM_WORLD);//Esperar que ho tinguin creat

	MPI_Bcast(&kern->kernelX, 1, MPI_INT, 0, MPI_COMM_WORLD); //Enviar les variables del kernel als fils
	MPI_Bcast(&kern->kernelY, 1, MPI_INT, 0, MPI_COMM_WORLD); //Enviar les variables del kernel als fils
	MPI_Barrier(MPI_COMM_WORLD);//Esperar que ho hagin rebut

	if(rank != 0) {
		kern->vkern = (float *)malloc(kern->kernelX*kern->kernelY*sizeof(float));
	}//Crear el tamany del vector de dades del cernel als fils
	MPI_Barrier(MPI_COMM_WORLD);//Esperar que tots els fils tinguin la informació guardada
	
	MPI_Bcast(kern->vkern, kern->kernelX*kern->kernelY, MPI_FLOAT, 0, MPI_COMM_WORLD); //Enviar el kernel als fils, no cal & perquè ja és un punter
	MPI_Bcast(&partitions, 1, MPI_INT, 0, MPI_COMM_WORLD); //Enviar les variables del while als fils
	MPI_Barrier(MPI_COMM_WORLD);

	int c=0, offset=0;

    while (c < partitions) {
    	if(rank == 0){
	        ////////////////////////////////////////////////////////////////////////////////
	        //Reading Next chunk.
	        //gettimeofday(&tim, NULL);
	        //start = tim.tv_sec+(tim.tv_usec/1000000.0);
	        start = MPI_Wtime();
	        if (c==0) {
	            halosize  = halo/2;
	            chunksize = partsize + (source->ancho*halosize);
	            offset   = 0;
	        }
	        else if(c<partitions-1) {
	            halosize  = halo;
	            chunksize = partsize + (source->ancho*halosize);
	            offset    = (source->ancho*halo/2);
	        }
	        else {
	            halosize  = halo/2;
	            chunksize = partsize + (source->ancho*halosize);
	            offset    = (source->ancho*halo/2);
	        }
	        
	        if (readImage(source, &fpsrc, chunksize, halo/2, &position)) {
	            return -1;
	        }
	        dataSizeXMPI = source->ancho;
	        dataSizeYMPI = (source->altura/partitions)+halosize;
	        //gettimeofday(&tim, NULL);
	        //tread = tread + (tim.tv_sec+(tim.tv_usec/1000000.0) - start);
	        tread = tread + (MPI_Wtime() - start);
	        
	        //Duplicate the image chunk
	        //gettimeofday(&tim, NULL);
	        //start = tim.tv_sec+(tim.tv_usec/1000000.0);
	        start = MPI_Wtime();
	        if ( duplicateImageChunk(source, output, chunksize) ) {
	            return -1;
	        }

	        //gettimeofday(&tim, NULL);
	        //tcopy = tcopy + (tim.tv_sec+(tim.tv_usec/1000000.0) - start);
	        tcopy = tcopy + (MPI_Wtime() - start);
    			        
	        //////////////////////////////////////////////////////////////////////////////////////////////////
	        // CHUNK CONVOLUTION
	        //////////////////////////////////////////////////////////////////////////////////////////////////
	        //gettimeofday(&tim, NULL);
	        //start = tim.tv_sec+(tim.tv_usec/1000000.0);
	        start = MPI_Wtime();
        }//RANK 0

    	MPI_Bcast(&dataSizeXMPI, 1, MPI_INT, 0, MPI_COMM_WORLD); //Enviar les variables de la imatge als fils
		MPI_Bcast(&dataSizeYMPI, 1, MPI_INT, 0, MPI_COMM_WORLD); //Enviar les variables de la imatge als fils
		MPI_Bcast(&chunkMPI, 1, MPI_INT, 0, MPI_COMM_WORLD); //Enviar les variables de la imatge als fils

 		if(rank != 0){
			source = (ImagenData) calloc(1, sizeof(struct imagenppm));
			output = (ImagenData) calloc(1, sizeof(struct imagenppm));
 		}//Inicialitzar les imatges als fils per solucionar errors al scatter i gather 

 		//SCATTER GATHER
 		int *Inbuff, *Outbuff;
 		int chunk = ((dataSizeYMPI*dataSizeXMPI)/size);

 		Inbuff = (int *)calloc(chunk, sizeof(int));
 		Outbuff = (int *)calloc(chunk, sizeof(int));


        MPI_Barrier(MPI_COMM_WORLD); //Esperar que el fil principal hagi llegit tot lo necessari
 		
	 	//printf("rank = %d, kX*ky = %d/%d iX/iY = %d/%d chunk = %d\n", rank, kern->kernelX, kern->kernelY, dataSizeXMPI, dataSizeYMPI, chunk);
 		MPI_Scatter(source->R, chunk, MPI_INT, Inbuff, chunk, MPI_INT, 0, MPI_COMM_WORLD);
		convolve2DMPI(Inbuff, Outbuff, dataSizeYMPI/size);
		MPI_Gather(Outbuff, chunk, MPI_INT, output->R, chunk, MPI_INT, 0, MPI_COMM_WORLD);

		MPI_Scatter(source->G, chunk, MPI_INT, Inbuff, chunk, MPI_INT, 0, MPI_COMM_WORLD);
		convolve2DMPI(Inbuff, Outbuff, dataSizeYMPI/size);
		MPI_Gather(Outbuff, chunk, MPI_INT, output->G, chunk, MPI_INT, 0, MPI_COMM_WORLD);

		MPI_Scatter(source->B, chunk, MPI_INT, Inbuff, chunk, MPI_INT, 0, MPI_COMM_WORLD);
		convolve2DMPI(Inbuff, Outbuff, dataSizeYMPI/size);
 		MPI_Gather(Outbuff, chunk, MPI_INT, output->B, chunk, MPI_INT, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD); //Esperar que el fil principal hagi llegit tot lo necessari
		free(Inbuff);
		free(Outbuff);
		if(rank == 0){
        	int kCenter = (int) kern->kernelX/2;

			for(int th=1;th<size;th++){
				int row = th*(dataSizeYMPI/size);
				for(int j = -kCenter; j <= kCenter; j++){
					processFrontiers(row + j, kCenter);
				}//Iterar tot el rank del kernel que esdevé frontera
			}//Iterar entre els fils per obtenir les fronteres
        }//Afegir aquesta zona per fer que master corregeixi els errors d'informació de les zones frontera

 		//REDUCE
 		/*MPI_Bcast(source->R, chunkMPI, MPI_INT, 0, MPI_COMM_WORLD); //Enviar les variables de la imatge als fils
 		MPI_Bcast(source->G, chunkMPI, MPI_INT, 0, MPI_COMM_WORLD); //Enviar les variables de la imatge als fils
 		MPI_Bcast(source->B, chunkMPI, MPI_INT, 0, MPI_COMM_WORLD); //Enviar les variables de la imatge als fils
        MPI_Barrier(MPI_COMM_WORLD); //Esperar que el fil principal hagi llegit tot lo necessari

        printf("rank = %d, kX*ky = %d/%d iX/iY = %d/%d\n", rank, kern->kernelX, kern->kernelY, dataSizeXMPI, dataSizeYMPI);

        MPI_Reduce(source->R, output->R, chunkMPI, MPI_INT, convolve2DOp, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD); //Esperar 
        MPI_Reduce(source->G, output->G, chunkMPI, MPI_INT, convolve2DOp, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD); //Esperar 
        MPI_Reduce(source->B, output->B, chunkMPI, MPI_INT, convolve2DOp, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD); //Esperar*/

        if(rank == 0){
	        //gettimeofday(&tim, NULL);
	        //tconv = tconv + (tim.tv_sec+(tim.tv_usec/1000000.0) - start);
        	tconv = tconv + (MPI_Wtime() - start);
        	        
        	//////////////////////////////////////////////////////////////////////////////////////////////////
	        // CHUNK SAVING
	        //////////////////////////////////////////////////////////////////////////////////////////////////
	        //Storing resulting image partition.
	        //gettimeofday(&tim, NULL);
	        //start = tim.tv_sec+(tim.tv_usec/1000000.0);
	        start = MPI_Wtime();
	        if (savingChunk(output, &fpdst, partsize, offset)) {
	            perror("Error: ");
	            //        free(source);
	            //        free(output);
	            return -1;
	        }
	        //gettimeofday(&tim, NULL);
	        //tstore = tstore + (tim.tv_sec+(tim.tv_usec/1000000.0) - start);
	        tstore = tstore + (MPI_Wtime() - start);
	        //Next partition
        }//RANK 0

        c++;
    }//While

	if(rank == 0){
	    fclose(fpsrc);
	    fclose(fpdst);
	    
	    //gettimeofday(&tim, NULL);
	    //tend = tim.tv_sec+(tim.tv_usec/1000000.0);
	    tend = MPI_Wtime();
	    
	    printf("Imatge: %s\n", argv[1]);
	    printf("ISizeX : %d\n", source->ancho);
	    printf("ISizeY : %d\n", source->altura);
	    printf("kSizeX : %d\n", kern->kernelX);
	    printf("kSizeY : %d\n", kern->kernelY);
	    printf("%.6lf seconds elapsed for Reading image file.\n", tread);
	    printf("%.6lf seconds elapsed for copying image structure.\n", tcopy);
	    printf("%.6lf seconds elapsed for Reading kernel matrix.\n", treadk);
	    printf("%.6lf seconds elapsed for make the convolution.\n", tconv);
	    printf("%.6lf seconds elapsed for writing the resulting image.\n", tstore);
	    printf("%.6lf seconds elapsed\n", tend-tstart);
	    
	    freeImagestructure(&source);
	    freeImagestructure(&output);
	}//RANK 0
    
    MPI_Finalize();
	exit(0);
    return 0;
}
