#include <cstdlib>
#include <iostream>
#include <random>
#include <ctime>
#include <chrono>
#include <cassert>
#include <omp.h>
#include <mpi.h>

#define EXEC_EXPERIMENTO 25
#define MAX_VALUE 1000L
#define TEST_SEED 123456
std::mt19937 mt(TEST_SEED); //Generator as a global variable
using namespace std;

/**
 * @brief Função para multiplicar uma matriz (nxm) por um vetor (nx1) utilizando cache blocking e retornar o resultado 
 * 
 * @param matrix: Matriz usada na multiplicação
 * @param vector: Vetor usado na multiplicação
 * @param n: Número de linhas da matriz
 * @param m: Número de linhas do vetor
 * @param tile_size: Tamanho do tile
 * @return long*: Retorna o vetor resultante da multiplicação
 */
long* multiplyCacheBlocking(long* matrix, long* vector, long m, long n, int tile_size){
    
    long* c = new long[m];
    #pragma omp parallel shared(c)
    {
        
        #pragma omp for
        for(long jj = 0; jj < n; jj+=(long)tile_size){
            for(long i  = 0; i < m; i++){
                    for(long j = jj; j < jj + (long)tile_size; j++){
                            #pragma omp atomic
                            c[i] += matrix[i*n + j]*vector[j];
                            //cout << "c[" << i << "]: = " << "M[" << i << "][" << j << "]*b[" << j << "]" << endl;
                            //cout << c[i] << " = " << matrix[i][j] << " * " << vector[j] << endl;
                    }
            }
        }
    }
    return c;
}
/**
 * @brief Cria uma matriz de dimensao nxm com valores aleatórios entre 0 e MAX_VALUE
 * 
 * @param m: Número de linhas
 * @param n: Número de colunas
 * @return long**: Retorna a matriz criada
 */

long* create_matrix(long m, long n){
    uniform_int_distribution<long> uniform{0L,MAX_VALUE};
    long* matrix  = new long[m*n];
    for(long i = 0; i < m; i++){
        for(long j = 0; j < n; j++){
            //matrix[i*n + j] = uniform(mt);
            matrix[i*n + j] = 1;
        }
    }
    return matrix;

}
/**
 * @brief Cria um vetor de dimensao mx1 com valores aleatórios entre 0 e MAX_VALUE
 * 
 * @param m: Número de linhas do vetor
 * @return long*: Retorna o vetor criado 
 */
long* create_vector(long m){
    uniform_int_distribution<long> uniform{0L,MAX_VALUE};
    long* vector = new long[m];
    for(long i = 0; i < m; i++){
        //vector[i] = uniform(mt);
        vector[i] = 1;
    }
    return vector;

}
/**
 * @brief Exibe os elementos (matriz, vetor e vetor resultante) usados para os cálculos
 * 
 * @param matrix: Matriz de dimensão nxm
 * @param vector: Vetor de dimensão mx1
 * @param c: Vetor de dimensão nx1 resultante da multiplicação da matriz pelo vetor 
 * @param n: Número de linhas da matriz
 * @param m: Número de colunas da matriz 
 */
void printElements(long** matrix, long* vector, long *c, long m, long n){
    if(matrix){
        cout << "Matrix: " << endl;
        for(long i = 0; i < m; i++){
            for(long j = 0; j < n; j++){
                cout << matrix[i*n + j] << "\t";
            }
            cout << endl;
        }

    }
    if(vector){
        cout << "Vector: " << endl;
        cout << "----------------------------------" << endl;
        for(long i = 0; i < n; i++){
            cout << vector[i] << endl;
        }

    }
    if(c){
        cout << "----------------------------------" << endl;
        cout << "Resulting vector: " << endl;
        for(long i = 0; i < m; i++){
            cout << c[i] << endl;
        }
        cout << "----------------------------------" << endl;
    }
}

int main(int argc, char* argv[]){
    long m = 2000L, n = 2000L;
    int tile_size=4, myrank, numprocs, num_tiles;
    double ini = 0, end = 0;
    long* c = new long[m];
    long* c_mpi = new long[m];
    long i, j, jj;
    for(i =0; i < m; i++){
            c[i] = 0L;
            c_mpi[i] = 0L;
    }
    if (argc > 3){
        m = atol(argv[1]);
        n = atol(argv[2]);
        tile_size = atoi(argv[3]);
    }
    else{
        cerr << "Arguments needed (in order): \"number of lines\" \"number of columns\" \"tile size\"" << endl;
        exit(-1);
    }
    assert(n % tile_size == 0);
    num_tiles = n / tile_size;
    long* vector  = create_vector(n);
    long* matrix = create_matrix(m, n);
    MPI_Aint extent;
    MPI_Init(&argc, &argv); 
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs); 
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Type_extent(MPI_LONG, &extent);

    if(myrank == 0)
        cout << "Experimento - Move branches out of loop" << endl;    
    #pragma omp parallel shared(c)
    {
        #pragma omp for
        for(jj = (myrank*((num_tiles/numprocs)*tile_size)); jj < ((tile_size*num_tiles)/(2 - myrank)); jj+=(long)tile_size){
            for(i  = 0; i < m; i++){
                    for(j = jj; j < jj + (long)tile_size; j++){
                            #pragma omp atomic
                            c_mpi[i] += matrix[i*n + j]*vector[j];
                    }
            }
        }
    }
    for(i =0; i < m; i++)
        cout << "(" << myrank << ") my_c[" << i << "]: " << c_mpi[i] <<"\t";
    cout << endl;
    MPI_Reduce(c_mpi, &c[0], m, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD); 
    if(myrank == 0){
        printElements(nullptr, nullptr, c, m, n);
    }
    MPI_Finalize();
    return 0;
}
