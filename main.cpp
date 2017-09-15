#include <iostream>
#include <algorithm>
#include <random>
#include <time.h>
#include <thread>
#include <chrono>
#include "libs/Eigen3.3.4/Eigen/Core"
#include "libs/Eigen3.3.4/Eigen/Eigen"

#define SIZEVECT(vect) (sizeof(vect)/sizeof((vect)[0]))

using namespace Eigen;
using namespace std;

/**
 * Felli Fitness Function
 * @param individual
 * @param sizeIndividual
 * @return individual fitness
 */
double felliFunc (double individual[], int sizeIndividual){
    double result = 0.0;
    double auxSize = sizeIndividual; //for double division
    double auxVector[sizeIndividual];
    if (sizeIndividual < 2){
        cout << "dimension must be greater one" << endl;
        return result;
    }
    for (int i=0; i<sizeIndividual; i++){
//        cout << pow(1e6,(i/(auxSize-1.0))) << endl;
        auxVector[i] = pow(1e6,(i/(auxSize-1.0)));// condition number 1e6
//        auxVector[i] = (i/(auxSize-1.0));//OTHER FUNCTION TO TEST
    }

    for(int i=0; i<sizeIndividual; i++){
        result += auxVector[i]*(pow(individual[i],2));
//        result += individual[i];//OTHER FUNCTION TEST
    }
    return result;
}

/**
 * Print Matrix in console
 * @param matrix: obj to print
 * @param line: lines size
 * @param col: columns size
 */
void printMatrix (double **matrix, int line, int col){
    for (int i=0; i<line; i++){
        for (int j=0; j<col; j++){
            cout << matrix[i][j] << "   ";
        }
        cout << endl;
    }
    cout << endl;
}

/**
 * Print Vector in console
 * @param vector: obj to print
 * @param size: vector dimension
 */
void printVector (double *vector, int size){
    for (int i=0; i<size; i++){
        cout << vector[i] << endl;
    }
    cout << endl;
}

/**
 * Multiply two matrices
 * @param matrixA: input
 * @param matixB: input
 * @param matrixAxB: result
 * @param lines: number of rows in matrixA
 * @param col: number of columns in matrixB
 */
void multMatrix (double **matrixA, double **matixB, double **matrixAxB, int lines, int col){
    double sumprod;
    for(int l=0; l<lines; l++){
        for(int c=0; c<col; c++){
            sumprod=0.0;
            for(int i=0; i<lines; i++){
                sumprod+=matrixA[l][i]*matixB[i][c];
            }
            matrixAxB[l][c]=sumprod; // A*B
        }
    }
}

/**
 * Transpose Matrix
 * @param matrix: input
 * @param matixTransp: matrix transposed
 * @param lines: number of rows in matrix
 * @param col: number of columns in matrix
 */
void transpMatrix (double **matrix, double **matixTransp, int lines, int col){
    for (int l=0; l<lines; l++){
        for (int c=0; c<col; c++){
            matixTransp[c][l] = matrix[l][c];
        }
    }
}

/**
 * Create a Identity Matrix
 * @param matrix: matrix that will be the identity
 * @param lines: number of rows in matrix
 * @param col: number of columns in matrix
 */
void identityMatrix (double **matrix, int lines, int col){
    for (int l=0; l<lines; l++){
        for (int c=0; c<col; c++){
            if (l==c){
                matrix[l][c] = 1.0;
            } else{
                matrix[l][c] = 0.0;
            }
        }
    }
}

/**
 * To Alloc Space in Memory to Pointer of Pointer (matrix)
 * @param matrix: pointer of pointer
 * @param lines: number of doubles in pointer 1 (rows)
 * @param col: number of doubles in pointer 2 (columns)
 */
double ** allocPointerOfPointer (int lines, int col) {
    double **matrix;
    matrix = (double **)malloc(sizeof(double *)* lines);
    for (int i = 0; i < lines; ++i){
        matrix[i] = (double *)malloc(sizeof(double)* col);
    }
    return matrix;
}

/**
 * Free Space in Pointer of Pointer's (matrix) Memory
 * @param matrix: pointer of pointer
 * @param lines: number of doubles in *matrix (rows)
 * @return: pointer of pointer NULL
 */
double ** freePointerOfPointer(double **matrix, int lines)
{
    for (int i = 0; i < lines; ++i){
        free(matrix[i]);
    }
    free(matrix);
    return NULL;
}


/**
 * Euclidean Norm of a Vector
 * @param vector: vector to calc the norm
 * @param sizeVector: vector dimension
 * @return
 */
double euclideanNorm (double vector[], int sizeVector){
    double norm=0;
    for (int i=0; i<sizeVector; i++){
        norm += pow(vector[i],2);
    }
    return sqrt(norm);
}

/**
 * Sum Values in a Vector
 * @param vector: origin vector
 * @param sumVector: sum value
 * @param sizeVector: vector size
 */
void sumVector (double *vector, double *sumVector, int sizeVector){
    *sumVector = 0.0;
    for(int i=0; i<sizeVector; i++){
        *sumVector += vector[i];
    }
}

/**
 * Extern Product of a Vector
 * @param vector: input vector
 * @param matrix: result of the extern product (matrix[sizeVector][sizeVector])
 * @param sizeVector
 */
void externProd (double *vector, double **matrix, int sizeVector){
    for (int i=0; i<sizeVector; i++){
        for (int j=0; j<sizeVector; j++){
            matrix[i][j] = vector[i]*vector[j];
        }
    }
}

/**
 * Convert double **matrix[size][size] to double EigenMatrix[size][size]
 * @param matrix: obj to convert
 * @param size: matrix order
 * @return EigenMatrix: matrix converted
 */
Eigen::MatrixXd convertToEigenMatrix(double **matrix, int size)
{
    Eigen::MatrixXd EigenMatrix(size, size);
    for (int i = 0; i < size; ++i)
        EigenMatrix.row(i) = Eigen::VectorXd::Map(&matrix[i][0], size);
    return EigenMatrix;
}

/**
 * Update the Matrices B and D
 * @param matrixB
 * @param matrixD
 * @param size: matrices orders
 * @param eigenSolver: obj with eigenValues and eigenVectors
 */
void updateBandD(double **matrixB, double **matrixD, int size, EigenSolver<MatrixXd> eigenSolver){
    MatrixXcd eigenMatrixcD = eigenSolver.eigenvalues().asDiagonal();
    MatrixXcd eigenMatrixcB = eigenSolver.eigenvectors();
//    cout << eigenMatrixcD << endl;
//    cout << endl;
//    cout << eigenMatrixcB << endl;
    complex<double> complexNum;
    for (int i=0; i<size; i++){
        for (int j=0; j<size; j++){
            complexNum = eigenMatrixcB(i,j);
            matrixB[i][j] = complexNum.real();
            //D = diag(sqrt(diag(D))); // D contains standard deviations now
            complexNum = eigenMatrixcD(i,j);
            matrixD[i][j] = sqrtf(complexNum.real());
        }
    }
//    printMatrix(matrixD,size,size);
//    printMatrix(matrixB,size,size);
}

/**
 * Generate One Number With the Normal Distribution
 * @param mean: average
 * @param stdDev: standard deviation
 * @return double random number
 */
double randNormalNumber (double mean, double stdDev){
//    srand (static_cast <unsigned> (time(0))); // to not generate the same values
//    return lowerBound + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(upperBound-lowerBound)));
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::normal_distribution<double> distribution(mean,stdDev);
    return distribution(generator);
}

/**
 * Generate One Number With the Uniform Distribution
 * @param lowerBound
 * @param upperBound
 * @return double random number
 */
double randUniformNumber (double lowerBound, double upperBound){
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::uniform_real_distribution<double> distribution(lowerBound,upperBound);
    return distribution(generator);
}

int main()
{
    // -------------------- Initialization --------------------------------

    //----- User defined input parameters (need to be edited)
    //
    //char strfitnessfct[] = "felli"; // name of objective/fitness function (don't used)
    int N = 10; // number of objective variables/problem dimension
    double xmean[N]; // objective variables initial point
    double zmean[N];
    double auxSum;
    double sigma = 0.5; // coordinate wise standard deviation (step-size)
    double stopfitness = 1e-10; // stop if fitness < stopfitness (minimization)
    double stopeval = 1e3*pow(N,2); // stop after stopeval number of function evaluations

    //xmean
    double lowerBound, upperBound;
    lowerBound = 0.0;
    upperBound = 1.0;
    for (int i=0; i<N; i++){
        xmean[i] = randUniformNumber(lowerBound, upperBound);
        std::this_thread::sleep_for (std::chrono::nanoseconds(1));
//        xmean[i] = 0.1; //for test
    }


    //----- Strategy parameter setting: Selection
    //
    int lambda = 4+floor(3.0*log(N)); // population size, offspring number
    double mudouble = lambda/2.0; // lambda=12; mu=3; weights = ones(mu,1); would be (3_I,12)-ES
    int mu = floor(mudouble); // number of parents/points for recombination

    // muXone recombination weights
    double weights[mu];
    double logs[mu];
    double oneLog;
    oneLog = log(mudouble+1.0/2.0);
    for (int i=0; i<mu;i++){
        weights[i] = oneLog - log(i+1);
    }

    // normalize recombination weights array
    double sumWeights;
    sumVector(weights, &sumWeights, SIZEVECT(weights));
    double sumOfQuad = 0.0;
    for (int i=0; i<SIZEVECT(weights); i++){
        weights[i] /= sumWeights;
        sumOfQuad += pow(weights[i],2);
    }
    sumVector(weights, &sumWeights, SIZEVECT(weights)); // sumWeights of new weights
    double mueff= pow(sumWeights,2)/sumOfQuad; // variance-effective size of mu


    //----- Strategy parameter setting: Adaptation
    //
    double cc = (4.0+mueff/N) / (N+4.0 + 2.0*mueff/N); // time constant for cumulation for C
    double cs = (mueff+2)/(N+mueff+5.0); // t-const for cumulation for sigma control
    double c1 = 2.0 / (pow((N+1.3),2)+mueff); // learning rate for rank-one update of C
    double cmu = 2.0 * (mueff-2.0+1.0/mueff) / (pow((N+2.0),2)+2.0*mueff/2.0); // and for rank-mu update
    double damps = 1.0 + 2.0*max(0.0, sqrt((mueff-1.0)/(N+1.0))-1.0) + cs; // damping for sigma

    //------ Initialize dynamic (internal) strategy parameters and constants
    //
    // evolution paths for C and sigma
    double pc[N];
    double ps[N];
    for (int i=0; i<N; i++){
        pc[i] = 0.0;
        ps[i] = 0.0;
    }

    // B defines the coordinate system
    // diagonal matrix D defines the scaling
    // covariance matrix
    double **B;
    double **D;
    double **BxD;
    double **BxDTransp;
    double **C;
    B = allocPointerOfPointer(N,N);
    D = allocPointerOfPointer(N,N);
    BxD = allocPointerOfPointer(N,N);
    BxDTransp = allocPointerOfPointer(N,N);
    C = allocPointerOfPointer(N,N);

    identityMatrix(B,N,N);
    identityMatrix(D,N,N);

//    cout << "Inicialização B" << endl;
//    printMatrix(B,N,N);

//    cout << "Inicialização D" << endl;
//    printMatrix(D,N,N);

//    cout << "Inicialização BxD" << endl;
//    printMatrix(BxD,N,N);

//    cout << "Inicialização BxDTransp" << endl;
//    printMatrix(BxDTransp,N,N);

//    cout << "Inicialização C" << endl;
//    printMatrix(C,N,N);

    // auxiliary variable
    double sumprod;

    // B*D
    multMatrix(B,D,BxD,N,N);
//    cout << "BxD = multMatrix(B,D,BxD,N,N);" << endl;
//    printMatrix(BxD,N,N);

    // (B*D)'
    transpMatrix(BxD,BxDTransp,N,N);
//    cout << "BxDTransp = transpMatrix(BxD,BxDTransp,N,N);" << endl;
//    printMatrix(BxDTransp,N,N);

    // C=(B*D)*(B*D)'
    multMatrix(BxD,BxDTransp,C,N,N);
//    cout << "C = multMatrix(BxD,BxDTransp,C,N,N);" << endl;
//    printMatrix(C,N,N);

    double eigeneval = 0.0; // B and D updated at counteval == 0
    double chiN=pow(N,0.5)*(1.0-1.0/(4.0*N)+1.0/(21.0*pow(N,2))); // expectation of ||N(0,I)|| == norm(randn(N,1))


    // -------------------- Generation Loop --------------------------------
    //TESTE PRINTANDO MATRIZES
//    printMatrix(B, N, N);
//    printMatrix(D, N, N);
//    printMatrix(C, N, N);
    //FIM TESTE

    int counteval = 0; // the next 40 lines contain the 20 lines of interesting code
    double **arz;// standard normally distributed vectors
    double **arx;// add mutation // Eq. 40
    double **BxDxarz;
    double **BxDxArz;
    double **CmuxBxDxArzxWeig;
    double **BxDxArzTransp;
    double **plusRankMiUpd;
    double **extProdPc;

    arz = allocPointerOfPointer(N,lambda);
    arx = allocPointerOfPointer(N,lambda);
    BxDxarz = allocPointerOfPointer(N,lambda);
    BxDxArz = allocPointerOfPointer(N,mu);
    CmuxBxDxArzxWeig = allocPointerOfPointer(N,mu);
    BxDxArzTransp = allocPointerOfPointer(mu,N);
    plusRankMiUpd = allocPointerOfPointer(N,N);
    extProdPc = allocPointerOfPointer(N,N);

    double arfitness[lambda];// fitness of individuals
    double individualForTest[N];
    int arindex[lambda];
    double sortArfitness[lambda];
    double distMean = 0.0;
    double distDev = 1.0;

//    cout << "INICIAR WHILE: " << endl;
    while (counteval < stopeval){

        // (B*D)
        multMatrix(B,D,BxD,N,N); //for update to each loop

        // Generate and evaluate lambda offspring
        for (int i=0; i<lambda; i++){
            // standard normally distributed vector
            for (int j=0; j<N; j++){
                arz[j][i] = randNormalNumber(distMean, distDev);
                std::this_thread::sleep_for (std::chrono::nanoseconds(1));
//                arz[j][i] = 0.1; //for test
            }
        }

        multMatrix(BxD,arz,BxDxarz,N,lambda);

        for (int i=0; i<lambda; i++){
            for (int j=0; j<N; j++){
                arx[j][i] = xmean[j] + sigma * (BxDxarz[i][j]); // add mutation // Eq. 40
                individualForTest[j] = arx[j][i];
            }
            //verificar strfitnessfct para disparar a função desejada
            arfitness[i] = felliFunc(individualForTest, SIZEVECT(individualForTest)); // objective function call
            counteval += 1;
        }

        //----- Sort by fitness and compute weighted mean into xmean
        //
        for (int i=0; i<SIZEVECT(arfitness); i++) {
            sortArfitness[i] = arfitness[i];
        }
        sort(sortArfitness, sortArfitness + lambda); // minimization
        for (int i=0; i<SIZEVECT(sortArfitness); i++){
            for (int j=0; j<lambda; j++){
                if(sortArfitness[i] == arfitness[j]){
                    arindex[i] = j;
                    break;
                }
            }
        }

        // recombination // Eq. 42
        // == D^-1*B'*(xmean-xold)/sigma
        double auxSumX;
        double auxSumZ;

//        cout << "ARX:" << endl;
//        printMatrix(arx,N,N);
//        cout << "WEIGHTS:" << endl;
//        printVector(weights,N);

        for (int i=0; i<N; i++){
            auxSumX = 0.0;
            auxSumZ = 0.0;
            for (int j=0; j<mu; j++){
                auxSumX += arx[i][arindex[j]]*weights[j]; // recombination // Eq. 42
                auxSumZ += arz[i][arindex[j]]*weights[j]; // == D^-1*B'*(xmean-xold)/sigma
            }
            xmean[i] = auxSumX;
            zmean[i] = auxSumZ;
        }

//        cout << "XMEAN:" << endl;
//        printVector(xmean,N);

        //----- Cumulation: Update evolution paths
        //
        // ps (Eq. 43)
        // (sqrt(cs*(2-cs)*mueff)) * (B * zmean)
        double auxValue = sqrt(cs*(2.0-cs)*mueff);
        double BxZmeanxAux[N];
        for (int i=0; i<N; i++){
            auxSum = 0.0;
            for (int j=0; j<N; j++){
                auxSum += B[i][j]*zmean[j]*auxValue;
            }
            BxZmeanxAux[i] = auxSum;
        }
        //
        // ps = [(1-cs)*ps] + [(sqrt(cs*(2-cs)*mueff)) * (B * zmean)]
        for (int i=0; i<SIZEVECT(ps); i++){
            ps[i] = (1.0-cs)*ps[i];
            ps[i] += BxZmeanxAux[i];
        }

        // hsig = norm(ps)/sqrt(1-(1-cs)^(2*counteval/lambda))/chiN < 1.4+2/(N+1)
        double hsig;
        auxValue = euclideanNorm(ps,SIZEVECT(ps))/sqrt(1.0-pow((1.0-cs),(2.0*counteval/lambda)))/chiN;
        if (auxValue < (1.4+2/(N+1.0))){
            hsig = 1.0;
        } else {
            hsig = 0.0;
        }

        // pc (Eq. 45)
        // hsig * sqrt(cc*(2-cc)*mueff) * (B*D*zmean)
        auxValue = sqrt(cc*(2.0-cc)*mueff);
        double BxDxZmeanxAuxxHsig[N];
        for (int i=0; i<N; i++){
            auxSum = 0.0;
            for (int j=0; j<N; j++){
                auxSum += BxD[i][j]*zmean[j]*auxValue*hsig;
            }
            BxDxZmeanxAuxxHsig[i] = auxSum;
            //cout << BxDxZmeanxAuxxHsig[i] << endl;
        }
        //
        // pc = (1-cc)*pc + hsig * sqrt(cc*(2-cc)*mueff) * (B*D*zmean)
        for (int i=0; i<SIZEVECT(pc); i++){
            pc[i] = (1.0-cc)*pc[i];
            pc[i] += BxDxZmeanxAuxxHsig[i];
        }


        //----- Adapt covariance matrix C
        //
        //C = (1-c1-cmu) * C ... // regard old matrix // Eq. 47
        //+ c1 * (pc*pc' ... // plus rank one update
        //+ (1-hsig) * cc*(2-cc) * C) ... // minor correction
        //+ cmu ... // plus rank mu update
        //* (B*D*arz(:,arindex(1:mu))) ...
        //* diag(weights) * (B*D*arz(:,arindex(1:mu)))';
        //
        // regard old matrix (Eq. 47)
        //(1-c1-cmu) * C
        double regOldC[N][N];
        for (int i=0; i<N; i++){
            for (int j=0; j<N; j++){
                regOldC[i][j] = (1.0-c1-cmu) * C[i][j];
            }
        }

        // pc*pc'
        externProd(pc, extProdPc, SIZEVECT(pc));

        // minor correction
        auxValue = (1.0-hsig) * cc*(2.0-cc);
        double minorCor[N][N];
        for (int i=0; i<N; i++){
            for (int j=0; j<N; j++){
                minorCor[i][j] = auxValue * C[i][j];
                //cout << minorCor[i][j] << endl;
            }
        }

        // (pc*pc') + (minor correction)
        double sumPcMinor[N][N];
        for (int i=0; i<N; i++){
            for (int j=0; j<N; j++){
                sumPcMinor[i][j] = extProdPc[i][j] + minorCor[i][j];
            }
        }

        // plus rank one update
        double plusRankOneUp[N][N];
        for (int i=0; i<N; i++){
            for (int j=0; j<N; j++){
                plusRankOneUp[i][j] = c1 * sumPcMinor[i][j];
            }
        }

        // B*D*arz(:,arindex(1:mu))
        multMatrix(BxD,arz,BxDxArz,N,mu);

        // cmu*(B*D*arz(:,arindex(1:mu)))*diag(weights)
        for (int i=0; i<N; i++){
            for (int j=0; j<mu; j++){
                CmuxBxDxArzxWeig[i][j] = cmu * BxDxArz[i][j] * weights[j];
            }
        }

        // (B*D*arz(:,arindex(1:mu)))'
        transpMatrix(BxDxArz,BxDxArzTransp,N,mu);

        // cmu * (B*D*arz(:,arindex(1:mu))) * diag(weights) * (B*D*arz(:,arindex(1:mu)))'
        for(int l=0; l<N; l++){
            for(int c=0; c<N; c++){
                sumprod=0.0;
                for(int i=0; i<mu; i++){
                    sumprod += CmuxBxDxArzxWeig[l][i]*BxDxArzTransp[i][c];
                }
                plusRankMiUpd[l][c]=sumprod;
            }
        }

        // Adapt covariance matrix C
        for (int i=0; i<N; i++){
            for (int j=0; j<N; j++){
                C[i][j] = regOldC[i][j] + plusRankOneUp[i][j] + plusRankMiUpd [i][j];
            }
        }

//        cout << "C before" << endl;
//        printMatrix(C,N,N);


        //----- Adapt step-size sigma
        //
        sigma = sigma * exp((cs/damps)*(euclideanNorm(ps,SIZEVECT(ps))/chiN - 1.0)); // Eq. 44


        //----- Update B and D from C
        //
        if ((counteval - eigeneval) > (lambda/(c1+cmu)/N/10)){ // to achieve O(N^2)
            eigeneval = counteval;

            // enforce symmetry
            // C=triu(C)+triu(C,1)'
            for (int i=0; i<N; i++){
                for (int j=0; j<i; j++){
                    C[i][j] = C[j][i];
                }
            }
//            cout << "C after" << endl;
//            printMatrix(C,N,N);
            MatrixXd eigenMatrixC = convertToEigenMatrix(C,N); //converting C[][] to EigenMatrix
//            cout << eigenMatrixC << endl;
//            cout << endl;
            EigenSolver<MatrixXd> eigenSolver(eigenMatrixC);
            updateBandD(B,D,N,eigenSolver);
//            printMatrix(D,N,N);
//            printMatrix(B,N,N);
        }

//        cout << "B" << endl;
//        printMatrix(B,N,N);

//        cout << "D" << endl;
//        printMatrix(D,N,N);


        //----- Break, if fitness is good enough
        if (sortArfitness[0] <= stopfitness){
            break;
        }

        //----- Escape flat fitness, or better terminate?
        int flatFitness = floorf(0.7*lambda);
        if (sortArfitness[0] == sortArfitness[flatFitness]){
            sigma = sigma * exp(0.2+cs/damps);
            cout << "warning: flat fitness, consider reformulating the objective" << endl;
        }

        cout << counteval << ": " << sortArfitness[0] << endl;

    }


    // -------------------- Final Message ---------------------------------
    cout << counteval << ": " << sortArfitness[0] << endl;

    /*
     * Return best point of last generation.
     * Notice that xmean is expected to be even
     * better.
    */
    //xmin = arx(:, arindex(1));
    double xmin[N];
    for (int i=0; i<N; i++){
        xmin[i] = arx[i][arindex[0]];
    }
    cout << "The best individual:" << endl;
    printVector(xmin,SIZEVECT(xmin));

    //free memory/pointers
    B = freePointerOfPointer(B,N);
    D = freePointerOfPointer(D,N);
    BxD = freePointerOfPointer(BxD,N);
    BxDTransp = freePointerOfPointer(BxDTransp,N);
    C = freePointerOfPointer(C,N);
    arx = freePointerOfPointer(arx,N);
    arz = freePointerOfPointer(arz,N);
    BxDxArz = freePointerOfPointer(BxDxArz,N);
    CmuxBxDxArzxWeig = freePointerOfPointer(CmuxBxDxArzxWeig,N);
    BxDxArzTransp = freePointerOfPointer(BxDxArzTransp,mu);
    plusRankMiUpd = freePointerOfPointer(plusRankMiUpd,N);
    extProdPc = freePointerOfPointer(extProdPc,N);
    BxDxarz = freePointerOfPointer(BxDxarz,N);

    return 0;
}