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

/*
 * This is a "felli" function.
 * There should be a switch for a choice of function to be used.
 * This information should be passed in "strfitnessfct[]"
 */
float feval (char strfitnessfct[], float individual[], int sizeIndividual){
    float result = 0.0;
    float auxSize = sizeIndividual; //for float division
    float auxVector[sizeIndividual];
    if (sizeIndividual < 2){
        cout << "dimension must be greater one" << endl;
        return result;
    }
    for (int i=0; i<sizeIndividual; i++){
        //cout << i/(auxSize-1) << endl;
        auxVector[i] = pow(1e6,(i/(auxSize-1.0)));// condition number 1e6
    }

    for(int i=0; i<sizeIndividual; i++){
        result += auxVector[i]*(pow(individual[i],2));
        //BxD[l][c]=sumprod;
    }
    return result;
}

/**
 * Print Matrix in console
 * @param matrix: obj to print
 * @param line: lines size
 * @param col: columns size
 */
void printMatrix (float **matrix, int line, int col){
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
void printVector (float *vector, int size){
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
void multMatrix (float **matrixA, float **matixB, float **matrixAxB, int lines, int col){
    int sumprod;
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
void transpMatrix (float **matrix, float **matixTransp, int lines, int col){
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
void identityMatrix (float **matrix, int lines, int col){
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
 * @param pointer1: number of floats in pointer 1 (rows)
 * @param pointer2: number of floats in pointer 2 (columns)
 */
void allocPointerOfPointer (float **matrix, int pointer1, int pointer2){
    matrix = (float **) malloc(pointer1*sizeof(float *));
    for (int i=0; i<pointer1; i++) {
        matrix[i] = (float *) malloc(pointer2 * sizeof(float));
    }
}

/**
 * Euclidean Norm of a Vector
 * @param vector: vector to calc the norm
 * @param sizeVector: vector dimension
 * @return
 */
float euclideanNorm (float vector[], int sizeVector){
    float norm=0;
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
void sumVector (float *vector, float *sumVector, int sizeVector){
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
void externProd (float *vector, float **matrix, int sizeVector){
    for (int i=0; i<sizeVector; i++){
        for (int j=0; j<sizeVector; j++){
            matrix[i][j] = vector[i]*vector[j];
        }
    }
}

/**
 * Convert float **matrix[size][size] to float EigenMatrix[size][size]
 * @param matrix: obj to convert
 * @param size: matrix order
 * @return EigenMatrix: matrix converted
 */
Eigen::MatrixXf convertToEigenMatrix(float **matrix, int size)
{
    Eigen::MatrixXf EigenMatrix(size, size);
    for (int i = 0; i < size; ++i)
        EigenMatrix.row(i) = Eigen::VectorXf::Map(&matrix[i][0], size);
    return EigenMatrix;
}

/**
 * Update the Matrices B and D
 * @param matrixB
 * @param matrixD
 * @param size: matrices orders
 * @param eigenSolver: obj with eigenValues and eigenVectors
 */
void updateBandD(float **matrixB, float **matrixD, int size, EigenSolver<MatrixXf> eigenSolver){
    MatrixXcf eigenMatrixcD = eigenSolver.eigenvalues().asDiagonal();
    MatrixXcf eigenMatrixcB = eigenSolver.eigenvectors();
//    cout << eigenMatrixcD << endl;
//    cout << endl;
//    cout << eigenMatrixcB << endl;
    complex<float> complexNum;
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
 * @return float random number
 */
float randNormalNumber (float mean, float stdDev){
//    srand (static_cast <unsigned> (time(0))); // to not generate the same values
//    return lowerBound + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(upperBound-lowerBound)));
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::normal_distribution<float> distribution(mean,stdDev);
    return distribution(generator);
}

/**
 * Generate One Number With the Uniform Distribution
 * @param lowerBound
 * @param upperBound
 * @return float random number
 */
float randUniformNumber (float lowerBound, float upperBound){
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::uniform_real_distribution<float> distribution(lowerBound,upperBound);
    return distribution(generator);
}

int main()
{
    // -------------------- Initialization --------------------------------

    //----- User defined input parameters (need to be edited)
    //
    char strfitnessfct[] = "felli"; // name of objective/fitness function
    int N = 10; // number of objective variables/problem dimension
    float xmean[N]; // objective variables initial point
    float zmean[N];
    float auxSum;
    float sigma = 0.5; // coordinate wise standard deviation (step-size)
    float stopfitness = 1e-10; // stop if fitness < stopfitness (minimization)
    float stopeval = 1e3*pow(N,2); // stop after stopeval number of function evaluations

    //xmean
    float lowerBound, upperBound;
    lowerBound = 0.0;
    upperBound = 1.0;
    for (int i=0; i<N; i++){
//        xmean[i] = randUniformNumber(lowerBound, upperBound);
//        std::this_thread::sleep_for (std::chrono::nanoseconds(1));
        xmean[i] = 0.1; //for test
    }


    //----- Strategy parameter setting: Selection
    //
    int lambda = 4+floor(3.0*log(N)); // population size, offspring number
    float mufloat = lambda/2.0; // lambda=12; mu=3; weights = ones(mu,1); would be (3_I,12)-ES
    int mu = floor(mufloat); // number of parents/points for recombination

    // muXone recombination weights
    float weights[mu];
    float logs[mu];
    float oneLog;
    oneLog = log(mufloat+1.0/2.0);
    for (int i=0; i<mu;i++){
        weights[i] = oneLog - log(i+1);
    }

    // normalize recombination weights array
    float sumWeights;
    sumVector(weights, &sumWeights, SIZEVECT(weights));
    float sumOfQuad = 0.0;
    for (int i=0; i<SIZEVECT(weights); i++){
        weights[i] /= sumWeights;
        sumOfQuad += pow(weights[i],2);
    }
    sumVector(weights, &sumWeights, SIZEVECT(weights)); // sumWeights of new weights
    float mueff= pow(sumWeights,2)/sumOfQuad; // variance-effective size of mu


    //----- Strategy parameter setting: Adaptation
    //
    float cc = (4.0+mueff/N) / (N+4.0 + 2.0*mueff/N); // time constant for cumulation for C
    float cs = (mueff+2)/(N+mueff+5.0); // t-const for cumulation for sigma control
    float c1 = 2.0 / (pow((N+1.3),2)+mueff); // learning rate for rank-one update of C
    float cmu = 2.0 * (mueff-2.0+1.0/mueff) / (pow((N+2.0),2)+2.0*mueff/2.0); // and for rank-mu update
    float damps = 1.0 + 2.0*max(0.0, sqrt((mueff-1.0)/(N+1.0))-1.0) + cs; // damping for sigma

    //------ Initialize dynamic (internal) strategy parameters and constants
    //
    // evolution paths for C and sigma
    float pc[N];
    float ps[N];
    for (int i=0; i<N; i++){
        pc[i] = 0.0;
        ps[i] = 0.0;
    }

    // B defines the coordinate system
    // diagonal matrix D defines the scaling
    // covariance matrix
    float **B;
    float **D;
    float **BxD;
    float **BxDTransp;
    float **C;
    allocPointerOfPointer(B,N,N);
    allocPointerOfPointer(D,N,N);
    allocPointerOfPointer(BxD,N,N);
    allocPointerOfPointer(BxDTransp,N,N);
    allocPointerOfPointer(C,N,N);

    identityMatrix(B,N,N);
    identityMatrix(D,N,N);

    // auxiliary variable
    float sumprod;

    // B*D
    multMatrix(B,D,BxD,N,N);

    // (B*D)'
    transpMatrix(BxD,BxDTransp,N,N);

    // C=(B*D)*(B*D)'
    multMatrix(BxD,BxDTransp,C,N,N);

    float eigeneval = 0.0; // B and D updated at counteval == 0
    float chiN=pow(N,0.5)*(1.0-1.0/(4.0*N)+1.0/(21.0*pow(N,2))); // expectation of ||N(0,I)|| == norm(randn(N,1))


    // -------------------- Generation Loop --------------------------------
    //TESTE PRINTANDO MATRIZES
//    printMatrix(B, N, N);
//    printMatrix(D, N, N);
//    printMatrix(C, N, N);
    //FIM TESTE

    int counteval = 0; // the next 40 lines contain the 20 lines of interesting code
    float **arz;// standard normally distributed vectors
    float **arx;// add mutation // Eq. 40
    float **BxDxarz;
    float **BxDxArz;
    float **CmuxBxDxArzxWeig;
    float **BxDxArzTransp;
    float **plusRankMiUpd;
    float **extProdPc;

    allocPointerOfPointer(arz,N,lambda);
    allocPointerOfPointer(arx,N,lambda);
    allocPointerOfPointer(BxDxarz,N,lambda);
    allocPointerOfPointer(BxDxArz,N,mu);
    allocPointerOfPointer(CmuxBxDxArzxWeig,N,mu);
    allocPointerOfPointer(BxDxArzTransp,mu,N);
    allocPointerOfPointer(plusRankMiUpd,N,N);
    allocPointerOfPointer(extProdPc,N,N);

    float arfitness[lambda];// fitness of individuals
    float individualForTest[N];
    int arindex[lambda];
    float auxArfitness[lambda];
    float distMean = 0.0;
    float distDev = 1.0;

    while (counteval < stopeval){
        // Generate and evaluate lambda offspring
        for (int i=0; i<lambda; i++){
            // standard normally distributed vector
            for (int j=0; j<N; j++){
//                arz[j][i] = randNormalNumber(distMean, distDev);
//                std::this_thread::sleep_for (std::chrono::nanoseconds(1));
                arz[j][i] = 0.1; //for test
            }
        }


        multMatrix(BxD,arz,BxDxarz,N,lambda);

        for (int i=0; i<lambda; i++){
            for (int j=0; j<N; j++){
                arx[j][i] = xmean[j] + sigma * (BxDxarz[i][j]); // add mutation // Eq. 40
                individualForTest[j] = arx[j][i];
            }
            //verificar strfitnessfct para disparar a função desejada
            arfitness[i] = feval(strfitnessfct, individualForTest, SIZEVECT(individualForTest)); // objective function call
            counteval += 1;
        }

        //----- Sort by fitness and compute weighted mean into xmean
        //
        for (int i=0; i<SIZEVECT(arfitness); i++) {
            auxArfitness[i] = arfitness[i];
        }
        int idxArindex = 0;
        sort(auxArfitness, auxArfitness + lambda); // minimization
        for (int i=0; i<SIZEVECT(auxArfitness); i++){
            for (int j=0; j<lambda; j++){
                if(auxArfitness[i] == arfitness[j]){
                    arindex[idxArindex] = j;
                    idxArindex += 1;
                    break;
                }
            }
        }

        // recombination // Eq. 42
        // == D^-1*B'*(xmean-xold)/sigma
        float auxSumX;
        float auxSumZ;

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
        float auxValue = sqrt(cs*(2.0-cs)*mueff);
        float BxZmeanxAux[N];
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
        float hsig;
        float noma = euclideanNorm(ps,SIZEVECT(ps));
        auxValue = euclideanNorm(ps,SIZEVECT(ps))/sqrt(1.0-pow((1.0-cs),(2.0*counteval/lambda)))/chiN;
        if (auxValue <= (1.4+2/(N+1.0))){
            hsig = 1.0;
        } else {
            hsig = 0.0;
        }

        // pc (Eq. 45)
        // hsig * sqrt(cc*(2-cc)*mueff) * (B*D*zmean)
        auxValue = sqrt(cc*(2.0-cc)*mueff);
        float BxDxZmeanxAuxxHsig[N];
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
        float regOldC[N][N];
        for (int i=0; i<N; i++){
            for (int j=0; j<N; j++){
                regOldC[i][j] = (1.0-c1-cmu) * C[i][j];
            }
        }

        // pc*pc'
        externProd(pc, extProdPc, SIZEVECT(pc));

        // minor correction
        auxValue = (1.0-hsig) * cc*(2.0-cc);
        float minorCor[N][N];
        for (int i=0; i<N; i++){
            for (int j=0; j<N; j++){
                minorCor[i][j] = auxValue * C[i][j];
                //cout << minorCor[i][j] << endl;
            }
        }

        // (pc*pc') + (minor correction)
        float sumPcMinor[N][N];
        for (int i=0; i<N; i++){
            for (int j=0; j<N; j++){
                sumPcMinor[i][j] = extProdPc[i][j] + minorCor[i][j];
            }
        }

        // plus rank one update
        float plusRankOneUp[N][N];
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

//            printMatrix(C,N,N);
            MatrixXf eigenMatrixC = convertToEigenMatrix(C,N); //converting C[][] to EigenMatrix
//            cout << eigenMatrixC << endl;
//            cout << endl;
            EigenSolver<MatrixXf> eigenSolver(eigenMatrixC);
            updateBandD(B,D,N,eigenSolver);
//            printMatrix(D,N,N);
//            printMatrix(B,N,N);
        }

        //----- Break, if fitness is good enough
        if (auxArfitness[0] <= stopfitness){
            break;
        }

        //----- Escape flat fitness, or better terminate?
        int flatFitness = floorf(0.7*lambda);
        if (auxArfitness[0] == auxArfitness[flatFitness]){
            sigma = sigma * exp(0.2+cs/damps);
            cout << "warning: flat fitness, consider reformulating the objective" << endl;
        }

        cout << counteval << ": " << auxArfitness[0] << endl;

    }


    // -------------------- Final Message ---------------------------------
    cout << counteval << ": " << auxArfitness[0] << endl;

    /*
     * Return best point of last generation.
     * Notice that xmean is expected to be even
     * better.
    */
    //xmin = arx(:, arindex(1));
    float xmin[N];
    for (int i=0; i<N; i++){
        xmin[i] = arx[i][arindex[0]];
    }
    cout << "The best individual:" << endl;
    printVector(xmin,SIZEVECT(xmin));

    //free memory
    free(B);
    free(D);
    free(BxD);
    free(BxDTransp);
    free(C);
    free(arx);
    free(arz);
    free(BxDxArz);
    free(CmuxBxDxArzxWeig);
    free(BxDxArzTransp);
    free(plusRankMiUpd);
    free(extProdPc);
    free(BxDxarz);

    return 0;
}