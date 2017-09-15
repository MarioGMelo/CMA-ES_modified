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
double felliFunc (VectorXd individual){
    double result = 0.0;
    double auxSize = individual.size(); //for double division
    VectorXd auxVector(individual.size());
    if (individual.size() < 2){
        cout << "dimension must be greater one" << endl;
        return result;
    }
    for (int i=0; i<individual.size(); i++){
//        cout << pow(1e6,(i/(auxSize-1.0))) << endl;
        auxVector[i] = pow(1e6,(i/(auxSize-1.0)));// condition number 1e6
//        auxVector[i] = (i/(auxSize-1.0));//OTHER FUNCTION TO TEST
    }

    for(int i=0; i<individual.size(); i++){
        result += auxVector[i]*(pow(individual[i],2));
//        result += individual[i];//OTHER FUNCTION TEST
    }
    return result;
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
    VectorXd xmean(N); // objective variables initial point
    VectorXd zmean(N);
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
    VectorXd weights(mu);
    double oneLog;
    oneLog = log(mudouble+1.0/2.0);
    for (int i=0; i<mu;i++){
        weights[i] = oneLog - log(i+1);
    }

    // normalize recombination weights array
    double sumWeights = weights.sum();
    double sumOfQuad = 0.0;
    weights /= sumWeights;
    for (int i=0; i<weights.size(); i++){
        sumOfQuad += pow(weights[i],2);
    }
    sumWeights = weights.sum(); // sumWeights of new weights
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
    VectorXd pc(N);
    VectorXd ps(N);
    pc.setZero();
    ps.setZero();

    // B defines the coordinate system
    // diagonal matrix D defines the scaling
    // covariance matrix
    MatrixXd B(N,N);
    MatrixXd D(N,N);
    //MatrixXd BxD(N,N); //VER SE EH NECESSAHRIO
    MatrixXd C(N,N);

    B.setIdentity();
    D.setIdentity();

    // auxiliary variable
    double sumprod;

    // B*D
    //BxD = B*D;

    // (B*D)'
    //transpMatrix(BxD,BxDTransp,N,N);

    // C=(B*D)*(B*D)'
    C = B * D * (B*D).transpose();

    double eigeneval = 0.0; // B and D updated at counteval == 0
    double chiN=pow(N,0.5)*(1.0-1.0/(4.0*N)+1.0/(21.0*pow(N,2))); // expectation of ||N(0,I)|| == norm(randn(N,1))


    // -------------------- Generation Loop --------------------------------

    int counteval = 0; // the next 40 lines contain the 20 lines of interesting code
    MatrixXd arz(N,lambda);// standard normally distributed vectors
    MatrixXd arx(N,lambda);// add mutation // Eq. 40

    //VER SE PRECISA ABAIXO
//    double **plusRankMiUpd;
//    double **extProdPc;
//    plusRankMiUpd = allocPointerOfPointer(N,N);
//    extProdPc = allocPointerOfPointer(N,N);

    VectorXd arfitness(lambda);// fitness of individuals
    VectorXd individualForTest(N);
    VectorXi arindex(lambda);
    VectorXd sortArfitness(lambda);
    double distMean = 0.0;
    double distDev = 1.0;

//    cout << "INICIAR WHILE: " << endl;
    while (counteval < stopeval){

        // Generate and evaluate lambda offspring
        for (int i=0; i<lambda; i++){
            // standard normally distributed vector
            for (int j=0; j<N; j++){
                arz(j,i) = randNormalNumber(distMean, distDev);
                std::this_thread::sleep_for (std::chrono::nanoseconds(1));
//                arz[j][i] = 0.1; //for test
            }
        }

        for (int i=0; i<lambda; i++){
            arx.col(i) = xmean + sigma * ((B*D*arz).col(i));
            arfitness[i] = felliFunc(arx.col(i)); // objective function call
            counteval += 1;
        }

        //----- Sort by fitness and compute weighted mean into xmean
        //
        for (int i=0; i<arfitness.size(); i++) {
            sortArfitness[i] = arfitness[i];
        }
        std::sort(sortArfitness.data(), sortArfitness.data()+sortArfitness.size()); // minimization
        for (int i=0; i<sortArfitness.size(); i++){
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
                auxSumX += arx(i,arindex[j])*weights[j]; // recombination // Eq. 42
                auxSumZ += arz(i,arindex[j])*weights[j]; // == D^-1*B'*(xmean-xold)/sigma
            }
            xmean[i] = auxSumX;
            zmean[i] = auxSumZ;
        }

//        cout << "XMEAN:" << endl;
//        printVector(xmean,N);

        //----- Cumulation: Update evolution paths
        ps = (1-cs)*ps + (sqrt(cs*(2-cs)*mueff)) * (B * zmean); // Eq. 43

        // hsig = norm(ps)/sqrt(1-(1-cs)^(2*counteval/lambda))/chiN < 1.4+2/(N+1)
        double hsig;
        double auxValue;
        auxValue = ps.norm()/sqrt(1.0-pow((1.0-cs),(2.0*counteval/lambda)))/chiN;
        if (auxValue < (1.4+2/(N+1.0))){
            hsig = 1.0;
        } else {
            hsig = 0.0;
        }

        pc = (1-cc)*pc + hsig * sqrt(cc*(2-cc)*mueff) * (B*D*zmean); // Eq. 45


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

        MatrixXd arzMu(N,mu);
        for (int i=0; i<mu; i++){
            arzMu.col(i) = arz.col(arindex[i]);
        }

        C = (1-c1-cmu) * C + c1 * (pc*pc.transpose() + (1-hsig) * cc*(2-cc) * C) + cmu * (B*D*arzMu) * weights.asDiagonal() * (B*D*arzMu).transpose();


        //----- Adapt step-size sigma
        //
        sigma = sigma * exp((cs/damps)*(ps.norm()/chiN - 1.0)); // Eq. 44


        //----- Update B and D from C
        //
        if ((counteval - eigeneval) > (lambda/(c1+cmu)/N/10)){ // to achieve O(N^2)
            eigeneval = counteval;

            // enforce symmetry
            // C=triu(C)+triu(C,1)'
            for (int i=0; i<N; i++){
                for (int j=0; j<i; j++){
                    C(i,j) = C(j,i);
                }
            }

            EigenSolver<MatrixXd> eigenSolver(C);
            B = eigenSolver.eigenvectors().real();
            D = eigenSolver.eigenvalues().real().asDiagonal();
            for(int i=0; i< D.rows(); i++){
                for (int j=0; j<D.cols(); j++){
                    if (i==j){
                        D(i,j) = sqrt(D(i,j));
                    }
                }
            }

        }

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
    VectorXd xmin(N);
    xmin = arx.col(0);
    cout << "The best individual:" << endl;
    cout << xmin << endl;

    return 0;
}