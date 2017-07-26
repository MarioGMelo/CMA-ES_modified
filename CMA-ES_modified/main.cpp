#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <math.h>
#include <numeric>
//#include <Eigen/Eigenvalues>
#include <ObjectiveFunction.h>

#define MAX 20

using namespace std;

double feval (char strfitnessfct[], double individual[]){
    int idx = sizeof(individual);
    double result;
    double auxVector[idx];
    if (sizeof(individual) < 2){
        cout << "dimension must be greater one" << endl;
        return 0.0;
    }
    for (int i=0; i<idx; i++){
        auxVector[i] = pow(1e6,(i/idx-1));// condition number 1e6
    }

    for(int i=0; i<idx; i++){
        result += auxVector[i]*(pow(individual[i],2));
        //BxD[l][c]=sumprod;
    }
    return result;
}

double euclidianNorm (double vectorA[]){
    double norm=0;
    int idx = sizeof(vectorA);
    for (int i=0; i<idx; i++){
        norm += pow(i,2);
    }
    return sqrt(norm);
}

/*
double[][] externProd (double vectorA[]){
    double matrix[sizeof(vectorA)][sizeof(vectorA)];
    for (int i=0; i<sizeof(vectorA); i++){
        for (int j=0; j<sizeof(vectorA); j++){
            matrix[i][j] = vectorA[i]*vectorA[j];
        }
    }
    return matrix;
}
*/

void externProd (double* vectorA, double** matrix){
    for (int i=0; i<sizeof(vectorA); i++){
        for (int j=0; j<sizeof(vectorA); j++){
            matrix[i][j] = vectorA[i]*vectorA[j];
        }
    }
}

int main()
{
    // -------------------- Initialization --------------------------------

    //----- User defined input parameters (need to be edited)
    //
    char strfitnessfct[] = "felli"; // name of objective/fitness function
    int N = 10; // number of objective variables/problem dimension
    double xmean[N]; // objective variables initial point
    double sigma = 0.5; // coordinate wise standard deviation (step-size)
    double stopfitness = 1e-10; // stop if fitness < stopfitness (minimization)
    double stopeval = 1e3*(N^2); // stop after stopeval number of function evaluations

    //xmean
    double val;
    for (int i=0; i<N; i++){
        val = rand() % 101;
        xmean[i] = val/100; // between 0 and 1
    }





    //----- Strategy parameter setting: Selection
    //
    int lambda = 4+floor(3*log(N)); // population size, offspring number
    double muDouble = lambda/2; // lambda=12; mu=3; weights = ones(mu,1); would be (3_I,12)-ES
    int mu = floor(muDouble); // number of parents/points for recombination

    // muXone recombination weights
    double weights[mu];
    double logs[mu];
    double oneLog;
    oneLog = muDouble+1/2;
    for (int i=0; i<mu;i++){
        weights[i] = oneLog - log(i+1);
    }



    // normalize recombination weights array
    double sumWeights =  accumulate(weights[0], weights[sizeof(weights)], 0);
    double sumOfQuad = 0;
    for (int i=0; i<sizeof(weights); i++){
        weights[i] /= sumWeights;
        sumOfQuad += pow(weights[i],2);
    }
    double mueff= pow(sumWeights,2)/sumOfQuad; // variance-effective size of mu





    //----- Strategy parameter setting: Adaptation
    //
    double cc = (4+mueff/N) / (N+4 + 2*mueff/N); // time constant for cumulation for C
    double cs = (mueff+2)/(N+mueff+5); // t-const for cumulation for sigma control
    double c1 = 2 / (pow((N+1.3),2)+mueff); // learning rate for rank-one update of C
    double cmu = 2 * (mueff-2+1/mueff) / (pow((N+2),2)+2*mueff/2); // and for rank-mu update
    double damps = 1 + 2*max(0.0, sqrt((mueff-1)/(N+1))-1) + cs; // damping for sigma





    //------ Initialize dynamic (internal) strategy parameters and constants
    //
    // evolution paths for C and sigma
    double pc[N];
    double ps[N];
    for (int i=0; i<N; i++){
        pc[i] = 0;
        ps[i] = 0;
    }

    // B defines the coordinate system
    // diagonal matrix D defines the scaling
    double B[N][N];
    double D[N][N];
    for (int l=0; l<N; l++){
        for (int c=0; c<N; c++){
            if (l==c){
                B[l][c] = 1;
                D[l][c] = 1;
            } else{
                B[l][c] = 0;
                D[l][c] = 0;
            }
        }
    }

    // covariance matrix
    double BxD[N][N];
    double BxDTransp[N][N];
    double C[N][N];
    double sumprod;

    // B*D
    for(int l=0; l<N; l++){
        for(int c=0; c<N; c++){
            sumprod=0;
            for(int i=0; i<N; i++){
                sumprod+=B[l][i]*D[i][c];
                BxD[l][c]=sumprod; // B*D
            }
        }
    }

    // (B*D)'
    for (int l=0; l<N; l++){
        for (int c=0; c<N; c++){
            BxDTransp[c][l] = BxD[l][c];
        }
    }

    // C=(B*D)*(B*D)'
    for(int l=0; l<N; l++){
        for(int c=0; c<N; c++){
            sumprod=0;
            for(int i=0; i<N; i++){
                sumprod+=BxD[l][i]*BxDTransp[i][c];
            }
            C[l][c]=sumprod;
        }
    }

    double eigeneval = 0; // B and D updated at counteval == 0
    double chiN=pow(N,0.5)*(1-1/(4*N)+1/(21*N^2)); // expectation of
    // ||N(0,I)|| == norm(randn(N,1))





    // -------------------- Generation Loop --------------------------------

    int counteval = 0; // the next 40 lines contain the 20 lines of interesting code
    double arz[N][lambda];// standard normally distributed vectors
    double arx[N][lambda];// add mutation // Eq. 40
    double arfitness[lambda];// fitness of individuals
    double individualForTest[N];
    while (counteval < stopeval){
        // Generate and evaluate lambda offspring
        for (int k=0; k<lambda; k++){
            // standard normally distributed vector
            for (int i=0; i<N; i++){
                val = rand() % 101;
                arz[i][k] = val/100; // between 0 and 1

                arx[i][k] = xmean[i] + sigma * (BxD * arz[i][k]); // add mutation // Eq. 40
                individualForTest[i] = arx[i][k];
            }
            /*
            verificar strfitnessfct para disparar a função desejada
            */
            arfitness[k] = feval(strfitnessfct, individualForTest); // objective function call
            counteval += 1;
        }





        //----- Sort by fitness and compute weighted mean into xmean
        //
        double auxArfitness[lambda] = arfitness;
        int arindex[lambda];
        int idxArindex = 0;
        sort(auxArfitness, auxArfitness + lambda); // minimization
        for (double fit:auxArfitness){
            for (int i=0; i<lambda; i++){
                if(fit == arfitness[i]){
                    arindex[idxArindex] = i;
                    idxArindex += 1;
                    break;
                }
            }
        }

        // recombination // Eq. 42
        // == D^-1*B'*(xmean-xold)/sigma
        double auxSumX;
        double auxSumZ;
        for (int i=0; i<N; i++){
            auxSumX = 0;
            auxSumZ = 0;
            for (int j=0; j<mu; j++){
                auxSumX += arx[i][arindex[j]]*weights[j]; // recombination // Eq. 42
                auxSumZ += arz[i][arindex[j]]*weights[j]; // == D^-1*B'*(xmean-xold)/sigma
            }
            xmean[i] = auxSumX;
            zmean[i] = auxSumZ;
        }





        //----- Cumulation: Update evolution paths
        //
        // ps (Eq. 43)
        // (sqrt(cs*(2-cs)*mueff)) * (B * zmean)
        double auxValue = sqrt(cs*(2-cs)*mueff);
        double BxZmeanxAux[N];
        for (int i=0; i<N; i++){
            auxSum = 0;
            for (int j=0; j<N; j++){
                auxSum += B[i][j]*zmean[j]*auxValue;
            }
            BxZmeanxAux[i] = auxSum;
        }
        //
        // ps = [(1-cs)*ps] + [(sqrt(cs*(2-cs)*mueff)) * (B * zmean)]
        for (int i=0; i<sizeof(ps); i++){
            ps[i] = (1-cs)*ps[i];
            ps[i] += BxZmeanxAux[i];
        }

        // hsig = norm(ps)/sqrt(1-(1-cs)^(2*counteval/lambda))/chiN < 1.4+2/(N+1)
        double hsig;
        auxValue = euclidianNorm(ps)/sqrt(1-(1-cs)^(2*counteval/lambda))/chiN;
        if (auxValue <= (1.4+2/(N+1)){
            hsig = 1;
        } else {
            hsig = 0;
        }

        // pc (Eq. 45)
        // hsig * sqrt(cc*(2-cc)*mueff) * (B*D*zmean)
        auxValue = sqrt(cc*(2-cc)*mueff);
        double BxDxZmeanxAuxxHsig[N];
        for (int i=0; i<N; i++){
            auxSum = 0;
            for (int j=0; j<N; j++){
                auxSum += BxD[i][j]*zmean[j]*auxValue*hsig;
            }
            BxDxZmeanxAuxxHsig[i] = auxSum;
        }
        //
        // pc = (1-cc)*pc + hsig * sqrt(cc*(2-cc)*mueff) * (B*D*zmean)
        for (int i=0; i<sizeof(pc); i++){
            pc[i] = (1-cc)*pc[i];
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
        double regOldC=[N][N];
        for (int i=0; i<N; i++){
            for (int j=0; j<N; j++){
                regOldC[i][j] = (1-c1-cmu) * C[i][j];
            }
        }

        // pc*pc'
        //double extProdPc[N][N] = externProd(pc);
        double extProdPc[N][N];
        externProd(pc, extProdPc);

        // minor correction
        auxValue = (1-hsig) * cc*(2-cc);
        double minorCor[N][N];
        for (int i=0; i<N; i++){
            for (int j=0; j<N; j++){
                minorCor[i][j] = auxValue * C[i][j];
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
        double BxDxArz[N][mu];
        for(int l=0; l<N; l++){
            for(int c=0; c<mu; c++){
                sumprod=0;
                for(int i=0; i<N; i++){
                    sumprod += BxD[l][i]*arz[i][c];
                }
                BxDxArz[l][c]=sumprod;
            }
        }

        // cmu*(B*D*arz(:,arindex(1:mu)))*diag(weights)
        double CmuxBxDxArzxWeig[N][mu];
        for (int i=0; i<N; i++){
            for (int j=0; j<mu; j++){
                CmuxBxDxArzxWeig[i][j] = cmu * BxDxArz[i][j] * weights[i];
            }
        }

        // (B*D*arz(:,arindex(1:mu)))'
        double BxDxArzTransp[mu][N];
        for (int i=0; i<N; i++){
            for (int j=0; j<mu; j++){
                BxDxArzTransp[j][i] = BxDxArz[i][j];
            }
        }

        // cmu * (B*D*arz(:,arindex(1:mu))) * diag(weights) * (B*D*arz(:,arindex(1:mu)))'
        double plusRankMiUpd[N][N];
        for(int l=0; l<N; l++){
            for(int c=0; c<N; c++){
                sumprod=0;
                for(int i=0; i<mu; i++){
                    sumprod += CmuxBxDxArzxWeig[l][i]*BxDxArzTransp[i][c];
                }
                plusRankMiUpd[l][c]=sumprod;
            }
        }





        //----- Adapt step-size sigma
        //
        sigma = sigma * exp((cs/damps)*(euclidianNorm(ps)/chiN - 1)); // Eq. 44





        //PAREI AKIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
        //----- Update B and D from C
        //
//        if ((counteval - eigeneval) > (lambda/(c1+cmu)/N/10)){ // to achieve O(N^2)
//            eigeneval = counteval;
//
//            // enforce symmetry
//            // C=triu(C)+triu(C,1)'
//            for (int i=0; i<N; i++){
//                for (int j=i; j<N; j++){
//                    C[j][i] = C[i][j];
//                }
//            }
//
//            // eigen decomposition, B==normalized eigenvectors
//            //[B,D] = eig(C)
//
//
//
//            D = diag(sqrt(diag(D))); // D contains standard deviations now
//        }
//
//
//
//
//
//
//        //----- Break, if fitness is good enough
//        //
//        if (arfitness[1] <= stopfitness){
//            break;
//        }
//
//
//
//
//
//
//        //----- Escape flat fitness, or better terminate?
//        //
//        if (arfitness[1] == arfitness[ceil(0.7*lambda)]){
//            sigma = sigma * exp(0.2+cs/damps);
//            cout << "warning: flat fitness, consider reformulating the objective" << endl;
//        }
//
//        disp([num2str(counteval) ': ' num2str(arfitness(1))]);

    }



    // -------------------- Final Message ---------------------------------

    disp([num2str(counteval) ': ' num2str(arfitness(1))]);
    xmin = arx(:, arindex(1)); // Return best point of last generation.
    // Notice that xmean is expected to be even
    // better.
}
