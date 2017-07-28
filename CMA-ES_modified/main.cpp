#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <math.h>
#include <numeric>
//#include <Eigen/Eigenvalues>
#include <ObjectiveFunction.h>

#define SIZEVECT(vect) (sizeof(vect)/sizeof((vect)[0]))

using namespace std;

float feval (char strfitnessfct[], float individual[], int sizeIndividual){
    float result;
    float auxVector[sizeIndividual];
    if (SIZEVECT(individual) < 2){
        cout << "dimension must be greater one" << endl;
        return 0.0;
    }
    for (int i=0; i<sizeIndividual; i++){
        auxVector[i] = pow(1e6,(i/sizeIndividual-1));// condition number 1e6
    }

    for(int i=0; i<sizeIndividual; i++){
        result += auxVector[i]*(pow(individual[i],2));
        //BxD[l][c]=sumprod;
    }
    return result;
}

float euclidianNorm (float vectorA[], int sizeVectorA){
    float norm=0;
    for (int i=0; i<sizeVectorA; i++){
        norm += pow(vectorA[i],2);
    }
    return sqrt(norm);
}

void sumVector (float *vectorA, float *sumVector, int sizeVectorA){
        *sumVector = 0.0;
    for(int i=0; i<sizeVectorA; i++){
        *sumVector += vectorA[i];
    }
}

/*
float[][] externProd (float vectorA[]){
    float matrix[SIZEVECT(vectorA)][SIZEVECT(vectorA)];
    for (int i=0; i<SIZEVECT(vectorA); i++){
        for (int j=0; j<SIZEVECT(vectorA); j++){
            matrix[i][j] = vectorA[i]*vectorA[j];
        }
    }
    return matrix;
}
*/

void externProd (float* vectorA, float** matrix, int sizeVectorA){
    for (int i=0; i<sizeVectorA; i++){
        //cout << vectorA[i] << endl;
        /*
        for (int j=0; j<sizeVectorA; j++){
            matrix[i][j] = vectorA[i]*vectorA[j];
        }
        */
    }
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
    float val;
    for (int i=0; i<N; i++){
        val = rand() % 101;
        //val = 10.0; //PARA TESTE
        xmean[i] = val/100.0; // between 0 and 1
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
    float B[N][N];
    float D[N][N];
    for (int l=0; l<N; l++){
        for (int c=0; c<N; c++){
            if (l==c){
                B[l][c] = 1.0;
                D[l][c] = 1.0;
            } else{
                B[l][c] = 0.0;
                D[l][c] = 0.0;
            }
        }
    }

    // covariance matrix
    float BxD[N][N];
    float BxDTransp[N][N];
    float C[N][N];
    float sumprod;

    // B*D
    for(int l=0; l<N; l++){
        for(int c=0; c<N; c++){
            sumprod=0.0;
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
            sumprod=0.0;
            for(int i=0; i<N; i++){
                sumprod+=BxD[l][i]*BxDTransp[i][c];
            }
            C[l][c]=sumprod;
        }
    }

    float eigeneval = 0.0; // B and D updated at counteval == 0
    float chiN=pow(N,0.5)*(1.0-1.0/(4.0*N)+1.0/(21.0*pow(N,2))); // expectation of
    // ||N(0,I)|| == norm(randn(N,1))





    // -------------------- Generation Loop --------------------------------

    int counteval = 0; // the next 40 lines contain the 20 lines of interesting code
    float arz[N][lambda];// standard normally distributed vectors
    float arx[N][lambda];// add mutation // Eq. 40
    float arfitness[lambda];// fitness of individuals
    float individualForTest[N];
    while (counteval < stopeval){
        // Generate and evaluate lambda offspring
        for (int i=0; i<lambda; i++){
            // standard normally distributed vector
            for (int j=0; j<N; j++){
                val = rand() % 101;
                //val = 10.0; //PARA TESTE
                arz[j][i] = val/100.0; // between 0 and 1

                arx[j][i] = xmean[j] + sigma * (BxD[i][j] * arz[j][i]); // add mutation // Eq. 40
                individualForTest[j] = arx[j][i];
            }
            /*
            verificar strfitnessfct para disparar a função desejada
            */
            arfitness[i] = feval(strfitnessfct, individualForTest, SIZEVECT(individualForTest)); // objective function call
            counteval += 1;
        }





        //----- Sort by fitness and compute weighted mean into xmean
        //
        float auxArfitness[lambda];
        for (int i=0; i<SIZEVECT(arfitness); i++){
            auxArfitness[i] = arfitness[i];
        }
        int arindex[lambda];
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
        float noma = euclidianNorm(ps,SIZEVECT(ps));
        auxValue = euclidianNorm(ps,SIZEVECT(ps))/sqrt(1.0-pow((1.0-cs),(2.0*counteval/lambda)))/chiN;
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
        //float extProdPc[N][N] = externProd(pc);
        float **extProdPc;
        externProd(pc, extProdPc, SIZEVECT(pc));

        // minor correction
        auxValue = (1.0-hsig) * cc*(2.0-cc);
        float minorCor[N][N];
        for (int i=0; i<N; i++){
            for (int j=0; j<N; j++){
                minorCor[i][j] = auxValue * C[i][j];
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
        float BxDxArz[N][mu];
        for(int l=0; l<N; l++){
            for(int c=0; c<mu; c++){
                sumprod=0.0;
                for(int i=0; i<N; i++){
                    sumprod += BxD[l][i]*arz[i][c];
                }
                BxDxArz[l][c]=sumprod;
            }
        }

        // cmu*(B*D*arz(:,arindex(1:mu)))*diag(weights)
        float CmuxBxDxArzxWeig[N][mu];
        for (int i=0; i<N; i++){
            for (int j=0; j<mu; j++){
                CmuxBxDxArzxWeig[i][j] = cmu * BxDxArz[i][j] * weights[i];
            }
        }

        // (B*D*arz(:,arindex(1:mu)))'
        float BxDxArzTransp[mu][N];
        for (int i=0; i<N; i++){
            for (int j=0; j<mu; j++){
                BxDxArzTransp[j][i] = BxDxArz[i][j];
            }
        }

        // cmu * (B*D*arz(:,arindex(1:mu))) * diag(weights) * (B*D*arz(:,arindex(1:mu)))'
        float plusRankMiUpd[N][N];
        for(int l=0; l<N; l++){
            for(int c=0; c<N; c++){
                sumprod=0.0;
                for(int i=0; i<mu; i++){
                    sumprod += CmuxBxDxArzxWeig[l][i]*BxDxArzTransp[i][c];
                }
                plusRankMiUpd[l][c]=sumprod;
            }
        }





        //----- Adapt step-size sigma
        //
        sigma = sigma * exp((cs/damps)*(euclidianNorm(ps,SIZEVECT(ps))/chiN - 1.0)); // Eq. 44





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

    //disp([num2str(counteval) ': ' num2str(arfitness(1))]);
    //min = arx(:, arindex(1)); // Return best point of last generation.
    // Notice that xmean is expected to be even
    // better.

    return 0;
}
