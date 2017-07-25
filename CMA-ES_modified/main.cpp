#include <iostream>
#include <ObjectiveFunction.h>
#include <time.h>

using namespace std;

int main()
{
    // -------------------- Initialization --------------------------------

    //----- User defined input parameters (need to be edited)
    //
    char strfitnessfct[] = 'felli'; // name of objective/fitness function
    int N = 10; // number of objective variables/problem dimension
    double xmean[N]; // objective variables initial point
    double sigma = 0.5; // coordinate wise standard deviation (step-size)
    double stopfitness = 1e-10; // stop if fitness < stopfitness (minimization)
    double stopeval = 1e3*N^2; // stop after stopeval number of function evaluations

    //xmean
    double val;
    for (int i=0; i<N; i++){
        srand(time(NULL));
        val = rand() % 101;
        xmean[i] = val/100; // between 0 and 1
    }





    //----- Strategy parameter setting: Selection
    //
    double lambda = 4+floor(3*log(N)); // population size, offspring number
    double mu = lambda/2; // lambda=12; mu=3; weights = ones(mu,1); would be (3_I,12)-ES

    // muXone recombination weights
    double weights[mu];
    double logs[mu];
    double oneLog;
    oneLog = mu+1/2;
    for (int i=0; i<mu;i++){
        weights[i] = oneLog - log(i+1);
    }

    mu = floor(mu); // number of parents/points for recombination

    // normalize recombination weights array
    double sumWeights = sum(weights);
    double sumOfQuad = 0;
    for (int i=0; i<sizeof(weights); i++){
        weights[i] /= sumWeights;
        sumOfQuad += weights[i]^2;
    }
    double mueff=sum(weights)^2/sumOfQuad; // variance-effective size of mu





    //----- Strategy parameter setting: Adaptation
    //
    double cc = (4+mueff/N) / (N+4 + 2*mueff/N); // time constant for cumulation for C
    double cs = (mueff+2)/(N+mueff+5); // t-const for cumulation for sigma control
    double c1 = 2 / ((N+1.3)^2+mueff); // learning rate for rank-one update of C
    double cmu = 2 * (mueff-2+1/mueff) / ((N+2)^2+2*mueff/2); // and for rank-mu update
    double damps = 1 + 2*max(0, sqrt((mueff-1)/(N+1))-1) + cs; // damping for sigma





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
    int sumprod;

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
                C[l][c]=sumprod;
            }
        }
    }

    double eigeneval = 0; // B and D updated at counteval == 0
    double chiN=N^0.5*(1-1/(4*N)+1/(21*N^2)); // expectation of
    // ||N(0,I)|| == norm(randn(N,1))




    //PAREI AKIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
    // -------------------- Generation Loop --------------------------------

    counteval = 0; // the next 40 lines contain the 20 lines of interesting code
    while counteval < stopeval

    // Generate and evaluate lambda offspring
    for k=1:lambda,
    arz(:,k) = randn(N,1); // standard normally distributed vector
    arx(:,k) = xmean + sigma * (B*D * arz(:,k)); // add mutation // Eq. 40
    arfitness(k) = feval(strfitnessfct, arx(:,k)); // objective function call
    counteval = counteval+1;
    end

    // Sort by fitness and compute weighted mean into xmean
    [arfitness, arindex] = sort(arfitness); // minimization
    xmean = arx(:,arindex(1:mu))*weights; // recombination // Eq. 42
    zmean = arz(:,arindex(1:mu))*weights; // == D^-1*B'*(xmean-xold)/sigma

    // Cumulation: Update evolution paths
    ps = (1-cs)*ps + (sqrt(cs*(2-cs)*mueff)) * (B * zmean); // Eq. 43
    hsig = norm(ps)/sqrt(1-(1-cs)^(2*counteval/lambda))/chiN < 1.4+2/(N+1);
    pc = (1-cc)*pc + hsig * sqrt(cc*(2-cc)*mueff) * (B*D*zmean); // Eq. 45

    // Adapt covariance matrix C
    C = (1-c1-cmu) * C ... // regard old matrix // Eq. 47
    + c1 * (pc*pc' ... // plus rank one update
    + (1-hsig) * cc*(2-cc) * C) ... // minor correction
    + cmu ... // plus rank mu update
    * (B*D*arz(:,arindex(1:mu))) ...
    * diag(weights) * (B*D*arz(:,arindex(1:mu)))';

    // Adapt step-size sigma
    sigma = sigma * exp((cs/damps)*(norm(ps)/chiN - 1)); // Eq. 44

    // Update B and D from C
    if counteval - eigeneval > lambda/(c1+cmu)/N/10 // to achieve O(N^2)
    eigeneval = counteval;
    C=triu(C)+triu(C,1)'; // enforce symmetry
    [B,D] = eig(C); // eigen decomposition, B==normalized eigenvectors
    D = diag(sqrt(diag(D))); // D contains standard deviations now
    end

    // Break, if fitness is good enough
    if arfitness(1) <= stopfitness
    break;
    end

    // Escape flat fitness, or better terminate?
    if arfitness(1) == arfitness(ceil(0.7*lambda))
    sigma = sigma * exp(0.2+cs/damps);
    disp('warning: flat fitness, consider reformulating the objective');
    end

    disp([num2str(counteval) ': ' num2str(arfitness(1))]);

    end // while, end generation loop

    // -------------------- Final Message ---------------------------------

    disp([num2str(counteval) ': ' num2str(arfitness(1))]);
    xmin = arx(:, arindex(1)); // Return best point of last generation.
    // Notice that xmean is expected to be even
    // better.
}
