#include "mex.h"
#include "stdlib.h"
/*=================================================================
% function [Px,G1,G2,...,Gd] = ComputePx_mex(d,n,r,v1,v2,...,vd,SizeOmega,Omega,p,PA)
% This function computes the tensor values on Omega with tensor ring decomposition, and the Euclidean gradients.
% d: int; n: d-by-1 uint32, r: (d+1)-by-1, r[0]=r[d] uint32
% v1, v2, ..., vd: r(k-1)*r(k)*n(k) double vector
% SizeOmega: uint32
% Omega: d-by-SizeOmega uint32
% p: double
% PA: SizeOmega-by-1 double
%
% Original author: Renfeng Peng, May 26th, 2023.
 *=================================================================*/

// Column-major as in Matlab
//#define mat(A,i,j) *(A+i+j*N)
double* matmul(double* A, double* B, uint32_T m, uint32_T n, uint32_T p)
{
	double* C;
	int i,j,k;
    C=(double *)malloc(m*p*sizeof(double));
	//C=(int *)malloc(N,N*sizeof(double))
	for (i=0;i<m;i++)
		for (j=0;j<p;j++)
		{
			*(C+i+j*m)=0;
			for (k=0;k<n;k++)
//				mat(C,i,j)+=mat(A,i,k)*mat(B,k,j);
                *(C+i+j*m)+=*(A+i+k*m)*(*(B+k+j*n));
		}
	return C;
}


double* transpose(double* A, int m, int n)
{
    double* transA;
    transA=(double *)malloc(m*n*sizeof(double));
	int i,j;
    for(i=0; i<m; i++)
        for(j=0; j<n; j++)
        {
            *(transA+j+i*n)=*(A+i+j*m);
        }
    return transA;
}

double* getCoreMatrix(double* tensU, int ind, int r1, int r2)
{
    double* coreMat;
    int i, r1r2=r1*r2;

    coreMat=(double *)malloc(r1*r2*sizeof(double));
    for(i=0; i<r1r2; i++)
        *(coreMat+i)=*(tensU+(ind-1)*r1r2+i);

    return coreMat;
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int d, i, j, r0r1, r1r2, r2r3, r3r4;
    uint32_T *n, *r, SizeOmega, *Omega;
    double p, *Px, *PA, *temp1, *temp2, *temp3, *tempu1, *tempu2, *tempu3, *tempu4;
    double *u1, *u2, *u3, *u4, *G1, *G2, *G3, *G4, temp_const;
    d=mxGetScalar(prhs[0]);

    if (d>4)
        mexErrMsgTxt("ComputePx_mex is not available for d>4");
    if (nrhs-d!=7)
        mexErrMsgTxt("The dimension must be consistent with the number of cores. ");
    if (nlhs!=d+1)
        mexErrMsgTxt("The number of output should be d+1!");

    n=(uint32_T*)mxGetData(prhs[1]);
    r=(uint32_T*)mxGetData(prhs[2]);

//    mexPrintf("%d %d %d %d\n",r[0],r[1],r[2],r[3]);
//    mexPrintf("%d %d %d\n",n[0],n[1],n[2]);

    if (*r!=*(r+d))
        mexErrMsgTxt("The tensor ring rank must satisfy r(1)=r(d+1)!");

    u1=mxGetPr(prhs[3]);
    u2=mxGetPr(prhs[4]);
    u3=mxGetPr(prhs[5]);
    if (d==4)
    {
        u4=mxGetPr(prhs[6]);
        SizeOmega=mxGetScalar(prhs[7]);
        Omega=(uint32_T*)mxGetPr(prhs[8]);
        p=mxGetScalar(prhs[9]);
        PA=mxGetPr(prhs[10]);
    }
    else
    {
        SizeOmega=mxGetScalar(prhs[6]);
        Omega=(uint32_T*)mxGetPr(prhs[7]);
        p=mxGetScalar(prhs[8]);
        PA=mxGetPr(prhs[9]);
    }

    // Compute Px
    r0r1=r[0]*r[1];
    r1r2=r[1]*r[2];
    r2r3=r[2]*r[3];



    plhs[0]=mxCreateDoubleMatrix(SizeOmega, 1, mxREAL);
    Px=mxGetPr(plhs[0]);
    plhs[1]=mxCreateDoubleMatrix(n[0]*r0r1, 1, mxREAL);
    G1=mxGetPr(plhs[1]);
//    for (i=0; i<n[0]*r0r1; i++) G1[i]=0;
    plhs[2]=mxCreateDoubleMatrix(n[1]*r1r2, 1, mxREAL);
    G2=mxGetPr(plhs[2]);
//    for (i=0; i<n[1]*r1r2; i++) G2[i]=0;
    plhs[3]=mxCreateDoubleMatrix(n[2]*r2r3, 1, mxREAL);
    G3=mxGetPr(plhs[3]);
//    for (i=0; i<n[2]*r2r3; i++) G3[i]=0;
    if (d==4)
    {
        r3r4=r[3]*r[4];
        plhs[4]=mxCreateDoubleMatrix(n[3]*r3r4, 1, mxREAL);
        G4=mxGetPr(plhs[4]);
//        for (i=0; i<n[3]*r3r4; i++) G4[i]=0;
    }

//    mexPrintf("%d\n",SizeOmega);
    for (i=0; i<SizeOmega; i++)
    {
        // d
        *(Px+i)=0;
//        mexPrintf("%d %d %d\n",*(Omega+i*d+0),*(Omega+i*d+1),*(Omega+i*d+2));
        tempu1=getCoreMatrix(u1,*(Omega+i*d+0),r[0],r[1]);
        tempu2=getCoreMatrix(u2,*(Omega+i*d+1),r[1],r[2]);
        tempu3=getCoreMatrix(u3,*(Omega+i*d+2),r[2],r[3]);
        if (d==3)
        {

            // Compute G1 and Px
            temp1=matmul(tempu2,tempu3,r[1],r[2],r[3]);
            temp2=transpose(temp1,r[1],r[0]);

            for (j=0; j<r0r1; j++)
                *(Px+i)+=(*(tempu1+j))*(*(temp2+j));

            temp_const=((*(Px+i))-(*(PA+i)))/p;

            for (j=0; j<r0r1; j++)
            {
                G1[*(Omega+i*d+0)-1+j*n[0]]+=temp_const*temp2[j];
//                mexPrintf("%d / %d\n",j,r0r1);
            }

            free(temp1);
            free(temp2);
            // Compute G2
            temp1=matmul(tempu3,tempu1,r[2],r[3],r[1]);
            temp2=transpose(temp1,r[2],r[1]);
            for (j=0; j<r1r2; j++)
                G2[*(Omega+i*d+1)-1+j*n[1]]+=temp_const*temp2[j];

            free(temp1);
            free(temp2);
            // Compute G3
            temp1=matmul(tempu1,tempu2,r[3],r[1],r[2]);
            temp2=transpose(temp1,r[3],r[2]);
            for (j=0; j<r2r3; j++)
                G3[*(Omega+i*d+2)-1+j*n[2]]+=temp_const*temp2[j];


            free(temp1);
            free(temp2);
        }
        else
        {
//            mexErrMsgTxt("d=4 is not prepared!");
            tempu4=getCoreMatrix(u4,*(Omega+i*d+3),r[3],r[4]);
            temp1=matmul(tempu2,tempu3,r[1],r[2],r[3]);
            temp3=matmul(temp1,tempu4,r[1],r[3],r[4]);
            temp2=transpose(temp3,r[1],r[0]);

            for (j=0; j<r0r1; j++)
                *(Px+i)+=(*(tempu1+j))*(*(temp2+j));

            temp_const=((*(Px+i))-(*(PA+i)))/p;

            for (j=0; j<r0r1; j++)
            {
                G1[*(Omega+i*d+0)-1+j*n[0]]+=temp_const*temp2[j];
//                mexPrintf("%d / %d\n",j,r0r1);
            }

            free(temp1);
            free(temp2);
            free(temp3);
            // Compute G2
            temp1=matmul(tempu3,tempu4,r[2],r[3],r[4]);
            temp3=matmul(temp1,tempu1,r[2],r[4],r[1]);
            temp2=transpose(temp3,r[2],r[1]);
            for (j=0; j<r1r2; j++)
                G2[*(Omega+i*d+1)-1+j*n[1]]+=temp_const*temp2[j];

            free(temp1);
            free(temp2);
            free(temp3);
            // Compute G3
            temp1=matmul(tempu4,tempu1,r[3],r[4],r[1]);
            temp3=matmul(temp1,tempu2,r[3],r[1],r[2]);
            temp2=transpose(temp3,r[3],r[2]);
            for (j=0; j<r2r3; j++)
                G3[*(Omega+i*d+2)-1+j*n[2]]+=temp_const*temp2[j];

            free(temp1);
            free(temp2);
            free(temp3);
            // Compute G4
            temp1=matmul(tempu1,tempu2,r[0],r[1],r[2]);
            temp3=matmul(temp1,tempu3,r[0],r[2],r[3]);
            temp2=transpose(temp3,r[0],r[3]);
            for (j=0; j<r3r4; j++)
                G4[*(Omega+i*d+3)-1+j*n[3]]+=temp_const*temp2[j];

            free(tempu4);
            free(temp1);
            free(temp2);
            free(temp3);
        }
        free(tempu1);
        free(tempu2);
        free(tempu3);
    }


//    free(u1);
//    free(u2);
//    free(u3);
//    if (d==4)
//    {
//        free(u4);
//    }

//    mexPrintf("Hello World\n");









}