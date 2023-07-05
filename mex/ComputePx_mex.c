#include "mex.h"
#include "stdlib.h"
/*=================================================================
% function Px = ComputePx_mex(d,n,r,v1,v2,...,vd,SizeOmega,Omega)
% This function computes the tensor values on Omega with tensor ring decomposition.
% d: int; n: d-by-1 uint32, r: (d+1)-by-1, r[0]=r[d] uint 32
% v1, v2, ..., vd: r(k-1)*r(k)*n(k) double vector
% SizeOmega: uint32
% Omega: d-by-SizeOmega uint32
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
    int d, i, j, r0r1, r0r2;
    uint32_T *n, *r, SizeOmega, *Omega;
    double *Px, *temp1, *temp2, *temp12, *tempu1, *tempu2, *tempu3, *tempu4, *u1, *u2, *u3, *u4;
    d=mxGetScalar(prhs[0]);

    if (d>4)
        mexErrMsgTxt("ComputePx_mex is not available for d>4");
    if (nrhs-d!=5)
        mexErrMsgTxt("The dimension must be consistent with the number of cores. ");
    if (nlhs!=1)
        mexErrMsgTxt("Only a single output is accepted!");

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
    }
    else
    {
        SizeOmega=mxGetScalar(prhs[6]);
        Omega=(uint32_T*)mxGetPr(prhs[7]);
    }

    // Compute Px
    plhs[0]=mxCreateDoubleMatrix(SizeOmega, 1, mxREAL);
    Px=mxGetPr(plhs[0]);

    r0r1=r[0]*r[1];
    r0r2=r[0]*r[2];
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
            temp1=transpose(tempu1,r[0],r[1]);
            temp2=matmul(tempu2,tempu3,r[1],r[2],r[3]);

            for(j=0; j<r0r1; j++)
                *(Px+i)+=(*(temp1+j))*(*(temp2+j));
        }
        else
        {
            tempu4=getCoreMatrix(u4,*(Omega+i*d+3),r[3],r[4]);
            temp12=matmul(tempu1,tempu2,r[0],r[1],r[2]);
            temp1=transpose(temp12,r[0],r[2]);
            temp2=matmul(tempu3,tempu4,r[2],r[3],r[4]);

            for(j=0; j<r0r2; j++)
                *(Px+i)+=(*(temp1+j))*(*(temp2+j));

            free(tempu4);
            free(temp12);
        }
        free(tempu1);
        free(tempu2);
        free(tempu3);
        free(temp1);
        free(temp2);
    }



//    mexPrintf("Hello World\n");









}