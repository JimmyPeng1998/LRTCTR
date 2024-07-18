#include "mex.h"
#include "stdlib.h"
#include "matrix.h"
/*=================================================================
% function [Px,CoefMat] = ComputePx_mex(d,n,r,v1,v2,v3,SizeOmega,Omega)
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
    int d, i, j, r0r1, r1r2, r2r3, n0r0r1, n1r1r2, startind, ind=0, columns, r0r1_r1r2_r2r3;
    uint32_T *n, *r, SizeOmega, *Omega;
    mwIndex *Coremat_i, *Coremat_j;
    double *Px, *temp1, *temp2, *temp3, *tempu1, *tempu2, *tempu3, *tempu4;
    double *u1, *u2, *u3, *Coremat, temp_const, *Coremat_data;
    d=mxGetScalar(prhs[0]);

    if (d!=3)
        mexErrMsgTxt("RGN_matrix_mex is only available for d=2");
    if (nrhs!=8)
        mexErrMsgTxt("The dimension must be consistent with the number of cores. ");
    if (nlhs!=2)
        mexErrMsgTxt("The number of output should be 2!");

    n=(uint32_T*)mxGetData(prhs[1]);
    r=(uint32_T*)mxGetData(prhs[2]);

//    mexPrintf("%d %d %d %d\n",r[0],r[1],r[2],r[3]);
//    mexPrintf("%d %d %d\n",n[0],n[1],n[2]);

    if (*r!=*(r+d))
        mexErrMsgTxt("The tensor ring rank must satisfy r(1)=r(d+1)!");

    u1=mxGetPr(prhs[3]);
    u2=mxGetPr(prhs[4]);
    u3=mxGetPr(prhs[5]);
    SizeOmega=mxGetScalar(prhs[6]);
    Omega=(uint32_T*)mxGetPr(prhs[7]);


    // Compute Px
    r0r1=r[0]*r[1];
    r1r2=r[1]*r[2];
    r2r3=r[2]*r[3];
    n0r0r1=n[0]*r0r1;
    n1r1r2=n[1]*r1r2;
    r0r1_r1r2_r2r3=r0r1+r1r2+r2r3;

    columns=(n0r0r1+n1r1r2+n[2]*r2r3);

    plhs[0]=mxCreateDoubleMatrix(SizeOmega, 1, mxREAL);
    Px=mxGetPr(plhs[0]);
//    plhs[1]=mxCreateDoubleMatrix(SizeOmega * (n[0]*r0r1+n[1]*r1r2+n[2]*r2r3), 1, mxREAL);
//    Coremat=mxGetPr(plhs[1]);

    plhs[1]=mxCreateSparse(columns, SizeOmega, SizeOmega*(r0r1_r1r2_r2r3), mxREAL);
    Coremat_data=mxGetPr(plhs[1]);
    Coremat_i=mxGetIr(plhs[1]);
    Coremat_j=mxGetJc(plhs[1]);



//    nonzeros=(int *)malloc((SizeOmega+1)*sizeof(int));
//    for (j=0; j<SizeOmega+1; j++) nonzeros[j]=0;

//    mexPrintf("%d\n",SizeOmega);
//    system("pause");
    for (i=0; i<SizeOmega; i++)
    {
        *(Px+i)=0;
        tempu1=getCoreMatrix(u1,*(Omega+i*d+0),r[0],r[1]);
        tempu2=getCoreMatrix(u2,*(Omega+i*d+1),r[1],r[2]);
        tempu3=getCoreMatrix(u3,*(Omega+i*d+2),r[2],r[3]);

        // Compute Px
        temp1=matmul(tempu2,tempu3,r[1],r[2],r[3]);
        temp2=transpose(temp1,r[1],r[0]);

        for (j=0; j<r0r1; j++)
            *(Px+i)+=(*(tempu1+j))*(*(temp2+j));

        // Equipping first core
        startind=(*(Omega+i*d+0)-1)*r0r1;
        for (j=startind; j<r0r1+startind; j++)
        {
//            Coremat[i+j*SizeOmega]+=temp2[j-startind];
            Coremat_i[ind]=j;
//            Coremat_j[ind]=j;
//            nonzeros[i]++;
            Coremat_data[ind]+=temp2[j-startind];
            ind++;
        }

        free(temp1);
        free(temp2);

        // Equipping second core
        temp1=matmul(tempu3,tempu1,r[2],r[3],r[1]);
        temp2=transpose(temp1,r[2],r[1]);
        startind=(*(Omega+i*d+1)-1)*r1r2+n0r0r1;
        for (j=startind; j<r1r2+startind; j++)
        {
//            Coremat[i+j*SizeOmega]+=temp2[j-startind];
            Coremat_i[ind]=j;
//            Coremat_j[ind]=j;
//            nonzeros[i]++;
            Coremat_data[ind]+=temp2[j-startind];
            ind++;
        }



        free(temp1);
        free(temp2);

        // Compute G3
        temp1=matmul(tempu1,tempu2,r[3],r[1],r[2]);
        temp2=transpose(temp1,r[3],r[2]);
        startind=(*(Omega+i*d+2)-1)*r2r3+n0r0r1+n1r1r2;
        for (j=startind; j<r2r3+startind; j++)
        {
//            Coremat[i+j*SizeOmega]+=temp2[j-startind];
            Coremat_i[ind]=j;
//            Coremat_j[ind]=j;
//            nonzeros[i]++;
            Coremat_data[ind]+=temp2[j-startind];
            ind++;
        }


        free(tempu1);
        free(tempu2);
        free(tempu3);
        free(temp1);
        free(temp2);

//        system("pause");
    }


    for (i=0; i<SizeOmega+1; i++)
    {
        Coremat_j[i]=i*r0r1_r1r2_r2r3;
//        nonzeros_ans+=nonzeros[i];
    }


//    free(nonzeros);
//    free(u1);
//    free(u2);
//    free(u3);
//    if (d==4)
//    {
//        free(u4);
//    }

//    mexPrintf("Hello World\n");









}