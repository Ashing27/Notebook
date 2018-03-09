// Copyright 
//*****************************************************************************************************************
//*Declaration :This file is used for CV system. Beijing Neusoft Medical Equipment co., LTD all right reserved.
//*File Name   : ImageSaver.h
//*Version       : 1.0
//*Author         : Mike
//*Company     :Beijing Neusoft Medical Equipment co., LTD
//*Create Time  : 2017-1-20
//*Modifier        :
//*Modify Time  : 
//*Description   :
//*****************************************************************************************************************
#include "BaseElement.h"
#include "cuda_runtime.h"
#include <algorithm>
#include "LoggerWrapper.h"
#include "Common.h"

using namespace NMS::CV::CoreLogic::Logger;
using namespace NeuSoftCv::Infrastructure::Common;

namespace NMS
{
    namespace CV
    {
        namespace CoreLogic
        {
            namespace ImageProcessor
            {

__constant__ float DevConvKernel3_3[CONV_KERNEL_THREE][CONV_KERNEL_THREE]; // parasoft-suppress  NAMING-33 "CUDA key word" // parasoft-suppress  NAMING-HN-30 "CUDA key word"
__constant__ float DevConvKernel5_5[CONV_FIVE][CONV_FIVE]; // parasoft-suppress  NAMING-HN-30 "CUDA key word" // parasoft-suppress  NAMING-33 "CUDA key word"
__constant__ float DevConvKernel9_9[CONV_NINE][CONV_NINE]; // parasoft-suppress  NAMING-HN-30 "CUDA key word" // parasoft-suppress  NAMING-33 "CUDA key word"
__constant__ float DevConvKernel5[CONV_FIVE]; // parasoft-suppress  NAMING-HN-30 "CUDA key word" // parasoft-suppress  NAMING-33 "CUDA key word"
__constant__ float DevConvKernel9[CONV_NINE]; // parasoft-suppress  NAMING-HN-30 "CUDA key word" // parasoft-suppress  NAMING-33 "CUDA key word"


//Suppose that the fImage has been rounded!!!
__global__ void LookupKernel( const int nImgWidth, const int nImgHeight, const float * pfLut, float * fImage )
{
    int thdx = blockIdx.x * blockDim.x + threadIdx.x;
	int thdy = blockIdx.y * blockDim.y + threadIdx.y;

	if ( thdx >= nImgWidth || thdy >= nImgHeight )
	{
	    return;
	}

	int nIndex = thdy*nImgWidth+thdx;

	fImage[nIndex] = pfLut[(int)fImage[nIndex]];

}


__global__ void AddNumberKernel( const int nImgWidth, const int nImgHeight, const float fAddend, float * clsSumImage )
{
    int thdx = blockIdx.x * blockDim.x + threadIdx.x;
	int thdy = blockIdx.y * blockDim.y + threadIdx.y;

	if ( thdx >= nImgWidth || thdy >= nImgHeight )
	{
	    return;
	}

	clsSumImage[thdy * nImgWidth + thdx] +=  fAddend; 
}

__global__ void AveAndRoundKernel( const int nImgWidth, const int nImgHeight, const float fDivisor, float * clsSumImage )
{
    int thdx = blockIdx.x * blockDim.x + threadIdx.x;
	int thdy = blockIdx.y * blockDim.y + threadIdx.y;

	if ( thdx >= nImgWidth || thdy >= nImgHeight )
	{
	    return;
	}
	float f = clsSumImage[thdy * nImgWidth + thdx] / fDivisor; 
	clsSumImage[thdy * nImgWidth + thdx] = lroundf(f);
}

__global__ void Float2UShortKenerl( const cudaPitchedPtr ptrOrgImage, cudaPitchedPtr ptrDevRes, int imageHeight, int imageWidth )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if ( idy >= imageHeight || idx >= imageWidth )
    {
        return;
    }

    float* pOrgRow = (float*)((char*)ptrOrgImage.ptr + ptrOrgImage.pitch*idy );
    unsigned short* pResRow = (unsigned short*)((char*)ptrDevRes.ptr + ptrDevRes.pitch*idy );

    float fRes = pOrgRow[idx];
    if( fRes < 0 )
    {
        fRes = 0;
    }

    pResRow[idx] = lroundf( fRes );
}

__global__ void UShort2FloatKenerl( const cudaPitchedPtr ptrOrgImage, cudaPitchedPtr ptrDevRes, int imageHeight, int imageWidth )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if ( idy >= imageHeight || idx >= imageWidth )
    {
        return;
    }
    unsigned short* pOrgRow = (unsigned short*)((char*)ptrOrgImage.ptr + ptrOrgImage.pitch*idy );
    float* pResRow = (float*)((char*)ptrDevRes.ptr + ptrDevRes.pitch*idy );

    pResRow[idx] = pOrgRow[idx];
}


template<class T>
__global__ void Conv3Multi3Kernel( const T* pDevOrg, float* pDevRes, int imageHeight, int imageWidth, T EdgeValue )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if ( idy >= imageHeight || idx >= imageWidth )
    {
        return;
    }
    
    int nIndex = idy * imageWidth + idx;

    T subImage[3][3];
    int nRow = 0;    // solution1
    int nCol = 0;    // solution1
    int nSubIndex = 0;    // solution1
    //int nRow = idy - 1 ;    // solution2
    //int nCol = idx - 1;    // solution2
    //int nSubIndex = nIndex - imageWidth - 1;    // solution2
    for( int nSubRow = 0; nSubRow < 3; nSubRow++ )
    {        
        nRow = idy + nSubRow - 1;    // solution1
        for( int nSubCol = 0; nSubCol < 3; nSubCol++ )
        {
            nCol = idx + nSubCol - 1;    // solution1
            if(  nCol < 0 || nCol >= imageWidth ||
                 nRow < 0 || nRow >= imageHeight )
            {
                subImage[nSubRow][nSubCol] = EdgeValue;
            }
            else
            {
                nSubIndex = nRow*imageWidth + nCol;        // solution1
                subImage[nSubRow][nSubCol] = pDevOrg[nSubIndex];
            }
            //nCol++;                //solution2
            //nSubIndex++;        //solution2
        }
        //nRow++;            //solution2
        //nCol -= 3;        //solution2
        //nSubIndex += imageWidth - 3;        //solution2
    }

    float res = 0;
    for( int i = 0; i < 3; i++ )
    {
        for( int j = 0; j < 3; j++ )
        {
            res += subImage[i][j]*DevConvKernel3_3[i][j];
        }
    }
    pDevRes[nIndex] = res;
}

template<class T>
__global__ void Conv3By3KernelSymmetric( const T *in, float *out, int imageHeight, int imageWidth )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if ( idy >= imageHeight || idx >= imageWidth )
    {
        return;
    }
    
    int nIndex = idy * imageWidth + idx;

    T subImage[3][3];
    int nRow = 0;    // solution1
    int nCol = 0;    // solution1
    int nSubIndex = 0;    // solution1

    for( int nSubRow = 0; nSubRow < 3; nSubRow++ )
    {        
        nRow = idy + nSubRow - 1;    // solution1
        if( nRow < 0 )
        {
            nRow = -nRow - 1;
        }
        else if( nRow >= imageHeight )
        {
            nRow = imageHeight*2 - nRow - 1;
        }
        for( int nSubCol = 0; nSubCol < 3; nSubCol++ )
        {
            nCol = idx + nSubCol - 1;    // solution1
            if( nCol < 0 )
            {
                nCol = -nCol - 1;
            }
            else if( nCol >= imageWidth )
            {
                nCol = imageWidth*2 - nCol - 1;
            }

            nSubIndex = nRow*imageWidth + nCol;        // solution1
            subImage[nSubRow][nSubCol] = in[nSubIndex];
        }
    }

    float res = 0;
    for( int i = 0; i < 3; i++ )
    {
        for( int j = 0; j < 3; j++ )
        {
            res += subImage[i][j]*DevConvKernel3_3[i][j];
        }
    }
    out[nIndex] = res;
}


template<class T>
__global__ void Conv5Multi5Kernel( const T *in, float *out, int imageHeight, int imageWidth, T EdgeValue )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if ( idy >= imageHeight || idx >= imageWidth )
    {
        return;
    }
    
    int nIndex = idy * imageWidth + idx;

    T subImage[5][5];
    int nRow = 0;    // solution1
    int nCol = 0;    // solution1
    int nSubIndex = 0;    // solution1

    //int nRow = idy - 1 ;    // solution2
    //int nCol = idx - 1;    // solution2
    //int nSubIndex = nIndex - imageWidth - 1;    // solution2

    for( int nSubRow = 0; nSubRow < 5; nSubRow++ )
    {        
        nRow = idy + nSubRow - 2;    // solution1
        for( int nSubCol = 0; nSubCol < 5; nSubCol++ )
        {
            nCol = idx + nSubCol - 2;    // solution1
            if(  nCol < 0 || nCol >= imageWidth ||
                 nRow < 0 || nRow >= imageHeight )
            {
                subImage[nSubRow][nSubCol] = EdgeValue;
            }
            else
            {
                nSubIndex = nRow*imageWidth + nCol;        // solution1
                subImage[nSubRow][nSubCol] = in[nSubIndex];
            }
            //nCol++;                //solution2
            //nSubIndex++;        //solution2
        }
        //nRow++;            //solution2
        //nCol -= 5;        //solution2
        //nSubIndex += imageWidth - 5;        //solution2
    }

    float res = 0;
    for( int i = 0; i < 5; i++ )
    {
        for( int j = 0; j < 5; j++ )
        {
            res += subImage[i][j]*DevConvKernel5_5[i][j];
        }
    }
    out[nIndex] = res;
}

template<class T>
__global__ void Conv5Multi5KernelSymmetric( const T *in, float *out, int imageHeight, int imageWidth )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if ( idy >= imageHeight || idx >= imageWidth )
    {
        return;
    }
    
    int nIndex = idy * imageWidth + idx;

    T subImage[5][5];
    int nRow = 0;    // solution1
    int nCol = 0;    // solution1
    int nSubIndex = 0;    // solution1

    for( int nSubRow = 0; nSubRow < 5; nSubRow++ )
    {        
        nRow = idy + nSubRow - 2;    // solution1
        if( nRow < 0 )
        {
            nRow = -nRow - 1;
        }
        else if( nRow >= imageHeight )
        {
            nRow = imageHeight*2 - nRow - 1;
        }
        for( int nSubCol = 0; nSubCol < 5; nSubCol++ )
        {
            nCol = idx + nSubCol - 2;    // solution1
            if( nCol < 0 )
            {
                nCol = -nCol - 1;
            }
            else if( nCol >= imageWidth )
            {
                nCol = imageWidth*2 - nCol - 1;
            }

            nSubIndex = nRow*imageWidth + nCol;        // solution1
            subImage[nSubRow][nSubCol] = in[nSubIndex];
        }
    }

    float res = 0;
    for( int i = 0; i < 5; i++ )
    {
        for( int j = 0; j < 5; j++ )
        {
            res += subImage[i][j]*DevConvKernel5_5[i][j];
        }
    }
    out[nIndex] = res;
}

template<class T>
__global__ void Conv5Multi5KernelReplicate( const T *in, float *out, int imageHeight, int imageWidth )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if ( idy >= imageHeight || idx >= imageWidth )
    {
        return;
    }
    
    int nIndex = idy * imageWidth + idx;

    T subImage[5][5];
    int nRow = 0;    // solution1
    int nCol = 0;    // solution1
    int nSubIndex = 0;    // solution1

    for( int nSubRow = 0; nSubRow < 5; nSubRow++ )
    {        
        nRow = idy + nSubRow - 2;    // solution1
        if( nRow < 0 )
        {
            nRow = 0;
        }
        else if( nRow >= imageHeight )
        {
            nRow = imageHeight - 1;
        }

        for( int nSubCol = 0; nSubCol < 5; nSubCol++ )
        {
            nCol = idx + nSubCol - 2;    // solution1
            if( nCol < 0 )
            {
                nCol = 0;
            }
            else if( nCol >= imageWidth )
            {
                nCol = imageWidth - 1;
            }
            nSubIndex = nRow*imageWidth + nCol;        // solution1
            subImage[nSubRow][nSubCol] = in[nSubIndex];
        }
    }

    float res = 0;
    for( int i = 0; i < 5; i++ )
    {
        for( int j = 0; j < 5; j++ )
        {
            res += subImage[i][j]*DevConvKernel5_5[i][j];
        }
    }
    out[nIndex] = res;
}

template<class T>
__global__ void Conv5byRowKernelReplicate( const T *in, float *out, int imageHeight, int imageWidth )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if ( idy >= imageHeight || idx >= imageWidth )
    {
        return;
    }
    
    int nIndex = idy * imageWidth + idx;

    T subImage[CONV_FIVE];
    int nCol = 0;    // solution1
    int nSubIndex = 0;    // solution1

    for( int nSubCol = 0; nSubCol < CONV_FIVE; nSubCol++ )
    {
        nCol = idx + nSubCol - CONV_HALF_FIVE;    // solution1
        if( nCol < 0 )
        {
            nCol = 0;
        }
        else if( nCol >= imageWidth )
        {
            nCol = imageWidth - 1;
        }
        nSubIndex = idy*imageWidth + nCol;        // solution1
        subImage[nSubCol] = in[nSubIndex];
    }

    float res = 0;
    for( int i = 0; i < CONV_FIVE; i++ )
    {
        res += subImage[i]*DevConvKernel5[i];
    }

    out[nIndex] = res;
}

template<class T>
__global__ void Conv5byColKernelReplicate( const T *in, float *out, int imageHeight, int imageWidth )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if ( idy >= imageHeight || idx >= imageWidth )
    {
        return;
    }
    
    int nIndex = idy * imageWidth + idx;

    T subImage[CONV_FIVE];
    int nRow = 0;    // solution1
    int nSubIndex = 0;    // solution1

    for( int nSubRow = 0; nSubRow < CONV_FIVE; nSubRow++ )
    {
        nRow = idy + nSubRow - CONV_HALF_FIVE;    // solution1
        if( nRow < 0 )
        {
            nRow = 0;
        }
        else if( nRow >= imageHeight )
        {
            nRow = imageHeight - 1;
        }

        nSubIndex = nRow*imageWidth + idx;        // solution1
        subImage[nSubRow] = in[nSubIndex];
    }

    float res = 0;
    for( int i = 0; i < CONV_FIVE; i++ )
    {
        res += subImage[i]*DevConvKernel5[i];
    }

    out[nIndex] = res;
}

template<class T>
__global__ void Conv9byRowKernelReplicate( const T *in, float *out, int imageHeight, int imageWidth )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if ( idy >= imageHeight || idx >= imageWidth )
    {
        return;
    }
    
    int nIndex = idy * imageWidth + idx;

    T subImage[CONV_NINE];
    int nCol = 0;    // solution1
    int nSubIndex = 0;    // solution1

    for( int nSubCol = 0; nSubCol < CONV_NINE; nSubCol++ )
    {
        nCol = idx + nSubCol - CONV_HALF_NINE;    // solution1
        if( nCol < 0 )
        {
            nCol = 0;
        }
        else if( nCol >= imageWidth )
        {
            nCol = imageWidth - 1;
        }
        nSubIndex = idy*imageWidth + nCol;        // solution1
        subImage[nSubCol] = in[nSubIndex];
    }

    float res = 0;
    for( int i = 0; i < CONV_NINE; i++ )
    {
        res += subImage[i]*DevConvKernel9[i];
    }

    out[nIndex] = res;
}

template<class T>
__global__ void Conv9byColKernelReplicate( const T *in, float *out, int imageHeight, int imageWidth )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if ( idy >= imageHeight || idx >= imageWidth )
    {
        return;
    }
    
    int nIndex = idy * imageWidth + idx;

    T subImage[CONV_NINE];
    int nRow = 0;    // solution1
    int nSubIndex = 0;    // solution1

    for( int nSubRow = 0; nSubRow < CONV_NINE; nSubRow++ )
    {
        nRow = idy + nSubRow - CONV_HALF_NINE;    // solution1
        if( nRow < 0 )
        {
            nRow = 0;
        }
        else if( nRow >= imageHeight )
        {
            nRow = imageHeight - 1;
        }

        nSubIndex = nRow*imageWidth + idx;        // solution1
        subImage[nSubRow] = in[nSubIndex];
    }

    float res = 0;
    for( int i = 0; i < CONV_NINE; i++ )
    {
        res += subImage[i]*DevConvKernel9[i];
    }

    out[nIndex] = res;
}

template<class T>
__global__ void Conv9Multi9Kernel( const T *in, float *out, int imageHeight, int imageWidth, T EdgeValue )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if ( idy >= imageHeight || idx >= imageWidth )
    {
        return;
    }
    
    int nIndex = idy * imageWidth + idx;

    T subImage[CONV_NINE][CONV_NINE];
    int nRow = 0;    // solution1
    int nCol = 0;    // solution1
    int nSubIndex = 0;    // solution1

    for( int nSubRow = 0; nSubRow < CONV_NINE; nSubRow++ )
    {        
        nRow = idy + nSubRow - CONV_HALF_NINE;    // solution1
        for( int nSubCol = 0; nSubCol < CONV_NINE; nSubCol++ )
        {
            nCol = idx + nSubCol - CONV_HALF_NINE;    // solution1
            if(  nCol < 0 || nCol >= imageWidth ||
                 nRow < 0 || nRow >= imageHeight )
            {
                subImage[nSubRow][nSubCol] = EdgeValue;
            }
            else
            {
                nSubIndex = nRow*imageWidth + nCol;        // solution1
                subImage[nSubRow][nSubCol] = in[nSubIndex];
            }
        }
    }

    float res = 0;
    for( int i = 0; i < CONV_NINE; i++ )
    {
        for( int j = 0; j < CONV_NINE; j++ )
        {
            res += subImage[i][j]*DevConvKernel9_9[i][j];
        }
    }
    out[nIndex] = res;
}

template<class T>
__global__ void Conv9By9KernelReplicate( const T *in, float *out, int imageHeight, int imageWidth )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if ( idy >= imageHeight || idx >= imageWidth )
    {
        return;
    }
    
    int nIndex = idy * imageWidth + idx;

    T subImage[CONV_NINE][CONV_NINE];
    int nRow = 0;    // solution1
    int nCol = 0;    // solution1
    int nSubIndex = 0;    // solution1

    for( int nSubRow = 0; nSubRow < CONV_NINE; nSubRow++ )
    {        
        nRow = idy + nSubRow - CONV_HALF_NINE;    // solution1
        if( nRow < 0 )
        {
            nRow = 0;
        }
        else if( nRow >= imageHeight )
        {
            nRow = imageHeight - 1;
        }

        for( int nSubCol = 0; nSubCol < CONV_NINE; nSubCol++ )
        {
            nCol = idx + nSubCol - CONV_HALF_NINE;    // solution1
            if( nCol < 0 )
            {
                nCol = 0;
            }
            else if( nCol >= imageWidth )
            {
                nCol = imageWidth - 1;
            }
            nSubIndex = nRow*imageWidth + nCol;        // solution1
            subImage[nSubRow][nSubCol] = in[nSubIndex];
        }
    }

    float res = 0;
    for( int i = 0; i < CONV_NINE; i++ )
    {
        for( int j = 0; j < CONV_NINE; j++ )
        {
            res += subImage[i][j]*DevConvKernel9_9[i][j];
        }
    }
    out[nIndex] = res;
}

__global__ void SubstractKernel( cudaPitchedPtr first, cudaPitchedPtr second, cudaPitchedPtr result, int nHeight, int nWidth )
{
    int row_o=blockIdx.y*blockDim.y + threadIdx.y;
    int col_o=blockIdx.x*blockDim.x + threadIdx.x;
    if ( row_o >= nHeight || col_o >= nWidth )
    {
        return;
    }

    float* pFirst = (float*)( (char*)first.ptr + first.pitch*row_o ) + col_o;
    float* pSecond = (float*)( (char*)second.ptr + second.pitch*row_o ) + col_o;
    float* pResult = (float*)( (char*)result.ptr + result.pitch*row_o ) + col_o;

    *pResult = *pFirst - *pSecond;
}


__global__ void AddKernel( float* first, float* second, float* result, int nHeight, int nWidth )
{
    int row_o=blockIdx.y*blockDim.y + threadIdx.y;
    int col_o=blockIdx.x*blockDim.x + threadIdx.x;
    if ( row_o >= nHeight || col_o >= nWidth )
    {
        return;
    }
    int nIndex = row_o*nWidth + col_o;
    float tempValue = first[nIndex] + second[nIndex];
    result[nIndex] = tempValue;
}

template<class T1, class T2>    // T2 can be double or float
__global__ void PaddingKernel( const T1* pDevOrgImage, T2* pDevResultImage, int nOrgHeight, int nOrgWidth, int nResHeight, int nResWidth )
{
    int row_o=blockIdx.y*blockDim.y + threadIdx.y;
    int col_o=blockIdx.x*blockDim.x + threadIdx.x;

    if ( row_o >= nResHeight || col_o >= nResWidth )
    {
        return;
    }
    
    int nResIndex = row_o*nResWidth + col_o;

    short nOrgCol = col_o;
    short nOrgRow = row_o;
    if ( row_o >= nOrgHeight )
    {
        nOrgRow = 2*nOrgHeight - row_o - 1;
    }
    if ( col_o >= nOrgWidth )
    {
        nOrgCol = 2*nOrgWidth - col_o - 1;
    }

    int nOrgIndex = nOrgRow*nOrgWidth + nOrgCol;
    pDevResultImage[nResIndex] = pDevOrgImage[nOrgIndex];
}

template<class T1>
__global__ void UnPaddingKernel( const T1* pDevOrgImage, float* pDevResultImage, int nOrgHeight, int nOrgWidth, int nResHeight, int nResWidth )
{
    int row_o=blockIdx.y*blockDim.y + threadIdx.y;
    int col_o=blockIdx.x*blockDim.x + threadIdx.x;

    if ( row_o >= nResHeight || col_o >= nResWidth )
    {
        return;
    }
    
    int nResIndex = row_o*nResWidth + col_o;
    int nOrgIndex = row_o*nOrgWidth + col_o;

    pDevResultImage[nResIndex] = pDevOrgImage[nOrgIndex];
}

CBaseElement::DevLut::DevLut(  )
            : pDevTable( NULL )
            , nOffsetValue( 0 )
            , nLength( 0 )
{
            
}
CBaseElement::DevLut::~DevLut(  )
{
            
}

bool CBaseElement::DevLut::Init( float* pHostTable, int nTargetOffset, int nTargetLength )
{
    bool bRet = false;
    cudaError cudaStatus;
    nOffsetValue = nTargetOffset;
    nLength = nTargetLength;
    bRet = Destroy();
    if( !bRet )
    {
        EngineerLog( ImageProcessorID.c_str(), "CBaseElement::DevLut::Init Failed");
        return false;
    }

    cudaStatus = cudaMalloc( &pDevTable, nLength*sizeof(float) );
    if ( cudaSuccess != cudaStatus )
    {
        EngineerLog( ImageProcessorID.c_str(), "CBaseElement cudaMalloc Failed");
        return false;
    }

    cudaStatus = cudaMemcpy( pDevTable, pHostTable, nLength*sizeof(float), cudaMemcpyHostToDevice );
    if ( cudaSuccess != cudaStatus )
    {
        EngineerLog( ImageProcessorID.c_str(), "CBaseElement cudaMemcpy Failed");
        return false;
    }
    return true;
}

bool CBaseElement::DevLut::Destroy( )
{
    cudaError cudaStatus;
    if( NULL != pDevTable )
    {
        cudaStatus = cudaFree( pDevTable );
        if ( cudaSuccess != cudaStatus )
        {
            EngineerLog( ImageProcessorID.c_str(), "CBaseElement::DevLut::Destroy Failed");
            return false;
        }
        pDevTable = NULL;
    }
    return true;
}


CBaseElement::CBaseElement(void)
{
}

CBaseElement::~CBaseElement(void)
{
}

bool CBaseElement::Init( )
{
    return true;
}

bool CBaseElement::OnUpdateLUT()
{
    return true;
}
bool CBaseElement::Conv3Multi3(  const unsigned short* pDevOrg, float* pDevRes, const float fKernel[CONV_KERNEL_THREE][CONV_KERNEL_THREE], int imageHeight, int imageWidth, unsigned short usDefault /*= 0*/)
{

    cudaError_t cudaStatus;
    // Launch a kernel on the GPU with one thread for each element.
    dim3 threadSquare( TILE_WIDTH, TILE_HEIGHT );
    dim3 blockSquare( (imageWidth + TILE_WIDTH - 1)/TILE_WIDTH, (imageHeight + TILE_HEIGHT - 1) /TILE_HEIGHT );

    cudaStatus = cudaMemcpyToSymbolAsync( DevConvKernel3_3, fKernel, CONV_KERNEL_THREE * CONV_KERNEL_THREE * sizeof(float) );
    if (cudaStatus != cudaSuccess) 
    {
        EngineerLog( ImageProcessorID.c_str(), "CBaseElement::Conv3Multi3 Failed");
        return false;
    }
    
    Conv3Multi3Kernel<<<blockSquare, threadSquare>>>( pDevOrg, pDevRes, imageHeight, imageWidth, usDefault );

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        EngineerLog( ImageProcessorID.c_str(), "CBaseElement::Conv3Multi3 Failed");
        return false;
    }

    return true;
}

bool CBaseElement::Conv3Multi3( const float* pDevOrg, float* pDevRes, const float fKernel[CONV_KERNEL_THREE][CONV_KERNEL_THREE], int imageHeight, int imageWidth, float fDefault/* = 0.f */ )
{

    cudaError_t cudaStatus;
    // Launch a kernel on the GPU with one thread for each element.
    dim3 threadSquare( TILE_WIDTH, TILE_HEIGHT );
    dim3 blockSquare( (imageWidth + TILE_WIDTH - 1)/TILE_WIDTH, (imageHeight + TILE_HEIGHT - 1) /TILE_HEIGHT );

    cudaStatus = cudaMemcpyToSymbolAsync( DevConvKernel3_3, fKernel, CONV_KERNEL_THREE * CONV_KERNEL_THREE * sizeof(float) );
    if (cudaStatus != cudaSuccess) 
    {
        EngineerLog( ImageProcessorID.c_str(), "CBaseElement::Conv3Multi3 Failed");
        return false;
    }

    Conv3Multi3Kernel<<<blockSquare, threadSquare>>>( pDevOrg, pDevRes, imageHeight, imageWidth, fDefault );

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        EngineerLog( ImageProcessorID.c_str(), "CBaseElement::Conv3Multi3 Failed");
        return false;
    }

    return true;
}


bool CBaseElement::Conv5Multi5( const unsigned short* pDevOrg, float* pDevRes, const float fKernel[5][5], int imageHeight, int imageWidth )
{
    cudaError_t cudaStatus;
    // Launch a kernel on the GPU with one thread for each element.
    dim3 threadSquare( TILE_WIDTH, TILE_HEIGHT );
    dim3 blockSquare( (imageWidth + TILE_WIDTH - 1)/TILE_WIDTH, (imageHeight + TILE_HEIGHT - 1) /TILE_HEIGHT );

    cudaStatus = cudaMemcpyToSymbolAsync( DevConvKernel5_5, fKernel, 5 * 5 * sizeof(float) );
    if (cudaStatus != cudaSuccess) 
    {
        EngineerLog( ImageProcessorID.c_str(), "CBaseElement::Conv5Multi5 Failed");
        return false;
    }
    unsigned short defaultValue = 0;
    Conv5Multi5Kernel<<<blockSquare, threadSquare>>>( pDevOrg, pDevRes, imageHeight, imageWidth, defaultValue );

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        EngineerLog( ImageProcessorID.c_str(), "CBaseElement::Conv5Multi5 Failed");
        return false;
    }

    return true;
}

bool CBaseElement::Conv5Multi5( const float* pDevOrg, float* pDevRes, const float fKernel[5][5],int imageHeight, int imageWidth )
{
    cudaError_t cudaStatus;
    // Launch a kernel on the GPU with one thread for each element.
    dim3 threadSquare( TILE_WIDTH, TILE_HEIGHT );
    dim3 blockSquare( (imageWidth + TILE_WIDTH - 1)/TILE_WIDTH, (imageHeight + TILE_HEIGHT - 1) /TILE_HEIGHT );

    cudaStatus = cudaMemcpyToSymbolAsync( DevConvKernel5_5, fKernel, 5 * 5 * sizeof(float) );
    if (cudaStatus != cudaSuccess) 
    {
        EngineerLog( ImageProcessorID.c_str(), "CBaseElement::Conv5Multi5 Failed");
        return false;
    }

    Conv5Multi5Kernel<<<blockSquare, threadSquare>>>( pDevOrg, pDevRes, imageHeight, imageWidth, 0.f );

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        EngineerLog( ImageProcessorID.c_str(), "CBaseElement::Conv5Multi5 Failed");
        return false;
    }

    return true;
}

// Warning: This function can be replaced by Substract2D in future
bool CBaseElement::Substract( float* first, float* second, float* result, int imageHeight, int imageWidth )
{
    // Launch a kernel on the GPU with one thread for each element.
    dim3 threadSquare( TILE_WIDTH, TILE_HEIGHT );
    dim3 blockSquare( (imageWidth + TILE_WIDTH - 1)/TILE_WIDTH, (imageHeight + TILE_HEIGHT - 1) /TILE_HEIGHT );

    cudaPitchedPtr ptrFirstImage, ptrSecondImage, ptrResultImage;
    ptrFirstImage.ptr = first;
    ptrSecondImage.ptr = second;
    ptrResultImage.ptr = result;

    ptrFirstImage.pitch = ptrSecondImage.pitch = ptrResultImage.pitch = sizeof(float)*imageWidth;

    SubstractKernel<<<blockSquare, threadSquare>>>( ptrFirstImage, ptrSecondImage, ptrResultImage, imageHeight, imageWidth );
    
    return true;
}

void CBaseElement::Substract2D( CDevImage<float>& first, CDevImage<float>& second, CDevImage<float>& result )
{
    int imageWidth = first.stHeader.nWidth;
    int imageHeight = first.stHeader.nHeight;

    // Launch a kernel on the GPU with one thread for each element.
    dim3 threadSquare( TILE_WIDTH, TILE_HEIGHT );
    dim3 blockSquare( (imageWidth + TILE_WIDTH - 1)/TILE_WIDTH, (imageHeight + TILE_HEIGHT - 1) /TILE_HEIGHT );

    SubstractKernel<<<blockSquare, threadSquare>>>( first.ptrPitchedImage, second.ptrPitchedImage, result.ptrPitchedImage, imageHeight, imageWidth );
}

bool CBaseElement::Add( float* first, float* second, float* result, int imageHeight, int imageWidth )
{
    // Launch a kernel on the GPU with one thread for each element.
    dim3 threadSquare( TILE_WIDTH, TILE_HEIGHT );
    dim3 blockSquare( (imageWidth + TILE_WIDTH - 1)/TILE_WIDTH, (imageHeight + TILE_HEIGHT - 1) /TILE_HEIGHT );
    AddKernel<<<blockSquare, threadSquare>>>( first, second, result, imageHeight, imageWidth );
    return true;
}

bool CBaseElement::Padding( const float* pDevOrgImage, float* pDevResultImage, int nOrgHeight, int nOrgWidht, int nResHeight, int nResWidth )
{
    if( nResHeight < nOrgHeight || nResWidth < nOrgWidht )
    {
        EngineerLog( ImageProcessorID.c_str(), "CBaseElement::Padding Failed");
        return false;
    }

    dim3 threadSquare(TILE_WIDTH, TILE_HEIGHT);
    dim3 blockSquare( (nResWidth + TILE_WIDTH - 1)/TILE_WIDTH, (nResHeight + TILE_HEIGHT - 1)/TILE_HEIGHT );
    PaddingKernel<<<blockSquare, threadSquare>>>( pDevOrgImage, pDevResultImage, nOrgHeight, nOrgWidht, nResHeight, nResWidth );
    return true;
}

bool CBaseElement::UnPadding( const float* pDevOrgImage, float* pDevResultImage , int nOrgHeight, int nOrgWidht, int nResHeight, int nResWidth )
{
    if( nResHeight > nOrgHeight || nResWidth > nOrgWidht )
    {
        EngineerLog( ImageProcessorID.c_str(), "CBaseElement::UnPadding Failed" );
        return false;
    }

    dim3 threadSquare(TILE_WIDTH, TILE_HEIGHT);
    dim3 blockSquare( (nResWidth + TILE_WIDTH - 1)/TILE_WIDTH, (nResHeight + TILE_HEIGHT - 1)/TILE_HEIGHT );
    UnPaddingKernel<<<blockSquare, threadSquare>>>( pDevOrgImage, pDevResultImage, nOrgHeight, nOrgWidht, nResHeight, nResWidth );

    return true;
}

bool CBaseElement::Conv5Multi5( const float* pDevOrg, float* pDevRes, 
        const float fKernel[5][5],int imageHeight, int imageWidth, ConvType enType )
{
    cudaError_t cudaStatus;
    // Launch a kernel on the GPU with one thread for each element.
    dim3 threadSquare( TILE_WIDTH, TILE_HEIGHT );
    dim3 blockSquare( (imageWidth + TILE_WIDTH - 1)/TILE_WIDTH, (imageHeight + TILE_HEIGHT - 1) /TILE_HEIGHT );

    cudaStatus = cudaMemcpyToSymbolAsync( DevConvKernel5_5, fKernel, 5 * 5 * sizeof(float) );
    if (cudaStatus != cudaSuccess) 
    {
        EngineerLog( ImageProcessorID.c_str(), "CBaseElement::Conv5Multi5f Failed");
        return false;
    }
    if( symmetric == enType )
    {
        Conv5Multi5KernelSymmetric<<<blockSquare, threadSquare>>>( pDevOrg, pDevRes, imageHeight, imageWidth );
    }
    else if( replicate == enType )
    {
        Conv5Multi5KernelReplicate<<<blockSquare, threadSquare>>>( pDevOrg, pDevRes, imageHeight, imageWidth );
    }
    else
    {
        // TO-DO: Do Nothing
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        EngineerLog( ImageProcessorID.c_str(), "CBaseElement::Conv5Multi5f Failed");
        return false;
    }

    return true;
}

bool CBaseElement::Conv9By9( const unsigned short* pDevOrg, float* pDevRes, 
        const float fKernel[CONV_NINE][CONV_NINE], int imageHeight, int imageWidth, ConvType enType )
{
    cudaError_t cudaStatus;
    // Launch a kernel on the GPU with one thread for each element.
    dim3 threadSquare( TILE_WIDTH, TILE_HEIGHT );
    dim3 blockSquare( (imageWidth + TILE_WIDTH - 1)/TILE_WIDTH, (imageHeight + TILE_HEIGHT - 1) /TILE_HEIGHT );

    cudaStatus = cudaMemcpyToSymbolAsync( DevConvKernel9_9, fKernel, CONV_NINE * CONV_NINE * sizeof(float) );
    if (cudaStatus != cudaSuccess) 
    {
        EngineerLog( ImageProcessorID.c_str(), "CBaseElement::Conv5Multi5f Failed");
        return false;
    }
    if( symmetric == enType )
    {
        // ToDo: Currently, not used
        return false;
    }
    else if( replicate == enType )
    {
        Conv9By9KernelReplicate<<<blockSquare, threadSquare>>>( pDevOrg, pDevRes, imageHeight, imageWidth );
    }
    else
    {
        // TO-DO: Do Nothing
        return false;
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        EngineerLog( ImageProcessorID.c_str(), "CBaseElement::Conv9By9f Failed");
        return false;
    }

    return true;
}

bool CBaseElement::Conv5ByRow( const unsigned short* pDevOrg, float* pDevRes, 
    const float fKernel[CONV_FIVE], int nHeight, int nWidth, ConvType enType )
{
    cudaError_t cudaStatus;
    // Launch a kernel on the GPU with one thread for each element.
    dim3 threadSquare( TILE_WIDTH, TILE_HEIGHT );
    dim3 blockSquare( (nWidth + TILE_WIDTH - 1)/TILE_WIDTH, (nHeight + TILE_HEIGHT - 1) /TILE_HEIGHT );

    cudaStatus = cudaMemcpyToSymbolAsync( DevConvKernel5, fKernel, CONV_FIVE * sizeof(float) );
    if (cudaStatus != cudaSuccess) 
    {
        EngineerLog( ImageProcessorID.c_str(), "CBaseElement::Conv5Multi5f Failed");
        return false;
    }
    if( symmetric == enType )
    {
        // ToDo: Implement in future
        return false;
    }
    else if( replicate == enType )
    {
        Conv5byRowKernelReplicate<<<blockSquare, threadSquare>>>( pDevOrg, pDevRes, nHeight, nWidth );
    }
    else
    {
        // TO-DO: Do Nothing
        return false;
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        EngineerLog( ImageProcessorID.c_str(), "CBaseElement::Conv5Multi5f Failed");
        return false;
    }
    return true;
}

bool CBaseElement::Conv5ByRow( const float* pDevOrg, float* pDevRes, 
    const float fKernel[CONV_FIVE], int nHeight, int nWidth, ConvType enType )
{
    cudaError_t cudaStatus;
    // Launch a kernel on the GPU with one thread for each element.
    dim3 threadSquare( TILE_WIDTH, TILE_HEIGHT );
    dim3 blockSquare( (nWidth + TILE_WIDTH - 1)/TILE_WIDTH, (nHeight + TILE_HEIGHT - 1) /TILE_HEIGHT );

    cudaStatus = cudaMemcpyToSymbolAsync( DevConvKernel5, fKernel, CONV_FIVE * sizeof(float) );
    if (cudaStatus != cudaSuccess) 
    {
        EngineerLog( ImageProcessorID.c_str(), "CBaseElement::Conv5Multi5f Failed");
        return false;
    }
    if( symmetric == enType )
    {
        // ToDo: Implement in future
        return false;
    }
    else if( replicate == enType )
    {
        Conv5byRowKernelReplicate<<<blockSquare, threadSquare>>>( pDevOrg, pDevRes, nHeight, nWidth );
    }
    else
    {
        // TO-DO: Do Nothing
        return false;
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        EngineerLog( ImageProcessorID.c_str(), "CBaseElement::Conv5Multi5f Failed");
        return false;
    }
    return true;
}

bool CBaseElement::Conv5ByCol( const float* pDevOrg, float* pDevRes, 
    const float fKernel[CONV_FIVE], int nHeight, int nWidth, ConvType enType )
{
    cudaError_t cudaStatus = cudaGetLastError();
    // Launch a kernel on the GPU with one thread for each element.
    dim3 threadSquare( TILE_WIDTH, TILE_HEIGHT );
    dim3 blockSquare( (nWidth + TILE_WIDTH - 1)/TILE_WIDTH, (nHeight + TILE_HEIGHT - 1) /TILE_HEIGHT );

    cudaStatus = cudaMemcpyToSymbolAsync( DevConvKernel5, fKernel, CONV_FIVE * sizeof(float) );
    if (cudaStatus != cudaSuccess) 
    {
        EngineerLog( ImageProcessorID.c_str(), "CBaseElement::Conv5Multi5f Failed");
        return false;
    }
    if( symmetric == enType )
    {
        // ToDo: Implement in future
        return false;
    }
    else if( replicate == enType )
    {
        Conv5byColKernelReplicate<<<blockSquare, threadSquare>>>( pDevOrg, pDevRes, nHeight, nWidth );
    }
    else
    {
        // TO-DO: Do Nothing
        return false;
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        EngineerLog( ImageProcessorID.c_str(), "CBaseElement::Conv5Multi5f Failed");
        return false;
    }
    return true;
}

bool CBaseElement::Conv9ByRow( const unsigned short* pDevOrg, float* pDevRes, 
    const float fKernel[CONV_NINE], int nHeight, int nWidth, ConvType enType )
{
    cudaError_t cudaStatus;
    // Launch a kernel on the GPU with one thread for each element.
    dim3 threadSquare( TILE_WIDTH, TILE_HEIGHT );
    dim3 blockSquare( (nWidth + TILE_WIDTH - 1)/TILE_WIDTH, (nHeight + TILE_HEIGHT - 1) /TILE_HEIGHT );

    cudaStatus = cudaMemcpyToSymbolAsync( DevConvKernel9, fKernel, CONV_NINE * sizeof(float) );
    if (cudaStatus != cudaSuccess) 
    {
        EngineerLog( ImageProcessorID.c_str(), "CBaseElement::Conv5Multi5f Failed");
        return false;
    }
    if( symmetric == enType )
    {
        // ToDo: Implement in future
        return false;
    }
    else if( replicate == enType )
    {
        Conv9byRowKernelReplicate<<<blockSquare, threadSquare>>>( pDevOrg, pDevRes, nHeight, nWidth );
    }
    else
    {
        // TO-DO: Do Nothing
        return false;
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        EngineerLog( ImageProcessorID.c_str(), "CBaseElement::Conv5Multi5f Failed");
        return false;
    }
    return true;
}

bool CBaseElement::Conv9ByCol( const float* pDevOrg, float* pDevRes, 
    const float fKernel[CONV_NINE], int nHeight, int nWidth, ConvType enType )
{
    cudaError_t cudaStatus = cudaGetLastError();
    // Launch a kernel on the GPU with one thread for each element.
    dim3 threadSquare( TILE_WIDTH, TILE_HEIGHT );
    dim3 blockSquare( (nWidth + TILE_WIDTH - 1)/TILE_WIDTH, (nHeight + TILE_HEIGHT - 1) /TILE_HEIGHT );

    cudaStatus = cudaMemcpyToSymbolAsync( DevConvKernel9, fKernel, CONV_NINE * sizeof(float) );
    if (cudaStatus != cudaSuccess) 
    {
        EngineerLog( ImageProcessorID.c_str(), "CBaseElement::Conv5Multi5f Failed");
        return false;
    }
    if( symmetric == enType )
    {
        // ToDo: Implement in future
        return false;
    }
    else if( replicate == enType )
    {
        Conv9byColKernelReplicate<<<blockSquare, threadSquare>>>( pDevOrg, pDevRes, nHeight, nWidth );
    }
    else
    {
        // TO-DO: Do Nothing
        return false;
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        EngineerLog( ImageProcessorID.c_str(), "CBaseElement::Conv5Multi5f Failed");
        return false;
    }
    return true;
}

bool CBaseElement::Conv9By9( const unsigned short* pDevOrg, float* pDevRes, 
        const float fKernel[CONV_NINE][CONV_NINE],int imageHeight, int imageWidth, unsigned short nDefaultValue /*= 0*/ )
{
    cudaError_t cudaStatus = cudaSuccess;
    // Launch a kernel on the GPU with one thread for each element.
    dim3 threadSquare( TILE_WIDTH, TILE_HEIGHT );
    dim3 blockSquare( (imageWidth + TILE_WIDTH - 1)/TILE_WIDTH, (imageHeight + TILE_HEIGHT - 1) /TILE_HEIGHT );

    cudaStatus = cudaMemcpyToSymbolAsync( DevConvKernel9_9, fKernel, CONV_NINE * CONV_NINE * sizeof(float) );
    if (cudaStatus != cudaSuccess) 
    {
        EngineerLog( ImageProcessorID.c_str(), "CBaseElement::Conv9By9 Failed");
        return false;
    }
    
    Conv9Multi9Kernel<<<blockSquare, threadSquare>>>( pDevOrg, pDevRes, imageHeight, imageWidth, nDefaultValue );
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if ( cudaStatus != cudaSuccess ) 
    {
        EngineerLog( ImageProcessorID.c_str(), "CBaseElement::Conv9By9 Failed");
        return false;
    }
    return true;
}

bool CBaseElement::Conv9By9( const float* pDevOrg, float* pDevRes, 
        const float fKernel[CONV_NINE][CONV_NINE],int imageHeight, int imageWidth, float fDefaultValue /*= 0*/ )
{
    cudaError_t cudaStatus = cudaSuccess;
    // Launch a kernel on the GPU with one thread for each element.
    dim3 threadSquare( TILE_WIDTH, TILE_HEIGHT );
    dim3 blockSquare( (imageWidth + TILE_WIDTH - 1)/TILE_WIDTH, (imageHeight + TILE_HEIGHT - 1) /TILE_HEIGHT );

    cudaStatus = cudaMemcpyToSymbolAsync( DevConvKernel9_9, fKernel, CONV_NINE * CONV_NINE * sizeof(float) );
    if (cudaStatus != cudaSuccess) 
    {
        EngineerLog( ImageProcessorID.c_str(), "CBaseElement::Conv9By9 Failed");
        return false;
    }
    
    Conv9Multi9Kernel<<<blockSquare, threadSquare>>>( pDevOrg, pDevRes, imageHeight, imageWidth, fDefaultValue );
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if ( cudaStatus != cudaSuccess ) 
    {
        EngineerLog( ImageProcessorID.c_str(), "CBaseElement::Conv9By9 Failed");
        return false;
    }
    return true;
}

bool CBaseElement::Float2Ushort( CDevImage<float> clsInImage, CDevImage<USHORT>& clsOutImage )
{
    clsOutImage.stHeader = clsInImage.stHeader;
    
    dim3 threadSquare( TILE_WIDTH, TILE_HEIGHT );
    dim3 blockSquare( (clsInImage.stHeader.nWidth + TILE_WIDTH - 1)/TILE_WIDTH, (clsInImage.stHeader.nHeight + TILE_HEIGHT - 1) /TILE_HEIGHT );
    
    cudaPitchedPtr ptrInImage, ptrOutImage;
    ptrInImage.ptr = clsInImage.pImage;
    ptrInImage.pitch = clsInImage.stHeader.nWidth*sizeof(float);
    ptrOutImage.ptr = clsOutImage.pImage;
    ptrOutImage.pitch = clsOutImage.stHeader.nWidth*sizeof(USHORT);

    Float2UShortKenerl<<<blockSquare, threadSquare>>>( ptrInImage, ptrOutImage, clsInImage.stHeader.nHeight, clsInImage.stHeader.nWidth );

    return true;
}

bool CBaseElement::Ushort2Float( CDevImage<USHORT> clsInImage, CDevImage<float>& clsOutImage )
{
    clsOutImage.stHeader = clsInImage.stHeader;

    dim3 threadSquare( TILE_WIDTH, TILE_HEIGHT );
    dim3 blockSquare( (clsInImage.stHeader.nWidth + TILE_WIDTH - 1)/TILE_WIDTH, (clsInImage.stHeader.nHeight + TILE_HEIGHT - 1) /TILE_HEIGHT );
    
    cudaPitchedPtr ptrInImage, ptrOutImage;
    ptrInImage.ptr = clsInImage.pImage;
    ptrInImage.pitch = clsInImage.stHeader.nWidth*sizeof(USHORT);
    ptrOutImage.ptr = clsOutImage.pImage;
    ptrOutImage.pitch = clsOutImage.stHeader.nWidth*sizeof(float);
    
    UShort2FloatKenerl<<<blockSquare, threadSquare>>>( ptrInImage, ptrOutImage, clsInImage.stHeader.nHeight, clsInImage.stHeader.nWidth );

    return true;
}

void CBaseElement::Float2Ushort( CDeviceImage clsInImage, CDeviceImage& clsOutImage )
{
    clsOutImage.stHeader = clsInImage.stHeader;
    
    dim3 threadSquare( TILE_WIDTH, TILE_HEIGHT );
    dim3 blockSquare( (clsInImage.stHeader.nWidth + TILE_WIDTH - 1)/TILE_WIDTH, (clsInImage.stHeader.nHeight + TILE_HEIGHT - 1) /TILE_HEIGHT );
    Float2UShortKenerl<<<blockSquare, threadSquare>>>( clsInImage.ptrPitchedImage, clsOutImage.ptrPitchedImage, clsInImage.stHeader.nHeight, clsInImage.stHeader.nWidth );
}

void CBaseElement::Ushort2Float( CDeviceImage clsInImage, CDeviceImage& clsOutImage )
{
    clsOutImage.stHeader = clsInImage.stHeader;
    
    dim3 threadSquare( TILE_WIDTH, TILE_HEIGHT );
    dim3 blockSquare( (clsInImage.stHeader.nWidth + TILE_WIDTH - 1)/TILE_WIDTH, (clsInImage.stHeader.nHeight + TILE_HEIGHT - 1) /TILE_HEIGHT );
    UShort2FloatKenerl<<<blockSquare, threadSquare>>>( clsInImage.ptrPitchedImage, clsOutImage.ptrPitchedImage, clsInImage.stHeader.nHeight, clsInImage.stHeader.nWidth );
}

bool CBaseElement::Conv3By3( const float* pDevOrg, float* pDevRes, const float fKernel[3][3], int imageHeight, int imageWidth, ConvType enType )
{
     cudaError_t cudaStatus;
    // Launch a kernel on the GPU with one thread for each element.
    dim3 threadSquare( TILE_WIDTH, TILE_HEIGHT );
    dim3 blockSquare( (imageWidth + TILE_WIDTH - 1)/TILE_WIDTH, (imageHeight + TILE_HEIGHT - 1) /TILE_HEIGHT );

    cudaStatus = cudaMemcpyToSymbolAsync( DevConvKernel3_3, fKernel, 3 * 3 * sizeof(float) );
    if (cudaStatus != cudaSuccess) 
    {
        EngineerLog( ImageProcessorID.c_str(), "CBaseElement::Conv3By3 Failed");
        return false;
    }
    if( symmetric == enType )
    {
        Conv3By3KernelSymmetric<<<blockSquare, threadSquare>>>( pDevOrg, pDevRes, imageHeight, imageWidth );
    }
    else if( replicate == enType )
    {
        // Todo: Add in future
        return false;
    }
    else
    {
        // TO-DO: Do Nothing
    }

    return true;
}


bool CBaseElement::AveImgAndRound( float * pImage, const float fDivisor, const int nImgWidth, const int nImgHeight)
{
	dim3 threadSquare(TILE_WIDTH, TILE_HEIGHT);
	dim3 blockSquare( (nImgWidth+TILE_WIDTH-1)/TILE_WIDTH, (nImgHeight+TILE_HEIGHT-1)/TILE_HEIGHT);
	AveAndRoundKernel<<<blockSquare, threadSquare>>>( nImgWidth, nImgHeight, fDivisor, pImage);

	cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        EngineerLog( ImageProcessorID.c_str(), "CBaseElement::DivideNumber Failed. CUDA status = %d", cudaStatus);
        return false;
    }

    return true;
}


bool CBaseElement::AddNumber( float * pImage,const float fAddend, const int nImgWidth, const int nImgHeight )
{
	dim3 threadSquare(TILE_WIDTH, TILE_HEIGHT);
	dim3 blockSquare( (nImgWidth+TILE_WIDTH-1)/TILE_WIDTH, (nImgHeight+TILE_HEIGHT-1)/TILE_HEIGHT);
	AddNumberKernel<<<blockSquare, threadSquare>>>( nImgWidth, nImgHeight, fAddend, pImage);

	cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        EngineerLog( ImageProcessorID.c_str(), "CBaseElement::AddNumber Failed. CUDA status = %d", cudaStatus);
        return false;
    }

    return true;

}


bool CBaseElement::Lookup( const int nImgWidth, const int nImgHeight, const float * pfLut, float * fImage )
{
    dim3 threadSquare(TILE_WIDTH, TILE_HEIGHT);
	dim3 blockSquare( (nImgWidth+TILE_WIDTH-1)/TILE_WIDTH, (nImgHeight+TILE_HEIGHT-1)/TILE_HEIGHT);
	LookupKernel<<<blockSquare, threadSquare>>>( nImgWidth, nImgHeight, pfLut, fImage);

	cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        EngineerLog( ImageProcessorID.c_str(), "CBaseElement::Lookup Failed. CUDA status = %d", cudaStatus);
        return false;
    }
	return true;
}

bool CBaseElement::Process( CDevImage<float>& clsInOutImage, PreProcParam* /*pPreParam = NULL*/, PostProcParam* /*pPostParam = NULL*/ )
{
    return true;
}

            } // End ImageProcessor
        } // End CoreLogic
    } // End CV
} // End NMS
