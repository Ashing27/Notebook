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

#include "SubtractElement.h"
#include "cuda_runtime.h"
#include <stdio.h>
#include "LoggerWrapper.h"
#include <math.h>
#include "Common.h"
#include "ElementParams.h"
#include "ImageFactory.h"

using namespace NeuSoftCv::Infrastructure::Common;
using namespace std;
using namespace NMS::CV::CoreLogic::Logger;

namespace NMS
{
    namespace CV
    {
        namespace CoreLogic
        {
            namespace ImageProcessor
            {

__global__ void SubtractKernel( const cudaPitchedPtr ptrLive, cudaPitchedPtr ptrMask, cudaPitchedPtr ptrResult,
                               int nHeight, int nWidth, float fOffset, float fLandmark, float fPixelMin, float fPixelMax )
{
    int row_o=blockIdx.y*blockDim.y + threadIdx.y;
    int col_o=blockIdx.x*blockDim.x + threadIdx.x;
    if ( row_o >= nHeight || col_o >= nWidth )
    {
        return;
    }

    float* pLiveRow = (float*)( (char*)ptrLive.ptr + ptrLive.pitch*row_o );
    float* pMaskRow = (float*)( (char*)ptrMask.ptr + ptrMask.pitch*row_o );
    float* pResultRow = (float*)( (char*)ptrResult.ptr + ptrResult.pitch*row_o );

    const float nLivePixel = pLiveRow[col_o];
    const float nMaskPixel = pMaskRow[col_o];
    float fDelta = (nLivePixel - nMaskPixel + fOffset);

    // Modify Mantis bug: 0000406
    // TODO(Mike): Need check Max and Min Pixel
    float fRes = lroundf((1-fLandmark)*fDelta + fLandmark*nLivePixel ); 

    if ( fRes < fPixelMin )
    {
        fRes = fPixelMin;
    }
    else if ( fRes > fPixelMax )
    {
        fRes = fPixelMax;
    }

    pResultRow[col_o] = fRes;
}


// Warning: This function can be deleted in future
__global__ void CombinationKernel( float* pLiveImage, float * pTraceImage, float *pMask1,  float fOffset, float factor_AB, float factor_VC, float factor_CC, int nHeight, int nWidth );


CSubtractElement::CSubtractElement(void)
{
}

CSubtractElement::~CSubtractElement(void)
{
}

bool CSubtractElement::Init( )
{
    bool bRet = false;
    bRet = CBaseElement::Init( );
    if( !bRet )
    {
        EngineerLog( ImageProcessorID.c_str(), "CSubtractElement::Prepare" );
        return false;
    }

    return true;
}

bool CSubtractElement::Process( CDevImage<float>& clsInOutImage, PreProcParam* /*pPreParam = NULL*/, PostProcParam* pPostParam )
{
    const bool bSubtractionSwitchIsOn =pPostParam->bSubtractionSwitchIsOn;
    bool bRet = false;
    CDevImage<float> clsMaskImage;
    bRet = CImageFactory::GetInstance()->GetPixelShiftMaskImage( clsMaskImage );
    if ( !bRet )
    {
        if( (clsInOutImage.stHeader.nDetectorID >= pPostParam->nMaskImageID) )
        {
            bRet = CImageFactory::GetInstance()->SavePixelShiftMaskImage( clsInOutImage );
            if ( !bRet )
            {
                EngineerLog( ImageProcessorID.c_str(), "SaveMaskImage Failed" );
                return false;
            }
        }
        // Warning, Miss NOSUB_LUT
        return true;
    }

    if( !bSubtractionSwitchIsOn )
    {
        return true;
    }

    if( clsInOutImage.stHeader.nWidth != clsMaskImage.stHeader.nWidth 
        || clsInOutImage.stHeader.nHeight != clsMaskImage.stHeader.nHeight )
    {
        EngineerLog( ImageProcessorID.c_str(), "CSubtractElement::Process(), As image size not match: Live:%d*%d, Mask:%d*%d.",  
            clsInOutImage.stHeader.nHeight, clsInOutImage.stHeader.nWidth, clsMaskImage.stHeader.nHeight, clsMaskImage.stHeader.nWidth );
        return false;
    }

    dim3 threadSquare( TILE_WIDTH, TILE_HEIGHT );
    dim3 blockSquare( (clsInOutImage.stHeader.nWidth + TILE_WIDTH - 1)/TILE_WIDTH, (clsInOutImage.stHeader.nHeight + TILE_HEIGHT - 1) /TILE_HEIGHT );

    float fOffset = CElementParams::GetInstance()->GetSubtractParam().fOffset;
    float fLandscape = CElementParams::GetInstance()->GetSubtractParam().fLandscape;

    cudaPitchedPtr ptrLive, ptrMask;
    ptrLive.ptr = clsInOutImage.pImage;
    ptrMask.ptr = clsMaskImage.pImage;
    ptrMask.pitch = ptrLive.pitch = clsInOutImage.stHeader.nWidth*sizeof(float);

    SubtractKernel<<<blockSquare, threadSquare>>>( ptrLive, ptrMask, ptrLive, \
        clsInOutImage.stHeader.nHeight, clsInOutImage.stHeader.nWidth, fOffset, fLandscape, 0.f, 4095.f );

    CImageFactory::GetInstance()->SaveTraceSubImage( clsInOutImage );

    return true;
}

bool CSubtractElement::Process( CDevImage<float> clsLiveImage, CDevImage<float> clsMaskImage, CDevImage<float>& clsSubImage, float fPixelMin, float fPixelMax )
{
    if( clsLiveImage.stHeader.nWidth != clsMaskImage.stHeader.nWidth 
        || clsLiveImage.stHeader.nHeight != clsMaskImage.stHeader.nHeight )
    {
        EngineerLog( ImageProcessorID.c_str(), "CSubtractElement::Process(), As image size not match: Live:%d*%d, Mask:%d*%d.",  
            clsLiveImage.stHeader.nHeight, clsLiveImage.stHeader.nWidth, clsMaskImage.stHeader.nHeight, clsMaskImage.stHeader.nWidth );
        return false;
    }
    clsSubImage.stHeader = clsLiveImage.stHeader;

    dim3 threadSquare( TILE_WIDTH, TILE_HEIGHT );
    dim3 blockSquare( (clsLiveImage.stHeader.nWidth + TILE_WIDTH - 1)/TILE_WIDTH, (clsLiveImage.stHeader.nHeight + TILE_HEIGHT - 1) /TILE_HEIGHT );

    float fOffset = CElementParams::GetInstance()->GetSubtractParam().fOffset;
    float fLandscape = CElementParams::GetInstance()->GetSubtractParam().fLandscape;

    cudaPitchedPtr ptrLive, ptrMask, ptrSub;
    ptrLive.ptr = clsLiveImage.pImage;
    ptrMask.ptr = clsMaskImage.pImage;
    ptrSub.ptr = clsSubImage.pImage;
    ptrMask.pitch = ptrLive.pitch = ptrSub.pitch = clsLiveImage.stHeader.nWidth*sizeof(float);

    SubtractKernel<<<blockSquare, threadSquare>>>( ptrLive, ptrMask, ptrSub, \
        clsLiveImage.stHeader.nHeight, clsLiveImage.stHeader.nWidth, fOffset, fLandscape, fPixelMin, fPixelMax );

    CImageFactory::GetInstance()->SaveTraceSubImage( clsLiveImage );

    return true;
}

            } // End ImageProcessor
        } // End CoreLogic
    } // End CV
} // End NMS
