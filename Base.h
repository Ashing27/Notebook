#pragma once

#include <math.h>
#include "PreProcess.h"
#include "PostProcess.h"
#include "IPCommon.h"

namespace NMS
{
    namespace CV
    {
        namespace CoreLogic
        {
            namespace ImageProcessor
            {

#define PI   3.141592653589793f
#define MAX_PIXEL_VALUE     65535
#define MAX_RES_PIXEL_VALUE     4095

// 2018.1.10 Mike Tested BlockSize of (8,8), (16,16), (32, 16), (16,32), (32, 32)
// On M4000, (32, 16) Get the best performance( (32, 16) > (32, 32) > (16,16) > (16,32) > (8,8) )
// To do: check P4000 Performance
const unsigned short TILE_WIDTH    = 32;
const unsigned short TILE_HEIGHT   = 16;

const unsigned short    CONV_KERNEL_THREE     = 3;

#define CONV_NINE   9
#define CONV_HALF_NINE  4

#define CONV_FIVE   5
#define CONV_HALF_FIVE   2

#define CONV_SEVEN    7
#define CONV_HALF_SEVEN    3

#define NUM_ZERO    0
#define NUM_ONE        1
#define NUM_TWO        2
#define NUM_THREE    3
#define NUM_FOUR    4
#define NUM_FIVE    5
#define NUM_SIX        6
#define NUM_SEVEN    7
#define NUM_EIGHT    8
#define NUM_NINE    9
#define NUM_TEN        10

enum ConvType
{
    symmetric = 0,
    replicate = 1,
};

enum ProcessType
{
    e_Invalid_Process = -1,
    e_Fluoro_Process = 0,
    e_DSA_Process,
    e_Roadmap_1,
    e_Roadmap_2,
};

class CBaseElement
{
public:
    struct DevLut
    {
        float* pDevTable;
        int nOffsetValue;
        int nLength;

    public:
        DevLut();
        ~DevLut();
        bool Init( float* pHostTable, int nTargetOffset, int nTargetLength );
        bool Destroy( );         
    private:     
        DevLut& operator =( const DevLut& );
    };

public:
    CBaseElement(void);
    virtual ~CBaseElement(void);
    virtual bool Init( );
    virtual bool Process( CDevImage<float>& clsInOutImage, PreProcParam* pPreParam = NULL, PostProcParam* pPostParam = NULL );
    virtual bool OnUpdateLUT();
    // Warning: This function can be replaced by CDeviceImage version
    bool Float2Ushort( CDevImage<float> clsInImage, CDevImage<USHORT>& clsOutImage );
    bool Ushort2Float( CDevImage<USHORT> clsInImage, CDevImage<float>& clsOutImage );

    void Float2Ushort( CDeviceImage clsInImage, CDeviceImage& clsOutImage );
    void Ushort2Float( CDeviceImage clsInImage, CDeviceImage& clsOutImage );

	bool Add( float* first, float* second, float* result, int imageHeight, int imageWidth );
	bool AveImgAndRound( float * pImage, const float fDivisor, const int nImgWidth, int nImgHeight );
	bool AddNumber(  float * pImage,const float fAddend, const int nImgWidth, const int nImgHeight );
	bool Lookup( const int nImgWidth, const int nImgHeight, const float * pfLut, float * fImage );
protected:
    bool Padding( const float* pDevOrgImage, float* pDevResultImage, int nOrgHeight, int nOrgWidht, int nResHeight, int nResWidth );
    bool UnPadding( const float* pDevOrgImage, float* pDevResultImage, int nOrgHeight, int nOrgWidht, int nResHeight, int nResWidth );

    bool Conv3Multi3( const unsigned short* pDevOrg, float* pDevRes, 
        const float fKernel[CONV_KERNEL_THREE][CONV_KERNEL_THREE],int imageHeight, int imageWidth, unsigned short fDefault = 0 );
    bool Conv3Multi3( const float* pDevOrg, float* pDevRes, 
        const float fKernel[CONV_KERNEL_THREE][CONV_KERNEL_THREE],int imageHeight, int imageWidth, float fDefault = 0.f );

    bool Conv3By3( const unsigned short* pDevOrg, float* pDevRes, const float fKernel[3][3], int imageHeight, int imageWidth, ConvType enType );
    bool Conv3By3( const float* pDevOrg, float* pDevRes, const float fKernel[3][3], int imageHeight, int imageWidth, ConvType enType );


    bool Conv5Multi5( const unsigned short* pDevOrg, float* pDevRes, const float fKernel[5][5],int imageHeight, int imageWidth );
    bool Conv5Multi5( const float* pDevOrg, float* pDevRes, const float fKernel[5][5], int imageHeight, int imageWidth );

    bool Conv5Multi5( const unsigned short* pDevOrg, float* pDevRes, const float fKernel[5][5], int imageHeight, int imageWidth, ConvType enType );
    bool Conv5Multi5( const float* pDevOrg, float* pDevRes, const float fKernel[5][5], int imageHeight, int imageWidth, ConvType enType );

    bool Conv9By9( const unsigned short* pDevOrg, float* pDevRes, 
        const float fKernel[CONV_NINE][CONV_NINE], int imageHeight, int imageWidth, ConvType enType );
    bool Conv9By9( const float* pDevOrg, float* pDevRes, 
        const float fKernel[CONV_NINE][CONV_NINE], int imageHeight, int imageWidth, ConvType enType );

    bool Conv9By9( const unsigned short* pDevOrg, float* pDevRes, 
        const float fKernel[CONV_NINE][CONV_NINE],int imageHeight, int imageWidth, unsigned short nDefaultValue = 0 );
    bool Conv9By9( const float* pDevOrg, float* pDevRes, 
        const float fKernel[CONV_NINE][CONV_NINE],int imageHeight, int imageWidth, float fDefault = 0 );

    bool Conv5ByRow( const unsigned short* pDevOrg, float* pDevRes, 
        const float fKernel[CONV_FIVE], int nHeight, int nWidth, ConvType enType );
    bool Conv5ByRow( const float* pDevOrg, float* pDevRes, 
        const float fKernel[CONV_FIVE], int nHeight, int nWidth, ConvType enType );

    bool Conv5ByCol( const unsigned short* pDevOrg, float* pDevRes, 
        const float fKernel[CONV_FIVE], int nHeight, int nWidth, ConvType enType );
    bool Conv5ByCol( const float* pDevOrg, float* pDevRes, 
        const float fKernel[CONV_FIVE], int nHeight, int nWidth, ConvType enType );

    bool Conv9ByRow( const unsigned short* pDevOrg, float* pDevRes, 
        const float fKernel[CONV_NINE], int nHeight, int nWidth, ConvType enType );
    bool Conv9ByRow( const float * pDevOrg, float* pDevRes, 
        const float fKernel[CONV_NINE], int nHeight, int nWidth, ConvType enType );

    bool Conv9ByCol( const unsigned short* pDevOrg, float* pDevRes, 
        const float fKernel[CONV_NINE], int nHeight, int nWidth, ConvType enType );
    bool Conv9ByCol( const float* pDevOrg, float* pDevRes, 
        const float fKernel[CONV_NINE], int nHeight, int nWidth, ConvType enType );

    bool Substract( float* first, float* second, float* result, int imageHeight, int imageWidth );
    
    double Round( double r );
    float  Round( float r);
    void Padding2D( const CDevImage<float>& pDevOrgImage, CDevImage<float>& pDevResultImage );
    void Substract2D( CDevImage<float>& first, CDevImage<float>& second, CDevImage<float>& result );
};


inline double CBaseElement::Round( double r )
{
    return  static_cast<float>((r > 0.0) ? floor(r + 0.5f) : ceil(r - 0.5f));
}

inline float CBaseElement::Round( float r )
{
    return  (r > 0.0) ? floor(r + 0.5f) : ceil(r - 0.5f);
}

            } // End ImageProcessor
        } // End CoreLogic
    } // End CV
} // End NMS
