#include "stdafx.h"
#include "CommonUltility/CommonUltility.h"
#include "IPCommon.h"
#include "Common.h"
#include "Logger\LoggerWrapper\Src\LoggerWrapper.h"
#include "ElementFactory.h"
#include "Element/SqrtEncodingElement.h"
#include "Element/SNRElement.h"
#include "Element/MRElement.h"
#include "Element/SqrtDecodingElement.h"
#include "Element/AutoGainCorrElement.h"
#include "Element/LogEncodingElement.h"
#include "Element/DRMElement.h"
#include "Element/EdgeEnhanceElement.h"
#include "Element/AutoWindowElement.h"
#include "Element/SubtractElement.h"
#include "Element/TraceElement.h"
#include "Element/ResizeElement.h"
#include "Element/DTWaveletNRElement.h"
#include "Element/FNRElement.h"
#include "Element/FNRCompGElement.h"
#include "Element/CombinationElement.h"
#include "Element/CheckContrastElement.h"
#include "Element/PixelShift.h"
#include "Element/InverseElement.h"
#include "cuda_runtime.h"
#include "LoggerWrapper.h"
#include "PreProcess.h"
#include "PostProcess.h"

using namespace std;
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

CElementFactory::CGarbo<CElementFactory> CElementFactory::m_Garbo;
CElementFactory* CElementFactory::GetInstance()
{
    return m_Garbo.m_pSingleClass;
}

CElementFactory::CElementFactory(void)
    : m_mapElement()
{
}


CElementFactory::~CElementFactory(void)
{
}

bool CElementFactory::UnInit()
{
    EngineerLog( ImageProcessorID.c_str(), "CElementFactory::UnInit" );
    m_mapElement.clear();
    return true;
}

void CElementFactory::UpdateLUT()
{
    bool bRet = false;
    for ( std::map<std::string, std::unique_ptr<CBaseElement> >::iterator iter = m_mapElement.begin();
        iter != m_mapElement.end(); iter++ )
    {
        bRet = iter->second->OnUpdateLUT();
        if ( !bRet )
        {
            EngineerLog( ImageProcessorID.c_str(), "UpdateLUT of %s Failed", iter->first.c_str() );
        }
    }
}

CBaseElement* CElementFactory::CreateElement( std::string strElementName )
{
    if ( SQRT_ENCODING_ELEMENT == strElementName )
    {
        return (new CSqrtEncodingElement());
    }
    else if ( SNR_ELEMENT == strElementName )
    {
        return new CDTWaveletNRElement();
    }
    else if ( MR_ELEMENT == strElementName ) 
    {
        return (new CMRElement());
    }
    else if ( SQRT_DECODING_ELEMENT == strElementName )
    {
        return (new CSqrtDecodingElement());
    }
    else if ( LOG_ENCODING_ELEMENT == strElementName ) 
    {
        return (new CLogEncodingElement());
    }
    else if ( AUTO_GAIN_CORR_ELEMENT == strElementName ) 
    {
        return (new CAutoGainCorrElement());
    }
    else if ( DRM_ELEMENT == strElementName )
    {           
        return (new CDRMElement());
    }
    else if ( EDGE_ENHANCEMENT_ELEMENT == strElementName )
    {
        return (new CEdgeEnhanceElement());
    }
    else if ( TRACE_ELEMENT == strElementName )
    {
        return (new CTraceElement());
    }
    else if ( SUBTRACT_ELEMENT == strElementName )
    {
        return (new CSubtractElement());
    }
    else if ( AUTO_WINDOW_ELEMENT == strElementName )
    {
        return (new CAutoWindowElement());
    }
    else if ( RESIZE_ELEMENT == strElementName ) 
    {
        return (new CResizeElement());
    }
    else if ( FNR_ELEMENT == strElementName ) 
    {
        return (new CFNRElement());
    }
    else if ( FNR_COMPG_ELEMENT == strElementName )
    {
        return (new CFNRCompGElement());
    }
    else if ( COMBINATION_ELEMENT == strElementName )
    {
        return (new CCombinationElement() );
    }
    else if ( CHECK_CONTRAST_ELEMENT == strElementName )
    {
        return (new CCheckContrastElement() );
    }
	else if ( PIXEL_SHIFT == strElementName ) 
	{
		return (new CPixelShift());
	}
    else if ( INVERSE_ELEMENT == strElementName )
    {
        return (new CInverseElement() );
    }
    else if ( BASE_ELEMENT == strElementName ) 
    {
        return (new CBaseElement());
    }
    else
    {
        return (new CBaseElement());
    }
}

CBaseElement* CElementFactory::GetElement( std::string strElementName )
{
    std::map<std::string, std::unique_ptr<CBaseElement> >::iterator iter = m_mapElement.begin();
    iter = m_mapElement.find( strElementName );
    if ( iter == m_mapElement.end() )
    {
        m_mapElement[strElementName] = unique_ptr<CBaseElement>( CreateElement( strElementName ) );
        m_mapElement[strElementName]->Init();
        m_mapElement[strElementName]->OnUpdateLUT();
    }
    
    return m_mapElement[strElementName].get();
}

CSqrtEncodingElement* CElementFactory::GetSqrtEncoding( )
{
    return dynamic_cast<CSqrtEncodingElement*>(GetElement( SQRT_ENCODING_ELEMENT ));
}

CAutoWindowElement* CElementFactory::GetAutoWindow( )
{
    return dynamic_cast<CAutoWindowElement*>(GetElement( AUTO_WINDOW_ELEMENT ));
}

CAutoGainCorrElement* CElementFactory::GetAutoGainCorr()
{
    return dynamic_cast<CAutoGainCorrElement*>(GetElement( AUTO_GAIN_CORR_ELEMENT ));
}

CTraceElement* CElementFactory::GetTraceElement()
{
    return dynamic_cast<CTraceElement*>(GetElement( TRACE_ELEMENT ));
}

CSubtractElement* CElementFactory::GetSubElement( )
{
    return dynamic_cast<CSubtractElement*>(GetElement( SUBTRACT_ELEMENT ));
}

CResizeElement* CElementFactory::GetResizeElement( )
{
    return dynamic_cast<CResizeElement*>(GetElement( RESIZE_ELEMENT ));
}

CFNRElement* CElementFactory::GetFNRElement()
{
    return dynamic_cast<CFNRElement*>(GetElement( FNR_ELEMENT ));
}

CCheckContrastElement* CElementFactory::GetCheckContrastElement()
{
    return dynamic_cast<CCheckContrastElement*>(GetElement(CHECK_CONTRAST_ELEMENT));
}
// Jane
CPixelShift* CElementFactory::GetPixelShift()
{
    return dynamic_cast<CPixelShift*>(GetElement(PIXEL_SHIFT));
}
// End_J
            } // End XrayController
        } // End CoreLogic
    } // End CV
} // End NMS
