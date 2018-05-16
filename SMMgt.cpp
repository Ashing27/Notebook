
#include "StdAfx.h"
#include <string>
#include "SharedMemoryMgt.h"
#include "LoggerWrapper.h"
#include "Common.h"
#include "cuda_runtime.h"
#include <atlstr.h>
using namespace std;
using namespace NeuSoftCv::Infrastructure::Common;
using namespace NMS::CV::CoreLogic::Logger;

const int WAITING_TIME_OUT = 300;

CSharedMemoryMgt::CSharedMemoryMgt( LPCSTR pSharedMemName, LPCSTR pSemaphoreName, LPCSTR pMutexName, IMGSIZE imageSize, int imageNumber )\
    : m_sSharedMemName( pSharedMemName )
    , m_sSemaphoreName( pSemaphoreName )
    , m_sMutexName( pMutexName )
    , m_ulImageSize( imageSize )\
	, m_nImageNumber( imageNumber )\
	, m_pSharedMemByte( 0 )\
	, m_hMapping( 0 )\
	, m_hSemaphore( 0 )\
	, m_hMutex( 0 )\
	, m_bReadyToRW( false )\
	, m_nBufferSize( 0 )
{    
    //To-Do: if we can save these information in class, rather than in shared memory.
    bool bInit = Init( m_sSharedMemName, m_sSemaphoreName, m_sMutexName, m_ulImageSize, m_nImageNumber );
    if ( !bInit )
    {
        CloseHandles();
    }
}

CSharedMemoryMgt::CSharedMemoryMgt( LPCSTR pSharedMemName, IMGSIZE imageSize, int imageNumber )\
    : m_sSharedMemName( pSharedMemName )
    , m_ulImageSize( imageSize )\
	, m_nImageNumber( imageNumber )\
	, m_pSharedMemByte( 0 )\
	, m_hMapping( 0 )\
	, m_hSemaphore( 0 )\
	, m_hMutex( 0 )\
	, m_bReadyToRW( false )\
	, m_nBufferSize( 0 )
{    
	//To-Do: if we can save these information in class, rather than in shared memory.

	m_bReadyToRW = false;
	m_nBufferSize = imageSize * imageNumber + ShareMemoryHeadSize;

    if ( !OpenShareMemory( m_sSharedMemName ) )
    {
        if ( !CreateShareMemory( m_sSharedMemName ) )
        {
            EngineerLog( SharedMemoryID.c_str(), "CSharedMemoryMgt::Init failed." );
            return;
        }
    }

    m_bReadyToRW = true;
}

void CSharedMemoryMgt::ResetShareMemory()
{  
	if ( 0 == m_hSemaphore || 0 == m_hMutex )
	{
		EngineerLog( SharedMemoryID.c_str(), "Empty semaphore or mutex pointer" );
		return;
	}

	int nLoopCount = m_nImageNumber;
	while ( WAIT_OBJECT_0 == WaitForSingleObject( m_hSemaphore, WAITING_TIME_OUT ))
	{
		--nLoopCount;
		if ( 0 >= nLoopCount )
		{
			break;
		}
	}
	DWORD dwRes = WaitForSingleObject( m_hMutex, WAITING_TIME_OUT );
	if ( WAIT_OBJECT_0 == dwRes )
	{ 
		::memset( m_pSharedMemByte, 0, m_nBufferSize );
		ReleaseMutex( m_hMutex );
	}
}

void CSharedMemoryMgt::InitSharedMemHead()
{
    if ( 0 != m_pSharedMemByte )
    {
        *m_pSharedMemByte = 0;
        *( m_pSharedMemByte + 1 ) = 0;
        *( m_pSharedMemByte + 2 ) = 0;
    }
}


inline int CSharedMemoryMgt::GetWritingIdx() const
{
    BYTE index = *(m_pSharedMemByte+1);
    ////assert(index>=0 && index<m_nImageNumber);
    //It is safe to convert a BYTE type to int type.
    return (int)index;
}

inline void CSharedMemoryMgt::IncreaseWritingIdx()
{
    *( m_pSharedMemByte + 1 ) = ( *( m_pSharedMemByte + 1 ) + 1 ) % m_nImageNumber;
}

inline int CSharedMemoryMgt::GetReadingIdx() const
{
    BYTE index = *( m_pSharedMemByte );
    //assert(index>=0 && index<m_nImageNumber);
    //It is safe to convert a BYTE type to int type.
    return (int)index;
}

inline void CSharedMemoryMgt::IncreaseReadingIdx()
{
    *m_pSharedMemByte = ( *m_pSharedMemByte + 1 ) % m_nImageNumber;
}

int CSharedMemoryMgt::GetImgNumbers() const
{
	BYTE byNum = 0;
	if ( 0 == m_hMutex )
	{
		EngineerLog( SharedMemoryID.c_str(), "CSharedMemoryMgt::GetImgNumbers: Empty mutex pointer" );
		return byNum;
	}

    DWORD dwRes = WaitForSingleObject( m_hMutex, WAITING_TIME_OUT );
    if ( WAIT_OBJECT_0 == dwRes )
    {   
        BYTE byNum = *( m_pSharedMemByte + 2 ) + 1;
        
        ReleaseMutex( m_hMutex );
    }
    else
    {
        EngineerLog( AcquisitionID.c_str(), "CSharedMemoryMgt::IncreaseImgNumbers failed.");
    }

    return byNum;
}

int CSharedMemoryMgt::TryGetImgNumbersLessThenTotol() const
{
	if ( 0 == m_hMutex )
	{
		EngineerLog( SharedMemoryID.c_str(), "CSharedMemoryMgt::TryGetImgNumbersLessThenTotol: Empty mutex pointer" );
		return false;
	}

    BYTE nImgNum = m_nImageNumber;
    const int nTryingNumber = 3;
    int i = 0;    
    bool bBuffFull = true;
    DWORD waitingResult;
    //Try to read the image number for three time if the shared memory is full or if . It resolves the issue in the file comments.
    //To-Do:If it is a good way???
    while ( i < nTryingNumber && bBuffFull )
    {
        //Get the mutex 
        waitingResult = WaitForSingleObject( m_hMutex, WAITING_TIME_OUT );    
        switch ( waitingResult )
        {
        case WAIT_OBJECT_0:
            {
                DebugLog( SharedMemoryID.c_str(),"ReadImgNumbers start." );
                nImgNum = *( m_pSharedMemByte + 2 );

                if ( nImgNum < m_nImageNumber && nImgNum >=0 )
                {
                    bBuffFull = false;
                }
                else if (nImgNum >= m_nImageNumber)
                {
                    i++;
                }
                else 
                {
                    //Wrong imgNum value!!!
                }
                //Must release since the mutex is succeeded to be acquired.
                ReleaseMutex( m_hMutex );
                break;
            }
        case WAIT_TIMEOUT:
            {
                i++;
                break;
            }
            //WAIT_ABANDONED & WAIT_FAILED
        default:
            EngineerLog( SharedMemoryID.c_str(), "WaitForSingleObject: %d.", waitingResult );

            break;
        }
    }
    //assert(index>=0 && index<=m_nImageNumber);
    //It is safe to convert a BYTE type to int type.
    return (int)nImgNum;
}

inline bool CSharedMemoryMgt::IncreaseImgNumbers()
{
	if ( 0 == m_hMutex )
	{
		EngineerLog( SharedMemoryID.c_str(), "CSharedMemoryMgt::IncreaseImgNumbers: Empty mutex pointer" );
		return false;
	}

    //Update image numbers
    DWORD dwRes = WaitForSingleObject( m_hMutex, WAITING_TIME_OUT );
    if ( WAIT_OBJECT_0 == dwRes )
    {   
        BYTE byNum = *( m_pSharedMemByte + 2 ) + 1;
        if ( byNum > m_nImageNumber )
        {
            byNum = m_nImageNumber;
        }
        *( m_pSharedMemByte + 2 ) = byNum;
        ReleaseMutex( m_hMutex );
        return true;
    }
    else
    {
        EngineerLog( AcquisitionID.c_str(), "CSharedMemoryMgt::IncreaseImgNumbers failed.");
        return false;
    }
}

inline bool CSharedMemoryMgt::DecreaseImgNumbers()
{
	if ( 0 == m_hMutex )
	{
		EngineerLog( SharedMemoryID.c_str(), "CSharedMemoryMgt::DecreaseImgNumbers: Empty mutex pointer" );
		return false;
	}

    bool bResult = false;
    DWORD waitingResult = WaitForSingleObject( m_hMutex, WAITING_TIME_OUT );
    if ( WAIT_OBJECT_0 == waitingResult )
    {        
        BYTE byNum = *( m_pSharedMemByte + 2 ) - 1;
        if ( byNum < 0 )
        {
            byNum = 0;
        }
        *( m_pSharedMemByte + 2 ) = byNum;
        ReleaseMutex( m_hMutex );
        return true;
    }
    else
    {
        EngineerLog( AcquisitionID.c_str(), "CSharedMemoryMgt::DecreaseImgNumbers failed.");
        return false;
    }
}

bool CSharedMemoryMgt::WriteImage( const CHostImage& clsImage )
{
    try
    {
		if ( 0 == m_hSemaphore || 0 == m_hMutex )
		{
			EngineerLog( SharedMemoryID.c_str(), "CSharedMemoryMgt::WriteImage: Empty semaphore or mutex pointer" );
			return false;
		}

        if ( !m_bReadyToRW )
        {
            EngineerLog( SharedMemoryID.c_str(), "CSharedMemoryMgt::WriteImage: Share Memory not ready for write." );
            return false;
        }

        if ( 0 == clsImage.pImage )
        {
            EngineerLog( SharedMemoryID.c_str(), "clsImage.pImage is null." );
            return false;
        }

        if ( clsImage.stHeader.stImgSize.nHeight * clsImage.stHeader.stImgSize.nWidth * sizeof(USHORT) + sizeof(ImageHeader) > m_ulImageSize )
        {
            EngineerLog( SharedMemoryID.c_str(), "Image Data Length is too long." );
            return false;
        }

        DebugLog( SharedMemoryID.c_str(), "writeImgData start");

        if ( TryGetImgNumbersLessThenTotol() < m_nImageNumber )
        {
            int nWrtIdx = GetWritingIdx();    
            ::memcpy( m_pSharedMemByte + ShareMemoryHeadSize + nWrtIdx * m_ulImageSize, &clsImage.stHeader, sizeof( ImageHeader ));
            ::memcpy( m_pSharedMemByte + ShareMemoryHeadSize + nWrtIdx * m_ulImageSize + sizeof(ImageHeader), clsImage.pImage, \
                clsImage.stHeader.stImgSize.nHeight * clsImage.stHeader.stImgSize.nWidth * sizeof(USHORT));
            IncreaseImgNumbers();
            IncreaseWritingIdx();
            ReleaseSemaphore( m_hSemaphore, 1, 0 );        
            return true;
        }
        else
        {
            EngineerLog( SharedMemoryID.c_str(), "ShareMemory[%s] has no space, last image is lost, Det ID is: %d", m_sSharedMemName.c_str(), clsImage.stHeader.nDetectorID );
            return false;
        }
    }
    catch (...)
    {
        CatchExcepLog( SharedMemoryID.c_str(), "Exception for CSharedMemoryMgt::WriteImage" );
        //throw;
    }
    
    return false;
}

bool CSharedMemoryMgt::ReadImage( CHostImage& clsImage )
{
	try
	{
		if ( 0 == m_hSemaphore || 0 == m_hMutex )
		{
			EngineerLog( SharedMemoryID.c_str(), "CSharedMemoryMgt::ReadImage: Empty semaphore or mutex pointer" );
			return false;
		}

        if ( !m_bReadyToRW )
        {
            EngineerLog( SharedMemoryID.c_str(), "CSharedMemoryMgt::ReadImage: Share Memory not ready for write." );
            return false;
        }

        if ( 0 == clsImage.pImage )
		{
			EngineerLog( SharedMemoryID.c_str(), "CSharedMemoryMgt::ReadImage: Empty image pointer" );
			return false;
		}

		bool bResult = false;
		//wait for semaphore
		DWORD waitingResult = WaitForSingleObject( m_hSemaphore, WAITING_TIME_OUT );    
		if ( WAIT_OBJECT_0 == waitingResult )
		{
			//get reading index
			int nReadingIdx = GetReadingIdx();
			ImageHeader stHeader;
			::memcpy( &stHeader, m_pSharedMemByte + ShareMemoryHeadSize + nReadingIdx * m_ulImageSize, sizeof( ImageHeader ));

            if ( stHeader.stImgSize.nHeight*stHeader.stImgSize.nWidth*sizeof(unsigned short) > clsImage.nBufferLength )
            {
                EngineerLog( SharedMemoryID.c_str(), "Image size and buffer don't match: Image Width: %d, Image Height: %d, BufferLength: %d.",
                    stHeader.stImgSize.nWidth, stHeader.stImgSize.nHeight, clsImage.nBufferLength );
                return false;
            }

			::memcpy( &clsImage.stHeader, &stHeader, sizeof( ImageHeader ));
			::memcpy( clsImage.pImage, m_pSharedMemByte + ShareMemoryHeadSize + nReadingIdx * m_ulImageSize + \
				sizeof(ImageHeader), clsImage.stHeader.stImgSize.nHeight * clsImage.stHeader.stImgSize.nWidth * sizeof(USHORT));
			//update reading index
			IncreaseReadingIdx();
			//Update image numbers
            bResult = DecreaseImgNumbers();
		}
		else
		{
			bResult = false;
		}
		return bResult;

	}
	catch (...)
	{
		CatchExcepLog( SharedMemoryID.c_str(), "Exception for CSharedMemoryMgt::ReadImage" );
	}
    return false;
}

void CSharedMemoryMgt::CloseHandles()
{
    if ( 0 != m_pSharedMemByte )
    {
        UnmapViewOfFile( m_pSharedMemByte );
        m_pSharedMemByte = 0;
    }

    if ( 0 != m_hMapping )
    {
        CloseHandle( m_hMapping );
        m_hMapping = 0;
    }

    if ( 0 != m_hSemaphore )
    {
        CloseHandle( m_hSemaphore );
        m_hSemaphore = 0;
    }

    if ( 0 != m_hMutex )
    {
        CloseHandle( m_hMutex );
        m_hMutex = 0;
    }

    m_bReadyToRW = false;
}

CSharedMemoryMgt::~CSharedMemoryMgt(void)
{
    CloseHandles();
}

bool CSharedMemoryMgt::ReadImage( int nIndex, CHostImage& clsImage )
{
	try
	{
		if ( 0 == clsImage.pImage )
		{
			EngineerLog( SharedMemoryID.c_str(), "CSharedMemoryMgt::ReadImage: Empty image pointer" );
			return false;
		}

		ImageHeader stHeader;
		::memcpy( &stHeader, m_pSharedMemByte + ShareMemoryHeadSize + nIndex * m_ulImageSize, sizeof( ImageHeader ));
        if ( stHeader.stImgSize.nHeight*stHeader.stImgSize.nWidth*sizeof(unsigned short) > clsImage.nBufferLength )
        {
            EngineerLog( SharedMemoryID.c_str(), "Image size and buffer don't match: Image Width: %d, Image Height: %d, BufferLength: %d.",
                stHeader.stImgSize.nWidth, stHeader.stImgSize.nHeight, clsImage.nBufferLength );
            return false;
        }
		::memcpy( &clsImage.stHeader, &stHeader, sizeof( ImageHeader ));
		::memcpy( clsImage.pImage, m_pSharedMemByte + ShareMemoryHeadSize + nIndex * m_ulImageSize + \
			sizeof(ImageHeader), clsImage.stHeader.stImgSize.nHeight * clsImage.stHeader.stImgSize.nWidth * sizeof(USHORT));

		return true;
	}
	catch (...)
	{
		CatchExcepLog( SharedMemoryID.c_str(), "Exception for CSharedMemoryMgt::ReadImage" );
		throw;
	}
}

bool CSharedMemoryMgt::WriteImage( int nIndex, const CHostImage& clsImage )
{
	try
	{
		if ( 0 == clsImage.pImage )
		{
			EngineerLog( SharedMemoryID.c_str(), "CSharedMemoryMgt::WriteImage: Empty image pointer" );
			return false;
		}

		::memcpy( m_pSharedMemByte + ShareMemoryHeadSize + nIndex * m_ulImageSize, &clsImage.stHeader, sizeof( ImageHeader ));
		::memcpy( m_pSharedMemByte + ShareMemoryHeadSize + nIndex * m_ulImageSize + sizeof(ImageHeader), clsImage.pImage, \
			clsImage.stHeader.stImgSize.nHeight * clsImage.stHeader.stImgSize.nWidth * sizeof(USHORT));
		
		return true;
	}
	catch (...)
	{
		CatchExcepLog( SharedMemoryID.c_str(), "Exception for CSharedMemoryMgt::WriteImage" );
		throw;
	}
}

bool CSharedMemoryMgt::WriteImage( int nIndex, const CDeviceImage& clsImage )
{
	try
	{
		if ( 0 == clsImage.ptrPitchedImage.ptr )
		{
			EngineerLog( SharedMemoryID.c_str(), "CSharedMemoryMgt::WriteImage: Empty pitched image pointer" );
			return false;
		}

		::memcpy( m_pSharedMemByte + ShareMemoryHeadSize + nIndex * m_ulImageSize, &clsImage.stHeader, sizeof( ImageHeader ));

#if ACQ_MODULE_SIMULATOR_IP
        memcpy( m_pSharedMemByte + ShareMemoryHeadSize + nIndex * m_ulImageSize + sizeof(ImageHeader), clsImage.ptrPitchedImage.ptr, clsImage.stHeader.stImgSize.nWidth*sizeof(USHORT)*clsImage.stHeader.stImgSize.nHeight );
#else
		cudaMemcpy2D( m_pSharedMemByte + ShareMemoryHeadSize + nIndex * m_ulImageSize + sizeof(ImageHeader), clsImage.stHeader.stImgSize.nWidth * sizeof(USHORT), \
            clsImage.ptrPitchedImage.ptr,clsImage.ptrPitchedImage.pitch,
			clsImage.stHeader.stImgSize.nWidth * sizeof(USHORT), clsImage.stHeader.stImgSize.nHeight,
            cudaMemcpyDeviceToHost );
#endif
		return true;
	}
	catch (...)
	{
		CatchExcepLog( SharedMemoryID.c_str(), "Exception for CSharedMemoryMgt::WriteImage" );
		throw;
	}
}

bool CSharedMemoryMgt::WriteImage( const CDeviceImage& clsImage )
{
    try
    {
		if ( 0 == m_hSemaphore || 0 == m_hMutex )
		{
			EngineerLog( SharedMemoryID.c_str(), "CSharedMemoryMgt::WriteImage: Empty semaphore or mutex pointer" );
			return false;
		}

        if(!m_bReadyToRW)
        {
            EngineerLog( ImageProcessorID.c_str(), "CSharedMemoryMgt::WriteImage: Share Memory not ready for write." );
            return false;
        }
        if ( 0 == clsImage.ptrPitchedImage.ptr )
        {
            EngineerLog( SharedMemoryID.c_str(), "clsImage.ptrPitchedImage.ptr is null." );
            return false;
        }
        if ( clsImage.stHeader.stImgSize.nHeight*clsImage.stHeader.stImgSize.nWidth*sizeof(USHORT) + sizeof(ImageHeader) > m_ulImageSize )
        {
            EngineerLog( ImageProcessorID.c_str(), "Image Data Length is too long." );
            return false;
        }

        DebugLog( ImageProcessorID.c_str(), "writeImgData start");

        //To-Do: if only one process writes the buffer, the write index can be member variable?
        //If the buffer is not full, write image data into the buffer
        if( TryGetImgNumbersLessThenTotol() < m_nImageNumber )
        {
            int nWrtIdx = GetWritingIdx();    
            ::memcpy( m_pSharedMemByte + ShareMemoryHeadSize + nWrtIdx * m_ulImageSize, &clsImage.stHeader, sizeof( ImageHeader ));

#if ACQ_MODULE_SIMULATOR_IP
            memcpy( m_pSharedMemByte+ShareMemoryHeadSize+nWrtIdx*m_ulImageSize + sizeof(ImageHeader), clsImage.ptrPitchedImage.ptr, clsImage.stHeader.stImgSize.nWidth*sizeof(USHORT)*clsImage.stHeader.stImgSize.nHeight );
#else
            cudaMemcpy2D( m_pSharedMemByte+ShareMemoryHeadSize+nWrtIdx*m_ulImageSize + sizeof(ImageHeader), clsImage.stHeader.stImgSize.nWidth*sizeof(USHORT),
                clsImage.ptrPitchedImage.ptr, clsImage.ptrPitchedImage.pitch,
                clsImage.stHeader.stImgSize.nWidth*sizeof(USHORT), clsImage.stHeader.stImgSize.nHeight, 
                cudaMemcpyDeviceToHost );
#endif
            IncreaseImgNumbers();
            IncreaseWritingIdx();
            ReleaseSemaphore(m_hSemaphore,1,NULL);        
            return true;
        }
        else
        {
            EngineerLog( SharedMemoryID.c_str(), "ShareMemory[%s] has no space, last image is lost, Det ID is: %d", m_sSharedMemName.c_str(), clsImage.stHeader.nDetectorID );
            return false;
        }
    }
    catch (...)
    {
        CatchExcepLog( SharedMemoryID.c_str(), "Exception for CSharedMemoryMgt::WriteImage" );
        //throw;
    }
    return false;
}

bool CSharedMemoryMgt::IsExist( int nImageID, int& nImageIndex, bool& bAllInvalid )
{
	int nInvalidCount = 0;
	for ( int nIndex = 0; nIndex < m_nImageNumber; ++nIndex )
	{
		ImageHeader stHeader;
		::memcpy( &stHeader, m_pSharedMemByte + ShareMemoryHeadSize + nIndex * m_ulImageSize, sizeof( ImageHeader ));
		if ( nImageID == stHeader.nDetectorID )
		{
			nImageIndex = nIndex;
			return true;
		}
		if ( -1 == stHeader.nDetectorID )
		{
			++nInvalidCount;
		}
	}
	if ( nInvalidCount == m_nImageNumber )
	{
		bAllInvalid = true;
	}

	return false;
}

bool CSharedMemoryMgt::ReadImageByImageID( CHostImage& clsImage )
{
	try
	{
		if ( 0 == m_hSemaphore || 0 == m_hMutex )
		{
			EngineerLog( SharedMemoryID.c_str(), "CSharedMemoryMgt::ReadImageByImageID Empty semaphore or mutex pointer" );
			return false;
		}

		if ( 0 == clsImage.pImage )
		{
			EngineerLog( SharedMemoryID.c_str(), "CSharedMemoryMgt::ReadImageByImageID Empty image pointer" );
			return false;
		}

		if ( 0 >=  clsImage.stHeader.nDetectorID )
		{
			EngineerLog( SharedMemoryID.c_str(), "Invalid image ID." );
			return false;
		}

		if ( !m_bReadyToRW )
		{
			EngineerLog( SharedMemoryID.c_str(), "Share Memory not ready for write." );
			return false;
		}

		const int nTryCount = 3;
		bool bExist = false;
		int nImageIndex = -1;
		bool bAllInvalid = false;
		for ( int nTryIndex = 0; nTryIndex < nTryCount; ++nTryIndex )
		{
			//Get the mutex 
			DWORD dwRet = -1;
			dwRet = WaitForSingleObject( m_hMutex, WAITING_TIME_OUT );    
			if ( WAIT_OBJECT_0 == dwRet )
			{
				bExist = IsExist( clsImage.stHeader.nDetectorID, nImageIndex, bAllInvalid );
				if ( bExist )
				{
					break;
				}
				ReleaseMutex( m_hMutex );
			}
		}

		if ( bExist )
		{
			ImageHeader stHeader;
			::memcpy( &stHeader, m_pSharedMemByte + ShareMemoryHeadSize + nImageIndex * m_ulImageSize, sizeof( ImageHeader ));
            if ( stHeader.stImgSize.nHeight*stHeader.stImgSize.nWidth*sizeof(unsigned short) > clsImage.nBufferLength )
            {
                EngineerLog( SharedMemoryID.c_str(), "Image size and buffer don't match: Image Width: %d, Image Height: %d, BufferLength: %d.",
                    stHeader.stImgSize.nWidth, stHeader.stImgSize.nHeight, clsImage.nBufferLength );
                ReleaseMutex( m_hMutex );
                return false;
            }

			::memcpy( &clsImage.stHeader, &stHeader, sizeof( ImageHeader ));
			::memcpy( clsImage.pImage, m_pSharedMemByte + ShareMemoryHeadSize + nImageIndex * m_ulImageSize + \
				sizeof(ImageHeader), clsImage.stHeader.stImgSize.nHeight * clsImage.stHeader.stImgSize.nWidth * sizeof(USHORT));
			::memset( &stHeader, 0, sizeof( ImageHeader ));
			stHeader.nDetectorID = g_nInvalidImageID;
			::memcpy( m_pSharedMemByte + ShareMemoryHeadSize + nImageIndex * m_ulImageSize, &stHeader, sizeof( ImageHeader ));

			ReleaseMutex( m_hMutex );                   // Mike: Why Release Mutex and Semaphore?
			ReleaseSemaphore( m_hSemaphore, 1, 0 );        
			return true;
		}
		else
		{
			ReleaseMutex( m_hMutex );
			if ( bAllInvalid )
			{
				ReleaseSemaphore( m_hSemaphore, 1, 0 ); 
			}
			return false;
		}
		ReleaseMutex( m_hMutex );

		return true;

	}
	catch (...)
	{
		CatchExcepLog( SharedMemoryID.c_str(), "Exception for CSharedMemoryMgt::ReadImageByImageID" );
		ReleaseMutex( m_hMutex );
		throw;
	}
}

bool CSharedMemoryMgt::WriteImageToInvalidBuffer( const CHostImage& clsImage )
{
	try
	{
		if ( 0 == m_hSemaphore || 0 == m_hMutex )
		{
			EngineerLog( SharedMemoryID.c_str(), "CSharedMemoryMgt::WriteImageToInvalidBuffer: Empty semaphore or mutex pointer" );
			return false;
		}

		if ( 0 == clsImage.pImage )
		{
			EngineerLog( SharedMemoryID.c_str(), "CSharedMemoryMgt::WriteImageToInvalidBuffer: Empty image pointer" );
			return false;
		}

		DWORD dwRet = WaitForSingleObject( m_hSemaphore, WAITING_TIME_OUT );    
		if ( WAIT_OBJECT_0 == dwRet )
		{
			bool bExist = false;
			int nImageIndex = -1;
			bool bAllInvalid = false;
			const int nTryCount = 3;
			for ( int nTryIndex = 0; nTryIndex < nTryCount; ++nTryIndex )
			{
				dwRet = WaitForSingleObject( m_hMutex, WAITING_TIME_OUT );    
				if ( WAIT_OBJECT_0 == dwRet )
				{
					bExist = IsExist( -1, nImageIndex, bAllInvalid );
					if ( bExist )
					{
						break;
					}
					ReleaseMutex( m_hMutex );
				}
			}

			if ( bExist )
			{
				::memcpy( m_pSharedMemByte + ShareMemoryHeadSize + nImageIndex * m_ulImageSize, &clsImage.stHeader, \
					sizeof( ImageHeader ));
				::memcpy( m_pSharedMemByte + ShareMemoryHeadSize + nImageIndex * m_ulImageSize + sizeof(ImageHeader), \
					clsImage.pImage, clsImage.stHeader.stImgSize.nHeight * clsImage.stHeader.stImgSize.nWidth * sizeof(USHORT));

				ReleaseMutex( m_hMutex );
			}
			else
			{
				ReleaseMutex( m_hMutex );
				return false;
			}
		}
		else
		{
			return false;
		}

		return true;
	}
	catch (...)
	{
		CatchExcepLog( SharedMemoryID.c_str(), "Exception for CSharedMemoryMgt::WriteImageToInvalidBuffer" );
		ReleaseMutex( m_hMutex );
		throw;
	}
}

bool CSharedMemoryMgt::Init( std::string strSharedMemName, std::string strSemaphoreName, std::string strMutexName, IMGSIZE imageSize, int imageNumber )
{
    try
    {
        m_bReadyToRW = false;

        m_nBufferSize = imageSize * imageNumber + ShareMemoryHeadSize;

        if ( !OpenShareMemory(strSharedMemName) )
        {
            if ( !CreateShareMemory( strSharedMemName ) )
            {
                EngineerLog( SharedMemoryID.c_str(), "CSharedMemoryMgt::Init failed." );
                return false;
            }
        }
        
        if ( NULL == m_hSemaphore )
        {
            m_hSemaphore = CreateSemaphoreA( 0, 0, m_nImageNumber, strSemaphoreName.c_str() );
            if ( NULL == m_hSemaphore )
            {
                EngineerLog( SharedMemoryID.c_str(), "CreateSemaphoreA:%s failed, last error: %d.", strSemaphoreName.c_str(), GetLastError() );
                return false;
            }
        }

        if ( NULL == m_hMutex)
        {
            m_hMutex = CreateMutexA( 0, FALSE, strMutexName.c_str() );
            if ( 0 == m_hMutex)
            {
                EngineerLog( SharedMemoryID.c_str(), "CreateMutexA:%s failed, last error: %d.", strMutexName.c_str(), GetLastError() );
                return false;
            }
        }

        m_bReadyToRW = true;
        return true;
    }
    catch (std::exception& e)
    {
        CatchExcepLog( SharedMemoryID.c_str(), e.what() );
    }
    catch ( ... )
    {
        CatchExcepLog( SharedMemoryID.c_str(), "UnKnown Expception." );
    }

    return false;
}

bool CSharedMemoryMgt::OpenShareMemory( const std::string& strSharedMemName )
{
    if ( 0 == m_hMapping )
    {
        m_hMapping = OpenFileMappingA( FILE_MAP_ALL_ACCESS, FALSE, strSharedMemName.c_str() ); 
        if ( 0 == m_hMapping )
        {
            EngineerLog( SharedMemoryID.c_str(), "Fail to open file mapping: %s, last error: %d.", strSharedMemName.c_str(), GetLastError() );
            return false;
        }

        if ( NULL != m_pSharedMemByte )
        {
            UnmapViewOfFile( m_pSharedMemByte );
            m_pSharedMemByte = NULL;
        }

        m_pSharedMemByte = (BYTE*)( MapViewOfFile( m_hMapping, FILE_MAP_ALL_ACCESS, 0, 0, 0 ));
        if ( 0 == m_pSharedMemByte )
        {        
            EngineerLog( SharedMemoryID.c_str(), "MapViewOfFile Failed: %s, last error: %d", strSharedMemName.c_str(), GetLastError() );
            CloseHandles();
            return false;
        }
    }
    return true;
}

bool CSharedMemoryMgt::CreateShareMemory( const std::string& strSharedMemName )
{
    if ( NULL == m_hMapping )
    {
        m_hMapping = CreateFileMappingA( INVALID_HANDLE_VALUE, 0, PAGE_READWRITE, 0, m_nBufferSize, strSharedMemName.c_str() );
        if ( 0 == m_hMapping )
        {
            EngineerLog( SharedMemoryID.c_str(), "Create File mapping failed, share memory name: %s, last Error: %d", strSharedMemName.c_str(), GetLastError() );
            return false;
        }
    }

    if ( NULL == m_pSharedMemByte )
    {
        m_pSharedMemByte = (BYTE*)( MapViewOfFile( m_hMapping, FILE_MAP_ALL_ACCESS, 0, 0, 0 ));
        if ( 0 == m_pSharedMemByte )
        {        
            EngineerLog( SharedMemoryID.c_str(), "MapViewOfFile failed, share memory name: %s, last Error: %d", strSharedMemName.c_str(), GetLastError() );
            return false;
        }

        // These code will be called twice by two processes
        ::memset( m_pSharedMemByte, 0, m_nBufferSize );
        InitSharedMemHead();
    }

    return true;
}
