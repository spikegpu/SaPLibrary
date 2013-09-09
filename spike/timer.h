#ifndef TIMER_H
#define TIMER_H

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#endif


// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
class Timer {
public:
	virtual ~Timer() {}
	virtual void Start()=0;
	virtual void Stop()=0;
	virtual double getElapsed()=0;
};


// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
class GPUTimer : public Timer {
protected:
	int gpu_idx;
	cudaEvent_t timeStart;
	cudaEvent_t timeEnd;
public:
	GPUTimer(int g_idx = 0) {
		gpu_idx = g_idx;

		cudaEventCreate(&timeStart);
		cudaEventCreate(&timeEnd);
	}

	virtual ~GPUTimer() {
		cudaEventDestroy(timeStart);
		cudaEventDestroy(timeEnd);
	}

	virtual void Start() {
		cudaEventRecord(timeStart, 0);
	}

	virtual void Stop() {
		cudaEventRecord(timeEnd, 0);
		cudaEventSynchronize(timeEnd);
	}

	virtual double getElapsed() {
		float elapsed;
		cudaEventElapsedTime(&elapsed, timeStart, timeEnd);
		return elapsed;
	}
};


// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
#ifdef WIN32

class CPUTimer : public Timer
{
public:
	CPUTimer()   {QueryPerformanceFrequency(&m_frequency);}
	~CPUTimer()  {}

	virtual void Start() {QueryPerformanceCounter(&m_start);}
	virtual void Stop()  {QueryPerformanceCounter(&m_stop);}

	virtual double getElapsed() {
		return (m_stop.QuadPart - m_start.QuadPart) * 1000.0 / m_frequency.QuadPart;
	}

private:
	LARGE_INTEGER m_frequency;
	LARGE_INTEGER m_start;
	LARGE_INTEGER m_stop;
};

#else // WIN32

class CPUTimer : public Timer {
protected:
	timeval timeStart;
	timeval timeEnd;
public:
	virtual ~CPUTimer() {}

	virtual void Start() {
		gettimeofday(&timeStart, 0);
	}

	virtual void Stop() {
		gettimeofday(&timeEnd, 0);
	}

	virtual double getElapsed() {
		return 1000.0 * (timeEnd.tv_sec - timeStart.tv_sec) + (timeEnd.tv_usec - timeStart.tv_usec) / 1000.0;
	}
};

#endif // WIN32


#endif
