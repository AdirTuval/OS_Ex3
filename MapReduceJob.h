//
// Created by adirt on 22/04/2022.
//

#ifndef OS_EX3_MAPREDUCEJOB_H
#define OS_EX3_MAPREDUCEJOB_H
#include "ThreadContext.h"
#include "MapReduceClient.h"
#include <pthread.h>
#include <atomic>

using namespace std;

class MapReduceJob {
public:
	const MapReduceClient& client;
	const InputVec& inputVec;
	OutputVec& outputVec;
	const unsigned int multiThreadLevel;
	pthread_t * threads;
	vector<ThreadContext> threadContexts;
	bool isWaitingForJob;
	std::atomic<uint64_t> counter;
	pthread_mutex_t onlyOneThreadMutex = PTHREAD_MUTEX_INITIALIZER;
	pthread_mutex_t outputVecPushMutex = PTHREAD_MUTEX_INITIALIZER;
	pthread_cond_t shuffleThreadCV = PTHREAD_COND_INITIALIZER;
	pthread_cond_t restThreadsCV = PTHREAD_COND_INITIALIZER;
	unsigned int barrierCounter{0};
	vector<IntermediateVec> shuffledVec;
	MapReduceJob(const MapReduceClient &client,
				 const InputVec &inputVec, OutputVec &outputVec,
				 int multiThreadLevel);
	void initCounterFields();
	void waitForJob();
	void getJobState(JobState *ptr) const;
	~MapReduceJob();

    void createPThreads();

    void initThreadContexts();

    void startThreads();
};


#endif //OS_EX3_MAPREDUCEJOB_H
