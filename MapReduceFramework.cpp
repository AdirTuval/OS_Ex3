#include <pthread.h>
#include <iostream>
#include <algorithm>
#include <functional>
#include <queue>
#include "MapReduceFramework.h"
#include "MapReduceJob.h"
#define ERR_BAD_ALLOCATION "system error: bad allocation.\n"
#define ERR_BAD_MUTEX_LOCK_UNLOCK "system error: mutex lock or unlock failed.\n"
#define SUCCESS 0
using namespace std;

void emit2(K2 *key, V2 *value, void *context) {
    auto *threadContext = (ThreadContext *) context;
    threadContext->intermediateVec.push_back({key, value});
}

void emit3(K3 *key, V3 *value, void *context) {
    auto * _mapReduceJob = (MapReduceJob *) context;
    if(pthread_mutex_lock(&(_mapReduceJob->outputVecPushMutex)) != SUCCESS){
        cerr << ERR_BAD_MUTEX_LOCK_UNLOCK;
        exit(EXIT_FAILURE);
    };
    _mapReduceJob->outputVec.push_back({key, value});
    if(pthread_mutex_unlock(&(_mapReduceJob->outputVecPushMutex)) != SUCCESS){
        cerr << ERR_BAD_MUTEX_LOCK_UNLOCK;
        exit(EXIT_FAILURE);
    }
}


JobHandle startMapReduceJob(const MapReduceClient &client,
                            const InputVec &inputVec, OutputVec &outputVec,
                            int multiThreadLevel) {
	auto *mapReduceJob = new(std::nothrow) MapReduceJob(client, inputVec, outputVec, multiThreadLevel);
	if(mapReduceJob == nullptr){
	    cerr << ERR_BAD_ALLOCATION;
	    exit(EXIT_FAILURE);
	}
	return mapReduceJob;
}

void waitForJob(JobHandle job) {
	auto *mapReduceJob = (MapReduceJob *) job;
	mapReduceJob->waitForJob();
}

void getJobState(JobHandle job, JobState *state) {
	auto *mapReduceJob = (MapReduceJob *) job;
    mapReduceJob->getJobState(state);
}

void closeJobHandle(JobHandle job) {
    auto *mapReduceJob = (MapReduceJob *) job;
    mapReduceJob->waitForJob();
    delete mapReduceJob;
}

