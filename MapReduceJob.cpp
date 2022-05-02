//
// Created by adirt on 22/04/2022.
//

#include <iostream>
#include "MapReduceJob.h"
#define SUCCESS 0
#define ERR_BAD_ALLOCATION "system error: bad allocation.\n"
#define ERR_BAD_PTHREAD_CREATE "system error: pthread create failed.\n"
#define ERR_BAD_PTHREAD_JOIN "system error: pthread join failed.\n"
#define ERR_BAD_DESTRUCTION "system error: pthread destroy failed.\n"
void * threadEntryPoint(void * arg){
	auto *threadContext = (ThreadContext *) arg;
    threadContext->mapPhase();
	threadContext->sortPhase();
	threadContext->barrier();
    threadContext->shufflePhase();
    threadContext->reducePhase();
	return nullptr;
}

MapReduceJob::MapReduceJob(const MapReduceClient &client,
						   const InputVec &inputVec, OutputVec &outputVec,
						   int multiThreadLevel) :
        client(client),
        inputVec(inputVec),
        outputVec(outputVec),
        multiThreadLevel(multiThreadLevel),
        isWaitingForJob(false),
        counter(0)
{
    ThreadContext::uniqueID = 0;
    initCounterFields();
    createPThreads();
    initThreadContexts();
    startThreads();

}

void MapReduceJob::initCounterFields()
{
    counter.fetch_add((uint64_t)1 << 62);
    counter.fetch_add((uint64_t)(inputVec.size()) << 31);
}

void MapReduceJob::createPThreads()
{
    threads = new(std::nothrow) pthread_t[multiThreadLevel];
    if(threads == nullptr){
        cerr << ERR_BAD_ALLOCATION;
        exit(EXIT_FAILURE);
    }
}

void MapReduceJob::initThreadContexts()
{
    for(unsigned int i = 0; i < multiThreadLevel; i++){
        threadContexts.emplace_back(*this);
    }
}

void MapReduceJob::startThreads()
{
    for(unsigned int i = 0; i < multiThreadLevel; i++){
        if(pthread_create(threads + i, nullptr, threadEntryPoint, &(threadContexts[i])) != SUCCESS){
            cerr << ERR_BAD_PTHREAD_CREATE;
            exit(EXIT_FAILURE);
        }
    }
}

void MapReduceJob::waitForJob()
{
	if(isWaitingForJob){
		return;
	}
	for(unsigned int i = 0; i < multiThreadLevel; i++){
		if(pthread_join(threads[i], nullptr) != SUCCESS){
		    cerr << ERR_BAD_PTHREAD_JOIN;
		    exit(EXIT_FAILURE);
		}
	}
    isWaitingForJob = true;
}

MapReduceJob::~MapReduceJob() {
    if(pthread_mutex_destroy(&onlyOneThreadMutex) != SUCCESS or
       pthread_mutex_destroy(&outputVecPushMutex) != SUCCESS or
       pthread_cond_destroy(&shuffleThreadCV) != SUCCESS or
       pthread_cond_destroy(&restThreadsCV) != SUCCESS){
        cerr << ERR_BAD_DESTRUCTION;
        exit(EXIT_FAILURE);
    }
    delete[] threads;
}

void MapReduceJob::getJobState(JobState * job) const
{
    uint64_t curCounterValue = counter.load();
    uint64_t total = ((curCounterValue >> 31) & (((uint64_t)1 << 31) - 1));
    if(total == 0){
        return;
    }
    job->stage = (stage_t)(curCounterValue >> 62);
    uint64_t parts = min((curCounterValue & (((uint64_t)1 << 31) - 1)), total);
	job->percentage = ((float)parts / (float)total) * 100;
}
