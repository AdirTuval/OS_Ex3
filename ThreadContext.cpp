//
// Created by adirt on 23/04/2022.
//

#include <iostream>
#include <algorithm>
#include <queue>
#include <functional>
#include "ThreadContext.h"
#include "MapReduceJob.h"
#define ALL_EMPTY -1
#define ERR_BAD_MUTEX_LOCK_UNLOCK "system error: mutex lock or unlock failed.\n"
#define ERR_BAD_COND "system error: use of condition variable failed.\n"
#define SUCCESS 0
unsigned int ThreadContext::uniqueID = 0;
ThreadContext::ThreadContext(MapReduceJob& mapReduceJob):
        id(uniqueID++),
        mapReduceJob(mapReduceJob){}

void ThreadContext::mapPhase()
{
	while(true) {
		uint64_t oldCounterValue2 = mapReduceJob.counter++;
		oldCounterValue2 = oldCounterValue2 & (((uint64_t)1 << 31) - 1) ;

		if(oldCounterValue2 >= mapReduceJob.inputVec.size()){
			return;
		}
		InputPair currentPair = mapReduceJob.inputVec[oldCounterValue2];
		mapReduceJob.client.map(currentPair.first, currentPair.second, this);;
	}
}

void ThreadContext::sortPhase()
{
    std::sort(intermediateVec.begin(), intermediateVec.end(),
              [](const IntermediatePair &left, const IntermediatePair &right) {
        return *(left.first) < *(right.first);
    });
}

void ThreadContext::barrier() {
    if(pthread_mutex_lock(&mapReduceJob.onlyOneThreadMutex) != SUCCESS){
        cerr << ERR_BAD_MUTEX_LOCK_UNLOCK;
        exit(EXIT_FAILURE);
    }
    if (++(mapReduceJob.barrierCounter) < mapReduceJob.multiThreadLevel) {
        if (id == 0) {
            if(pthread_cond_wait(&mapReduceJob.shuffleThreadCV, &mapReduceJob.onlyOneThreadMutex) != SUCCESS){
                cerr << ERR_BAD_COND;
                exit(EXIT_FAILURE);
            }
        } else {
            if(pthread_cond_wait(&mapReduceJob.restThreadsCV, &mapReduceJob.onlyOneThreadMutex) != SUCCESS){
                cerr << ERR_BAD_COND;
                exit(EXIT_FAILURE);
            }
        }
    } else {
        mapReduceJob.barrierCounter = 0;
        if (id != 0) {
            if(pthread_cond_signal(&mapReduceJob.shuffleThreadCV) != SUCCESS){
                cerr << ERR_BAD_COND;
                exit(EXIT_FAILURE);
            }
            if(pthread_cond_wait(&mapReduceJob.restThreadsCV, &mapReduceJob.onlyOneThreadMutex) != SUCCESS){
                cerr << ERR_BAD_COND;
                exit(EXIT_FAILURE);
            }
        }
    }
    if(pthread_mutex_unlock(&mapReduceJob.onlyOneThreadMutex) != SUCCESS){
        cerr << ERR_BAD_MUTEX_LOCK_UNLOCK;
        exit(EXIT_FAILURE);
    }
}

bool compareShufflePairs(const ThreadContext::ShufflePair &left, const ThreadContext::ShufflePair &right){
    return *(left.first.first) < *(right.first.first);
}

bool areIntermediatePairEqualByKey(const IntermediatePair &right, const IntermediatePair &left){
    return not (*(right.first) < *(left.first)) and not (*(left.first) < *(right.first));
}

void ThreadContext::shufflePhase() {
    if(id != 0){
        return;
    }
    int sumOfIntermediatePairs = calcSumOfIntermediatePairs();
    advanceStage();
	mapReduceJob.counter.fetch_add((uint64_t)(sumOfIntermediatePairs) << 31);
    vector<IntermediateVec>& shuffledVec = mapReduceJob.shuffledVec;
    vector<ThreadContext>& threadContexts = mapReduceJob.threadContexts;
    std::priority_queue<ShufflePair, std::vector<ShufflePair>, std::function<bool(ShufflePair, ShufflePair)>> heap(compareShufflePairs);

    //init heap
    for(unsigned int i = 0; i < mapReduceJob.multiThreadLevel; i++){
        IntermediateVec& currIntermediateVec = threadContexts[i].intermediateVec;
        if(currIntermediateVec.empty()){
            continue;
        }
        auto topPair = currIntermediateVec.back();
        currIntermediateVec.pop_back();
        heap.push({topPair, i});
    }
    //main loop
    bool allEmpty = false;
    IntermediateVec currentIdenticalVector;
    while(not heap.empty()){
        auto currentShufflePair = heap.top();
        auto currentIntermediatePair = currentShufflePair.first;
        unsigned int cameFromIndex = currentShufflePair.second;
        heap.pop();
        if((not currentIdenticalVector.empty()) and
            (not areIntermediatePairEqualByKey(currentIntermediatePair,
                                               currentIdenticalVector[0]))){
            shuffledVec.push_back(currentIdenticalVector);
            currentIdenticalVector.clear();
        }
        currentIdenticalVector.push_back(currentIntermediatePair);
		mapReduceJob.counter++;
        // Inserting new element into heap.
        if(allEmpty){
            continue;
        }
        int nextPairVectorIndex = getNextShufflePairVectorIndex(cameFromIndex);
        if(nextPairVectorIndex == ALL_EMPTY){
            allEmpty = true;
            continue;
        }
        auto newPair = threadContexts[nextPairVectorIndex].intermediateVec.back();
        threadContexts[nextPairVectorIndex].intermediateVec.pop_back();
        heap.push({newPair,nextPairVectorIndex});
    }
    if(not currentIdenticalVector.empty()){
        shuffledVec.push_back(currentIdenticalVector);
    }
    advanceStage();
	mapReduceJob.counter.fetch_add((uint64_t)(sumOfIntermediatePairs) << 31);
    if(pthread_cond_broadcast(&mapReduceJob.restThreadsCV) != SUCCESS){
        cerr << ERR_BAD_COND;
        exit(EXIT_FAILURE);
    }

}

int ThreadContext::getNextShufflePairVectorIndex(unsigned int cameFromIndex) const {
    vector<ThreadContext>& threadContexts = mapReduceJob.threadContexts;
    if(not threadContexts[cameFromIndex].intermediateVec.empty()){
        return (int)cameFromIndex;
    }
    if(areIntermediateVectorsAreEmpty()){
        return ALL_EMPTY;
    }
    while(threadContexts[cameFromIndex].intermediateVec.empty()){
        cameFromIndex = (cameFromIndex + 1) % (int)threadContexts.size();
    }
    return (int)cameFromIndex;
}

bool ThreadContext::areIntermediateVectorsAreEmpty() const{
    vector<ThreadContext>& contexts = mapReduceJob.threadContexts;
    return std::all_of(contexts.cbegin(), contexts.cend(),
                       [](const ThreadContext& threadContext){
        return threadContext.intermediateVec.empty();
    });
}

void ThreadContext::reducePhase() {
    while(true){
        if(pthread_mutex_lock(&mapReduceJob.onlyOneThreadMutex) != SUCCESS){
            cerr << ERR_BAD_MUTEX_LOCK_UNLOCK;
            exit(EXIT_FAILURE);
        }
        if(mapReduceJob.shuffledVec.empty()){
            if(pthread_mutex_unlock(&mapReduceJob.onlyOneThreadMutex) != SUCCESS){
                cerr << ERR_BAD_MUTEX_LOCK_UNLOCK;
                exit(EXIT_FAILURE);
            }
            break;
        }
        auto identicalKeysVector = mapReduceJob.shuffledVec.back();
        mapReduceJob.shuffledVec.pop_back();
        if(pthread_mutex_unlock(&mapReduceJob.onlyOneThreadMutex) != SUCCESS){
            cerr << ERR_BAD_MUTEX_LOCK_UNLOCK;
            exit(EXIT_FAILURE);
        }
        mapReduceJob.client.reduce(&identicalKeysVector, &mapReduceJob);
        mapReduceJob.counter.fetch_add(identicalKeysVector.size());
    }
}

int ThreadContext::calcSumOfIntermediatePairs() const
{
	int sum = 0;
	for(ThreadContext &ctx : mapReduceJob.threadContexts){
		sum += ctx.intermediateVec.size();
	}
	return sum;
}

void ThreadContext::advanceStage()
{
    mapReduceJob.counter = mapReduceJob.counter.load() & ((uint64_t)3 << 62);
    mapReduceJob.counter.fetch_add((uint64_t)1 << 62);
}

