//
// Created by adirt on 23/04/2022.
//

#ifndef OS_EX3_THREADCONTEXT_H
#define OS_EX3_THREADCONTEXT_H

#include <atomic>
#include "MapReduceFramework.h"
class MapReduceJob;
class ThreadContext
{
public:
    typedef std::pair<IntermediatePair, unsigned int> ShufflePair;
	const unsigned int id;
	IntermediateVec intermediateVec;
	MapReduceJob& mapReduceJob;
	static unsigned int uniqueID;
	ThreadContext(MapReduceJob& mapReduceJob);
	void mapPhase();
	void sortPhase();
	void barrier();
	void shufflePhase();
	int getNextShufflePairVectorIndex(unsigned int cameFromIndex) const;
    bool areIntermediateVectorsAreEmpty() const;
    void reducePhase();
	int calcSumOfIntermediatePairs() const;
	void advanceStage();
};


#endif //OS_EX3_THREADCONTEXT_H
