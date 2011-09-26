// Copyright (C) 2003
// Gerhard Neumann (gneumann@gmx.net)
// Stephan Neumann (sneumann@gmx.net) 
//                
// This file is part of RL Toolbox.
// http://www.igi.tugraz.at/ril_toolbox
//
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products
//    derived from this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
// IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
// NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "ril_debug.h"
#include "cepisodehistory.h"
#include "cstatecollection.h"
#include "ril_debug.h"
#include "caction.h"
#include "cepisode.h"

#include <sys/time.h>
#include <assert.h>



CEpisodeHistory::CEpisodeHistory(CStateProperties *prop, CActionSet *actions) : CStateModifiersObject(prop), CStepHistory(properties, actions)
{
	stepToEpisodeMap = NULL;
	
	episodeOffsetMap = NULL;
}

CEpisodeHistory::~CEpisodeHistory()
{
	if (stepToEpisodeMap)
	{
		delete stepToEpisodeMap;
		delete episodeOffsetMap;
	}
}

void CEpisodeHistory::createStepToEpisodeMap()
{
	if (stepToEpisodeMap == NULL)
	{
		stepToEpisodeMap = new std::map<int, CEpisode *>;
		episodeOffsetMap = new std::map<CEpisode *, int>;
	}
	stepToEpisodeMap->clear();
	int step = 0;
	for (int i = 0; i < getNumEpisodes(); i ++)
	{
		CEpisode *episode = getEpisode(i);
		
		(*episodeOffsetMap)[episode] = step;
		
		for (int j = 0; j < episode->getNumSteps(); j++)
		{
			(*stepToEpisodeMap)[step] = episode;
			step ++;
		}
		
	}
}

int CEpisodeHistory::getNumSteps()
{
	int sum = 0;

	int i = 0;

	for (i = 0; i < getNumEpisodes(); i++)
	{
		sum += getEpisode(i)->getNumSteps();
	}

	return sum;
}


void CEpisodeHistory::getStep(int index, CStep *step)
{
	assert(index < getNumSteps());

//	struct timeval t1;
//	struct timeval t2;
//	struct timeval t3;
	
	int lnum = index;
	
	int i = 0;
	
	CEpisode *episode;

//	gettimeofday(&t1, NULL);

	if (stepToEpisodeMap)
	{
		episode = (*stepToEpisodeMap)[index];
		lnum = lnum - (*episodeOffsetMap)[episode];
	}
	else
	{
		while (lnum >= (episode = getEpisode(i))->getNumSteps())
		{	
			lnum -= episode->getNumSteps();
			i ++;
		}
	}
	
//	gettimeofday(&t2, NULL);
	
	episode->getStep(lnum, step);
		
//	gettimeofday(&t3, NULL);
//	
//	static int count = 0;
//	static double dur1 = 0;
//	static double dur2 = 0;
//	
//	dur1 += (t2.tv_sec - t1.tv_sec) * 1000 + t2.tv_usec - t1.tv_usec;
//	dur2 += (t3.tv_sec - t1.tv_sec) * 1000 + t3.tv_usec - t1.tv_usec;
	
//	count ++;
//	if (count % 100 == 0)
//	{
//		printf("Time for step retrieval: %f %f\n", dur1, dur2);
//		dur1 = 0;
//		dur2 = 0;
//	}
}
/*
void CEpisodeHistory::simulateEpisode(int index, CSemiMDPListener *listener)
{
	assert(index < this->getNumEpisodes());
	
	CEpisode *episode = getEpisode(index);
	
	CStep *step = new CStep(getStateProperties(), getStateModifiers());

	listener->newEpisode();

	int i = 0;

	

	for (i = 0; i < episode->getNumSteps(); i++)
	{
		episode->getStep(i, step);

		listener->nextStep(step->oldState, step->action, step->newState);
	}

	listener->newEpisode();

	delete step;
}

void CEpisodeHistory::simulateAllEpisodes(CSemiMDPListener *listener)
{
	for (int i = 0; i < getNumEpisodes(); i++)
	{
		simulateEpisode(i, listener);
	}
}

void CEpisodeHistory::simulateNRandomEpisodes(int numEpisodes, CSemiMDPListener *listener)
{

	std::list<int> *episodeIndex = new std::list<int>();
	for (int i = 0; i < getNumEpisodes(); i++)
	{
		episodeIndex->push_back(i);
	}
	for (int i = 0; i < numEpisodes && episodeIndex->size() > 0; i++)
	{
		int nRand = rand() % episodeIndex->size();
		std::list<int>::iterator it = episodeIndex->begin();
		for (int j = 0; j < nRand && it != episodeIndex->end(); j ++, it ++);
		nRand = *it;
		episodeIndex->erase(it);
		simulateEpisode(nRand, listener);
	}
}
*/

CEpisodeHistorySubset::CEpisodeHistorySubset(CEpisodeHistory *l_episodes, std::vector<int> *l_indices) : CStateModifiersObject(l_episodes->getStateProperties(), l_episodes->getStateModifiers()) , CEpisodeHistory(l_episodes->getStateProperties(), l_episodes->getActions())
{
	episodes = l_episodes;
	indices = l_indices;
}

CEpisodeHistorySubset::~CEpisodeHistorySubset()
{
}

int CEpisodeHistorySubset::getNumEpisodes()
{
	return indices->size();
}

CEpisode* CEpisodeHistorySubset::getEpisode(int index)
{
	return episodes->getEpisode((*indices)[index]);
}




CStoredEpisodeModel::CStoredEpisodeModel(CEpisodeHistory *history) : CEnvironmentModel(history->getStateProperties()), CAgentController(history->getActions())
{
	this->history = history;

	this->numEpisode = -1;
	this->numStep = 0;

	doResetModel();
}

CStoredEpisodeModel::~CStoredEpisodeModel()
{
}

void CStoredEpisodeModel::doNextState(CPrimitiveAction *)
{
	numStep ++;

	if (numStep >= currentEpisode->getNumSteps())
	{
		reset = true;
	}

}

void CStoredEpisodeModel::doResetModel() 
{
	if (reset == true)
	{
		numEpisode ++;
		if (numEpisode >= history->getNumEpisodes())
		{
			numEpisode = 0;
		}
	}
	currentEpisode = history->getEpisode(numEpisode);
	numStep = 0;
	if (currentEpisode->getNumSteps() == 0)
	{
		doResetModel();
	}
}

CEpisodeHistory* CStoredEpisodeModel::getEpisodeHistory()
{
	return history;
}

void CStoredEpisodeModel::setEpisodeHistory(CEpisodeHistory *hist)
{
	history = hist;
}


void CStoredEpisodeModel::getState(CState *state)
{
	currentEpisode->getState(numStep, state);
}

void CStoredEpisodeModel::getState(CStateCollectionImpl *stateCollection)
{
	stateCollection->newModelState();
	currentEpisode->getStateCollection(numStep, stateCollection);
}


CAction* CStoredEpisodeModel::getNextAction(CStateCollection *)
{
	if (numStep >= currentEpisode->getNumSteps())
	{
		return NULL;
	}
	else
	{
		return currentEpisode->getAction(numStep);	
	}
}

CBatchEpisodeUpdate::CBatchEpisodeUpdate(CSemiMDPListener *listener, CEpisodeHistory *logger, int numEpisodes, std::list<CStateModifier *> *modifiers)
{
	this->listener = listener;
	this->logger = logger;
	this->numEpisodes = numEpisodes;

	dataSet= new CActionDataSet(logger->getActions());

	step = new CStep(logger->getStateProperties(), modifiers, logger->getActions());

}

CBatchEpisodeUpdate::~CBatchEpisodeUpdate()
{
	delete dataSet;
	delete step;
}

void CBatchEpisodeUpdate::newEpisode()
{
	simulateNRandomEpisodes(numEpisodes, listener);
}

void CBatchEpisodeUpdate::simulateEpisode(int index, CSemiMDPListener *listener)
{
	assert(index < logger->getNumEpisodes());

	CEpisode *episode = logger->getEpisode(index);


	listener->newEpisode();

	int i = 0;

	CActionSet::iterator it = logger->getActions()->begin();
	for (; it != logger->getActions()->end(); it ++)
	{
		dataSet->setActionData(*it, (*it)->getActionData());
	}

	for (i = 0; i < episode->getNumSteps(); i++)
	{
		episode->getStep(i, step);
		step->action->loadActionData(step->actionData->getActionData(step->action));

		listener->nextStep(step->oldState, step->action, step->newState);
	}
	it = logger->getActions()->begin();
	for (; it != logger->getActions()->end(); it ++)
	{
		(*it)->loadActionData(dataSet->getActionData(*it));
	}

	listener->newEpisode();
}

void CBatchEpisodeUpdate::simulateAllEpisodes(CSemiMDPListener *listener)
{
	for (int i = 0; i < logger->getNumEpisodes(); i++)
	{
		simulateEpisode(i, listener);
	}
}

void CBatchEpisodeUpdate::simulateNRandomEpisodes(int numEpisodes, CSemiMDPListener *listener)
{

	std::list<int> *episodeIndex = new std::list<int>();
	for (int i = 0; i < logger->getNumEpisodes(); i++)
	{
		episodeIndex->push_back(i);
	}
	for (int i = 0; i < numEpisodes && episodeIndex->size() > 0; i++)
	{
		int nRand = rand() % episodeIndex->size();
		std::list<int>::iterator it = episodeIndex->begin();
		for (int j = 0; j < nRand && it != episodeIndex->end(); j ++, it ++);
		nRand = *it;
		episodeIndex->erase(it);
		simulateEpisode(nRand, listener);
	}
	delete episodeIndex;
}
