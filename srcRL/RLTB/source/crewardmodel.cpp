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
#include "crewardmodel.h"
#include "cagentlistener.h"
#include "ctheoreticalmodel.h"
#include "cstateproperties.h"
#include "cstatecollection.h"
#include "cstate.h"
#include "cstatemodifier.h"
#include "caction.h"
#include "cutility.h"

#include <math.h>


CFeatureStateRewardFunction::CFeatureStateRewardFunction(CStateProperties *discretizer) : CFeatureRewardFunction(discretizer)
{
	rewards = new std::map<int, double>;
}
	
CFeatureStateRewardFunction::~CFeatureStateRewardFunction()
{
	delete rewards;
}

double CFeatureStateRewardFunction::getReward(int , CAction *, int newState)
{
	return getReward(newState);
}

double CFeatureStateRewardFunction::getReward(int state)
{
	return (*rewards)[state];
}

void CFeatureStateRewardFunction::setReward(int state, double reward)
{
	(*rewards)[state] = reward;
}

CFeatureRewardModel::CFeatureRewardModel(CActionSet *actions, CRewardFunction *function, CAbstractFeatureStochasticEstimatedModel *model, CStateModifier *discretizer) : CFeatureRewardFunction(discretizer), CSemiMDPRewardListener(function), CActionObject(actions)
{
	this->rewardTable = new CMyArray2D<CFeatureMap *>(getNumActions(), discretizer->getDiscreteStateSize());

	for (int i = 0; i < rewardTable->getSize(); i++)
	{
		rewardTable->set1D(i, new CFeatureMap());
	}	
	
	this->model = model;

	this->bExternVisitSparse = true;
}

CFeatureRewardModel::CFeatureRewardModel(CActionSet *actions, CRewardFunction *function, CStateModifier *discretizer) : CFeatureRewardFunction(discretizer), CSemiMDPRewardListener(function), CActionObject(actions)
{
	int i;

	this->rewardTable = new CMyArray2D<CFeatureMap *>(getNumActions(), discretizer->getDiscreteStateSize());

	for (i = 0; i < rewardTable->getSize(); i++)
	{
		rewardTable->set1D(i, new CFeatureMap());
	}	
	
	this->visitTable = new CMyArray2D<CFeatureMap *>(getNumActions(), discretizer->getDiscreteStateSize());

	for (i = 0; i < visitTable->getSize(); i++)
	{
		visitTable->set1D(i, new CFeatureMap());
	}
	this->bExternVisitSparse = false;
}
	

CFeatureRewardModel::~CFeatureRewardModel()
{
	for (int i = 0; i < rewardTable->getSize(); i++)
	{
		delete rewardTable->get1D(i);
	}
	delete rewardTable;

	if (!bExternVisitSparse)
	{
		for (int i = 0; i < visitTable->getSize(); i++)
		{
			delete visitTable->get1D(i);
		}
		delete visitTable;
	}
}

double CFeatureRewardModel::getTransitionVisits(int oldState, int action, int newState)
{
	double visits = 0.0;
	if (!this->bExternVisitSparse)
	{	
		visits = visitTable->get(action, oldState)->getValue(newState);
	}
	else
	{
		CTransition *trans = model->getForwardTransitions(action, oldState)->getTransition(newState);
		if (trans == NULL) 
		{
			visits = 0;
		}
		else 
		{
			visits = trans->getPropability() * model->getStateActionVisits(oldState, action);
		}
	}
	return visits;
}

double CFeatureRewardModel::getReward(int oldState, CAction *action, int newState)
{
	int actionIndex = getActions()->getIndex(action);
	double transVisits = getTransitionVisits(oldState, actionIndex, newState);

	//assert(visitSparse->getFaktor(oldState, actionIndex, newState) > 0);


	if (transVisits > 0)
	{
		return rewardTable->get(actionIndex, oldState)->getValue(newState) / transVisits;
	}
	else
	{
		return 0.0;
	}
}

void CFeatureRewardModel::nextStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *newState)
{
	CFeatureMap *featMap;

	CState *oldS = oldState->getState(properties);
	CState *newS = newState->getState(properties);

	double oldreward = 0.0;
	double visits = 0.0;

	int actionIndex = getActions()->getIndex(action);
	
	int type = oldS->getStateProperties()->getType() & (DISCRETESTATE | FEATURESTATE);
	switch (type)
	{
		case FEATURESTATE:
		{
			for (unsigned int oldIndex = 0; oldIndex < oldS->getNumDiscreteStates(); oldIndex++)
			{
				int oldFeature = oldS->getDiscreteState(oldIndex);
				featMap = rewardTable->get(actionIndex, oldFeature);

				for (unsigned int newIndex = 0; newIndex < newS->getNumDiscreteStates(); newIndex++)
				{
					int newFeature = newS->getDiscreteState(newIndex);

					oldreward = featMap->getValue(newFeature);
				

					(*featMap)[newFeature] = oldreward + reward * newS->getContinuousState(newIndex) * oldS->getContinuousState(oldIndex);
					
					if (!bExternVisitSparse)
					{
						visits = visitTable->get(actionIndex, oldFeature)->getValue(newFeature);
					
						(*visitTable->get(actionIndex, oldFeature))[newFeature] = visits + newS->getContinuousState(newIndex) * oldS->getContinuousState(oldIndex);;
					}
				}
			}
			break;
		}
		case DISCRETESTATE:
		default: 
		{
			featMap = rewardTable->get(actionIndex, oldS->getDiscreteStateNumber());

			oldreward = featMap->getValue(newS->getDiscreteStateNumber());
		
			int feata = oldS->getDiscreteStateNumber();
			int featb = newS->getDiscreteStateNumber();

			(*featMap)[featb] = oldreward + reward;
			
			if (!bExternVisitSparse)
			{
				visits = visitTable->get(actionIndex, feata)->getValue(featb);
			
				(*visitTable->get(actionIndex, feata))[featb] = visits + 1.0;
			}
			break;
		}
	}
}


void CFeatureRewardModel::saveData(FILE *stream)
{
	CFeatureMap::iterator mapIt;
	CFeatureMap *featMap;
	fprintf(stream, "Reward Table\n");

	for (unsigned int action = 0; action < getNumActions(); action ++)
	{
		fprintf(stream, "Action %d:\n", action);
		for (unsigned int startState = 0; startState < discretizer->getDiscreteStateSize(); startState ++)
		{
			featMap = rewardTable->get(action, startState);

			fprintf(stream, "Startstate %d [%d]: ", startState, featMap->size());
			
			for (mapIt = featMap->begin(); mapIt != featMap->end(); mapIt ++)
			{
				fprintf(stream, "(%d %f)", (*mapIt).first, (*mapIt).second);			
			}
			fprintf(stream, "\n");
		}
		fprintf(stream, "\n");
	}

	if (!this->bExternVisitSparse)
	{
		fprintf(stream, "Visit Table\n");

		for (unsigned int action = 0; action < getNumActions(); action ++)
		{
			fprintf(stream, "Action %d:\n", action);
			for (unsigned int startState = 0; startState < discretizer->getDiscreteStateSize(); startState ++)
			{
				featMap = visitTable->get(action, startState);
	
				fprintf(stream, "Startstate %d [%d]: ", startState, featMap->size());
			
				for (mapIt = featMap->begin(); mapIt != featMap->end(); mapIt ++)
				{
					fprintf(stream, "(%d %f)", (*mapIt).first, (*mapIt).second);			
				}
				fprintf(stream, "\n");
			}
			fprintf(stream, "\n");
		}
	}
}

void CFeatureRewardModel::loadData(FILE *stream)
{
	CFeatureMap *featMap;
	fscanf(stream, "Reward Table\n");

	int buf, numVal = 0, endState;
	double reward;

	for (unsigned int action = 0; action < getNumActions(); action ++)
	{
		fscanf(stream, "Action %d:\n", &buf);
		for (unsigned int startState = 0; startState < discretizer->getDiscreteStateSize(); startState ++)
		{
			featMap = rewardTable->get(action, startState);

			featMap->clear();

			fscanf(stream, "Startstate %d [%d]: ", &buf, &numVal);
			
			for (int i = 0; i < numVal; i ++)
			{
				fscanf(stream, "(%d %lf)", &endState, &reward);
				(*featMap)[endState] = reward;
			}
			fscanf(stream, "\n");
		}
		fscanf(stream, "\n");
	}

	if (!this->bExternVisitSparse)
	{
		fprintf(stream, "Visit Table\n");

		for (unsigned int action = 0; action < getNumActions(); action ++)
		{
			fscanf(stream, "Action %d:\n", &buf);
			for (unsigned int startState = 0; startState < discretizer->getDiscreteStateSize(); startState ++)
			{
				featMap = visitTable->get(action, startState);
	
				featMap->clear();

				fscanf(stream, "Startstate %d [%d]: ", &buf, &numVal);
			
				for (int i = 0; i < numVal; i ++)
				{
					fscanf(stream, "(%d %lf)", &endState, &reward);
					(*featMap)[endState] = reward;
				}
				fscanf(stream, "\n");
			}
			fscanf(stream, "\n");
		}
	}
}

void CFeatureRewardModel::resetData()
{
	CFeatureMap *featMap;

	for (unsigned int action = 0; action < getNumActions(); action ++)
	{
		for (unsigned int startState = 0; startState < discretizer->getDiscreteStateSize(); startState ++)
		{
			featMap = rewardTable->get(action, startState);

			featMap->clear();
		}
	}

	if (!this->bExternVisitSparse)
	{
		for (unsigned int action = 0; action < getNumActions(); action ++)
		{
			for (unsigned int startState = 0; startState < discretizer->getDiscreteStateSize(); startState ++)
			{
				featMap = visitTable->get(action, startState);

				featMap->clear();
			}
		}
	}
}

CFeatureStateRewardModel::CFeatureStateRewardModel(CRewardFunction *function, CStateModifier *discretizer) : CFeatureRewardFunction(discretizer), CSemiMDPRewardListener(function) 
{
	rewards = new double[discretizer->getDiscreteStateSize(0)];
	visits = new double[discretizer->getDiscreteStateSize(0)];

	for (unsigned int i = 0; i < discretizer->getDiscreteStateSize(0); i++)
	{
		rewards[i] = 0.0;
		visits[i] = 0.0;
	}

	rewardMean = 0.0;
	numRewards = 0;
}
	
CFeatureStateRewardModel::~CFeatureStateRewardModel()
{
	delete [] rewards;
	delete [] visits;
}

	
double CFeatureStateRewardModel::getReward(CState *, CAction *, CState *newState)
{
	double reward = 0.0;

	if (newState->getStateProperties()->isType(FEATURESTATE))
	{
		for (unsigned int i = 0; i < newState->getNumContinuousStates(); i++)
		{
			reward += newState->getContinuousState(i) * getReward(newState->getDiscreteState(i));
		}
	}
	else
	{
		if (newState->getStateProperties()->isType(DISCRETESTATE))
		{
			reward = getReward(newState->getDiscreteState(0));
		}
	}
	return reward;
}
	
double CFeatureStateRewardModel::getReward(int , CAction *, int newState)
{
	return getReward(newState);
}

double CFeatureStateRewardModel::getReward(int newState)
{
	double numVisits = visits[newState];
	double reward = 0.0;

	if (numVisits > 0)
	{
		reward = rewards[newState] / numVisits;
	}
	else
	{
		if (numRewards > 0)
		{
			reward = rewardMean/ numRewards;
		}
	}
	return reward;
}

void CFeatureStateRewardModel::nextStep(CStateCollection *, CAction *, double reward, CStateCollection *newStateCol)
{
	CState *newState = newStateCol->getState(discretizer);

	if (newState->getStateProperties()->isType(FEATURESTATE))
	{
		for (unsigned int i = 0; i < newState->getNumContinuousStates(); i++)
		{
			rewards[newState->getDiscreteState(i)] +=  reward * newState->getContinuousState(i);
			visits[newState->getDiscreteState(i)] += newState->getContinuousState(i);
		}
	}
	else
	{
		if (newState->getStateProperties()->isType(DISCRETESTATE))
		{
			rewards[newState->getDiscreteState(0)] +=  reward ;
			visits[newState->getDiscreteState(0)] += 1.0;
		}
	}
}

void CFeatureStateRewardModel::saveData(FILE *stream)
{
	fprintf(stream, "State-Reward Table (%d Features):\n", discretizer->getDiscreteStateSize(0));
	for (unsigned int i = 0; i < discretizer->getDiscreteStateSize(0); i++)
	{
		fprintf(stream, "%f ", rewards[i]);
	}
	fprintf(stream, "\n");
	fprintf(stream, "State-Reward Visit Table (%d Features):\n", discretizer->getDiscreteStateSize(0));
	for (unsigned int i = 0; i < discretizer->getDiscreteStateSize(0); i++)
	{
		fprintf(stream, "%f ", visits[i]);
	}
	fprintf(stream, "\n");
}

void CFeatureStateRewardModel::loadData(FILE *stream)
{
	double bufNumRewards = 0.0;
	rewardMean = 0.0;
	int buffer;
	fscanf(stream, "State-Reward Table (%d Features):\n", &buffer);
	for (unsigned int i = 0; i < discretizer->getDiscreteStateSize(0); i++)
	{
		fscanf(stream, "%lf ", &rewards[i]);

		rewardMean += rewards[i];
	}
	fscanf(stream, "\n");
	fscanf(stream, "State-Reward Visit Table (%d Features):\n", &buffer);
	for (unsigned int i = 0; i < discretizer->getDiscreteStateSize(0); i++)
	{
		fscanf(stream, "%lf ", &visits[i]);

		bufNumRewards += visits[i];
	}
	fscanf(stream, "\n");
	numRewards = (int) floor(bufNumRewards);
}

void CFeatureStateRewardModel::resetData()
{
	for (unsigned int i = 0; i < discretizer->getDiscreteStateSize(0); i++)
	{
		rewards[i] = 0.0;
		visits[i] = 0.0;
	}

	rewardMean = 0.0;
	numRewards = 0;
}


CSemiMDPLastNRewardFunction::CSemiMDPLastNRewardFunction(CRewardFunction *rewardFunction, double l_gamma) : CRewardEpisode(rewardFunction)
{
	this->gamma = l_gamma;
}

CSemiMDPLastNRewardFunction::~CSemiMDPLastNRewardFunction()
{
}


double CSemiMDPLastNRewardFunction::getReward(CStateCollection *state, CAction *action, CStateCollection *)
{
	double reward = 0;
	unsigned int duration = action->getDuration();
	
	/*if (action->isType(MULTISTEPACTION))
	{
		printf("Final Reward %d: \nState: \n", duration);
		state->getState()->saveASCII(stdout);
	}*/

	for (unsigned int index = 0; index < duration; index ++)
	{
		if (rewards->size() < duration)
		{
			printf("Semi MDP Reward Function: %d < %d!!!\n",rewards->size(), duration );
		}
	/*	if (action->isType(MULTISTEPACTION))
		{
			printf("Single Reward: %f\n", (*rewards)[rewards->size() - duration + index]);
		}*/
		
		assert(rewards->size() >= duration);
		reward = reward + pow(gamma, (int) index) * (*rewards)[rewards->size() - duration + index];
	}
	/*if (action->isType(MULTISTEPACTION))
	{
		printf("Semi MDP Reward: %f\n", reward);
	}*/	


	return reward;
}

CRewardEpisode::CRewardEpisode(CRewardFunction *rewardFunction) : CSemiMDPRewardListener(rewardFunction)
{
	rewards = new std::vector<double>();
}

CRewardEpisode::~CRewardEpisode()
{
	delete rewards;
}

void CRewardEpisode::nextStep(CStateCollection *, CAction *, double reward, CStateCollection *)
{
	rewards->push_back(reward);
}

void CRewardEpisode::newEpisode()
{
	rewards->clear();
}

int CRewardEpisode::getNumRewards()
{
	return rewards->size();
}

double CRewardEpisode::getReward(int index)
{
	return (*rewards)[index];
}

double CRewardEpisode::getMeanReward()
{
	assert(getNumRewards() > 0);
	double mean = 0;
	for (int i = 0; i < getNumRewards(); i ++)
	{
		mean += getReward(i);
	}
	return mean / getNumRewards();
}

double CRewardEpisode::getSummedReward(double gamma)
{
	double sum = 0;
	double l_gamma = 1.0;
	for (int i = 0; i < getNumRewards(); i ++)
	{
		sum += l_gamma * getReward(i);
		l_gamma = gamma * l_gamma;
	}
	return sum;
}

double CRewardEpisode::getLastStepsMeanReward(int Steps)
{
	assert(getNumRewards() > 0);
	double mean = 0;
	for (int i = getNumRewards() - Steps; i < getNumRewards(); i ++)
	{
		mean += getReward(i);
	}
	return mean / (getNumRewards() - Steps);
}

void CRewardEpisode::saveBIN(FILE *stream)
{
	int buf = getNumRewards();
	fwrite(&buf, sizeof(int), 1, stream);
	for (int i = 0; i < buf; i ++)
	{
		double dBuf = getReward(i);
		fwrite(&dBuf, sizeof(double), 1, stream);
	}
}

void CRewardEpisode::saveData(FILE *stream)
{
	fprintf(stream, "NumRewards: %d\n", getNumRewards());
	
	for (int i = 0; i < getNumRewards(); i ++)
	{
		double dBuf = getReward(i);
		fprintf(stream, "%lf ", dBuf);
	}
	fprintf(stream, "\n");

}

void CRewardEpisode::loadBIN(FILE *stream)
{
	newEpisode();
	
	int buf;
	size_t result = fread(&buf, sizeof(int), 1, stream);
	if (result > 0)
	{
		for (int i = 0; i < buf; i ++)
		{
			double dBuf;
			fread(&dBuf, sizeof(double), 1, stream);
			rewards->push_back(dBuf);
		}
	}
}

void CRewardEpisode::loadData(FILE *stream)
{
	int buf;
	size_t result = fscanf(stream, "NumRewards: %d\n", &buf);
	
	if (result > 0)
	{
		for (int i = 0; i < buf; i ++)
		{
			double dBuf;
			fscanf(stream, "%lf ", &dBuf);
			rewards->push_back(dBuf);
		}
	}
	fscanf(stream, "\n");
}

CRewardHistorySubset::CRewardHistorySubset(CRewardHistory *l_episodes, std::vector<int> *l_indices)
{
	episodes = l_episodes;
	indices = l_indices;
}

CRewardHistorySubset::~CRewardHistorySubset()
{
}

int CRewardHistorySubset::getNumEpisodes()
{
	return indices->size();
}

CRewardEpisode* CRewardHistorySubset::getEpisode(int index)
{
	return episodes->getEpisode((*indices)[index]);
}


CRewardLogger::CRewardLogger(CRewardFunction *reward, char* autoSavefile, int holdMemory) :  CSemiMDPRewardListener(reward)
{
	init();

	this->holdMemory = holdMemory;

	setAutoSaveFile(autoSavefile);	
}

CRewardLogger::CRewardLogger(char *loadFile, CRewardFunction *reward) :  CSemiMDPRewardListener(reward)
{
	init();

	FILE* load = fopen(loadFile, "rb");
//	setmode(load, _O_BINARY);
	
	sprintf(loadFileName, "%s", loadFile);

	this->holdMemory = -1;

	loadBIN(load);

	fclose(load);
}

CRewardLogger::CRewardLogger(CRewardFunction *reward) : CSemiMDPRewardListener(reward)
{
	init();

}

void CRewardLogger::init()
{
	file = NULL;
	episodes = new std::list<CRewardEpisode *>();

	currentEpisode = new CRewardEpisode(semiMDPRewardFunction);

    holdMemory = -1;
   	loadFileName[0] = '\0';
}


void CRewardLogger::setAutoSaveFile(char *filename)
{
	if (file != NULL)
	{
		fclose(file);
	}

	if (filename != NULL)
	{
		assert(strlen(filename) > 0);
		strcpy(this->filename, filename);
		file = fopen(filename,"ab");
//		setmode(file, _O_BINARY);
	}
	else file = NULL;
}

CRewardLogger::~CRewardLogger()
{
    if (file != NULL) 
	{
		fclose(file);
	}
	for (std::list<CRewardEpisode *>::iterator it = episodes->begin(); it != episodes->end(); it++)
	{
		delete *it;
	}
	if (currentEpisode != NULL)
	{
		delete currentEpisode;
	}

	delete episodes;
	
}

void CRewardLogger::resetData()
{
	for (std::list<CRewardEpisode *>::iterator it = episodes->begin(); it != episodes->end(); it++)
	{
		delete *it;
	}
	if (currentEpisode != NULL)
	{
		delete currentEpisode;
		currentEpisode = NULL;
	}
	episodes->clear();
	
	if (strlen(loadFileName) > 0)
	{
		FILE* load = fopen(loadFileName, "rb");
	
		loadBIN(load);

		fclose(load);
	}
}

void CRewardLogger::nextStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState) 
{
	currentEpisode->nextStep(oldState, action, reward, nextState);   
}

void CRewardLogger::newEpisode()
{
	if (currentEpisode != NULL)
	{
		if (currentEpisode->getNumRewards() > 0)
		{
			if (file != NULL)
			{
				currentEpisode->saveBIN(file);
				fflush(file);
			}	
			episodes->push_back(currentEpisode);
			currentEpisode = new CRewardEpisode(semiMDPRewardFunction);
		}
	}
	else
	{
		currentEpisode = new CRewardEpisode(semiMDPRewardFunction);
	}
	
	if (holdMemory > 0 && (int)episodes->size() > holdMemory)
	{
		CRewardEpisode *firstEpisode = *episodes->begin();
		episodes->pop_front();
		delete firstEpisode;
	}
}

void CRewardLogger::saveBIN(FILE *stream)
{
	std::list<CRewardEpisode *>::iterator it;

	for (it = episodes->begin(); it != episodes->end(); it ++)
	{
		(*it)->saveBIN(stream);
	}
}

void CRewardLogger::saveData(FILE *stream)
{
	std::list<CRewardEpisode *>::iterator it;

//	fprintf(stream, "Episodes: %d\n", episodes->size());
	int i = 0;
	for (i = 0,it = episodes->begin(); it != episodes->end(); it ++, i++)
	{
		fprintf(stream, "\nRewardEpisode: %d\n", i);
		(*it)->saveData(stream);
	}
}

void CRewardLogger::loadBIN(FILE *stream, int numEpisodes)
{
	int nEpisodes = numEpisodes;

	if (holdMemory >= 0 && holdMemory < numEpisodes)
	{
		holdMemory = numEpisodes;
	}

	while (feof(stream) == 0 && (nEpisodes > 0 || nEpisodes < 0))
	{
		CRewardEpisode  *tmp = new CRewardEpisode(semiMDPRewardFunction);
		
		tmp->loadBIN(stream);
		
		//printf("Loaded Reward Episode %d with %d steps\n", nEpisodes, tmp->getNumRewards());

		if (tmp->getNumRewards() > 0)
		{
			episodes->push_back(tmp);
		}
		else
		{
			delete tmp;
		}

		nEpisodes --;
	}
	newEpisode();
}

void CRewardLogger::loadData(FILE *stream, int numEpisodes)
{
	int nEpisodes = numEpisodes;
	int buf;

	if (holdMemory >= 0 && holdMemory < numEpisodes)
	{
		holdMemory = numEpisodes;
	}

	while (feof(stream) == 0 && (nEpisodes > 0 || nEpisodes < 0))
	{
		CRewardEpisode  *tmp = new CRewardEpisode(semiMDPRewardFunction);
		fscanf(stream, "\nRewardEpisode: %d\n", &buf);
		tmp->loadData(stream);

		episodes->push_back(tmp);

		nEpisodes --;
	}
	
	newEpisode();
}

void CRewardLogger::loadData(FILE *stream)
{
	loadData(stream, -1);
}

int CRewardLogger::getNumEpisodes()
{
	return episodes->size();
}

CRewardEpisode* CRewardLogger::getEpisode(int index)
{
	int i = 0;

	std::list<CRewardEpisode *>::iterator it;

	assert(index < getNumEpisodes());

	for (it = episodes->begin(); it != episodes->end(), i < index; it++, i++)
	{
	
	}

	return *it;
}

CRewardEpisode* CRewardLogger::getCurrentEpisode()
{
	return currentEpisode;
}

void CRewardLogger::clearAutoSaveFile()
{
	if (file != NULL)
	{
		fclose(file);
		file = fopen(filename, "wb");
//		setmode(file, _O_BINARY);
	}
}

void CRewardLogger::setLoadDataFile(char *loadData)
{
	sprintf(loadFileName, "%s", loadData);
}
