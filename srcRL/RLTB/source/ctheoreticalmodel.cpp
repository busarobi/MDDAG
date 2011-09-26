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
#include "ctheoreticalmodel.h"
#include "cepisodehistory.h"
#include "cutility.h"
#include "cvfunction.h"
#include "cqfunction.h"
#include "cdiscretizer.h"
#include "cstatecollection.h"
#include "cstate.h"
#include "cstateproperties.h"
#include "caction.h"

#include <math.h>

CTransition::CTransition(int startState, int endState, double prop)
{
	this->startState = startState;
	this->endState = endState;
	this->propability = prop;

	type = 1;
}

bool CTransition::isType(int Type)
{
	return (this->type & Type) > 0;
}


int CTransition::getStartState()
{
	return startState;
}

int CTransition::getEndState()
{
	return endState;
}

double CTransition::getPropability()
{
	return propability;
}

void CTransition::setPropability(double prop)
{
	propability = prop;
}

void CTransition::saveASCII(FILE *stream, bool forward)
{
	int state;
	if (forward)
	{
		state = getEndState();
	}
	else
	{
		state = getStartState();
	}
	fprintf(stream, "(%d %lf)", state, getPropability());
}

void CTransition::loadASCII(FILE *stream, int fixedState, bool forward)
{
	int state;
	fscanf(stream, "(%d %lf)", &state, &propability);
	if (forward)
	{
		endState = state;
		startState = fixedState;
	}
	else
	{
		startState = state;
		endState = fixedState;
	}
}

CSemiMDPTransition::CSemiMDPTransition(int startState, int endState, double prop) : CTransition(startState, endState, prop)
{
	durations = new std::map<int, double>();
	type = type | SEMIMDPTRANSITION;
}

CSemiMDPTransition::~CSemiMDPTransition()
{
	delete durations;
}

std::map<int, double> *CSemiMDPTransition::getDurations()
{
	return durations;
}

void CSemiMDPTransition::addDuration(int duration, double factor)
{
	double normalize = (1 - factor);
	std::map<int, double>::iterator it = durations->begin();
	for (; it != durations->end(); it++)
	{
		(*it).second *= normalize;
	}
		
	it = durations->find(duration);
	if (it != durations->end())
	{
		(*it).second += factor;
	}
	else
	{
		(*durations)[duration] = factor;
	}
}

double CSemiMDPTransition::getDurationFaktor(int duration)
{
	return (*durations)[duration];
}


double CSemiMDPTransition::getDurationPropability(int duration)
{
	return getPropability() * getDurationFaktor(duration);
}

double CSemiMDPTransition::getSemiMDPFaktor(double gamma)
{
	double factor = 0.0;
	std::map<int,double>::iterator itDurations = getDurations()->begin();
	for (; itDurations != getDurations()->end(); itDurations++)
	{
		factor += pow(gamma, (*itDurations).first - 1) * (*itDurations).second;
	}
	return factor;
}

void CSemiMDPTransition::loadASCII(FILE *stream, int fixedState, bool forward)
{
	int bufDuration;
	double bufFaktor;
	char buf;

	CTransition::loadASCII(stream, fixedState, forward);
	
	fscanf(stream, "[");
	fread(&buf, sizeof(char), 1, stream);
	while (buf != ']')
	{
		fscanf(stream, "%d %lf)", &bufDuration, &bufFaktor);
		(*durations)[bufDuration] = bufFaktor;
		fread(&buf, sizeof(char), 1, stream);
	}
}

void CSemiMDPTransition::saveASCII(FILE *stream, bool forward)
{
	CTransition::saveASCII(stream, forward);
	
	std::map<int, double>::iterator it =  this->getDurations()->begin();

	fprintf(stream, "[");
	for (; it != getDurations()->end(); it ++)
	{
		fprintf(stream, "(%d %f)", (*it).first, (*it).second);
	}
	fprintf(stream, "]");
}


CTransitionList::CTransitionList(bool forwardList)
{
	this->forwardList = forwardList;
}

CTransitionList::iterator CTransitionList::getTransitionIterator(int index)
{
	CTransitionList::iterator it = begin();
	
	if (forwardList)
	{
		while(it != end() && (*it)->getEndState() < index)
		{
			it ++;
		}
	}
	else
	{
		while(it != end() && (*it)->getStartState() < index)
		{
			it ++;
		}
	}
	return it;
}

bool CTransitionList::isMember(int featureIndex)
{
	CTransitionList::iterator it = getTransitionIterator(featureIndex);
	
	if (forwardList)
	{
		return it != end() && (*it)->getEndState() == featureIndex;
	}
	else
	{
		return it != end() && (*it)->getStartState() == featureIndex;
	}
}

bool CTransitionList::isForwardList()
{
	return forwardList;
}

void CTransitionList::addTransition(CTransition *transition)
{	
	CTransitionList::iterator it;
	
	if (isForwardList())
	{
		it = getTransitionIterator(transition->getEndState());
	}
	else
	{
		it = getTransitionIterator(transition->getStartState());
	}
	insert(it, transition);
}

CTransition *CTransitionList::getTransition(int featureIndex)
{
	CTransitionList::iterator it = getTransitionIterator(featureIndex);
	
	if (it != end())
	{
		return (*it);
	}
	else return NULL;
}

void CTransitionList::clearAndDelete()
{
	CTransitionList::iterator it = begin();
	
	for (;it != end(); it ++)
	{
		delete (*it);
	}
	clear();
}

CStateActionTransitions::CStateActionTransitions()
{
	forwardList = new CTransitionList(true);
	backwardList = new CTransitionList(false);
}

CStateActionTransitions::~CStateActionTransitions()
{
	forwardList->clearAndDelete();
	backwardList->clear();
	
	delete forwardList;
	delete backwardList;
}


CTransitionList* CStateActionTransitions::getForwardTransitions()
{
	return forwardList;
}

CTransitionList* CStateActionTransitions::getBackwardTransitions()
{
	return backwardList;
}

unsigned int CAbstractFeatureStochasticModel::getNumFeatures()
{
	return numFeatures;
}

CAbstractFeatureStochasticModel::CAbstractFeatureStochasticModel(CActionSet *actions, int numFeatures) : CActionObject(actions)
{
	this->numFeatures = numFeatures;
	discretizer = NULL;

	createdActions = false;
}

CAbstractFeatureStochasticModel::CAbstractFeatureStochasticModel(int numActions, int numFeatures) : CActionObject(new CActionSet())
{
	this->numFeatures = numFeatures;
	discretizer = NULL;

	for (int i = 0; i < numActions; i++)
	{
		actions->add(new CStochasticModelAction(this));
	}

	createdActions = true;
}


CAbstractFeatureStochasticModel::CAbstractFeatureStochasticModel(CActionSet *actions, CStateModifier *l_discretizer) : CActionObject(actions)
{
	this->numFeatures = l_discretizer->getDiscreteStateSize();
	discretizer = l_discretizer;

	createdActions = false;
}

CAbstractFeatureStochasticModel::~CAbstractFeatureStochasticModel()
{
	if (createdActions)
	{
		CActionSet::iterator it = actions->begin();

		for(; it != actions->end(); it ++)
		{
			delete *it;
		}
		delete actions;
	}
}

double CAbstractFeatureStochasticModel::getPropability(int oldState, CAction *action, int newState)
{
	int index = getActions()->getIndex(action);

	return getPropability(oldState, index, newState);
}

double CAbstractFeatureStochasticModel::getPropability(CFeatureList *oldList, CAction *action, CFeatureList *newList)
{
	double propability = 0.0;
	CFeatureList::iterator itOld ;
	CFeatureList::iterator itNew = newList->begin();

	int actoinIndex = getActions()->getIndex(action);

	for (itOld = oldList->begin(); itOld != oldList->end(); itOld ++)
	{
		for (itNew = newList->begin(); itNew != newList->end(); itNew ++)
		{
			propability += getPropability((*itOld)->featureIndex, actoinIndex, (*itNew)->featureIndex) * (*itNew)->factor * (*itOld)->factor;
		}
	}
	return propability;
}

double CAbstractFeatureStochasticModel::getPropability(CStateCollection *oldState, CAction *action, CStateCollection *newState)
{
	return getPropability(oldState->getState(discretizer), action, newState->getState(discretizer));
}

double CAbstractFeatureStochasticModel::getPropability(CState *oldState, CAction *action, CState *newState)
{
	double propability = 0;
	int actoinIndex = getActions()->getIndex(action);

	for (unsigned int i = 0; i < oldState->getNumDiscreteStates(); i ++)
	{
		for (unsigned int j = 0; j < newState->getNumDiscreteStates(); j ++)
		{
			propability += getPropability(i, actoinIndex, j) * newState->getContinuousState(j) * oldState->getContinuousState(i);
		}
	}
	return propability;
}



CTransitionList* CAbstractFeatureStochasticModel::getForwardTransitions(CAction *action, CState *state)
{
	return getForwardTransitions(actions->getIndex(action), state->getDiscreteState(0));
}

CTransitionList* 
CAbstractFeatureStochasticModel::getForwardTransitions(CAction *action, CStateCollection *state)
{
	return getForwardTransitions(action, state->getState());
}

CFeatureStochasticModel::CFeatureStochasticModel(CActionSet *actions, int numFeatures, FILE *stream) : CAbstractFeatureStochasticModel(actions, numFeatures)
{
	stateTransitions = new CMyArray2D<CStateActionTransitions *>(getNumActions(), numFeatures);
	
	for (int i = 0; i < stateTransitions->getSize(); i++)
	{
		stateTransitions->set1D(i, new CStateActionTransitions());
	}
	loadASCII(stream);
}

CFeatureStochasticModel::CFeatureStochasticModel(CActionSet *actions, int numFeatures) : CAbstractFeatureStochasticModel(actions,  numFeatures)
{
	stateTransitions = new CMyArray2D<CStateActionTransitions *>(getNumActions(), numFeatures);
	
	for (int i = 0; i < stateTransitions->getSize(); i++)
	{
		stateTransitions->set1D(i, new CStateActionTransitions());
	}
}
CFeatureStochasticModel::CFeatureStochasticModel(int numActions, int numFeatures) : CAbstractFeatureStochasticModel(numActions,  numFeatures)
{
	
	stateTransitions = new CMyArray2D<CStateActionTransitions *>(getNumActions(), numFeatures);

	for (int i = 0; i < stateTransitions->getSize(); i++)
	{
		stateTransitions->set1D(i, new CStateActionTransitions());
	}
	
}


CFeatureStochasticModel::~CFeatureStochasticModel()
{
	

	CStateActionTransitions *saPair = NULL;

	for (int i = 0; i < stateTransitions->getSize(); i++)
	{
		saPair = stateTransitions->get1D(i);
		delete saPair;
	}
	
	delete stateTransitions;
}

CTransition *CFeatureStochasticModel::getNewTransition(int startState, int endState, CAction *action, double prop)
{
	if (action->isType(MULTISTEPACTION))
	{
		return new CSemiMDPTransition(startState, endState, prop);
	}
	else
		return new CTransition(startState, endState, prop);
}

void CFeatureStochasticModel::setPropability(double propability, int oldState, int action, int newState)
{
	CStateActionTransitions *saTrans = stateTransitions->get(action, oldState);

	if (saTrans->getForwardTransitions()->isMember(newState))
	{
		saTrans->getForwardTransitions()->getTransition(newState)->setPropability(propability);
	}
	else
	{
		CTransition *trans = getNewTransition(oldState, newState, actions->get(action), propability); 

		saTrans->getForwardTransitions()->addTransition(trans);
		stateTransitions->get(action, newState)->getBackwardTransitions()->addTransition(trans);
	}
}

double CFeatureStochasticModel::getPropability(int oldState, int action, int newState)
{
	assert(action > 0);
	
	CStateActionTransitions *saTrans = stateTransitions->get(action, oldState);

	CTransition *trans = saTrans->getForwardTransitions()->getTransition(newState);
	
	if (trans == NULL)
	{
		return 0.0;
	}

	return trans->getPropability();
}

double CFeatureStochasticModel::getPropability(int oldFeature, int action, int duration, int newFeature)
{
	if (actions->get(action)->isType(MULTISTEPACTION))
	{
		CSemiMDPTransition *trans = (CSemiMDPTransition *) stateTransitions->get(action, oldFeature)->getForwardTransitions()->getTransition(newFeature);

		if (trans != NULL)
		{
			return trans->getPropability() * trans->getDurationFaktor(duration);
		}

		else
		{
			return 0.0;
		}
	}
	else
		return getPropability(oldFeature, action, newFeature);
}

void CFeatureStochasticModel::setPropability(double propability, int oldFeature, int action, int duration, int newFeature)
{
	if (actions->get(action)->isType(MULTISTEPACTION))
	{

		CSemiMDPTransition *trans = (CSemiMDPTransition *) stateTransitions->get(action, oldFeature)->getForwardTransitions()->getTransition(newFeature);
		if (trans != NULL)
		{
			double durationProp = trans->getDurationFaktor(duration) * trans->getPropability();
			trans->setPropability(trans->getPropability()  - durationProp  + propability);
			trans->addDuration(duration, (propability - durationProp) / trans->getPropability());
		}
		else
		{
			trans = (CSemiMDPTransition *) getNewTransition(oldFeature, newFeature, actions->get(action), propability);
			trans->addDuration(duration, 1.0);
		}
	}
	else
		setPropability(propability, oldFeature, action, newFeature);
}

CTransitionList *CFeatureStochasticModel::getForwardTransitions(int action, int oldState)
{
	return stateTransitions->get(action, oldState)->getForwardTransitions();
}

CTransitionList *CFeatureStochasticModel::getBackwardTransitions(int action, int oldState)
{
	return stateTransitions->get(action, oldState)->getBackwardTransitions();
}

void CFeatureStochasticModel::saveASCII(FILE *stream)
{
	CTransitionList *transList;
	CTransitionList::iterator it;

	for (unsigned int i = 0; i < getNumActions(); i++)
	{
		fprintf(stream,"Action %d\n", i);

		for (unsigned int startState = 0; startState < numFeatures; startState++)
		{
			transList = stateTransitions->get(i, startState)->getForwardTransitions();
			fprintf(stream, "Startstate %d [%d]: ", startState, transList->size());

			for (it = transList->begin(); it != transList->end(); it++)
			{
				(*it)->saveASCII(stream, true);
			}	
			fprintf(stream, "\n");
		}
	}
}

void CFeatureStochasticModel::loadASCII(FILE *stream)
{
	int action = 0;
	int startState = 0;
	int numTransitions = 0;
		
	for (unsigned int i = 0; i < getNumActions(); i++)
	{
		fscanf(stream,"Action %d\n", &action);

		for (unsigned int j = 0; j < numFeatures; j ++)
		{
			assert(fscanf(stream, "Startstate %d [%d]: ", &startState, &numTransitions) == 2);

			for (int k = 0; k < numTransitions; k++)
			{
				CTransition *newTrans = getNewTransition(0,0,actions->get(i), 0.0);
				newTrans->loadASCII(stream, j, true);
				stateTransitions->get(i,j)->getForwardTransitions()->addTransition(newTrans);
				stateTransitions->get(i, newTrans->getEndState())->getBackwardTransitions()->addTransition(newTrans);	
			}	
			fscanf(stream, "\n");
		}
	}
}

CStochasticModelAction::CStochasticModelAction(CAbstractFeatureStochasticModel *l_model)
{
	this->model = l_model;
}


bool CStochasticModelAction::isAvailable(CStateCollection *state)
{
	return model->getForwardTransitions(this, state)->size() > 0;
}

CAbstractFeatureStochasticEstimatedModel::CAbstractFeatureStochasticEstimatedModel(CStateProperties *properties, CFeatureQFunction *stateActionVisits, CActionSet *actions, int numFeatures) : CFeatureStochasticModel(actions, numFeatures), CStateObject(properties)
{
	this->stateActionVisits = stateActionVisits;

	addParameter("EstimatedModelForgetFactor", 1.0);
}


CAbstractFeatureStochasticEstimatedModel::~CAbstractFeatureStochasticEstimatedModel()
{
}

void CAbstractFeatureStochasticEstimatedModel::saveData(FILE *stream)
{
	CFeatureStochasticModel::saveASCII(stream);
}

void CAbstractFeatureStochasticEstimatedModel::loadData(FILE *stream)
{
	//double factor= 0.0;
	CFeatureStochasticModel::loadASCII(stream);
}

void CAbstractFeatureStochasticEstimatedModel::resetData()
{
	
	CStateActionTransitions *saPair = NULL;

	for (int i = 0; i < stateTransitions->getSize(); i++)
	{
		saPair = stateTransitions->get1D(i);
		delete saPair;
		stateTransitions->set1D(i, new CStateActionTransitions());
	}	
	stateActionVisits->resetData();
}



double CAbstractFeatureStochasticEstimatedModel::getStateActionVisits(int Feature, int action)
{
	CAction *actionObj = actions->get(action);
	return stateActionVisits->getValue(Feature, actionObj, NULL);
}

double CAbstractFeatureStochasticEstimatedModel::getStateVisits(int Feature)
{
	double sum = 0;

	for (unsigned int i = 0; i < getNumActions();  i++)
	{
		sum += getStateActionVisits(i, Feature);
	}
	return sum;
}

void CAbstractFeatureStochasticEstimatedModel::intermediateStep(CStateCollection *oldState, CAction *action, CStateCollection *nextState)
{
	nextStep(oldState, action, nextState);
}

void CAbstractFeatureStochasticEstimatedModel::updateStep(int oldFeature, CAction *action, int newFeature, double factor)
{
	double propability = 0.0;
//	double timeFactor = getParameter("EstimatedModelForgetFactor");

	bool found = false;

	int actionIndex = getActions()->getIndex(action);

	double newSAVisits = stateActionVisits->getValue(oldFeature, action, NULL);
	double oldSAVisits = newSAVisits - factor;

	if (newSAVisits < 0.0001)
	{
		return;
	}

	CTransitionList *transList = stateTransitions->get(actionIndex, oldFeature)->getForwardTransitions();
	
	CTransitionList::iterator trans = transList->begin();
	
	for (; trans != transList->end(); trans++)
	{
		propability = (*trans)->getPropability() * oldSAVisits;

		if ((*trans)->getEndState() == newFeature)
		{
			found = true;
			propability += factor;
			
			if (action->isType(MULTISTEPACTION))
			{
				int duration = dynamic_cast<CMultiStepAction *>(action)->getDuration();
				CSemiMDPTransition *semiTrans = (CSemiMDPTransition *) (*trans);
				semiTrans->addDuration(duration, factor / (propability));
			}
		}
		propability = propability / newSAVisits; 

		assert(propability >= 0);
		(*trans)->setPropability(propability);
	}
	
	if (! found)
	{
		setPropability(factor / newSAVisits, oldFeature, actionIndex, newFeature);
		if (action->isType(MULTISTEPACTION))
		{
			int duration = dynamic_cast<CMultiStepAction *>(action)->getDuration();
			CSemiMDPTransition *semiTrans = (CSemiMDPTransition *) transList->getTransition(newFeature);
			semiTrans->addDuration(duration, 1.0);			
			
		}
	}
}

CDiscreteStochasticEstimatedModel::CDiscreteStochasticEstimatedModel(CAbstractStateDiscretizer *discState, CFeatureQFunction *stateActionVisits, CActionSet *actions) : CAbstractFeatureStochasticEstimatedModel(discState, stateActionVisits, actions, discState->getDiscreteStateSize())
{
	
}

void CDiscreteStochasticEstimatedModel::nextStep(CStateCollection *oldState, CAction *action, CStateCollection *newState)
{
	int oldStateNum = oldState->getState(properties)->getDiscreteState(0);
    int newStateNum = newState->getState(properties)->getDiscreteState(0);;
    
	updateStep(oldStateNum, action, newStateNum, 1.0);
}

int CDiscreteStochasticEstimatedModel::getStateActionVisits(int state, int action)
{
	return (int)floor(CAbstractFeatureStochasticEstimatedModel::getStateActionVisits(state, action));
}

int CDiscreteStochasticEstimatedModel::getStateVisits(int state)
{
	int sum = 0;

	for (unsigned int i = 0; i < getNumActions();  i++)
	{
		sum += (int)floor(CAbstractFeatureStochasticEstimatedModel::getStateVisits(state));
	}
	return sum;
}

CFeatureStochasticEstimatedModel::CFeatureStochasticEstimatedModel(CFeatureCalculator *featCalc, CFeatureQFunction *stateActionVisits, CActionSet *actions) : CAbstractFeatureStochasticEstimatedModel(featCalc, stateActionVisits, actions, featCalc->getNumFeatures())
{
	addParameter("EstimatedModelMinimumUpdateFactor",0.005);
}

void CFeatureStochasticEstimatedModel::nextStep(CStateCollection *oldState, CAction *action, CStateCollection *newState)
{
	//int actionIndex = getModelProperties()->getActions()->index(action);

	CState *oldS = oldState->getState(properties);
	CState *newS = newState->getState(properties);

	double minimumUpdate = getParameter("EstimatedModelMinimumUpdateFactor");

	for (unsigned int i = 0; i < oldS->getNumContinuousStates(); i++)
	{
		for (unsigned int j = 0; j < oldS->getNumContinuousStates(); j++)
		{
			double factor = oldS->getContinuousState(i) * newS->getContinuousState(j);
			if (factor > minimumUpdate)
			{
				updateStep(oldS->getDiscreteState(i), action, newS->getDiscreteState(j), factor);
			}
		}
	}
}
