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
#include "cdiscretizer.h"

#include "cstatecollection.h"
#include "cstate.h"

CAbstractStateDiscretizer::CAbstractStateDiscretizer(unsigned int numStates) : CStateModifier(0, 1)
{
	type = DISCRETESTATE;
	setDiscreteStateSize(0, numStates);

	stateSubstitutions = new std::map<int, std::pair<CStateModifier *, CState *>*>();
}

CAbstractStateDiscretizer::~CAbstractStateDiscretizer()
{
	std::map<int, std::pair<CStateModifier *, CState *>*>::iterator it = stateSubstitutions->begin();

	for (; it != stateSubstitutions->end(); it++)
	{
		delete (*it).second->second;
		delete (*it).second;
	}

	delete stateSubstitutions;
}

unsigned int CAbstractStateDiscretizer::getDiscreteStateSize()
{
	return CStateProperties::getDiscreteStateSize(0);
}

void CAbstractStateDiscretizer::getModifiedState(CStateCollection *state, CState *featState)
{
//	assert(equals(featState->getStateProperties()));
	int discState = getDiscreteStateNumber(state);
	int stateOffset = 0;
	featState->resetState();

	std::map<int, std::pair<CStateModifier *, CState *>*>::iterator it = stateSubstitutions->begin();
	
	while (it != stateSubstitutions->end() && (*it).first < discState)
	{
		stateOffset += (*it).second->first->getDiscreteStateSize() - 1;
		it++;
	}

	if (it != stateSubstitutions->end() && (*it).first == discState)
	{
		CStateModifier *stateMod = (*it).second->first;
		CState *stateBuf;
	
		if (state->isMember(stateMod))
		{
			stateBuf = state->getState(stateMod);
		}
		else
		{
			stateBuf = (*it).second->second;
			stateMod->getModifiedState(state, stateBuf);
		}
	
		int type = stateMod->getType() & (FEATURESTATE | DISCRETESTATE);
		switch (type)
		{
			case FEATURESTATE:
			{
				featState->setNumActiveContinuousStates(stateBuf->getNumActiveContinuousStates());
				featState->setNumActiveDiscreteStates(stateBuf->getNumActiveDiscreteStates());
				unsigned int i;
				for (i = 0; i < featState->getNumActiveContinuousStates(); i++)
				{
					featState->setContinuousState(i, stateBuf->getContinuousState(i));
				}

				for (i = 0; i < featState->getNumActiveDiscreteStates(); i++)
				{
					featState->setDiscreteState(i, stateOffset + discState +  stateBuf->getDiscreteState(i));
				}
				break;
			}
			case DISCRETESTATE:
			{
				featState->setNumActiveContinuousStates(0);
				featState->setNumActiveDiscreteStates(1);
			
				featState->setDiscreteState(0, stateOffset + discState + stateBuf->getDiscreteState(0));
				break;
			}
		}
		//printf("Applied State Substitution: %d -> %d\n", discState, featState->getDiscreteState(0));
	}
	else
	{
		featState->setNumActiveContinuousStates(0);
		featState->setNumActiveDiscreteStates(1);

		featState->setDiscreteState(0, stateOffset + discState);
	}
}

void CAbstractStateDiscretizer::addStateSubstitution(int discState, CStateModifier *modifier)
{
	std::pair<CStateModifier *, CState *> *stateMod = new std::pair<CStateModifier *, CState *>(modifier, new CState(modifier));

	(*stateSubstitutions)[discState] = stateMod;

	if (modifier->getNumContinuousStates() > this->getNumContinuousStates())
	{
		continuousStates = modifier->getNumContinuousStates();
	}
	
	if (modifier->getNumDiscreteStates() > this->getNumDiscreteStates())
	{
		discreteStates = modifier->getNumDiscreteStates();
	}

	this->setDiscreteStateSize(0, getDiscreteStateSize() + modifier->getDiscreteStateSize() - 1);

	if (modifier->isType(FEATURESTATE))
	{
		type = FEATURESTATE;
	}
}

void CAbstractStateDiscretizer::removeStateSubstitution(int discState)
{
	std::map<int, std::pair<CStateModifier *, CState *>*>::iterator it = stateSubstitutions->find(discState);

	if (it != stateSubstitutions->end())
	{
		delete (*it).second->second;
		delete (*it).second;

		stateSubstitutions->erase(it);
	}

}


CModelStateDiscretizer::CModelStateDiscretizer(CStateProperties *prop, int *discreteStates, unsigned int num) : CAbstractStateDiscretizer(calcDiscreteStateSize(prop, discreteStates, num))
{
	originalState = prop;
	this->numDiscStateVar = num;
	this->discreteStates = new int[num];

	memcpy(this->discreteStates, discreteStates, sizeof(int) * num);
}

CModelStateDiscretizer::~CModelStateDiscretizer()
{
	delete discreteStates;
}

unsigned int CModelStateDiscretizer::calcDiscreteStateSize(CStateProperties *prop, int *discreteStates, unsigned int num)
{
	unsigned int dim = 1;

	if (discreteStates == NULL)
	{
		dim = prop->getDiscreteStateSize();
	}
	else
	{
		for (unsigned int i = 0; i < num; i ++)
		{
			dim *= prop->getDiscreteStateSize(discreteStates[i]);
		}
	}
	return dim;
	
}

unsigned int CModelStateDiscretizer::getDiscreteStateNumber(CStateCollection *state)
{
	unsigned int dim = 1;
	unsigned int statenum = 0;

	if (discreteStates == NULL)
	{
		for (unsigned int i = 0; i < state->getState(originalState)->getNumDiscreteStates(); i ++)
		{
			statenum += state->getState(originalState)->getDiscreteState(i) * dim;
			dim *= state->getState(originalState)->getStateProperties()->getDiscreteStateSize(i);
		}
	}
	else
	{
		for (unsigned int i = 0; i < this->numDiscStateVar; i ++)
		{
			statenum += state->getState(originalState)->getDiscreteState(discreteStates[i]) * dim;
			dim *= state->getState(originalState)->getStateProperties()->getDiscreteStateSize(discreteStates[i]);
		}
	}
	return statenum;
}	



CSingleStateDiscretizer::CSingleStateDiscretizer(int dimension, int numPartitions, double *partitions) : CAbstractStateDiscretizer(numPartitions + 1)
{
	this->dimension = dimension;
	this->partitions = new double[numPartitions];

	memcpy(this->partitions, partitions, sizeof(double) * numPartitions);

	originalState = NULL;
}

CSingleStateDiscretizer::~CSingleStateDiscretizer()
{
	delete partitions;
}


unsigned int CSingleStateDiscretizer::getDiscreteStateNumber(CStateCollection *state)
{
	double contState = state->getState()->getContinuousState(dimension);

	unsigned int activeFeature = 0;
	
	while (activeFeature < getDiscreteStateSize() - 1 && partitions[activeFeature] <= contState)
	{
		activeFeature++;
	}
	return activeFeature;
}

void CSingleStateDiscretizer::setOriginalState(CStateProperties *originalState)
{
	this->originalState = originalState;
}

CDiscreteStateOperatorAnd::CDiscreteStateOperatorAnd() : CAbstractStateDiscretizer(1)
{
}

CDiscreteStateOperatorAnd::~CDiscreteStateOperatorAnd()
{
}

unsigned int CDiscreteStateOperatorAnd::getDiscreteStateNumber(CStateCollection *state)
{
	int stateOffset = 1;
	int discState = 0;
	int ldiscState = 0;

	std::list<CStateModifier *>::iterator it = getStateModifiers()->begin();
	std::list<CState *>::iterator itStates = this->states->begin();

	CState *stateBuf;
		
	for (; it != getStateModifiers()->end(); itStates ++, it ++)
	{
		stateBuf = NULL;
		if (state->isMember(*it))
		{
			stateBuf = state->getState(*it);
			ldiscState = stateBuf->getDiscreteState(0);
		}
		else
		{
			stateBuf = (*itStates);
            (*it)->getModifiedState(state,stateBuf);
			ldiscState = stateBuf->getDiscreteState(0);
		}
		
		discState += ldiscState * stateOffset;
		stateOffset = stateOffset * (*it)->getDiscreteStateSize();
	}
	return discState;
}

void CDiscreteStateOperatorAnd::addStateModifier(CAbstractStateDiscretizer *featCalc)
{
	CStateMultiModifier::addStateModifier(featCalc);
	this->setDiscreteStateSize(0, getDiscreteStateSize() * featCalc->getDiscreteStateSize());
}

