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
#include "cdynamicprogramming.h"

#include "crewardfunction.h"
#include "ctheoreticalmodel.h"
#include "cvfunction.h"
#include "cqfunction.h"
#include "cfeaturefunction.h"
#include "cagentlistener.h"
#include "cpolicies.h"
#include "cvfunctionfromqfunction.h"
#include "ril_debug.h"
#include "cstate.h"
#include "cstatecollection.h"
#include "cstateproperties.h"
#include "caction.h"
#include "cstatemodifier.h"
#include "cutility.h"

#include <math.h>

double CDynamicProgramming::getActionValue(CAbstractFeatureStochasticModel *model, CFeatureRewardFunction *rewardFunc, CAbstractVFunction *vFunction, CState *discState, CAction *action, double gamma)
{
	double V = 0;

	int feature = discState->getDiscreteState(0);

	assert(vFunction != NULL && discState->getStateProperties()->isType(DISCRETESTATE));
	
	CTransitionList *transList;

	CTransitionList::iterator itTrans;

	transList = model->getForwardTransitions(model->getActions()->getIndex(action), discState->getDiscreteState(0));

	    
	for (itTrans = transList->begin(); itTrans != transList->end(); itTrans ++)
	{
		discState->setDiscreteState(0, (*itTrans)->getEndState());
		if (action->isType(MULTISTEPACTION))
		{
			CMultiStepAction *mAction = dynamic_cast<CMultiStepAction *>(action);
			int oldDur = mAction->getDuration();
			CSemiMDPTransition *trans = ((CSemiMDPTransition *)(*itTrans));
			std::map<int,double>::iterator itDurations = trans->getDurations()->begin();
				
			for (; itDurations != trans->getDurations()->end(); itDurations++)
			{
				mAction->getMultiStepActionData()->duration = (*itDurations).first;
				V += (*itDurations).second * (*itTrans)->getPropability() * (rewardFunc->getReward(feature, mAction, (*itTrans)->getEndState()) + pow(gamma, (*itDurations).first) * vFunction->getValue(discState));
			}
			mAction->getMultiStepActionData()->duration = oldDur;
		}
		else
		{
			V += (*itTrans)->getPropability() * (rewardFunc->getReward(feature, action, (*itTrans)->getEndState()) + gamma * vFunction->getValue(discState));
		}
		DebugPrint('d', "Transition: [%d->%d, %f] ", (*itTrans)->getStartState(), (*itTrans)->getEndState(), (*itTrans)->getPropability());
		
		DebugPrint('d', "Reward: %f, V: %f\n",rewardFunc->getReward(feature, action, (*itTrans)->getEndState()), V );	
	}
	discState->setDiscreteState(0, feature);
	
	DebugPrint('d', "ActionValue (State: %d, Action: %d): %f\n", feature, model->getActions()->getIndex(action), V);

	return V;
}

double CDynamicProgramming::getBellmanValue(CAbstractFeatureStochasticModel *model, CFeatureRewardFunction *rewardFunc, CAbstractVFunction *vFunction, CState *feature, double gamma)
{
	double maxV = 0, V = 0;

	assert(vFunction != NULL);
	
	CAction *action = NULL;

	for (unsigned int naction = 0; naction < model->getNumActions(); naction ++)
	{
		V = 0;
		
		action = model->getActions()->get(naction);

		V = getActionValue(model, rewardFunc, vFunction, feature, action, gamma);
	
		if (action == 0 || maxV < V)
		{
			maxV = V;
		}
	}
	return maxV;
}

double CDynamicProgramming::getBellmanError(CAbstractFeatureStochasticModel *model, CFeatureRewardFunction *rewardFunc, CAbstractVFunction *vFunction, CState *feature, double gamma)
{
	return getBellmanValue(model, rewardFunc, vFunction, feature, gamma) - vFunction->getValue(feature);
}


CValueIteration::CValueIteration(CFeatureQFunction *qFunction, CAbstractFeatureStochasticModel *model, CFeatureRewardFunction *rewardModel)
{
	addParameters(qFunction);
	addParameter("DiscountFactor", 0.95);

	this->qFunction = qFunction;

	this->actions = qFunction->getActions();

	this->vFunction = new COptimalVFunctionFromQFunction(qFunction, qFunction->getFeatureCalculator());

	learnVFunction = false;

	init(model, rewardModel);
}

CValueIteration::CValueIteration(CFeatureQFunction *qFunction, CAbstractFeatureStochasticModel *model, CFeatureRewardFunction *rewardModel, CStochasticPolicy *stochPolicy)
{
	addParameters(qFunction);
	addParameter("DiscountFactor", 0.95);

	this->qFunction = qFunction;

	this->actions = qFunction->getActions();
	
	this->vFunction= new CVFunctionFromQFunction(qFunction, stochPolicy, qFunction->getFeatureCalculator());
	
	learnVFunction = false;

	init(model, rewardModel);
}

CValueIteration::CValueIteration(CFeatureVFunction *vFunction, CAbstractFeatureStochasticModel *model, CFeatureRewardFunction *rewardModel)
{	
	addParameters(vFunction);
	addParameter("DiscountFactor", 0.95);


	this->vFunction = vFunction;	
	this->qFunctionFromVFunction= new CQFunctionFromStochasticModel(vFunction, model, rewardModel);
	this->vFunctionFromQFunction = new COptimalVFunctionFromQFunction(qFunctionFromVFunction, vFunction->getStateProperties());

	this->actions = model->getActions();

	learnVFunction = true;

	init(model, rewardModel);
}

CValueIteration::CValueIteration(CFeatureVFunction *vFunction, CAbstractFeatureStochasticModel *model, CFeatureRewardFunction *rewardModel, CStochasticPolicy *stochPolicy)
{
	addParameters(vFunction);
	addParameter("DiscountFactor", 0.95);

	this->vFunction = vFunction;	
	this->qFunctionFromVFunction = new CQFunctionFromStochasticModel(vFunction, model, rewardModel);
	this->vFunctionFromQFunction = new CVFunctionFromQFunction(qFunctionFromVFunction, stochPolicy, vFunction->getStateProperties());

	this->actions = model->getActions();

	init(model, rewardModel);

	learnVFunction = true;
}


void CValueIteration::init(CAbstractFeatureStochasticModel *model, CFeatureRewardFunction *rewardModel)
{
	addParameter("ValueIterationMaxListSize",  model->getNumFeatures() / 4);

	this->model = model;
	this->rewardModel = rewardModel;

	this->discState = new CState(new CStateProperties(0,1,DISCRETESTATE));
	this->discState->getStateProperties()->setDiscreteStateSize(0, model->getNumFeatures());	

	this->priorityList = new CFeatureList(model->getNumFeatures() / 20, true);
}

CValueIteration::~CValueIteration()
{
	delete priorityList;

	if (learnVFunction)
	{
		delete qFunctionFromVFunction;
		delete this->vFunctionFromQFunction;
	}
	else
	{
		delete vFunction;
	}

	delete discState->getStateProperties();
	delete discState;

}

void CValueIteration::updateFirstFeature()
{
	int feature = 0;
	if (priorityList->size() > 0)
	{
		feature = (*priorityList->begin())->featureIndex;
	}
	else
	{
		feature = ((rand() * RAND_MAX)) % model->getNumFeatures();
	}
	updateFeature(feature);
}

double CValueIteration::getPriority(CTransition *trans, double bellE)
{
	if (trans->isType(SEMIMDPTRANSITION))
	{
		CSemiMDPTransition *semiTrans = (CSemiMDPTransition *) trans;
		return semiTrans->getSemiMDPFaktor(getParameter("DiscountFactor")) * trans->getPropability() * bellE;
	}
	else
	{
		return trans->getPropability() * bellE;
	}
}

void CValueIteration::updateFeature(int feature)
{
	DebugPrint('d', "Updating State: %d\n", feature);

	discState->setDiscreteState(0, feature);
	double oldV = vFunction->getValue(discState);
	double bellV = 0;
	double bellE = 0;
	double actionValue;
	CActionSet::iterator it = actions->begin();

	if (!learnVFunction)
	{
		for (int i = 0; it != actions->end(); it++, i++)
		{	
			actionValue = CDynamicProgramming::getActionValue(model, rewardModel, vFunction, discState, *it, getParameter("DiscountFactor"));
			((CFeatureVFunction *)qFunction->getVFunction(*it))->setFeature(feature, actionValue);
		}
	}
	else
	{
		((CFeatureVFunction *)vFunction)->setFeature(feature, vFunctionFromQFunction->getValue(discState));
	}

	bellV = vFunction->getValue(discState);	
	
	bellE = bellV - oldV;
	DebugPrint('d', "OldV: %f, NewV %f, bellE: %f\n", oldV, bellV, bellE);
	
	priorityList->remove(feature);

	CTransitionList *backTrans = NULL;
	CTransitionList::iterator transIt;
	
	if (fabs(bellE) >= 0.00001)
	{
		for (unsigned int action = 0; action < model->getNumActions(); action ++)
		{
			backTrans = model->getBackwardTransitions(action, feature);
			for (transIt = backTrans->begin(); transIt != backTrans->end(); transIt ++)
			{
				DebugPrint('d', "Adding State %d with Priority: %lf (prob: %lf)\n", (*transIt)->getStartState(), (*transIt)->getPropability() * bellE, (*transIt)->getPropability());
				addPriority((*transIt)->getStartState(),getPriority((*transIt), fabs(bellE)));
			}
		}
	}
}

void CValueIteration::addPriorities(CFeatureList *featList)
{
	CFeatureList::iterator it = featList->begin();
	for (; it != featList->end(); featList ++)
	{
		addPriority((*it)->featureIndex, (*it)->factor);
	}
}

void CValueIteration::addPriority(int feature, double priority)
{
	priorityList->set(feature, priority + priorityList->getFeatureFactor(feature));



	while (priorityList->size() > getParameter("ValueIterationMaxListSize"))
	{
		priorityList->remove(*priorityList->rbegin());		
	}
}

CAbstractFeatureStochasticModel *CValueIteration::getTheoreticalModel()
{
	return model;
}

CAbstractVFunction *CValueIteration::getVFunction()
{
	return vFunction;
}

int CValueIteration::getMaxListSize()
{
	return my_round(getParameter("ValueIterationMaxListSize"));
}

void CValueIteration::setMaxListSize(int maxListSize)
{
	setParameter("ValueIterationMaxListSize", maxListSize);
}

void CValueIteration::doUpdateSteps(int kSteps)
{
	for (int k = 0; k < kSteps; k ++)
	{
		updateFirstFeature();
	}
}

void CValueIteration::doUpdateStepsUntilEmptyList(int kSteps)
{
	for (int k = 0; k < kSteps && priorityList->size() > 0; k ++)
	{
		updateFirstFeature();
	}

}

CFeatureQFunction *CValueIteration::getQFunction()
{
	return qFunction;
}

CStochasticPolicy *CValueIteration::getStochasticPolicy()
{
	return stochPolicy;
}

void CValueIteration::doUpdateBackwardStates(int feature)
{
	CTransitionList *backTrans = NULL;
	CTransitionList::iterator transIt;
	
	for (unsigned int action = 0; action < model->getNumActions(); action ++)
	{
		backTrans = model->getBackwardTransitions(action, feature);
		for (transIt = backTrans->begin(); transIt != backTrans->end(); transIt ++)
		{
			updateFeature((*transIt)->getStartState());
		}
	}
}

