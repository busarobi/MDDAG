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


#include "cmontecarlo.h"
#include <math.h>

#include "cagent.h"
#include "ctransitionfunction.h"
#include "crewardfunction.h"
#include "ril_debug.h"
#include "cepisodehistory.h"
#include "cepisode.h"
#include "crewardmodel.h"
#include "csupervisedlearner.h"
#include "cstate.h"
#include "cstatecollection.h"
#include "cstateproperties.h"
#include "cepisode.h"
#include "cvfunction.h"

CMonteCarloError::CMonteCarloError(CAgent *l_agent, CRewardFunction *reward, CStateProperties *modelState, CActionSet *actions, std::list<CStateModifier *> *modifiers, int l_nEpisodes, int l_nStepsPerEpisode, double discountFactor)
{
	addParameter("DiscountFactor", discountFactor);
	
 	this->rewardFunction = reward;
	this->agent = l_agent;
	this->nEpisodes = l_nEpisodes;
	this->nStepsPerEpisode = l_nStepsPerEpisode;
	
	episode = new CEpisode(modelState, actions, modifiers, true);
	
	oldState = new CStateCollectionImpl(modelState, modifiers);
	newState = new CStateCollectionImpl(modelState, modifiers);

	semiMDPSender = agent;
	
	useRewardEpisode = true;
	
	episodeHistory = NULL;

	this->errorFunction = MC_MSE;
}

CMonteCarloError::~CMonteCarloError()
{
	delete episode;
	
	delete oldState;
	delete newState;
}

void CMonteCarloError::setSemiMDPSender(CSemiMDPSender *sender)
{
	semiMDPSender = sender;
}

void CMonteCarloError::setEpisodeHistory(CEpisodeHistory *l_episodeHistory, CRewardHistory *l_rewardLogger)
{
	rewardLogger = l_rewardLogger;
	this->episodeHistory = l_episodeHistory;
}

double CMonteCarloError::getMeanMonteCarloError(CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger)
{
	double mean = 0;
	for (int i = 0; i < episodeHistory->getNumEpisodes(); i ++)
	{
		if (rewardLogger)
		{
			mean +=	getMonteCarloError(episodeHistory->getEpisode(i), rewardLogger->getEpisode(i));
		}
		else
		{
			mean +=	getMonteCarloError(episodeHistory->getEpisode(i), NULL);
		}
	}
	
	return mean / episodeHistory->getNumSteps();
}

double CMonteCarloError::getMonteCarloError(CEpisode *episode, CRewardEpisode *rewardEpisode)
{
	double error = 0;
	double cum_reward = 0;
	double gamma = getParameter("DiscountFactor");
	int step = episode->getNumSteps();
	episode->getStateCollection(step, oldState);
	
	if (!oldState->getState()->isResetState())
	{
		cum_reward = getValue(oldState, NULL);
	}
	
	for (step = step - 1; step >= 0; step --)
	{
		CStateCollectionImpl *temp = oldState;
		oldState = newState;
		newState = temp;
		episode->getStateCollection(step, oldState);
		CAction *action = episode->getAction(step);
		
		double l_reward = 0;
		
		if (rewardEpisode == NULL)
		{
		 	l_reward = rewardFunction->getReward(oldState, action, newState);
		}
		else
		{
			l_reward = rewardEpisode->getReward(step);
		}
		
		cum_reward = l_reward + gamma * cum_reward;
		
		double value = getValue(oldState, action);
		
		//printf("MC-Error %d : %f %f %f %f\n", step, cum_reward, l_reward, value, fabs(cum_reward - getValue(oldState, action)));
				
		switch (errorFunction)
		{
			case MC_MAE:
			{
				error = error + fabs(cum_reward - value);
				break;
			}
			case MC_MSE:
			{
				error = error + pow(cum_reward - value, 2.0);
				break;
			}
			default:
			{
				error = error + pow(cum_reward - value, 2.0);
			}
		}
		
	}
	return  error;
}

double CMonteCarloError::evaluate()
{
	if (episodeHistory != NULL)
	{
		return - getMeanMonteCarloError(episodeHistory, rewardLogger);
	}
	
	semiMDPSender->addSemiMDPListener(episode);
	CRewardEpisode *rewardEpisode = NULL;

	if (useRewardEpisode)
	{
		rewardEpisode = new CRewardEpisode(rewardFunction);
		semiMDPSender->addSemiMDPListener(rewardEpisode);
	}
	
	double error = 0;
	for (int i = 0; i < nEpisodes; i ++)
	{
		agent->startNewEpisode();
		agent->doControllerEpisode(1, nStepsPerEpisode);
		error += getMonteCarloError(episode, rewardEpisode);
	}
	error = error / nEpisodes;
	
	semiMDPSender->removeSemiMDPListener(episode);

	if (useRewardEpisode)
	{
		semiMDPSender->removeSemiMDPListener(rewardEpisode);
		delete rewardEpisode;	
	}

	
	return - error;
}


double CMonteCarloVError::getValue(CStateCollection *state, CAction *)
{
	return vFunction->getValue(state);
}

CMonteCarloVError::CMonteCarloVError(CAbstractVFunction *vFunction, CAgent *agent, CRewardFunction *reward, CStateProperties *modelState, CActionSet *actions, std::list<CStateModifier *> *modifiers, int numEpisodes, int numSteps, double discountFactor)
	: CMonteCarloError(agent, reward, modelState, actions, modifiers, numEpisodes, numSteps, discountFactor)
{
	this->vFunction = vFunction;
}
	
CMonteCarloVError::~CMonteCarloVError()
{
}

double CMonteCarloQError::getValue(CStateCollection *state, CAction *action)
{
	if (action != NULL)
	{
		return qFunction->getValue(state, action);
	}
	else
	{	CActionSet availAbleActions;
		episode->getActions()->getAvailableActions(&availAbleActions, state);
		
		if (availAbleActions.size() > 0)
		{
			return qFunction->getMaxValue(state, &availAbleActions);
		}
		else
		{
			printf("MC Q-Error: No Actions available!!\n");
			return 0;
		}
	}
}

CMonteCarloQError::CMonteCarloQError(CAbstractQFunction *l_qFunction, CAgent *agent, CRewardFunction *reward, CStateProperties *modelState, CActionSet *actions, std::list<CStateModifier *> *modifiers, int numEpisodes, int numSteps, double discountFactor)
	: CMonteCarloError(agent, reward, modelState, actions, modifiers, numEpisodes, numSteps, discountFactor)
{
	this->qFunction = l_qFunction;
}
	
CMonteCarloQError::~CMonteCarloQError()
{
}

CMonteCarloSupervisedLearner::CMonteCarloSupervisedLearner(CEpisodeHistory *l_episodeHistory, CRewardHistory *l_rewardLogger, CBatchDataGenerator *l_dataGenerator)
{
	episodeHistory = l_episodeHistory;
	rewardLogger = l_rewardLogger;
	dataGenerator = l_dataGenerator;
	
	addParameters(dataGenerator);
	addParameter("DiscountFactor", 0.95);
}

CMonteCarloSupervisedLearner::~CMonteCarloSupervisedLearner()
{

}

void CMonteCarloSupervisedLearner::evaluatePolicy(int )
{
	dataGenerator->resetPolicyEvaluation();	

	CActionDataSet dataSet(episodeHistory->getActions());
	
	CStateCollectionImpl *oldState = new CStateCollectionImpl(episodeHistory->getStateProperties(), episodeHistory->getStateModifiers());	

	for (int i = 0; i < episodeHistory->getNumEpisodes(); i++)
	{
		double cum_reward = 0;
		double gamma = getParameter("DiscountFactor");
		
		CEpisode *episode = episodeHistory->getEpisode(i);
		CRewardEpisode *rewardEpisode = rewardLogger->getEpisode(i); 
		
		int step = episode->getNumSteps();
		
		episode->getStateCollection(step, oldState);
		
		if (!oldState->getState()->isResetState())
		{
			cum_reward = dataGenerator->getValue(oldState, NULL);
		}
		
		for (step = step - 1; step >= 0; step --)
		{
			episode->getStateCollection(step, oldState);
			CAction *action = episode->getAction(step, &dataSet);
			
			double l_reward = 0;
			
			l_reward = rewardEpisode->getReward(step);
						
			cum_reward = l_reward + gamma * cum_reward;
			
			dataGenerator->addInput(oldState, action, cum_reward); 			
		}
	}
	delete oldState;
	dataGenerator->trainFA();
}

CMonteCarloVLearner::CMonteCarloVLearner(CAbstractVFunction *l_vFunction,  CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger, CSupervisedLearner *l_learner) : CMonteCarloSupervisedLearner(episodeHistory, rewardLogger, new CBatchVDataGenerator(l_vFunction, l_learner))
{
}

CMonteCarloVLearner::~CMonteCarloVLearner()
{
	delete dataGenerator;
}

CMonteCarloCAQLearner::CMonteCarloCAQLearner(CStateProperties *properties, CContinuousActionQFunction *l_qFunction,  CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger, CSupervisedLearner *learner) : CMonteCarloSupervisedLearner(episodeHistory, rewardLogger, new CBatchCAQDataGenerator(properties, l_qFunction, learner))
{

}

CMonteCarloCAQLearner::~CMonteCarloCAQLearner()
{
	delete dataGenerator;
}


CMonteCarloQLearner::CMonteCarloQLearner(CQFunction *qFunction,  CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger, CSupervisedQFunctionLearner *learner) : CMonteCarloSupervisedLearner(episodeHistory, rewardLogger, new CBatchQDataGenerator(qFunction, learner))
{

}

CMonteCarloQLearner::~CMonteCarloQLearner()
{
	delete dataGenerator;
}







