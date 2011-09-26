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

#ifndef C_MONTECARLO__H
#define C_MONTECARLO__H

#include "cbatchlearning.h"
#include "cevaluator.h"
#include "cparameters.h"

#define MC_MSE 0
#define MC_MAE 1

class CAgent;
class CEpisode;
class CRewardFunction;
	
class CStateCollectionImpl;
class CStateCollectionImpl;
	
class CSemiMDPSender;
class CRewardHistory;
class CRewardEpisode;
class CEpisodeHistory;

class CMonteCarloError : public CEvaluator, public CParameterObject
{
protected:
	CAgent *agent;
	CEpisode *episode;
	CRewardFunction *rewardFunction;
	
	CStateCollectionImpl *oldState;
	CStateCollectionImpl *newState;
	
	int nEpisodes;
	int nStepsPerEpisode;
	
	CSemiMDPSender *semiMDPSender;

	virtual double getValue(CStateCollection *state, CAction *action) = 0;
	
	CRewardHistory *rewardLogger;
	CEpisodeHistory *episodeHistory;
public:	
	bool useRewardEpisode;
	int errorFunction;

	CMonteCarloError(CAgent *agent, CRewardFunction *reward, CStateProperties *modelState, CActionSet *actions, std::list<CStateModifier *> *modifiers, int numEpisodes, int numSteps, double discountFactor);
	virtual ~CMonteCarloError();
	
	void setEpisodeHistory(CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger);

	double getMonteCarloError(CEpisode *episode, CRewardEpisode *rewardEpisode);
	double getMeanMonteCarloError(CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger);

	virtual double evaluate();
	
	void setSemiMDPSender(CSemiMDPSender *sender);
	
};

class CMonteCarloVError : public CMonteCarloError
{
protected:
	CAbstractVFunction *vFunction;
	
	virtual double getValue(CStateCollection *state, CAction *action);
public:	
	CMonteCarloVError(CAbstractVFunction *vFunction, CAgent *agent, CRewardFunction *reward, CStateProperties *modelState, CActionSet *actions, std::list<CStateModifier *> *modifiers, int numEpisodes, int numSteps, double discountFactor);
	virtual ~CMonteCarloVError();	
};

class CMonteCarloQError : public CMonteCarloError
{
protected:
	CAbstractQFunction *qFunction;
	
	virtual double getValue(CStateCollection *state, CAction *action);
public:	
	CMonteCarloQError(CAbstractQFunction *vFunction, CAgent *agent, CRewardFunction *reward, CStateProperties *modelState, CActionSet *actions, std::list<CStateModifier *> *modifiers, int numEpisodes, int numSteps, double discountFactor);
	virtual ~CMonteCarloQError();	
};

class CMonteCarloSupervisedLearner : public CPolicyEvaluation
{
protected:
		
	CEpisodeHistory *episodeHistory;
	CRewardHistory *rewardLogger;

	CBatchDataGenerator *dataGenerator;

public:
	CMonteCarloSupervisedLearner(CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger, CBatchDataGenerator *dataGenerator);

	virtual ~CMonteCarloSupervisedLearner();

	virtual void evaluatePolicy(int trials);
};



class CMonteCarloVLearner : public CMonteCarloSupervisedLearner
{
protected:
public:
	CMonteCarloVLearner(CAbstractVFunction *vFunction,  CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger, CSupervisedLearner *learner);

	virtual ~CMonteCarloVLearner();
};

class CMonteCarloCAQLearner : public CMonteCarloSupervisedLearner
{
protected:
	
public:
	CMonteCarloCAQLearner(CStateProperties *properties, CContinuousActionQFunction *qFunction, CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger, CSupervisedLearner *learner);

	virtual ~CMonteCarloCAQLearner();
};


class CMonteCarloQLearner : public CMonteCarloSupervisedLearner
{
protected:
public:
	CMonteCarloQLearner(CQFunction *qFunction, CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger, CSupervisedQFunctionLearner *learner);

	virtual ~CMonteCarloQLearner();
};


#endif

