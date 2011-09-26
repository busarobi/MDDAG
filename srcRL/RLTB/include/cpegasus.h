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

#ifndef C__PEGASUS__H
#define C__PEGASUS__H

#include "cparameters.h"
#include "cpolicygradient.h"
#include "cagentlistener.h"

class CContinuousTimeAndActionTransitionFunction;
class CState;
class CContinuousActionData;

class CContinuousActionGradientPolicy;
class CStateList;
class CTransitionFunctionEnvironment;
class CPolicySameStateEvaluator; 
class CAgent;
class CRewardFunction;
class CCAGradientPolicyInputDerivationCalculator;
class CFeatureList;
class CStateReward;

class CTransitionFunctionInputDerivationCalculator : virtual public CParameterObject
{
protected:
	CContinuousTimeAndActionTransitionFunction *dynModel;
	CState *nextState;
	CContinuousActionData *buffData;

public:
	CTransitionFunctionInputDerivationCalculator(CContinuousTimeAndActionTransitionFunction *dynModel);
	~CTransitionFunctionInputDerivationCalculator();

	virtual void getInputDerivation(CState *currentState, CContinuousActionData *data, Matrix *dModelInput) = 0;
};

class CTransitionFunctionNumericalInputDerivationCalculator : public CTransitionFunctionInputDerivationCalculator
{
protected:
	CState *buffState;

	CState *nextState1;
	CState *nextState2;
public:
	CTransitionFunctionNumericalInputDerivationCalculator(CContinuousTimeAndActionTransitionFunction *dynModel, double stepsize);
	~CTransitionFunctionNumericalInputDerivationCalculator();

	virtual void getInputDerivation(CState *currentState, CContinuousActionData *data, Matrix *dModelInput);
};

class CPEGASUSPolicyGradientCalculator : public CPolicyGradientCalculator
{
protected:
	CContinuousActionGradientPolicy *policy;
	

	CStateList *startStates;

	CTransitionFunctionEnvironment *dynModel;
	
	CPolicySameStateEvaluator *sameStateEvaluator; 

public:
	CPEGASUSPolicyGradientCalculator(CAgent *agent, CRewardFunction *reward, CContinuousActionGradientPolicy *policy, CTransitionFunctionEnvironment *dynModel, int numStartStates,  int horizon, double gamma);
	~CPEGASUSPolicyGradientCalculator();

	virtual void getGradient(CFeatureList *gradient);
	virtual void getPEGASUSGradient(CFeatureList *gradient, CStateList *startStates) = 0;

	virtual CStateList* getStartStates();
	virtual void setStartStates(CStateList *startStates);

	virtual void setRandomStartStates();
};

class CPEGASUSAnalyticalPolicyGradientCalculator : public CPEGASUSPolicyGradientCalculator, public CSemiMDPListener
{
protected:
	ColumnVector *dReward;
	Matrix *dPolicy;
	Matrix *dModelInput;
	std::list<CFeatureList *> *stateGradient1;
	std::list<CFeatureList *> *stateGradient2;
	std::list<CFeatureList *> *dModelGradient;

	CFeatureList *episodeGradient;

	CStateReward *rewardFunction;
	CTransitionFunctionInputDerivationCalculator *dynModeldInput;
	CCAGradientPolicyInputDerivationCalculator *policydInput;

	int steps;

	CAgent *agent;

	void multMatrixFeatureList(Matrix *matrix, CFeatureList *features, int index, std::list<CFeatureList *> *newFeatures);
public:
	CPEGASUSAnalyticalPolicyGradientCalculator(CAgent *agent, CContinuousActionGradientPolicy *policy, CCAGradientPolicyInputDerivationCalculator *policyInputDerivation, CTransitionFunctionEnvironment *dynModel, CTransitionFunctionInputDerivationCalculator *dynModeldInput, CStateReward *reward, int numStartStates, int horizon, double gamma);
	~CPEGASUSAnalyticalPolicyGradientCalculator();

	virtual void getPEGASUSGradient(CFeatureList *gradientFeatures, CStateList *startStates);
	
	virtual void nextStep(CStateCollection *oldState, CAction *action, CStateCollection *newState);
	virtual void newEpisode();
};

class CPEGASUSNumericPolicyGradientCalculator : public CPEGASUSPolicyGradientCalculator
{
protected:
	CFeatureList *gradientFeatures;
	double *weights;

	CRewardFunction *rewardFunction;
	CAgent *agent;
public:
	CPEGASUSNumericPolicyGradientCalculator(CAgent *agent, CContinuousActionGradientPolicy *policy, CTransitionFunctionEnvironment *dynModel, CRewardFunction *reward, double stepSize, int startStates, int horizon, double gamma);
	~CPEGASUSNumericPolicyGradientCalculator();

	virtual void getPEGASUSGradient(CFeatureList *gradientFeatures, CStateList *startStates);
};

#endif

