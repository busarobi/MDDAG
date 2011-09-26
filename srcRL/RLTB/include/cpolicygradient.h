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

#ifndef C_POLICYGRADIENT__H
#define C_POLICYGRADIENT__H

#include "cagentlistener.h"
#include "csupervisedlearner.h"

class CAgentController;
class CPolicyEvaluator;
class CFeatureList;
class CReinforcementBaseLineCalculator;

class CRewardFunction;
class CAgent;
class CTransitionFunctionEnvironment;
class CContinuousActionGradientPolicy;
class CGradientUpdateFunction;
class CStochasticPolicy;

class CPolicyGradientCalculator : public CGradientCalculator
{
protected:
	CAgentController *policy;
	CPolicyEvaluator *evaluator;
public:
	CPolicyGradientCalculator(CAgentController *policy, CPolicyEvaluator *evaluator);
	virtual ~CPolicyGradientCalculator() {};

	virtual void getGradient(CFeatureList *gradient) = 0;
	virtual double getFunctionValue();
};

class CGPOMDPGradientCalculator : public CPolicyGradientCalculator, public CSemiMDPRewardListener
{
protected:
	CFeatureList *localGradient;
	CFeatureList *localZTrace;

	CFeatureList *globalGradient;

	CAgent *agent; 
	CReinforcementBaseLineCalculator *baseLine;

	CStochasticPolicy *stochPolicy;

public:
	CGPOMDPGradientCalculator(CRewardFunction *reward, CStochasticPolicy *policy, CPolicyEvaluator *evaluator, CAgent *agent, CReinforcementBaseLineCalculator *baseLine, int TSteps, int nEpisodes, double beta);
	virtual ~CGPOMDPGradientCalculator();

	virtual void nextStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *newState);
	virtual void newEpisode();

	virtual void getGradient(CFeatureList *gradient);

	virtual CFeatureList* getGlobalGradient();
	virtual void setGlobalGradient(CFeatureList *globalGradient);
};

class CContinuousActionGradientPolicy;

class CNumericPolicyGradientCalculator : public CPolicyGradientCalculator
{
protected:
	CFeatureList *gradientFeatures;
	double *weights;

	CRewardFunction *rewardFunction;
	CAgent *agent;
	CTransitionFunctionEnvironment *dynModel;
	CContinuousActionGradientPolicy *gradientPolicy;
public:
	CNumericPolicyGradientCalculator(CAgent *agent, CContinuousActionGradientPolicy *policy, CTransitionFunctionEnvironment *dynModel, CRewardFunction *reward, double stepSize, CPolicyEvaluator *evaluator);
	~CNumericPolicyGradientCalculator();

	virtual void getGradient(CFeatureList *gradientFeatures);
};

class CRandomPolicyGradientCalculator : public CPolicyGradientCalculator
{
protected:
	double *stepSizes;
	double *minWeights;
	double *nullWeights;
	double *plusWeights;
	
	int *numMinWeights;
	int *numMaxWeights;
	int *numNullWeights;
	
	CContinuousActionGradientPolicy *gradientPolicy;
public:
	CRandomPolicyGradientCalculator(CContinuousActionGradientPolicy *policy, CPolicyEvaluator *evaluator, int numEvaluations, double stepSize);
	virtual ~CRandomPolicyGradientCalculator();

	virtual void getGradient(CFeatureList *gradient);
	virtual void setStepSize(int index, double stepSize);
	
	virtual void resetGradientCalculator() {};
};

class CRandomMaxPolicyGradientCalculator : public CPolicyGradientCalculator
{
protected:
	double *stepSizes;
	double *workStepSizes;
	
	CContinuousActionGradientPolicy *gradientPolicy;
public:
	CRandomMaxPolicyGradientCalculator(CContinuousActionGradientPolicy *policy, CPolicyEvaluator *evaluator, int numEvaluations, double stepSize);
	virtual ~CRandomMaxPolicyGradientCalculator();

	virtual void getGradient(CFeatureList *gradient);
	virtual void setStepSize(int index, double stepSize);
	
	virtual void resetGradientCalculator();
};


class CGSearchPolicyGradientUpdater : public CGradientFunctionUpdater
{
protected:
	CPolicyGradientCalculator *gradientCalculator;

	double *startParameters;
	double *workParameters;

	double lastStepSize;


	void setWorkingParamters(CFeatureList *gradient, double stepSize, double *startParameters, double *workParameters);
public:

	CGSearchPolicyGradientUpdater(CGradientUpdateFunction *updateFunction, CPolicyGradientCalculator *gradientCalculator, double s0, double epsilon);
	virtual ~CGSearchPolicyGradientUpdater();

	virtual void updateWeights(CFeatureList *gradient);
};



class CPolicyGradientWeightDecayListener : public CSemiMDPListener
{
protected:
	CGradientUpdateFunction *updateFunction;
	double *parameters;
public:
	CPolicyGradientWeightDecayListener(CGradientUpdateFunction *updateFunction, double weightdecay);
	~CPolicyGradientWeightDecayListener();

	virtual void newEpisode();
};



#endif

