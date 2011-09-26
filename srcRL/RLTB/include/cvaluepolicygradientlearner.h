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

#ifndef C_VPOLICYFUNCTIONLEARNER__H
#define C_VPOLICYFUNCTIONLEARNER__H


#include "cagentlistener.h"
#include "newmat/newmat.h"



 
class CFeatureList;
class CAbstractVFunction;
class CVFunctionInputDerivationCalculator;
class CContinuousActionGradientPolicy;
class CCAGradientPolicyInputDerivationCalculator;
class CContinuousActionData;
class CStateGradient;
class CStateReward;
class CTransitionFunction;
class CTransitionFunctionInputDerivationCalculator;
class CStateCollectionImpl;
class CFeatureList;
class CStateModifier;
class CStateCollection;

	
class CVPolicyLearner : public CSemiMDPRewardListener
{
protected:
	typedef std::list<CFeatureList *> CStateGradient;

	CAbstractVFunction *vFunction;
	CVFunctionInputDerivationCalculator *vFunctionInputDerivation;
	
	CContinuousActionGradientPolicy *gradientPolicy;
	CCAGradientPolicyInputDerivationCalculator *policydInput;

	ColumnVector *dReward;
	ColumnVector *dVFunction;
	Matrix *dPolicy;
	Matrix *dModelInput;

	CContinuousActionData *data;

	std::list<CStateGradient *> *stateGradients;

	CStateGradient *stateGradient1;
	CStateGradient *stateGradient2;
	CStateGradient *dModelGradient;
	
	CStateReward *rewardFunction;
	CTransitionFunction *dynModel;
	CTransitionFunctionInputDerivationCalculator *dynModeldInput;


	CStateCollectionImpl *tempStateCol;

	CFeatureList *policyGradient;

	void getDNextState(CStateGradient *stateGradient1, CStateGradient *stateGradient2, CStateCollection *currentState, CContinuousActionData *data);
	void multMatrixFeatureList(Matrix *matrix, CFeatureList *features, int index, std::list<CFeatureList *> *newFeatures);

	std::list<CStateCollectionImpl *> *pastStates;
	std::list<ColumnVector *> *pastDRewards;
	std::list<CContinuousActionData *> *pastActions;

	std::list<CStateCollectionImpl *> *statesResource;
	std::list<ColumnVector *> *rewardsResource;
	std::list<CContinuousActionData *> *actionsResource;

	std::list<CStateModifier *> *stateModifiers;


public:
	CVPolicyLearner(CStateReward *rewardFunction, CTransitionFunction *dynModel, CTransitionFunctionInputDerivationCalculator *dynModeldInput,CAbstractVFunction *vFunction, CVFunctionInputDerivationCalculator *vFunctionInputDerivation, CContinuousActionGradientPolicy *gradientPolicy, CCAGradientPolicyInputDerivationCalculator *policydInput, std::list<CStateModifier *> *stateModifiers, int nForwardView);
	virtual ~CVPolicyLearner();

	virtual void nextStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState);

	virtual void newEpisode();

	void calculateGradient(std::list<CStateCollectionImpl *> *states, std::list<ColumnVector *> *Drewards, std::list<CContinuousActionData *> *actionDatas,  CFeatureList *policyGradient);

};

#endif

