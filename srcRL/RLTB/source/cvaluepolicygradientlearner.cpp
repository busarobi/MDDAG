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
#include "cvaluepolicygradientlearner.h"

#include "cqfunction.h"
#include "cvfunction.h"
#include "cqetraces.h"
#include "cresiduals.h"
#include "cpolicygradient.h"
#include "ctransitionfunction.h"
#include "cpegasus.h"
#include "ccontinuousactiongradientpolicy.h"

#include "cstate.h"
#include "cstatecollection.h"
#include "caction.h"
#include "cstateproperties.h"
#include "cstatemodifier.h"

CVPolicyLearner::CVPolicyLearner(CStateReward *rewardFunction, CTransitionFunction *dynModel, CTransitionFunctionInputDerivationCalculator *dynModeldInput,CAbstractVFunction *vFunction, CVFunctionInputDerivationCalculator *vFunctionInputDerivation, CContinuousActionGradientPolicy *gradientPolicy, CCAGradientPolicyInputDerivationCalculator *policydInput, std::list<CStateModifier *> *stateModifiers, int nForwardView) : CSemiMDPRewardListener(rewardFunction)
{
	this->rewardFunction = rewardFunction;
	this->vFunction = vFunction;
	this->gradientPolicy = gradientPolicy;
	this->dynModeldInput = dynModeldInput;
	this->policydInput = policydInput;
	this->vFunctionInputDerivation = vFunctionInputDerivation;
	this->dynModel = dynModel;
	this->stateModifiers = stateModifiers;


	dPolicy = new Matrix(gradientPolicy->getNumOutputs(), dynModel->getNumContinuousStates());
	dModelInput = new Matrix(dynModel->getNumContinuousStates(), gradientPolicy->getNumOutputs() + dynModel->getNumContinuousStates());
	dReward = new ColumnVector(dynModel->getNumContinuousStates());
	dVFunction = new ColumnVector(dynModel->getNumContinuousStates());
	data = new CContinuousActionData(gradientPolicy->getContinuousActionProperties());

//	states = new std::list<CState *>();

	addParameter("DiscountFactor", 0.95);
	addParameter("PolicyLearningRate", 1.0);
	addParameter("PolicyLearningFowardView", nForwardView);
	addParameter("PolicyLearningBackwardView", 0.0);

	addParameters(policydInput, "DPolicy");
	addParameters(dynModeldInput, "DModel");
	addParameters(vFunctionInputDerivation, "DVFunction");


	tempStateCol = new CStateCollectionImpl(dynModel->getStateProperties(), stateModifiers);


	stateGradient1 = new CStateGradient();

	for (unsigned int i = 0; i < dynModel->getNumContinuousStates(); i ++)
	{
		stateGradient1->push_back(new CFeatureList());
	}

	stateGradient2 = new CStateGradient();

	for (unsigned int i = 0; i < dynModel->getNumContinuousStates(); i ++)
	{
		stateGradient2->push_back(new CFeatureList());
	}

	dModelGradient = new CStateGradient();

	for (int i = 0; i < gradientPolicy->getNumOutputs(); i ++)
	{
		dModelGradient->push_back(new CFeatureList());
	}

	policyGradient = new CFeatureList();

	pastStates = new std::list<CStateCollectionImpl *>();
	pastDRewards = new std::list<ColumnVector *>();
	pastActions = new std::list<CContinuousActionData *>();

	statesResource = new std::list<CStateCollectionImpl *>();
	rewardsResource = new std::list<ColumnVector *>();
	actionsResource = new std::list<CContinuousActionData *>();
}

CVPolicyLearner::~CVPolicyLearner()
{
	newEpisode();
//	delete states;
	delete dPolicy;
	delete dModelInput;
	delete dReward;
	delete stateGradient1;
	delete stateGradient2;
	delete dModelGradient;
	delete tempStateCol;
	delete data;
	delete dVFunction;
	delete policyGradient;
	delete pastStates;
	delete pastDRewards;
	delete pastActions;

	newEpisode();

	std::list<CStateCollectionImpl *>::iterator itStates = statesResource->begin();
	for (; itStates != statesResource->end(); itStates ++)
	{
		delete *itStates;
	}

	std::list<ColumnVector *>::iterator itRewards = rewardsResource->begin();
	for (; itRewards != rewardsResource->end(); itRewards ++)
	{
		delete *itRewards;
	}

	std::list<CContinuousActionData *>::iterator itActions = actionsResource->begin();
	for (; itActions != actionsResource->end(); itActions ++)
	{
		delete *itActions;
	}

	delete statesResource;
	delete rewardsResource;
	delete actionsResource;
}

void CVPolicyLearner::newEpisode()
{
	std::list<CStateCollectionImpl *>::iterator itStates = pastStates->begin();
	for (; itStates != pastStates->end(); itStates ++)
	{
		statesResource->push_back(*itStates);
	}
	pastStates->clear();

	std::list<ColumnVector *>::iterator itRewards = pastDRewards->begin();
	for (; itRewards != pastDRewards->end(); itRewards ++)
	{
		rewardsResource->push_back(*itRewards);
	}
	pastDRewards->clear();

	std::list<CContinuousActionData *>::iterator itActions = pastActions->begin();
	for (; itActions != pastActions->end(); itActions ++)
	{
		actionsResource->push_back(*itActions);
	}
	pastActions->clear();
}

void CVPolicyLearner::multMatrixFeatureList(Matrix *matrix, CFeatureList *features, int index, std::list<CFeatureList *> *newFeatures)
{
	CFeatureList::iterator itFeat = features->begin();

	for (; itFeat != features->end(); itFeat ++)
	{
		std::list<CFeatureList *>::iterator itList = newFeatures->begin();
		for (int row = 0; itList != newFeatures->end(); itList ++,row ++)
		{
			(*itList)->update((*itFeat)->featureIndex, (*itFeat)->factor * matrix->element(row, index));
		}
	}
}

void CVPolicyLearner::getDNextState(CStateGradient *stateGradient1, CStateGradient *stateGradient2, CStateCollection *currentState, CContinuousActionData *data)
{
	// Clear 2nd StateGradient list
	CStateGradient::iterator it = stateGradient2->begin();

	for (; it != stateGradient2->end(); it ++)
	{
		(*it)->clear();
	}

	//Clear Model Gradient
	it = dModelGradient->begin();

	for (; it != dModelGradient->end(); it ++)
	{
		(*it)->clear();
	}

	// Derivation of the Model
	dynModeldInput->getInputDerivation(currentState->getState(dynModel->getStateProperties()), data, dModelInput);

	it = stateGradient1->begin();
	for (unsigned int i = 0; i < dynModel->getNumContinuousStates(); i ++, it ++)
	{
		multMatrixFeatureList(dModelInput, *it, i, stateGradient2);
	}

	if (DebugIsEnabled('p'))
	{
		DebugPrint('p', "Pegasus Gradient Calculation:\n ");
		DebugPrint('p', "State Gradient:\n ");
		for (it = stateGradient1->begin(); it != stateGradient1->end(); it ++)
		{
			(*it)->saveASCII(DebugGetFileHandle('p'));
			DebugPrint('p', "\n");
		}

		DebugPrint('p', "\n");
		DebugPrint('p',"dModel: ");
		//dModelInput->saveASCII(DebugGetFileHandle('p'));
	}


	// Input-Derivation of the policy
	policydInput->getInputDerivation(currentState, dPolicy);


	if (DebugIsEnabled('p'))
	{
		DebugPrint('p',"dPolicy: ");
		//dPolicy->saveASCII(DebugGetFileHandle('p'));
		DebugPrint('p', "\n");
	}

	// Derivation with respect to the weights
	it = dModelGradient->begin();

	// Gradient = d_Pi(s)/dw
	for (int i = 0; it != dModelGradient->end(); it++, i++)
	{
		gradientPolicy->getGradient(currentState, i, *it);
	}

	it = stateGradient1->begin();
	//Pi'(s) * s'
	for (int i = 0; it != stateGradient1->end(); i ++, it ++)
	{
		multMatrixFeatureList(dPolicy, *it, i, dModelGradient);
	}
	
	it = dModelGradient->begin();

	for (int i = 0; it != dModelGradient->end(); it++, i++)
	{
		multMatrixFeatureList(dModelInput, *it, i + dynModel->getNumContinuousStates(), stateGradient2);
	}

	if (DebugIsEnabled('p'))
	{
		DebugPrint('p', "Model Gradients:\n ");
		for (it = dModelGradient->begin(); it != dModelGradient->end(); it ++)
		{
			(*it)->saveASCII(DebugGetFileHandle('p'));
			DebugPrint('p', "\n");

		}
		DebugPrint('p', "New State Gradient:\n ");

		for (it = stateGradient2->begin(); it != stateGradient2->end(); it ++)
		{
			(*it)->saveASCII(DebugGetFileHandle('p'));
			DebugPrint('p', "\n");

		}

	}
}

void CVPolicyLearner::calculateGradient(std::list<CStateCollectionImpl *> *states, std::list<ColumnVector *> *Drewards, std::list<CContinuousActionData *> *actionDatas,  CFeatureList *policyGradient)
{
//	double gamma = getParameter("DiscountFactor");
	
	policyGradient->clear();

	std::list<CStateCollectionImpl *>::iterator itStates = states->begin();
	std::list<ColumnVector *>::iterator itRewards = Drewards->begin();
	std::list<CContinuousActionData *>::iterator itActions = actionDatas->begin();

	CStateGradient::iterator it = stateGradient1->begin();

	for (; it != stateGradient1->end(); it ++)
	{
		(*it)->clear();
	}

	it = stateGradient2->begin();

	for (; it != stateGradient2->end(); it ++)
	{
		(*it)->clear();
	}
	
	for (unsigned int i = 0; itStates != states->end(); itStates ++, itRewards ++, itActions ++, i ++)
	{
		getDNextState(stateGradient1, stateGradient2, *itStates, *itActions);

		CStateGradient *tempStateGradient = stateGradient1;
		stateGradient1 = stateGradient2;
		stateGradient2 = tempStateGradient;

		if (i <  states->size() - 1)
		{
			CStateGradient::iterator itGradient = stateGradient1->begin();
			for (int j = 0; itGradient != stateGradient1->end(); itGradient ++, j++)
			{
				policyGradient->add(*itGradient, (*itRewards)->element(j));
			}
		}
		else
		{
			vFunctionInputDerivation->getInputDerivation(*itStates, dVFunction);

			DebugPrint('p', "dVFunction : ");

			if (DebugIsEnabled('p'))
			{
				//dVFunction->saveASCII(DebugGetFileHandle('p'));
			}

			CStateGradient::iterator itGradient = stateGradient1->begin();
			for (int j = 0; itGradient != stateGradient1->end(); itGradient ++, j++)
			{
				policyGradient->add(*itGradient, dVFunction->element(j));
			}
		}
	}
}

void CVPolicyLearner::nextStep(CStateCollection *, CAction *action, double , CStateCollection *nextState)
{
	

	int nForwardView = (int) getParameter("PolicyLearningFowardView");
	int nBackwardView = (int) getParameter("PolicyLearningBackwardView");
	
	CStateCollectionImpl *currentState;
	CContinuousActionData *currentAction;
	if (statesResource->size() > 0)
	{
		currentAction = *actionsResource->begin();
		actionsResource->pop_front();

		currentState = *statesResource->begin();
		statesResource->pop_front();
	}
	else
	{
		currentAction = new CContinuousActionData(gradientPolicy->getContinuousAction()->getContinuousActionProperties());
		currentState = new CStateCollectionImpl(dynModel->getStateProperties(), stateModifiers);
	}
	currentAction->setData(action->getActionData());
	currentState->setStateCollection(nextState);

	pastStates->push_back(currentState);
	pastActions->push_back(currentAction);

	// FORWARD View
	int gradientPolicyRandomMode = gradientPolicy->getRandomControllerMode();
	gradientPolicy->setRandomControllerMode(NO_RANDOM_CONTROLLER);

	ColumnVector *currentReward;

	for (int i = 0; i < nForwardView - 1; i ++)
	{
		
		CStateCollection *lastState = *pastStates->rbegin();
		if (statesResource->size() > 0)
		{
			currentAction = *actionsResource->begin();
			actionsResource->pop_front();

			currentState = *statesResource->begin();
			statesResource->pop_front();


		}
		else
		{
			currentAction = new CContinuousActionData(gradientPolicy->getContinuousAction()->getContinuousActionProperties());
			currentState = new CStateCollectionImpl(dynModel->getStateProperties(), stateModifiers);
		}
		if (rewardsResource->size() > 0)
		{
			currentReward = *rewardsResource->begin();
			rewardsResource->pop_front();
		}
		else
		{
			currentReward = new ColumnVector(dynModel->getNumContinuousStates());
		}

		gradientPolicy->getNextContinuousAction(lastState, currentAction);
		dynModel->transitionFunction(lastState->getState(dynModel->getStateProperties()), gradientPolicy->getContinuousAction(), currentState->getState(dynModel->getStateProperties()),currentAction);
		currentState->newModelState();

		rewardFunction->getInputDerivation(lastState->getState(dynModel->getStateProperties()), currentReward);

		pastStates->push_back(currentState);
		pastDRewards->push_back(currentReward);
		pastActions->push_back(currentAction);

	}
	calculateGradient(pastStates, pastDRewards, pastActions, policyGradient);

	DebugPrint('p', "policyGradient for Update : ");

	if (DebugIsEnabled('p'))
	{
		policyGradient->saveASCII(DebugGetFileHandle('p'));
	}

	gradientPolicy->updateGradient(policyGradient, getParameter("PolicyLearningRate"));

	gradientPolicy->setRandomControllerMode(gradientPolicyRandomMode);

	for (int i = 0; i < nForwardView - 1; i ++)
	{
		statesResource->push_back(*pastStates->rbegin());
		actionsResource->push_back(*pastActions->rbegin());
		
		pastStates->pop_back();
		pastActions->pop_back();


		rewardsResource->push_back(*pastDRewards->rbegin());
		pastDRewards->pop_back();
	}

	if (nBackwardView > 0)
	{
		if (rewardsResource->size() > 0)
		{
			currentReward = *rewardsResource->begin();
			rewardsResource->pop_front();
		}
		else
		{
			currentReward = new ColumnVector(dynModel->getNumContinuousStates());
		}
		rewardFunction->getInputDerivation((*pastStates->rbegin())->getState(dynModel->getStateProperties()), currentReward);

		pastDRewards->push_back(currentReward);
	}

	if (pastStates->size() > (unsigned int) nBackwardView)
	{
		statesResource->push_back(*pastStates->begin());
		actionsResource->push_back(*pastActions->begin());

		pastStates->pop_front();
		pastActions->pop_front();

		if (nBackwardView > 0)
		{
			rewardsResource->push_back(*pastDRewards->begin());
			pastDRewards->pop_front();
		}
	}
}

