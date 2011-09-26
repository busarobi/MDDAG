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

#include "ccontinuousactiongradientpolicy.h"
#include "cvfunction.h"
#include "clinearfafeaturecalculator.h"
#include "ctorchvfunction.h"
#include "cstate.h"
#include "cstatecollection.h"
#include "cstateproperties.h"
#include "caction.h"
#include "ccontinuousactions.h"
#include "ril_debug.h"

#include <math.h>

CContinuousActionGradientPolicy::CContinuousActionGradientPolicy(CContinuousAction *contAction, CStateProperties *modelState) : CContinuousActionController(contAction),  CGradientFunction(modelState->getNumContinuousStates(),  contAction->getNumDimensions()), CStateObject(modelState)
{
	this->modelState = modelState;
}

CContinuousActionGradientPolicy::~CContinuousActionGradientPolicy()
{
}

void CContinuousActionGradientPolicy::getFunctionValuePre(ColumnVector *input, ColumnVector *output)
{
	CState *state = new CState(modelState);
	CContinuousActionData *data = dynamic_cast<CContinuousActionData *>(contAction->getNewActionData());
	
	for (int i = 0; i < getNumInputs(); i ++)
	{	
		state->setContinuousState(i, input->element(i));
	}
	
	getNextContinuousAction(state, data);
	for (int i = 0; i < getNumOutputs(); i ++)
	{	
		output->element(i) = state->getContinuousState(i);
	}

	delete state;
	delete data;
}

void CContinuousActionGradientPolicy::getGradientPre(ColumnVector *input, ColumnVector *outputErrors, CFeatureList *gradientFeatures)
{
	CFeatureList *featureList = new CFeatureList();
	CState *state = new CState(modelState);
	for (int i = 0; i < getNumInputs(); i ++)
	{	
		state->setContinuousState(i, input->element(i));
	}
	
	for (int i = 0; i < getNumOutputs(); i++)
	{
		getGradient(state, i, featureList);
		gradientFeatures->add(featureList, outputErrors->element(i));
	}
	delete featureList;
	delete state;
}


CContinuousActionPolicyFromGradientFunction::CContinuousActionPolicyFromGradientFunction(CContinuousAction *contAction, CGradientFunction *gradientFunction, CStateProperties *modelState) : CContinuousActionGradientPolicy(contAction, modelState)
{
	assert(contAction->getNumDimensions() == (unsigned int ) gradientFunction->getNumOutputs());
	this->gradientFunction = gradientFunction;

	outputError = new ColumnVector(gradientFunction->getNumOutputs());
}

CContinuousActionPolicyFromGradientFunction::~CContinuousActionPolicyFromGradientFunction()
{
	delete outputError;
}

void CContinuousActionPolicyFromGradientFunction::updateWeights(CFeatureList *dParams)
{
	gradientFunction->updateGradient(dParams, 1.0);
}


void CContinuousActionPolicyFromGradientFunction::getNextContinuousAction(CStateCollection *inputState, CContinuousActionData *action)
{
	CState *state = inputState->getState(modelState);
	
	gradientFunction->getFunctionValue(state, action);	
}

int CContinuousActionPolicyFromGradientFunction::getNumWeights()
{
	return gradientFunction->getNumWeights();
}

void CContinuousActionPolicyFromGradientFunction::getWeights(double *parameters)
{
	gradientFunction->getWeights(parameters);
}

void CContinuousActionPolicyFromGradientFunction::setWeights(double *parameters)
{
	gradientFunction->setWeights(parameters);
}

void CContinuousActionPolicyFromGradientFunction::getGradient(CStateCollection *inputState, int outputDimension, CFeatureList *gradientFeatures)
{
	*outputError = 0;
	outputError->element(outputDimension) = 1.0;
	ColumnVector input(getNumInputs());
	
	CState *state = inputState->getState(modelState);
	for (int i = 0; i < getNumInputs(); i++)
	{
		input.element(i) = state->getContinuousState(i);
	}
	
	gradientFunction->getGradient(&input, outputError, gradientFeatures);
}

void CContinuousActionPolicyFromGradientFunction::getInputDerivation(CStateCollection *, Matrix *)
{
	//gradientFunction->getInputDerivation(inputState->getState(modelState), targetVector);
}

void CContinuousActionPolicyFromGradientFunction::resetData()
{
	gradientFunction->resetData();
}

CContinuousActionFeaturePolicy::CContinuousActionFeaturePolicy(CContinuousAction *contAction, CStateProperties *modelState, std::list<CFeatureCalculator *> *l_featureCalculators) : CContinuousActionGradientPolicy(contAction, modelState)
{
	this->featureCalculators = new std::list<CFeatureCalculator *>();
		
	localGradient = new CFeatureList();

	numWeights = 0;

	std::list<CFeatureCalculator *>::iterator it = l_featureCalculators->begin();

	std::list<CStateModifier *> *modifiers = new std::list<CStateModifier *>();
	
	for (it = l_featureCalculators->begin(); it != l_featureCalculators->end(); it ++)
	{
		featureCalculators->push_back(*it);
		modifiers->push_back(*it);
	}

	featureFunctions = new std::list<CFeatureVFunction *>();

	inputDerivationFunctions = new std::map<CFeatureVFunction *, CVFunctionInputDerivationCalculator *>();

	for (it = featureCalculators->begin(); it != featureCalculators->end(); it ++)
	{
		CFeatureVFunction *featureFunction = new CFeatureVFunction(*it);
		featureFunctions->push_back(featureFunction);
		(*inputDerivationFunctions)[featureFunction] = new CVFunctionNumericInputDerivationCalculator(modelState,featureFunction, 0.005, modifiers);
		numWeights += (*it)->getNumFeatures();
	}
	inputDerivation = new ColumnVector(modelState->getNumContinuousStates());
	delete modifiers;

}

CContinuousActionFeaturePolicy::~CContinuousActionFeaturePolicy()
{
	delete localGradient;

	std::list<CFeatureVFunction *>::iterator it = featureFunctions->begin();

	for (; it != featureFunctions->end(); it ++)
	{
		delete (*inputDerivationFunctions)[*it];
		delete (*it);
	}
	
	delete featureFunctions;
	delete featureCalculators;

	delete inputDerivation;
	delete inputDerivationFunctions;
}

void CContinuousActionFeaturePolicy::updateWeights(CFeatureList *dParams)
{
	unsigned int weightIndexStart = 0;
	unsigned int weightIndexStop = 0;

	std::list<CFeatureVFunction *>::iterator it = featureFunctions->begin();

	for (; it != featureFunctions->end(); it ++)
	{
		weightIndexStop += (*it)->getNumFeatures();
		CFeatureList::iterator itFeat = dParams->begin();
		localGradient->clear();
		for (; itFeat != dParams->end(); itFeat ++)
		{
			if ((*itFeat)->featureIndex >= weightIndexStart && (*itFeat)->featureIndex < weightIndexStop)
			{
				localGradient->update((*itFeat)->featureIndex, (*itFeat)->factor);
			}
		}
		(*it)->updateFeatureList(localGradient, 1.0);
	}
}

void CContinuousActionFeaturePolicy::getNextContinuousAction(CStateCollection *stateCol, CContinuousActionData *action)
{
	std::list<CFeatureVFunction *>::iterator itFunc = featureFunctions->begin();
	std::list<CFeatureCalculator *>::iterator itCalc = featureCalculators->begin();

	for (int i = 0; itFunc != featureFunctions->end(); itFunc ++, itCalc ++, i++)
	{
		action->element(i) = (*itFunc)->getValue(stateCol->getState((*itFunc)->getStateProperties()));
	}
}

int CContinuousActionFeaturePolicy::getNumWeights()
{
	return numWeights;
}

void CContinuousActionFeaturePolicy::getWeights(double *parameters)
{
	std::list<CFeatureVFunction *>::iterator it = featureFunctions->begin();

	int weightIndex = 0;
	for (; it != featureFunctions->end(); it ++)
	{
		(*it)->getWeights(parameters + weightIndex);
		weightIndex += (*it)->getNumWeights();
	}
}

void CContinuousActionFeaturePolicy::setWeights(double *parameters)
{
	std::list<CFeatureVFunction *>::iterator it = featureFunctions->begin();

	int weightIndex = 0;
	for (; it != featureFunctions->end(); it ++)
	{
		(*it)->setWeights(parameters + weightIndex);
		weightIndex += (*it)->getNumWeights();
	}
}

void CContinuousActionFeaturePolicy::getGradient(CStateCollection *inputState, int outputDimension, CFeatureList *gradientFeatures)
{
	std::list<CFeatureVFunction *>::iterator it = featureFunctions->begin();

	int weightIndex = 0;
	for (int i = 0; it != featureFunctions->end(); it ++, i ++)
	{
		if (outputDimension == i)
		{
			localGradient->clear();
			(*it)->getGradient(inputState, localGradient);
			localGradient->addIndexOffset(weightIndex);
			gradientFeatures->add(localGradient, 1.0);
			weightIndex += (*it)->getNumWeights();
		}
	}
}

void CContinuousActionFeaturePolicy::getInputDerivation(CStateCollection *, Matrix *)
{
	// NOT WORKING, not in use
	assert(false);
	std::list<CFeatureVFunction *>::iterator it = featureFunctions->begin();

	for (unsigned int row = 0; it != featureFunctions->end(); it ++, row ++)
	{
//		(*inputDerivationFunctions)[(*it)]->getInputDerivation(inputState, inputDerivation);
		
//		for (unsigned int col = 0; col < inputDerivation->nrows(); col ++)
//		{
//			targetVector->element(row, col) = inputDerivation->element(col);
//		}
	}
}

void CContinuousActionFeaturePolicy::resetData()
{
	std::list<CFeatureVFunction *>::iterator it = featureFunctions->begin();

	for (; it != featureFunctions->end(); it ++)
	{
		(*it)->resetData();
	}
}

CContinuousActionSigmoidPolicy::CContinuousActionSigmoidPolicy(CContinuousActionGradientPolicy *policy, CCAGradientPolicyInputDerivationCalculator *inputDerivation) : CContinuousActionGradientPolicy(policy->getContinuousAction(), policy->getStateProperties())
{
	this->policy = policy;
	this->inputDerivation = inputDerivation;

	contData = new CContinuousActionData(policy->getContinuousActionProperties());

	randomControllerMode = INTERN_RANDOM_CONTROLLER;
}

CContinuousActionSigmoidPolicy::~CContinuousActionSigmoidPolicy()
{
	delete contData;
}

void CContinuousActionSigmoidPolicy::updateWeights(CFeatureList *dParams)
{
	policy->updateGradient(dParams, 1.0);
}


int CContinuousActionSigmoidPolicy::getNumWeights()
{
	return policy->getNumWeights();
}

void CContinuousActionSigmoidPolicy::getWeights(double *parameters)
{
	policy->getWeights(parameters);
}

void CContinuousActionSigmoidPolicy::setWeights(double *parameters)
{
	policy->setWeights(parameters);
}

void CContinuousActionSigmoidPolicy::resetData()
{
	policy->resetData();
}

void CContinuousActionSigmoidPolicy::getNoise(CStateCollection *state, CContinuousActionData *action, CContinuousActionData *l_noise)
{
	if (randomControllerMode == INTERN_RANDOM_CONTROLLER)
	{
		ColumnVector tempVector(this->contAction->getNumDimensions());
		
		policy->getNextContinuousAction(state, l_noise);
		
		tempVector = *action;

		for (int i = 0; i < tempVector.nrows(); i ++)
		{
			double umax = getContinuousActionProperties()->getMaxActionValue(i);
			double umin = getContinuousActionProperties()->getMinActionValue(i);
			double width = umax - umin;

			double actionValue = tempVector.element(i);
			actionValue = (actionValue - umin) / (umax - umin);

			if (actionValue <= 0.0001)
			{
				actionValue = 0.0001;
			}
			else
			{
				if (actionValue >= 0.9999)
				{
					actionValue = 0.9999;
				}
			}

			actionValue = (- log(1 / actionValue - 1) + 2) * width  / 4 + umin;

			tempVector.element(i) = actionValue;
		}

		*l_noise *= -1;
		*l_noise += tempVector;
	}
	else
	{
		CContinuousActionController::getNoise(state, action, l_noise);
	}
}

void CContinuousActionSigmoidPolicy::getNextContinuousAction(CStateCollection *state, CContinuousActionData *action)
{
	policy->getNextContinuousAction(state, action);

	if (DebugIsEnabled('p'))
	{
		DebugPrint('p', "Sigmoid Policy, Action Values:");
		action->saveASCII(DebugGetFileHandle('p'));
		DebugPrint('p', "\n");
	}
	
	
	noise->initData(0.0);
	
	if (randomController && this->randomControllerMode == INTERN_RANDOM_CONTROLLER)
	{
		randomController->getNextContinuousAction(state, noise);
	}

	if (DebugIsEnabled('p'))
	{
		DebugPrint('p', "Sigmoid Policy, Noise Values:");
		noise->saveASCII(DebugGetFileHandle('p'));
		DebugPrint('p', "\n");
	}

	(*action) << *action + *noise;


	for (int i = 0; i < action->nrows(); i ++)
	{
		double min = contAction->getContinuousActionProperties()->getMinActionValue(i);
		double width = contAction->getContinuousActionProperties()->getMaxActionValue(i) - min;



		action->element(i) = - 2 + (action->element(i) - min) / width * 4;

		action->element(i) = min + width * (1.0 / (1.0 + my_exp(-action->element(i))));
	}
}


void CContinuousActionSigmoidPolicy::getGradient(CStateCollection *inputState, int outputDimension, CFeatureList *gradientFeatures)
{
	policy->getNextContinuousAction(inputState, contData);
	policy->getGradient(inputState, outputDimension, gradientFeatures);

	double min = contAction->getContinuousActionProperties()->getMinActionValue(outputDimension);
	double width = contAction->getContinuousActionProperties()->getMaxActionValue(outputDimension) - min;
	
	DebugPrint('p', "Sigmoid Gradient Calculation: Action Value %f\n", contData->getActionValue(outputDimension));
	contData->element(outputDimension) = - 2 + (contData->element(outputDimension) - min) / width * 4;
	
	double dSig = 1 / pow(1 + my_exp(- contData->element(outputDimension)), 2) * my_exp(- contData->element(outputDimension));

	if (fabs(dSig) > 10000000)
	{
		printf("Infintity gradient!! : %f, %f, %f,%f \n", dSig, contData->element(outputDimension), min,width);
		assert(false);
	}

	if (DebugIsEnabled('p'))
	{
		DebugPrint('p', "ContinuousActionPolicyGradient: ");
		gradientFeatures->saveASCII(DebugGetFileHandle('p'));
		DebugPrint('p', "\nSaturationFactor; %f\n", 4* dSig);
	}

	gradientFeatures->multFactor(4 * dSig);
}

void CContinuousActionSigmoidPolicy::getInputDerivation(CStateCollection *, Matrix *)
{
	// Not used an not working
	
	assert(false);
/*	policy->getNextContinuousAction(inputState, contData);
	inputDerivation->getInputDerivation(inputState, targetVector);

	if (DebugIsEnabled('p'))
	{
		DebugPrint('p', "Inner Policy Input Derivation: ");
		//targetVector->saveASCII(DebugGetFileHandle('p'));
		DebugPrint('p', "Action Values: ");
		contData->saveASCII(DebugGetFileHandle('p'));

	}

	for (unsigned int i = 0; i < contData->nrows(); i ++)
	{
		double min = contAction->getContinuousActionProperties()->getMinActionValue(i);
		double width = contAction->getContinuousActionProperties()->getMaxActionValue(i) - min;
		contData->setElement(i, - 2 + (contData->element(i) - min) / width * 4);
		double dSig = 1 / pow(1 + my_exp(- contData->element(i)), 2) * my_exp(- contData->element(i));

		if (DebugIsEnabled('p'))
		{
			DebugPrint('p', "SaturationFactor for dimension %d: %f (actionValue %f)\n", i, dSig);
		}

		for (unsigned int j = 0; j < targetVector->ncols(); j++)
		{
			targetVector->setElement(i, j,4 * dSig * targetVector->element(i,j));
		}
	}*/
}

CCAGradientPolicyNumericInputDerivationCalculator::CCAGradientPolicyNumericInputDerivationCalculator(CContinuousActionGradientPolicy *policy, double stepSize,  std::list<CStateModifier *> *modifiers)
{
	this->policy = policy;
	contDataPlus = new CContinuousActionData(policy->getContinuousActionProperties());

	contDataMinus = new CContinuousActionData(policy->getContinuousActionProperties());

	this->stateBuffer = new CStateCollectionImpl(policy->getStateProperties(), modifiers);

	addParameter("NumericInputDerivationStepSize", stepSize);
}

CCAGradientPolicyNumericInputDerivationCalculator::~CCAGradientPolicyNumericInputDerivationCalculator()
{
	delete contDataPlus;
	delete contDataMinus;
	delete stateBuffer;
}

void CCAGradientPolicyNumericInputDerivationCalculator::getInputDerivation(CStateCollection *inputStateCol, Matrix *targetVector)
{
	CStateProperties *modelState = policy->getStateProperties();
	CState *inputState = stateBuffer->getState(modelState);
	inputState->setState(inputStateCol->getState(modelState));

	double stepSize = getParameter("NumericInputDerivationStepSize");

	DebugPrint('p', "Calculating Numeric Policy Input Derivation\n");;
	for (unsigned int col = 0; col < modelState->getNumContinuousStates(); col++)
	{
		double stepSize_i = (modelState->getMaxValue(col) - modelState->getMinValue(col)) * stepSize;
		inputState->setContinuousState(col, inputState->getContinuousState(col) + stepSize_i);
		stateBuffer->newModelState();
		policy->getNextContinuousAction(stateBuffer, contDataPlus);
		
		if (DebugIsEnabled('p'))
		{
			DebugPrint('p', "State : ");
			inputState->saveASCII(DebugGetFileHandle('p'));

			DebugPrint('p', "Action : ");
			contDataPlus->saveASCII(DebugGetFileHandle('p'));
		}

		inputState->setContinuousState(col, inputState->getContinuousState(col) - 2 * stepSize_i);
		stateBuffer->newModelState();
		policy->getNextContinuousAction(stateBuffer, contDataMinus);

		if (DebugIsEnabled('p'))
		{
			DebugPrint('p', "State : ");
			inputState->saveASCII(DebugGetFileHandle('p'));

			DebugPrint('p', "Action : ");
			contDataMinus->saveASCII(DebugGetFileHandle('p'));
		}

		inputState->setContinuousState(col, inputState->getContinuousState(col) + stepSize_i);
		for (int row = 0; row < policy->getNumOutputs(); row ++)
		{
			targetVector->element(row, col) = (contDataPlus->getActionValue(row) - contDataMinus->getActionValue(row)) / (2 * stepSize_i);
		}
	}
}

