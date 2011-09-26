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
#include "ccontinuoustime.h"
#include "cstate.h"
#include "cstatecollection.h"
#include "cstateproperties.h"
#include "ctransitionfunction.h"
#include "cvfunction.h"
#include "ccontinuousactiongradientpolicy.h"
#include "caction.h"
#include <assert.h>
#include <math.h>


double CContinuousTimeParameters::getGammaFromSgamma(double sgamma, double dt)
{
	return 1 - dt * sgamma;
}

double CContinuousTimeParameters::getLambdaFromKappa(double kappa, double sgamma, double dt)
{
	return (kappa - dt) / (1 / sgamma - dt);
}

CContinuousTimeVMPolicy::CContinuousTimeVMPolicy(CActionSet *actions, CActionDistribution *distribution, CVFunctionInputDerivationCalculator *vFunction, CContinuousTimeTransitionFunction *model, CRewardFunction *rewardFunction) : CQStochasticPolicy(actions, distribution, new CContinuousTimeQFunctionFromTransitionFunction(actions, vFunction, model, rewardFunction))
{
	this->vfunction = vFunction;
	this->model = model;

	addParameters(vFunction);

//	assert(vfunction->getNumContinuousStates() == model->getNumContinuousStates());
}

CContinuousTimeVMPolicy::~CContinuousTimeVMPolicy()
{
	delete this->qfunction;
}

CContinuousTimeQFunctionFromTransitionFunction *CContinuousTimeVMPolicy::getQFunctionFromTransitionFunction()
{
	return dynamic_cast<CContinuousTimeQFunctionFromTransitionFunction *>(qfunction);
}

CContinuousTimeAndActionVMPolicy::CContinuousTimeAndActionVMPolicy(CContinuousAction *action, CVFunctionInputDerivationCalculator *dvFunction, CTransitionFunction *model) : CContinuousActionController(action, false)
{
	this->dVFunction = dvFunction;
	this->model =  model;

	assert(this->dVFunction->getNumInputs() == model->getNumContinuousStates());
	assert(model->isType(DM_DERIVATIONUMODEL));

	actionValues = new ColumnVector(getContinuousActionProperties()->getNumActionValues());
	derivationU = new Matrix(model->getNumContinuousStates(), getContinuousActionProperties()->getNumActionValues());
	derivationX = new ColumnVector(model->getNumContinuousStates());
	//noise = new CContinuousActionData(getContinuousActionProperties());

	randomControllerMode = INTERN_RANDOM_CONTROLLER;

	addParameters(dvFunction);
}

CContinuousTimeAndActionVMPolicy::~CContinuousTimeAndActionVMPolicy()
{
	delete actionValues;
	delete derivationU;
	delete derivationX;
	//delete noise;
}




void CContinuousTimeAndActionVMPolicy::getNextContinuousAction(CStateCollection *state, CContinuousActionData *action)
{
	model->getDerivationU(state->getState(model->getStateProperties()), derivationU);
	dVFunction->getInputDerivation(state, derivationX);

	Matrix temp = (*derivationU) * (*derivationX);
	(*actionValues) << temp.column(1).t();

	noise->initData(0.0);

	if (randomController && randomControllerMode == INTERN_RANDOM_CONTROLLER)
	{
		randomController->getNextContinuousAction(state, noise);
	}

	getActionValues(actionValues, noise);

	(*action) << (*actionValues);
}



CContinuousTimeAndActionSigmoidVMPolicy::CContinuousTimeAndActionSigmoidVMPolicy(CContinuousAction *action, CVFunctionInputDerivationCalculator *vfunction, CTransitionFunction *model) : CContinuousTimeAndActionVMPolicy(action, vfunction, model)
{
	c = new ColumnVector(action->getContinuousActionProperties()->getNumActionValues());
	*c = (1.0);

	addParameter("SigmoidPolicyCFactor", 100.0);
}

CContinuousTimeAndActionSigmoidVMPolicy::~CContinuousTimeAndActionSigmoidVMPolicy()
{
	delete c;
}



void CContinuousTimeAndActionSigmoidVMPolicy::getActionValues(ColumnVector *actionValues, ColumnVector *noise)
{
	*actionValues = SP(*actionValues, *c) * getParameter("SigmoidPolicyCFactor");
	
	*actionValues = *actionValues + *noise;

	for (int i = 0; i < actionValues->nrows(); i++)
	{
		double umax = getContinuousActionProperties()->getMaxActionValue(i);
		double umin = getContinuousActionProperties()->getMinActionValue(i);
		if (actionValues->element(i) < - 400)
		{
			actionValues->element(i) = -400;
		}
		double s = 1/(1 + exp(-(actionValues->element(i))));
		actionValues->element(i) = umin + s * (umax - umin);
	}
}

void CContinuousTimeAndActionSigmoidVMPolicy::getNoise(CStateCollection *, CContinuousActionData *, CContinuousActionData *)
{
	assert(false);
	/*
	double generalC = getParameter("SigmoidPolicyCFactor");
	if (randomControllerMode == INTERN_RANDOM_CONTROLLER)
	{
		ColumnVector tempVector(contAction->nrows());
		model->getDerivationU(state->getState(model->getStateProperties()), derivationU);
		dVFunction->getInputDerivation(state, derivationX);

		derivationX->multMatrix(derivationU, l_noise);

		tempVector.setVector(action);

		for (unsigned int i = 0; i < tempVector.nrows(); i ++)
		{
			double umax = getContinuousActionProperties()->getMaxActionValue(i);
			double umin = getContinuousActionProperties()->getMinActionValue(i);

			double actionValue = tempVector.element(i);
			actionValue = (actionValue - umin) / (umax - umin);

			actionValue = - log(1 / actionValue - 1) / generalC / c->element(i);

			tempVector.element(i) = actionValue;
		}

		*l_noise = *l_noise * -1.0;
		*l_noise = *l_noise + tempVector;
		
	}
	else
	{
		CContinuousActionController::getNoise(state, action, l_noise);
	}*/
}


void CContinuousTimeAndActionSigmoidVMPolicy::setC(int index, double value)
{
	c->element(index) = value;
}

double CContinuousTimeAndActionSigmoidVMPolicy::getC(int index)
{
	return c->element(index);
}



CContinuousTimeAndActionSigmoidVMGradientPolicy::CContinuousTimeAndActionSigmoidVMGradientPolicy(CContinuousAction *action, CGradientVFunction *gradVFunction, CVFunctionInputDerivationCalculator *dvFunction, CTransitionFunction *model, std::list<CStateModifier *> *modifiers) : CContinuousActionGradientPolicy(action, model->getStateProperties())
{
	vFunction = gradVFunction;

	derivationState = new CStateCollectionImpl(model->getStateProperties(), modifiers);

	gradient1 = new CFeatureList();
	gradient2 = new CFeatureList();

	c = new ColumnVector(action->getContinuousActionProperties()->getNumActionValues());
	*c = (1.0);

	addParameter("SigmoidPolicyCFactor", 1.0);

	this->dVFunction = dvFunction;
	this->model =  model;

	assert(this->dVFunction->getNumInputs() == model->getNumContinuousStates());
	assert(model->isType(DM_DERIVATIONUMODEL));

	actionValues = new ColumnVector(getContinuousActionProperties()->getNumActionValues());
	derivationU = new Matrix(model->getNumContinuousStates(), getContinuousActionProperties()->getNumActionValues());
	derivationX = new ColumnVector(model->getNumContinuousStates());
	//noise = new CContinuousActionData(getContinuousActionProperties());

	randomControllerMode = INTERN_RANDOM_CONTROLLER;

	addParameters(dvFunction);
}

CContinuousTimeAndActionSigmoidVMGradientPolicy::~CContinuousTimeAndActionSigmoidVMGradientPolicy()
{
	delete derivationState;
	delete c;
	delete actionValues;
	delete derivationU;
	delete derivationX;
}

void CContinuousTimeAndActionSigmoidVMGradientPolicy::getNextContinuousAction(CStateCollection *state, CContinuousActionData *action)
{
	model->getDerivationU(state->getState(model->getStateProperties()), derivationU);
	dVFunction->getInputDerivation(state, derivationX);

	*actionValues = (*derivationU) *(*derivationX);

	noise->initData(0.0);

	if (randomController && randomControllerMode == INTERN_RANDOM_CONTROLLER)
	{
		randomController->getNextContinuousAction(state, noise);
	}

	getActionValues(actionValues, noise);

	(*action) << (*actionValues);
}


void CContinuousTimeAndActionSigmoidVMGradientPolicy::getActionValues(ColumnVector *actionValues, ColumnVector *noise)
{
	
	*actionValues = SP(*actionValues, *c);
	
	(*actionValues) = (*actionValues) * getParameter("SigmoidPolicyCFactor");
	(*actionValues) = (*actionValues) + (*noise);


	for (int i = 0; i < actionValues->nrows(); i++)
	{
		double umax = getContinuousActionProperties()->getMaxActionValue(i);
		double umin = getContinuousActionProperties()->getMinActionValue(i);
		if (actionValues->element(i) < - 400)
		{
			actionValues->element(i) = -400;
		}
		double s = 1/(1 + exp(-(actionValues->element(i))));
		actionValues->element(i) =  umin + s * (umax - umin);
	}
}

void CContinuousTimeAndActionSigmoidVMGradientPolicy::getNoise(CStateCollection *, CContinuousActionData *, CContinuousActionData *)
{
	assert(false);
	/*
	
	double generalC = getParameter("SigmoidPolicyCFactor");
	if (randomControllerMode == INTERN_RANDOM_CONTROLLER)
	{
		ColumnVector tempVector(contAction->nrows());
		model->getDerivationU(state->getState(model->getStateProperties()), derivationU);
		dVFunction->getInputDerivation(state, derivationX);

		derivationX->multMatrix(derivationU, l_noise);
		tempVector.setVector(action);

		for (unsigned int i = 0; i < tempVector.nrows(); i ++)
		{
			double umax = getContinuousActionProperties()->getMaxActionValue(i);
			double umin = getContinuousActionProperties()->getMinActionValue(i);

			double actionValue = tempVector.element(i);
			actionValue = (actionValue - umin) / (umax - umin);

			actionValue = - log(1 / actionValue - 1) / generalC / c->element(i);

			tempVector.element(i) = actionValue;
		}

		l_noise->multScalar(-1.0);
		l_noise->addVector(&tempVector);

	}
	else
	{
		CContinuousActionController::getNoise(state, action, l_noise);
	}*/
}


void CContinuousTimeAndActionSigmoidVMGradientPolicy::setC(int index, double value)
{
	c->element(index) = value;
}

double CContinuousTimeAndActionSigmoidVMGradientPolicy::getC(int index)
{
	return c->element(index);
}

int CContinuousTimeAndActionSigmoidVMGradientPolicy::getNumWeights()
{
	return vFunction->getNumWeights();
}

void CContinuousTimeAndActionSigmoidVMGradientPolicy::getWeights(double *parameters)
{
	vFunction->getWeights(parameters);
}

void CContinuousTimeAndActionSigmoidVMGradientPolicy::setWeights(double *parameters)
{
	vFunction->setWeights(parameters);
}

void CContinuousTimeAndActionSigmoidVMGradientPolicy::updateWeights(CFeatureList *dParams)
{
	vFunction->updateGradient(dParams);
}

void CContinuousTimeAndActionSigmoidVMGradientPolicy::getGradient(CStateCollection *currentState, int outputDimension, CFeatureList *gradientFeatures)
{
	gradientFeatures->clear();

	model->getDerivationU(currentState->getState(model->getStateProperties()), derivationU);
	dVFunction->getInputDerivation(currentState, derivationX);

	*actionValues = (*derivationU) * (*derivationX);
	
	double prodFactor = my_exp(- actionValues->element(outputDimension));
	double stepSize = 0.01;
	prodFactor = prodFactor / pow(1.0 + prodFactor, 2.0); // s'(dV/dx  * df/du)

	CState *inputState = derivationState->getState(modelState);
	inputState->setState(currentState->getState(modelState));
	for (unsigned int x_i = 0; x_i < modelState->getNumContinuousStates(); x_i++)
	{
		gradient1->clear();
		gradient2->clear();
		double stepSize_i = (modelState->getMaxValue(x_i) - modelState->getMinValue(x_i)) * stepSize;
		inputState->setContinuousState(x_i, inputState->getContinuousState(x_i) + stepSize_i);
		derivationState->newModelState();
		
		vFunction->getGradient(derivationState, gradient1);

		inputState->setContinuousState(x_i, inputState->getContinuousState(x_i) - 2 * stepSize_i);
		derivationState->newModelState();
		vFunction->getGradient(derivationState, gradient2);

        inputState->setContinuousState(x_i, inputState->getContinuousState(x_i) + stepSize_i);
		
		gradient1->add(gradient2, -1.0);

		gradientFeatures->add(gradient1, prodFactor * derivationU->element(x_i, outputDimension) / (2 * stepSize_i)); 
	}
	
}

void CContinuousTimeAndActionSigmoidVMGradientPolicy::resetData()
{
	vFunction->resetData();
}


CContinuousTimeAndActionBangBangVMPolicy::CContinuousTimeAndActionBangBangVMPolicy(CContinuousAction *action, CVFunctionInputDerivationCalculator *vfunction, CTransitionFunction *model) : CContinuousTimeAndActionVMPolicy(action, vfunction, model)
{
	
}

void CContinuousTimeAndActionBangBangVMPolicy::getNoise(CStateCollection *state, CContinuousActionData *action, CContinuousActionData *l_noise)
{
	
	CContinuousActionController::getNoise(state, action, l_noise);
}


void CContinuousTimeAndActionBangBangVMPolicy::getActionValues(ColumnVector *actionValues, ColumnVector *noise)
{
	*actionValues = *actionValues + (*noise);

	for (int i = 0; i < actionValues->nrows(); i++)
	{
		double umax = getContinuousActionProperties()->getMaxActionValue(i);
		double umin = getContinuousActionProperties()->getMinActionValue(i);
		if (actionValues->element(i) > 0)
		{
			actionValues->element(i) = umax;
		}
		else 
		{
			actionValues->element(i) = umin;
		}
	}
}

CContinuousActionSmoother::CContinuousActionSmoother(CContinuousAction *action, CContinuousActionController *policy, double alpha) : CContinuousActionController(action)
{
	this->policy = policy;
	this->alpha = alpha;

	this->actionValues = new double[contAction->getContinuousActionProperties()->getNumActionValues()];

	for (unsigned int i = 0; i < contAction->getNumDimensions(); i++)
	{
		actionValues[i] = 0.0;
	}
}

CContinuousActionSmoother::~CContinuousActionSmoother()
{
	delete [] actionValues;
}

void CContinuousActionSmoother::getNextContinuousAction(CStateCollection *state, CContinuousActionData *data)
{
	policy->getNextContinuousAction(state, data);

	for (unsigned int i = 0; i < contAction->getNumDimensions(); i++)
	{
		data->element(i) = data->element(i) * (1 - getAlpha()) + getAlpha() * actionValues[i];		
		actionValues[i] = data->element(i);
	}
}

void CContinuousActionSmoother::setAlpha(double alpha)
{
	this->alpha = alpha;
}

double CContinuousActionSmoother::getAlpha()
{
	return alpha;
}

