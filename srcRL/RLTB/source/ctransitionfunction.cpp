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

#include "ctransitionfunction.h"


#include "caction.h"
#include "cstatecollection.h"
#include "cstateproperties.h"
#include "cstate.h"
#include "cvfunction.h"
#include "ccontinuousactions.h"
#include "cutility.h"
#include "ril_debug.h"
#include "cregions.h"
#include "ctheoreticalmodel.h"

#include <math.h>


CTransitionFunction::CTransitionFunction(CStateProperties *properties, CActionSet *actions) : CStateObject(properties), CActionObject(actions)
{
	type = 0;

	resetType = DM_RESET_TYPE_ZERO;
}

int CTransitionFunction::getType()
{
	return type;
}

void CTransitionFunction::getDerivationU(CState *, Matrix *)
{
}

void CTransitionFunction::addType(int Type)
{
	type = type | Type;	
}

bool CTransitionFunction::isType(int type)
{
	return (this->type & type) == type;
}

void CTransitionFunction::getResetState(CState *modelState)
{
	switch(resetType) 
	{
	case DM_RESET_TYPE_ZERO:
		{			
			for (unsigned int i = 0; i < modelState->getNumContinuousStates(); i++)
			{
				modelState->setContinuousState(i, 0.0);
			}
			for (unsigned int i = 0; i < modelState->getNumDiscreteStates(); i++)
			{
				modelState->setDiscreteState(i, 0);
			}
			break;
		}
	case DM_RESET_TYPE_ALL_RANDOM:
	case DM_RESET_TYPE_RANDOM:
		{			
			for (unsigned int i = 0; i < modelState->getNumContinuousStates(); i++)
			{
				double stateSize = properties->getMaxValue(i) - properties->getMinValue(i);
				double randNum = (((double)rand()) / RAND_MAX );
				modelState->setContinuousState(i, randNum * stateSize + properties->getMinValue(i));
			}
			for (unsigned int i = 0; i < modelState->getNumDiscreteStates(); i++)
			{
				modelState->setDiscreteState(i, rand() % properties->getDiscreteStateSize(i));
			}
			break;
		}	
		default:
		{
		}
	}
}

void CTransitionFunction::setResetType(int resetType)
{
	this->resetType = resetType;
}

CExtendedActionTransitionFunction::CExtendedActionTransitionFunction(CActionSet *actions, CTransitionFunction *model, std::list<CStateModifier *> *modifiers, CRewardFunction *l_rewardFunction) : CTransitionFunction(model->getStateProperties(), actions)
{
	addType(DM_EXTENDEDACTIONMODEL);
	
	this->dynModel = model;

	addParameters(dynModel);
	addParameter("MaxHierarchicExecution", 50);

	intermediateState = new CStateCollectionImpl(properties, modifiers);
	nextState = new CStateCollectionImpl(properties, modifiers);

	rewardFunction = l_rewardFunction;

	addParameter("DiscountFactor", 0.95);
}

CExtendedActionTransitionFunction::~CExtendedActionTransitionFunction()
{
	delete intermediateState;
	delete nextState;
}

double CExtendedActionTransitionFunction::getReward( CStateCollection *, CAction *, CStateCollection *)
{
	return lastReward;
}

void CExtendedActionTransitionFunction::transitionFunction(CState *oldstate, CAction *action, CState *newState, CActionData *data)
{
	lastReward = transitionFunctionAndReward(oldstate, action, newState, data, rewardFunction, getParameter("DiscountFactor"));
}

double CExtendedActionTransitionFunction::transitionFunctionAndReward(CState *oldState, CAction *action, CState *newState, CActionData *data, CRewardFunction *rewardFunction, double gamma)
{
	double reward = 0;

	CStateCollectionImpl *buf = NULL;

	intermediateState->getState(properties)->setState(oldState);

	if (action->isType(PRIMITIVEACTION))
	{
		dynModel->transitionFunction(oldState, action, newState, data);
		
		nextState->getState(properties)->setState(newState);

		if (rewardFunction)
		{
			reward = rewardFunction->getReward(intermediateState, action, nextState);
		}
	}

	if (action->isType(EXTENDEDACTION))
	{
		int duration = 0;

		CAction *primAction = NULL;
		CExtendedAction *extAction = NULL;

		extAction = dynamic_cast<CExtendedAction *>(action);
		
		do
		{
			primAction = action;

			CExtendedAction *extAction2;
			int lduration = 1;
				
			while (primAction->isType(EXTENDEDACTION))
			{
				extAction2 = dynamic_cast<CExtendedAction *>(primAction);
				primAction = extAction2->getNextHierarchyLevel(intermediateState);
			}


			CActionData *primActionData = actionDataSet->getActionData(primAction);
			if (primActionData)
			{
				primActionData->setData(primAction->getActionData());
			}

			dynModel->transitionFunction(intermediateState->getState(dynModel->getStateProperties()), action, nextState->getState(dynModel->getStateProperties()), primActionData);
			nextState->newModelState();

			if (primActionData && primAction->isType(MULTISTEPACTION))
			{
				lduration = dynamic_cast<CMultiStepActionData *>(data)->duration;
			}
			else
			{
				lduration = primAction->getDuration();
			}			

			duration += lduration;
			
			if (rewardFunction)
			{
				reward = reward * pow(gamma, lduration) + rewardFunction->getReward(intermediateState, primAction, nextState);
			}

			//exchange Model State

			buf = intermediateState;
			intermediateState = nextState;
			nextState = buf;
		}
		// Execute the action until the state changed
		while (duration < 	getParameter("MaxHierarchicExecution") && !extAction->isFinished(intermediateState, nextState));
		
		if (data)
		{
			dynamic_cast<CMultiStepActionData *>(data)->duration = duration;
		}

		newState->setState(nextState->getState(properties));
	}

	return reward;
}

void CExtendedActionTransitionFunction::getDerivationU(CState *oldstate, Matrix *derivation)
{
	dynModel->getDerivationU(oldstate, derivation);
}

bool CExtendedActionTransitionFunction::isResetState(CState *state) 
{
	return dynModel->isResetState(state);
}
	
bool CExtendedActionTransitionFunction::isFailedState(CState *state)
{
	return dynModel->isFailedState(state);
}

void CExtendedActionTransitionFunction::getResetState(CState *resetState)
{
	dynModel->getResetState(resetState);
}

void CExtendedActionTransitionFunction::setResetType(int resetType)
{
	dynModel->setResetType(resetType);
}

CComposedTransitionFunction::CComposedTransitionFunction(CStateProperties *properties) : CTransitionFunction(properties, new CActionSet())
{
	this->TransitionFunction = new std::list<CTransitionFunction *>();
}

CComposedTransitionFunction::~CComposedTransitionFunction()
{
	delete TransitionFunction;
	delete actions;
}

void CComposedTransitionFunction::addTransitionFunction(CTransitionFunction *model)
{
	TransitionFunction->push_back(model);
	actions->add(model->getActions());
}

void CComposedTransitionFunction::transitionFunction(CState *oldstate, CAction *action, CState *newState, CActionData *data)
{
	std::list<CTransitionFunction *>::iterator it;

	for (it = TransitionFunction->begin(); it != TransitionFunction->end(); it ++)
	{
		if ((*it)->getActions()->isMember(action))
		{
			(*it)->transitionFunction(oldstate, action, newState, data);
		}
	}
}


CContinuousTimeTransitionFunction::CContinuousTimeTransitionFunction(CStateProperties *properties, CActionSet *actions, double dt) : CTransitionFunction(properties, actions)
{
	this->dt = dt;
	this->simulationSteps = 1;

	addType(DM_CONTINUOUSMODEL);

	derivation = new ColumnVector(getStateProperties()->getNumContinuousStates());
}

CContinuousTimeTransitionFunction::~CContinuousTimeTransitionFunction()
{
	delete derivation;
}
	
void CContinuousTimeTransitionFunction::doSimulationStep(CState *state, double timeStep, CAction *action, CActionData *data)
{
	for (unsigned int i = 0; i < state->getNumContinuousStates(); i++)
	{
		this->getDerivationX(state, action, derivation, data);

		state->setContinuousState(i, state->getContinuousState(i) + timeStep * derivation->element(i));
	}
}



void CContinuousTimeTransitionFunction::transitionFunction(CState *oldState, CAction *action, CState *newState, CActionData *data)
{
	double timestep = this->dt / simulationSteps;

	newState->setState(oldState);

	for (int i = 0; i < simulationSteps; i++)
	{
		this->doSimulationStep(newState, timestep, action, data);
	}
}

double CContinuousTimeTransitionFunction::getTimeIntervall()
{
	return dt;
}
	
void CContinuousTimeTransitionFunction::setTimeIntervall(double dt)
{
	this->dt = dt;
} 

void CContinuousTimeTransitionFunction::setSimulationSteps(int steps)
{
	assert(steps > 0);
	this->simulationSteps = steps;
}

int CContinuousTimeTransitionFunction::getSimulationSteps()
{
	return simulationSteps;
}


CContinuousTimeAndActionTransitionFunction::CContinuousTimeAndActionTransitionFunction(CStateProperties *properties, CContinuousAction *action, double dt) : CContinuousTimeTransitionFunction(properties, new CActionSet(), dt)
{
	actions->add(action);
	this->actionProp = action->getContinuousActionProperties();
	this->contAction = action;
}

CContinuousTimeAndActionTransitionFunction::~CContinuousTimeAndActionTransitionFunction()
{
	delete actions;
}


void CContinuousTimeAndActionTransitionFunction::getDerivationX(CState *oldState, CAction *action, ColumnVector *derivationX, CActionData *data)
{
	assert(action->isType(CONTINUOUSACTION));
	
	if (data)
	{
		getCADerivationX(oldState, dynamic_cast<CContinuousActionData *>(data), derivationX);
	}
	else
	{
		getCADerivationX(oldState, dynamic_cast<CContinuousActionData *>(action->getActionData()), derivationX);
	}
}

CContinuousAction *CContinuousTimeAndActionTransitionFunction::getContinuousAction()
{
	return contAction;
}

CLinearActionContinuousTimeTransitionFunction::CLinearActionContinuousTimeTransitionFunction(CStateProperties *properties, CContinuousAction *action, double dt) : CContinuousTimeAndActionTransitionFunction(properties, action, dt)
{
	A = new ColumnVector(properties->getNumContinuousStates());
	B = new Matrix(properties->getNumContinuousStates(), actionProp->getNumActionValues());

	addType(DM_DERIVATIONUMODEL);
}

CLinearActionContinuousTimeTransitionFunction::~CLinearActionContinuousTimeTransitionFunction()
{
	delete A;
	delete B;
}

void CLinearActionContinuousTimeTransitionFunction::getCADerivationX(CState *oldState, CContinuousActionData *contAction, ColumnVector *derivationX)
{
	assert(oldState->nrows() == derivationX->nrows());
	Matrix *B = getB(oldState);
	ColumnVector *a = getA(oldState);


	*derivationX = (*B) * (*contAction);
	*derivationX = (*derivationX) + (*a); // x' = B(x) * u + a(x)
}

void CLinearActionContinuousTimeTransitionFunction::getDerivationU(CState *oldstate, Matrix *derivation)
{
	*derivation = *getB(oldstate);
}



CDynamicLinearContinuousTimeModel::CDynamicLinearContinuousTimeModel(CStateProperties *properties, CContinuousAction *action, double dt, Matrix *A, Matrix *l_B) : CLinearActionContinuousTimeTransitionFunction(properties, action, dt)
{
	assert((unsigned int ) A->nrows() == properties->getNumContinuousStates() && (unsigned int ) A->ncols() == properties->getNumContinuousStates() && (unsigned int ) B->ncols() == action->getContinuousActionProperties()->getNumActionValues() && (unsigned int ) B->nrows() == properties->getNumContinuousStates());

	*B = *l_B;
	AMatrix = new Matrix(properties->getNumContinuousStates(), properties->getNumContinuousStates());
	(*AMatrix) = *A;
}

CDynamicLinearContinuousTimeModel::~CDynamicLinearContinuousTimeModel()
{
	delete AMatrix;
}

Matrix *CDynamicLinearContinuousTimeModel::getB(CState *)
{
	return B;
}

ColumnVector *CDynamicLinearContinuousTimeModel::getA(CState *state)
{
	*A = (*AMatrix) * (*state);// a(x) = A * x
	return A;
}

CTransitionFunctionEnvironment::CTransitionFunctionEnvironment(CTransitionFunction *model) : CEnvironmentModel(model->getStateProperties())
{
	this->TransitionFunction = model;
	modelState = new CState(getStateProperties());
	nextState = new CState(getStateProperties());

	startStates = NULL;
	nEpisode = 0;
	createdStartStates = false;

	failedRegion = NULL;
	sampleRegion = NULL;
	targetRegion = NULL;

	resetModel();
}
	
CTransitionFunctionEnvironment::~CTransitionFunctionEnvironment()
{
	delete modelState;
	delete nextState;

	if (createdStartStates)
	{
		delete startStates;
	}
}

void CTransitionFunctionEnvironment::doNextState(CPrimitiveAction *action)
{
	TransitionFunction->transitionFunction(modelState, action, nextState);
	CState *buf = modelState;
	modelState = nextState;
	nextState = buf;

	if (targetRegion == NULL)
	{
		reset = TransitionFunction->isResetState(modelState);
	}
	else
	{
		reset = targetRegion->isStateInRegion(modelState);
	}

	if (failedRegion == NULL)
	{
		failed = TransitionFunction->isFailedState(modelState);
	}
	else
	{
		failed = failedRegion->isStateInRegion(modelState);
	}
}

void CTransitionFunctionEnvironment::doResetModel()
{
	if (startStates != NULL)
	{
		startStates->getState(nEpisode, modelState);
		nEpisode ++;
		nEpisode = nEpisode % startStates->getNumStates();
	}
	else
	{
		if (sampleRegion == NULL)
		{
			TransitionFunction->getResetState(modelState);
		}
		else
		{
			sampleRegion->getRandomStateSample(modelState);
		}
	}
}

void CTransitionFunctionEnvironment::getState(CState *state)
{
	assert(state->getStateProperties()->equals(getStateProperties()));
	state->setState(modelState);
}

void CTransitionFunctionEnvironment::setState(CState *state)
{
	assert(state->getStateProperties()->equals(getStateProperties()));
	modelState->setState(state);
}

void CTransitionFunctionEnvironment::setStartStates(CStateList *startStates)
{
	if (createdStartStates)
	{
		delete this->startStates;
		createdStartStates = false;
	}
	this->startStates = startStates;
	nEpisode = 0;
}

void CTransitionFunctionEnvironment::setStartStates(char *filename)
{
	FILE *startStateFile = fopen(filename, "r");
	startStates = new CStateList(getStateProperties());
	startStates->loadASCII(startStateFile);
	fclose(startStateFile);
	nEpisode = 0;

}


void CTransitionFunctionEnvironment::setSampleRegion(CRegion *l_sampleRegion)
{
	this->sampleRegion = l_sampleRegion;
}

void CTransitionFunctionEnvironment::setFailedRegion(CRegion *l_failedRegion)
{
	this->failedRegion = l_failedRegion;
}

void CTransitionFunctionEnvironment::setTargetRegion(CRegion *l_targetRegion)
{
	this->targetRegion = l_targetRegion;
}

CTransitionFunctionFromStochasticModel::CTransitionFunctionFromStochasticModel(CStateProperties *properties,  CAbstractFeatureStochasticModel *model) : CTransitionFunction(properties, model->getActions())
{
	this->stochasticModel = model;

	startStates = new std::list<int>();
	startProbabilities = new std::list<double>();
	endStates = new std::map<int, double>();
}

CTransitionFunctionFromStochasticModel::~CTransitionFunctionFromStochasticModel()
{
	delete startStates;
	delete startProbabilities;
	delete endStates;
}

void CTransitionFunctionFromStochasticModel::addEndState(int state, double probability)
{
	(*endStates)[state] = probability;
}

void CTransitionFunctionFromStochasticModel::addStartState(int state, double probability)
{
	startStates->push_back(state);
	startProbabilities->push_back(probability);
}

bool CTransitionFunctionFromStochasticModel::isResetState(CState *state)
{
	int stateIndex = state->getDiscreteStateNumber();

	std::map<int, double>::iterator it = endStates->find(stateIndex);

	if (it != endStates->end())
	{
		double prob = (*it).second;
		double z = ((double) rand()) / RAND_MAX; 

		return z < prob;
	}
	return false;
}

void CTransitionFunctionFromStochasticModel::getResetState(CState *state)
{
	if (startStates->size() > 0)
	{
		double *prob = new double[startStates->size()];
		std::list<double>::iterator it = startProbabilities->begin();
		for (int i = 0; it != startProbabilities->end(); it ++, i++)
		{
			prob[i] = *it;
		}
		int stateIndex = CDistributions::getSampledIndex(prob, startStates->size());

		std::list<int>::iterator it2 = startStates->begin();
		for (int i = 0; i < stateIndex; it2 ++, i++)
		{
		}
		stateIndex = *it2;

		int stateDim = properties->getDiscreteStateSize();

		for (unsigned int i = 0; i < state->getNumDiscreteStates(); i ++)
		{
			stateDim = properties->getDiscreteStateSize(i);
			state->setDiscreteState(i, stateIndex % stateDim);
			stateIndex = stateIndex / stateDim;  
		}
		delete prob;
	}
	else
	{
		CTransitionFunction::getResetState(state);
	}
}

void CTransitionFunctionFromStochasticModel::transitionFunction(CState *oldstate, CAction *action, CState *newState, CActionData *)
{	
	unsigned int statenum = oldstate->getDiscreteStateNumber();

	int actionIndex = actions->getIndex(action);
	CTransitionList *list = stochasticModel->getForwardTransitions(actionIndex, statenum);

	assert(list->size() > 0);

	double *prob = new double[list->size()];
	CTransitionList::iterator it = list->begin();
	for (int i = 0; it != list->end(); it ++, i++)
	{
		prob[i] = (*it)->getPropability();
	}



	int index = CDistributions::getSampledIndex(prob, list->size());
	delete prob;

	it = list->begin();
	for (int i = 0; i < index; i++)
	{	
		it ++;
	}
	assert(it != list->end());

	int stateIndex = (*it)->getEndState();
	int stateDim = properties->getDiscreteStateSize();

	for (unsigned int i = 0; i < oldstate->getNumDiscreteStates(); i ++)
	{
		stateDim = properties->getDiscreteStateSize(i);
		newState->setDiscreteState(i, stateIndex % stateDim);
		stateIndex = stateIndex / stateDim;  
	}
}


CQFunctionFromTransitionFunction::CQFunctionFromTransitionFunction(CActionSet *actions, CAbstractVFunction *vfunction, CTransitionFunction *model, CRewardFunction *rewardfunction, std::list<CStateModifier *> *modifiers) : CAbstractQFunction(actions), CStateModifiersObject(model->getStateProperties())
{
	this->vfunction = vfunction;
	this->model = model;
	this->rewardfunction = rewardfunction;

	this->actionDataSet = new CActionDataSet(actions);

	nextState = new CStateCollectionImpl(model->getStateProperties());
	intermediateState = new CStateCollectionImpl(model->getStateProperties());

	this->stateCollectionList = new CStateCollectionList(model->getStateProperties());

	addParameter("SearchDepth", 1);
	addParameter("DiscountFactor", 0.95);
	addParameter("VFunctionScale", 1.0);

	addStateModifiers(modifiers);
}

CQFunctionFromTransitionFunction::~CQFunctionFromTransitionFunction()
{
	delete actionDataSet;
	delete nextState;
	delete intermediateState;

	delete stateCollectionList;
}

void CQFunctionFromTransitionFunction::addStateModifier(CStateModifier *modifier)
{
	CStateModifiersObject::addStateModifier(modifier);

	nextState->addStateModifier(modifier);
	intermediateState->addStateModifier(modifier);

	stateCollectionList->addStateModifier(modifier);
}

double CQFunctionFromTransitionFunction::getValue(CStateCollection *state, CAction *action, CActionData *data)
{
	stateCollectionList->clearStateLists();
	stateCollectionList->addStateCollection(state);

	return getValueDepthSearch(stateCollectionList, action, data, my_round(getParameter("SearchDepth")));
}

double CQFunctionFromTransitionFunction::getValueDepthSearch(CStateCollectionList *stateList, CAction *action, CActionData *data, int depth)
{
	stateList->getStateCollection(stateList->getNumStateCollections() - 1, intermediateState);
	if (depth == 0)
	{
		double vFunctionScale = getParameter("VFunctionScale");
		return vfunction->getValue(intermediateState) * vFunctionScale;
	}

	if (data)
	{
		actionDataSet->getActionData(action)->setData(data);
	}

	CActionData *ldata = actionDataSet->getActionData(action);

	int duration = 1;
	
	double rewardValue = 0;
	if (model->isType(DM_EXTENDEDACTIONMODEL))
	{
		CExtendedActionTransitionFunction *extModel = dynamic_cast<CExtendedActionTransitionFunction *>(model);
		rewardValue = extModel->transitionFunctionAndReward(intermediateState->getState(model->getStateProperties()), action, nextState->getState(model->getStateProperties()), ldata, rewardfunction, getParameter("DiscountFactor"));
		nextState->newModelState();
	}
	else
	{
		model->transitionFunction(intermediateState->getState(model->getStateProperties()), action, nextState->getState(model->getStateProperties()), ldata);
		nextState->newModelState();
		rewardValue = rewardfunction->getReward(intermediateState, action, nextState);
	}

	if ((action)->isType(MULTISTEPACTION))
	{
		CActionData *actionData = actionDataSet->getActionData(action);
		CMultiStepActionData *multiStepActionData  = dynamic_cast<CMultiStepActionData *>(actionData);
		duration = multiStepActionData->duration;
	}
	else
	{
		duration = action->getDuration();
	}

	if (DebugIsEnabled('q'))
	{
		DebugPrint('q', "Calculated NextState for Action: %d (", actions->getIndex(action));
		
		if (ldata)
		{
			ldata->saveASCII(DebugGetFileHandle('q'));
		}
		
//		data->saveASCII(DebugGetFileHandle('q'));

		DebugPrint('q', ")\n");
		nextState->getState()->saveASCII(DebugGetFileHandle('q'));
		DebugPrint('q',"\n");
	}
	
	double value = 0.0;

	if (!model->isResetState(nextState->getState()))
	{
		
		if(depth > 1)
		{
			stateList->addStateCollection(nextState);
	
			CActionSet::iterator it = actions->begin();
	
			value = getValueDepthSearch(stateList, *it, NULL, depth - 1);
			double max = value;
	
			it ++;
	
			for (; it != actions->end();it ++)
			{
				value = getValueDepthSearch(stateList, *it, NULL, depth - 1);
				if (max < value)
				{
					max = value;
				}
			}
			value = max;
	
			stateList->removeLastStateCollection();
		}
		else
		{
			double vFunctionScale = getParameter("VFunctionScale");
			value = vfunction->getValue(nextState) * vFunctionScale;
		}
	}
	else
	{
			DebugPrint('q', "Endstate... ");
	}
	DebugPrint('q', "Value: %f Reward %f\n", value, rewardValue);

	return rewardValue + pow(getParameter("DiscountFactor"), duration) * value;
}

CContinuousTimeQFunctionFromTransitionFunction::CContinuousTimeQFunctionFromTransitionFunction(CActionSet *actions, CVFunctionInputDerivationCalculator *vfunction, CContinuousTimeTransitionFunction *model, CRewardFunction *rewardfunction, std::list<CStateModifier *> *modifiers) : CAbstractQFunction(actions), CStateModifiersObject(model->getStateProperties())
{
	this->vfunction = vfunction;
	this->model = model;
	this->rewardfunction = rewardfunction;


	nextState = new CStateCollectionImpl(model->getStateProperties());

	derivationXModel = new CState(model->getStateProperties());
	derivationXVFunction = new CState(model->getStateProperties());

	addStateModifiers(modifiers);
}

CContinuousTimeQFunctionFromTransitionFunction::CContinuousTimeQFunctionFromTransitionFunction(CActionSet *actions, CVFunctionInputDerivationCalculator *vfunction, CContinuousTimeTransitionFunction *model, CRewardFunction *rewardfunction) : CAbstractQFunction(actions), CStateModifiersObject(model->getStateProperties())
{
	this->vfunction = vfunction;
	this->model = model;
	this->rewardfunction = rewardfunction;

	nextState = new CStateCollectionImpl(model->getStateProperties());

	derivationXModel = new CState(model->getStateProperties());
	derivationXVFunction = new CState(model->getStateProperties());
}

CContinuousTimeQFunctionFromTransitionFunction::~CContinuousTimeQFunctionFromTransitionFunction()
{
	delete nextState;

	delete derivationXModel;
	delete derivationXVFunction;
}

double CContinuousTimeQFunctionFromTransitionFunction::getValueVDerivation(CStateCollection *state, CAction *action, CActionData *data, ColumnVector *derivationXVFunction)
{
	model->getDerivationX(state->getState(model->getStateProperties()), action, derivationXModel, data);
	model->transitionFunction(state->getState(model->getStateProperties()), action, nextState->getState(model->getStateProperties()),  data);

//	double reward = rewardfunction->getReward(state, action, nextState);
	return dotproduct(*derivationXVFunction, *derivationXModel);
}


void CContinuousTimeQFunctionFromTransitionFunction::getActionValues(CStateCollection *state, CActionSet *actions, double *actionValues, CActionDataSet *actionDataSet)
{
	vfunction->getInputDerivation(state, derivationXVFunction);

	CActionSet::iterator it = actions->begin();

	for (int i = 0; it != actions->end(); it ++, i ++)
	{
		actionValues[i] = getValueVDerivation(state, *it, actionDataSet->getActionData(*it), derivationXVFunction);
	}

	if (DebugIsEnabled('v'))
	{
		DebugPrint('v', "CTQ Function: ");
		for (unsigned int i = 0; i < actions->size(); i++)
		{
			DebugPrint('v', "%f ", actionValues[i]);
		}
		DebugPrint('v', "\n");
	}
}


double CContinuousTimeQFunctionFromTransitionFunction::getValue(CStateCollection *state, CAction *action, CActionData *data)
{
	vfunction->getInputDerivation(state, derivationXVFunction);

	return getValueVDerivation(state, action, data, derivationXVFunction);
}

void CContinuousTimeQFunctionFromTransitionFunction::addStateModifier(CStateModifier *modifier)
{
	CStateModifiersObject::addStateModifier(modifier);

	nextState->addStateModifier(modifier);
}
