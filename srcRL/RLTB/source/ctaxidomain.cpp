#include "ril_debug.h"

#include "ctaxidomain.h"

#include "cstate.h"
#include "cstatecollection.h"
#include "cstateproperties.h"
#include "cstatemodifier.h"
#include "caction.h"

CTaxiDomain::CTaxiDomain(char* filename) : CGridWorldModel(filename, 100)
{
	targetXYValues = NULL;
	delete properties;
	properties = new CStateProperties(0, 5);
	initTargetVector();
}

CTaxiDomain::~CTaxiDomain()
{
	if (targetXYValues != NULL)
	{
		std::vector<std::pair<int, int> *>::iterator it = targetXYValues->begin();
		for (;it != targetXYValues->end(); it++)
		{
			delete *it;
		}
		delete targetXYValues;
	}
}

void CTaxiDomain::initTargetVector()
{
	if (targetXYValues == NULL)
	{
		targetXYValues = new std::vector<std::pair<int, int> *>();
	}
	else
	{
		std::vector<std::pair<int, int> *>::iterator it = targetXYValues->begin();
		for (;it != targetXYValues->end(); it++)
		{
			delete *it;
		}
		targetXYValues->clear();
	}

	for (unsigned int x = 0; x < this->getSizeX(); x++)
	{
		for (unsigned int y = 0; y < this->getSizeY(); y++)
		{
			if (target_values->find(getGridValue(x,y)) != target_values->end())
			{
				targetXYValues->push_back(new std::pair<int, int>(x,y));
			}
		}
	}
	properties->setDiscreteStateSize(0, getSizeX());
	properties->setDiscreteStateSize(1, getSizeY());
	properties->setDiscreteStateSize(2, 0);
	properties->setDiscreteStateSize(3, targetXYValues->size() + 1);
	properties->setDiscreteStateSize(4, targetXYValues->size());
}

void CTaxiDomain::load(FILE *stream)
{
	CGridWorldModel::load(stream);
	initTargetVector();
}

int CTaxiDomain::getTargetPositionX(int numTarget)
{
	return (*targetXYValues)[numTarget]->first;
}

int CTaxiDomain::getTargetPositionY(int numTarget)
{
	return (*targetXYValues)[numTarget]->second;
}

void CTaxiDomain::transitionFunction(CState *oldState, CAction *action, CState *newState, CActionData *data)
{
	if (action->isType(GRIDWORLDACTION))
	{
		CGridWorldModel::transitionFunction(oldState, action, newState, data);
	}
	int pos_x = oldState->getDiscreteState(0);
	int pos_y = oldState->getDiscreteState(1);
	
	int pasLocation = oldState->getDiscreteState(3);
	int pasDestination = oldState->getDiscreteState(4);

	if (action->isType(PICKUPACTION))
	{
		if (pasLocation < getNumTargets())
		{
			if (getTargetPositionX(pasLocation) == pos_x && getTargetPositionY(pasLocation) == pos_y)
			{
				pasLocation = getNumTargets();
			}
		}
	}
	if (action->isType(PUTDOWNACTION))
	{
		if (pasLocation == getNumTargets())
		{
			if (getTargetPositionX(pasDestination) == pos_x && getTargetPositionY(pasDestination) == pos_y)
			{
				pasLocation = pasDestination;
			}
		}
	}

	newState->setDiscreteState(3, pasLocation);
	newState->setDiscreteState(4, pasDestination);
}

bool CTaxiDomain::isResetState(CState *state)
{
	return (CGridWorldModel::isFailedState(state) || state->getDiscreteState(3) == state->getDiscreteState(4)); 
}

void CTaxiDomain::getResetState(CState *resetState)
{
	CGridWorldModel::getResetState(resetState);
	int location = rand() % targetXYValues->size();
	resetState->setDiscreteState(3, location);
	
	int target = rand() % targetXYValues->size();
	
	if (target == location)
	{
		target = (target + 1) % this->getNumTargets();
	}
	resetState->setDiscreteState(4, target);
}


double CTaxiDomain::getReward(CStateCollection *oldStateCol, CAction *action, CStateCollection *newStateCol)
{
	CState *oldState = oldStateCol->getState();
	CState *newState = newStateCol->getState();

	double reward = this->getRewardStandard();
	
	if (action->isType(GRIDWORLDACTION))
	{
		if (oldState->getDiscreteState(0) == newState->getDiscreteState(0) && oldState->getDiscreteState(1) == newState->getDiscreteState(1))
		{
			reward += this->getRewardBounce();
		}
	}

	if (action->isType(PICKUPACTION))
	{
		// Wrong Pickup action
		if (!(oldState->getDiscreteState(3) < getNumTargets() && newState->getDiscreteState(3) == getNumTargets()))
		{
			reward += this->getRewardBounce();;
		}
	}
	if (action->isType(PUTDOWNACTION))
	{
		if (oldState->getDiscreteState(3) == getNumTargets() && oldState->getDiscreteState(0) == getTargetPositionX(oldState->getDiscreteState(4)) && oldState->getDiscreteState(1) == getTargetPositionY(oldState->getDiscreteState(4)))
		{
			reward += getRewardSuccess();
		}
		else
		{
			reward += this->getRewardBounce();;
		}
	}
	return reward;
}

CTaxiHierarchicalBehaviour::CTaxiHierarchicalBehaviour(CEpisode *currentEpisode, int target, CTaxiDomain *taximodel) : CHierarchicalSemiMarkovDecisionProcess(currentEpisode)
{
	this->model = taximodel;
	this->target = target;
}

CTaxiHierarchicalBehaviour::~CTaxiHierarchicalBehaviour()
{

}

bool CTaxiHierarchicalBehaviour::isFinished(CStateCollection *, CStateCollection *newStateCol)
{
	CState *state = newStateCol->getState();
	if (state->getDiscreteState(0) == model->getTargetPositionX(target) && state->getDiscreteState(1) == model->getTargetPositionY(target))
	{
		return true;
	}
	else
	{
		return false;
	}
}

double  CTaxiHierarchicalBehaviour::getReward(CStateCollection *oldStateCol, CAction *action, CStateCollection *newStateCol)
{
	CState *oldState = oldStateCol->getState();
	CState *newState = newStateCol->getState();

	double reward = model->getRewardStandard();
	
	if (action->isType(GRIDWORLDACTION))
	{
		if (oldState->getDiscreteState(0) == newState->getDiscreteState(0) && oldState->getDiscreteState(1) == newState->getDiscreteState(1))
		{
			reward += model->getRewardBounce();
		}
		if (newState->getDiscreteState(0) == model->getTargetPositionX(target) && newState->getDiscreteState(1) == model->getTargetPositionY(target))
		{
			reward += model->getRewardSuccess() / 2;
		}
	}
	return reward;
}


CTaxiIsTargetDiscreteState::CTaxiIsTargetDiscreteState(CTaxiDomain *model) : CAbstractStateDiscretizer(model->getNumTargets() + 1)
{
	this->model = model;
}

unsigned int CTaxiIsTargetDiscreteState::getDiscreteStateNumber(CStateCollection *stateCol)
{
	int target = -1;
	CState *state = stateCol->getState();
	for (int i = 0; i < model->getNumTargets(); i++)
	{
		if (state->getDiscreteState(0) == model->getTargetPositionX(i) && state->getDiscreteState(1) == model->getTargetPositionY(i))
		{
			target = i;
			break;
		}
	}
	return target + 1;
}
