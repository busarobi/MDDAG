#include "cevaluator.h"
#include "cagent.h"
#include "cagentcontroller.h"
#include "crewardfunction.h"
#include "cstatecollection.h"
#include "cenvironmentmodel.h"
#include "ctransitionfunction.h"

#include <math.h>
 
CPolicyEvaluator::CPolicyEvaluator(CAgent *agent, CRewardFunction *rewardFunction, int nEpisodes, int nStepsPerEpisode) : CSemiMDPRewardListener(rewardFunction)
{
	this->agent = agent;
	this->nEpisodes = nEpisodes;
	this->nStepsPerEpisode = nStepsPerEpisode;

	controller = NULL;
	detController = agent;
}

void CPolicyEvaluator::setAgentController(CAgentController *l_controller)
{
	controller = l_controller;
}

void CPolicyEvaluator::setDeterministicController(CDeterministicController *l_detController)
{
	detController = l_detController;
}


double CPolicyEvaluator::evaluatePolicy()
{
	double value = 0;
	agent->addSemiMDPListener(this);
	
	CAgentController *tempController = NULL;
	if (controller)
	{
		tempController = detController->getController();
		detController->setController(controller);	
	}

	for (int i = 0; i < nEpisodes; i ++)
	{
		agent->startNewEpisode();
		agent->doControllerEpisode(1, nStepsPerEpisode);
		value += this->getEpisodeValue();
	}
	value /= nEpisodes;
	agent->removeSemiMDPListener(this);
	
	if (tempController)
	{
		detController->setController(tempController);
	}

	return value;
}

CAverageRewardCalculator::CAverageRewardCalculator(CAgent *agent, CRewardFunction *rewardFunction, int nEpisodes, int nStepsPerEpisode, double minReward) : CPolicyEvaluator(agent, rewardFunction, nEpisodes, nStepsPerEpisode)
{
	nSteps = 0;
	averageReward = 0;
	this->minReward = minReward;
}

double CAverageRewardCalculator::getEpisodeValue()
{
	if (agent->getEnvironmentModel()->isReset())
	{
		averageReward += minReward * (nStepsPerEpisode - nSteps);
		nSteps = nStepsPerEpisode;
	}
	return averageReward / nSteps;
}


void CAverageRewardCalculator::nextStep(CStateCollection *, CAction *, double reward, CStateCollection *)
{
	averageReward += reward;
	nSteps ++;
}

void CAverageRewardCalculator::newEpisode()
{
	averageReward = 0;
	nSteps = 0;
}

CRewardPerEpisodeCalculator::CRewardPerEpisodeCalculator(CAgent *agent, CRewardFunction *rewardFunction, int nEpisodes, int nStepsPerEpisode) : CPolicyEvaluator(agent, rewardFunction, nEpisodes, nStepsPerEpisode)
{
	reward = 0;
}

double CRewardPerEpisodeCalculator::getEpisodeValue()
{
	return reward;
}


void CRewardPerEpisodeCalculator::nextStep(CStateCollection *, CAction *, double l_reward, CStateCollection *)
{
	reward += l_reward;
}

void CRewardPerEpisodeCalculator::newEpisode()
{
	reward = 0;
}

CValueCalculator::CValueCalculator(CAgent *agent, CRewardFunction *rewardFunction, int nEpisodes, int nStepsPerEpisode, double gamma) : CPolicyEvaluator(agent, rewardFunction, nEpisodes, nStepsPerEpisode)
{
	nSteps = 0;
	value = 0;
	addParameter("DiscountFactor", gamma);
}

double CValueCalculator::getEpisodeValue()
{
	return value;
}


void CValueCalculator::nextStep(CStateCollection *, CAction *action, double reward, CStateCollection *)
{
	value += pow(getParameter("DiscountFactor"), nSteps) * reward;
	nSteps += action->getDuration();
}

void CValueCalculator::newEpisode()
{
	value = 0;
	nSteps = 0;
}

CPolicySameStateEvaluator::CPolicySameStateEvaluator(CAgent *agent, CRewardFunction *rewardFunction, CTransitionFunctionEnvironment *environment, CStateList *l_startStates, int nStepsPerEpisode) : CPolicyEvaluator(agent, rewardFunction, l_startStates->getNumStates(), nStepsPerEpisode)
{
	this->startStates = new CStateList(environment->getStateProperties());
	
	this->environment = environment;
	
	setStartStates(l_startStates);

	sender = agent;
}

CPolicySameStateEvaluator::CPolicySameStateEvaluator(CAgent *agent, CRewardFunction *rewardFunction, CTransitionFunctionEnvironment *environment, int numStartStates, int nStepsPerEpisode) : CPolicyEvaluator(agent, rewardFunction, numStartStates, nStepsPerEpisode)
{
	this->startStates = new CStateList(environment->getStateProperties());
	
	this->environment = environment;
	
	getNewStartStates(numStartStates);

	sender = agent;
}


CPolicySameStateEvaluator::~CPolicySameStateEvaluator()
{
	delete startStates;
}

double CPolicySameStateEvaluator::getValueForState(CState *startState, int nSteps)
{
	if (nSteps > 0)
	{
		//printf("Evaluator Start State: \n");		
		//startState->saveASCII(stdout);


		CAgentController *tempController = NULL;
		if (controller)
		{
			tempController = detController->getController();
			detController->setController(controller);	
		}


		sender->addSemiMDPListener(this);
		agent->startNewEpisode();
		environment->setState(startState);
		agent->doControllerEpisode(1, nSteps);
		sender->removeSemiMDPListener(this);
		if (tempController)
		{
			detController->setController(tempController);
		}

		return this->getEpisodeValue();
	}
	else
	{
		return 0;
	}
}

double CPolicySameStateEvaluator::getActionValueForState(CState *startState, CAction *action, int nSteps)
{
	if (nSteps > 0)
	{
		CAgentController *tempController = NULL;
		if (controller)
		{
			tempController = detController->getController();
			detController->setController(controller);	
		}


		sender->addSemiMDPListener(this);
	
		agent->startNewEpisode();
		
		environment->setState(startState);
		
		agent->doAction( action);
		if (nSteps > 1 && !agent->getEnvironmentModel()->isReset())
		{
			agent->doControllerEpisode(1, nSteps - 1);
		}
		sender->removeSemiMDPListener(this);

		if (tempController)
		{
			detController->setController(tempController);
		}

	}
	return this->getEpisodeValue();
}

double CPolicySameStateEvaluator::evaluatePolicy()
{
	double value = 0;
	
	CState *startState = new CState(environment->getStateProperties());
	for (int i = 0; i < nEpisodes; i ++)
	{
		
		startStates->getState(i, startState);
		value += getValueForState(startState, nStepsPerEpisode);
	}
	value /= nEpisodes;
	
	delete startState;
	return value;
}

void CPolicySameStateEvaluator::getNewStartStates(int numStartStates)
{
	CState *startState = new CState(environment->getStateProperties());

	startStates->clear();

	for (int i = 0; i < numStartStates; i++)
	{
		environment->resetModel();
		environment->getState(startState);
		startStates->addState(startState);
	}
	nEpisodes = startStates->getNumStates();
	delete startState;
}

void CPolicySameStateEvaluator::setStartStates(CStateList *newList)
{
	int numStartStates = newList->getNumStates();

	CState *startState = new CState(environment->getStateProperties());

	startStates->clear();

	for (int i = 0; i < numStartStates; i++)
	{
		newList->getState(i, startState);
		startStates->addState(startState);
	}
	nEpisodes = startStates->getNumStates();
	delete startState;
}

CAverageRewardSameStateCalculator::CAverageRewardSameStateCalculator(CAgent *agent, CRewardFunction *rewardFunction, CTransitionFunctionEnvironment *environment, CStateList *startStates, int nStepsPerEpisode, double minReward) : CPolicySameStateEvaluator(agent, rewardFunction, environment, startStates, nStepsPerEpisode)
{
	nSteps = 0;
	averageReward = 0;

	this->minReward = minReward;

}

double CAverageRewardSameStateCalculator::getEpisodeValue()
{
	if (agent->getEnvironmentModel()->isFailed())
	{
		averageReward += minReward * (nStepsPerEpisode - nSteps);
		nSteps = nStepsPerEpisode;
	}
	return averageReward / nSteps;
}


void CAverageRewardSameStateCalculator::nextStep(CStateCollection *, CAction *, double reward, CStateCollection *)
{
	averageReward += reward;
	nSteps ++;
}

void CAverageRewardSameStateCalculator::newEpisode()
{
	averageReward = 0;
	nSteps = 0;
}

CValueSameStateCalculator::CValueSameStateCalculator(CAgent *agent, CRewardFunction *rewardFunction, CTransitionFunctionEnvironment *environment, CStateList *startStates, int nStepsPerEpisode, double gamma) : CPolicySameStateEvaluator(agent, rewardFunction, environment, startStates, nStepsPerEpisode)
{
	nSteps = 0;
	value = 0;
	addParameter("DiscountFactor", gamma);
}

double CValueSameStateCalculator::getEpisodeValue()
{
//	printf("Value %f\n", value);
	return value;
}


void CValueSameStateCalculator::nextStep(CStateCollection *, CAction *action, double reward, CStateCollection *)
{
	value += pow(getParameter("DiscountFactor"), nSteps) * reward;
	nSteps += action->getDuration();

	//printf("Steps %d, Reward %f\n", nSteps, reward);
}

void CValueSameStateCalculator::newEpisode()
{
//	printf("Value reset %f\n", value); 

	value = 0;
	nSteps = 0;
}

CPolicyGreedynessEvaluator::CPolicyGreedynessEvaluator(CAgent *agent, CRewardFunction *reward, int nEpisodes, int nStepsPerEpsiode, CAgentController *l_greedyPolicy) : CPolicyEvaluator(agent, reward, nEpisodes, nStepsPerEpsiode)
{
	nGreedyActions = 0;
	greedyPolicy = l_greedyPolicy;

	actionDataSet = new CActionDataSet(greedyPolicy->getActions());
}

CPolicyGreedynessEvaluator::~CPolicyGreedynessEvaluator()
{
	delete actionDataSet;
}

double CPolicyGreedynessEvaluator::getEpisodeValue()
{
	return (double) nGreedyActions / (double) nStepsPerEpisode;
}

void CPolicyGreedynessEvaluator::nextStep(CStateCollection *oldState, CAction *action, double , CStateCollection *)
{
	CAction *greedyAction = greedyPolicy->getNextAction(oldState);
	if (action->isSameAction(greedyAction, actionDataSet->getActionData(greedyAction)))
	{
		nGreedyActions ++;
	}
}

void CPolicyGreedynessEvaluator::newEpisode()
{
	nGreedyActions = 0;
}


