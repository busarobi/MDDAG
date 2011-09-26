#include "cbatchlearning.h"


#include "crewardmodel.h"
#include "cresiduals.h"
#include "ccontinuousactions.h"
#include "ctreebatchlearning.h"
#include "cmontecarlo.h"
#include "chistory.h"
#include "clstd.h"
#include "cagentlogger.h"
#include "ckdtrees.h"
#include "cnearestneighbor.h"
#include "csamplingbasedmodel.h"

#include "caction.h"
#include "cstate.h"
#include "cstatecollection.h"
#include "cstateproperties.h"
#include "cepisode.h"
#include "cepisodehistory.h"
#include "cvfunction.h"
#include "cqfunction.h"
#include "cpolicies.h"
#include "cgradientfunction.h"

#include <math.h>
#include <time.h>



CBatchLearningPolicy::CBatchLearningPolicy(CActionSet *actions) : CDeterministicController(actions)
{
	actionDataSet = new CActionDataSet(actions);	
	nextAction = NULL;	
}

CBatchLearningPolicy::~CBatchLearningPolicy()
{
	delete actionDataSet;
}

		
CAction *CBatchLearningPolicy::getNextAction(CStateCollection *, CActionDataSet *actionData)
{
	assert(nextAction != NULL);
	if (actionData)
	{
		actionData->setActionData(nextAction, actionDataSet->getActionData(nextAction));
	}
	return nextAction;
}
		
void CBatchLearningPolicy::setAction(CAction *action, CActionData *data)
{
	nextAction = action;
	if (data)
	{
		actionDataSet->setActionData(nextAction, data);
	}
}

CPolicyEvaluation::CPolicyEvaluation(int maxEpisodes)
{
	addParameter("PolicyEvaluationMaxEpisodes", maxEpisodes);
}

CPolicyEvaluation::~CPolicyEvaluation()
{

}

void CPolicyEvaluation::evaluatePolicy()
{
	int maxEvaluationEpisodes =  (int) getParameter("PolicyEvaluationMaxEpisodes");
	evaluatePolicy(maxEvaluationEpisodes);
}

double CPolicyEvaluationGradientFunction::getWeightDifference(double *oldWeights)
{
	double sum = 0;
	double *newWeights = new double[learnData->getNumWeights()];
	
	learnData->getWeights(newWeights);
	
	for (int i = 0; i < learnData->getNumWeights(); i ++)
	{
		sum += pow(oldWeights[i] - newWeights[i], 2);
	}
	
	delete [] newWeights;
	
	return sqrt(sum);
}

	
CPolicyEvaluationGradientFunction::CPolicyEvaluationGradientFunction(CGradientUpdateFunction *l_learnData,  double l_treshold, int maxEpisodes)
{
	addParameter("PolicyEvaluationTreshold", l_treshold);
	addParameter("PolicyEvaluationMaxEpisodes", maxEpisodes);
	learnData = l_learnData;
		
	oldWeights = new double[learnData->getNumWeights()];
}

CPolicyEvaluationGradientFunction::~CPolicyEvaluationGradientFunction()
{
	delete [] oldWeights;
}
		
void CPolicyEvaluationGradientFunction::resetLearnData()
{
	if (resetData)
	{
		learnData->resetLearnData();
	}
}
	



COnlinePolicyEvaluation::COnlinePolicyEvaluation(CAgent *l_agent, CSemiMDPListener *l_learner, CGradientUpdateFunction *learnData, int l_maxEvaluationEpisodes, int l_numSteps, int l_checkWeightsPerEpisode) : CPolicyEvaluationGradientFunction(learnData, 0.01, l_maxEvaluationEpisodes)
{
	resetData = false;

	addParameter("PolicyEvaluationCheckWeights", l_checkWeightsPerEpisode);
	addParameter("PolicyEvaluationMaxSteps", l_numSteps);

	learner = l_learner;
	agent = l_agent;
	semiMDPSender = agent;
	
	addParameters(learner);
}	
	
COnlinePolicyEvaluation::~COnlinePolicyEvaluation()
{
}

void COnlinePolicyEvaluation::setSemiMDPSender(CSemiMDPSender *sender)
{
	semiMDPSender = sender;
}
		
		
void COnlinePolicyEvaluation::evaluatePolicy(int maxEvaluationEpisodes)
{
	resetLearnData();
	semiMDPSender->addSemiMDPListener(learner);
	learnData->getWeights(oldWeights);
	
	int checkWeightsEpisode = (int) getParameter("PolicyEvaluationCheckWeights");
	int numSteps = (int) getParameter("PolicyEvaluationMaxSteps");

	double treshold = getParameter("PolicyEvaluationTreshold");
		
	for (int i = 0; i < maxEvaluationEpisodes; i++)
	{
		agent->doControllerEpisode(1, numSteps);	
		
		if ((i + 1) % checkWeightsEpisode == 0)
		{
			double difference = getWeightDifference(oldWeights);
			if (difference < treshold)
			{
				printf("PE: FINISHED Update Difference in Weight Vector after %d Episodes: %f\n",i, difference);
//				updatePolicy();
				break;
			}
			printf("PE: Update Difference in Weight Vector after %d  Episode: %f\n", i + 1, difference);

			learnData->getWeights(oldWeights);		
			
		}
	}
	
	semiMDPSender->removeSemiMDPListener(learner);
}
		

CLSTDOnlinePolicyEvaluation::CLSTDOnlinePolicyEvaluation(CAgent *agent, CLSTDLambda *learner, CGradientUpdateFunction *learnData, int maxEvaluationSteps, int numSteps) : COnlinePolicyEvaluation(agent, learner, learnData, maxEvaluationSteps, numSteps, maxEvaluationSteps / 5)
{
	lstdLearner = learner;

	
	resetData = true;
}

CLSTDOnlinePolicyEvaluation::~CLSTDOnlinePolicyEvaluation()
{
}

void CLSTDOnlinePolicyEvaluation::resetLearnData()
{
	COnlinePolicyEvaluation::resetLearnData();
	if (resetData)
	{
		lstdLearner->resetData();
	}
}

double CLSTDOnlinePolicyEvaluation::getWeightDifference(double *oldWeights)
{
	lstdLearner->doOptimization();
	return COnlinePolicyEvaluation::getWeightDifference(oldWeights);
}

/*
COfflinePolicyEvaluation::COfflinePolicyEvaluation(CStepHistory *l_stepHistory, CSemiMDPListener *l_learner, CGradientUpdateFunction *learnData, std::list<CStateModifier *> *l_modifiers, int l_maxEvaluationEpisodes) : CPolicyEvaluation(learnData)
{
	resetData = false;
	learner = l_learner;
	stepHistory = l_stepHistory;
	
	modifiers = l_modifiers;
	
	addParameter("PolicyEvaluationMaxOfflineEpisodes", l_maxEvaluationEpisodes);
	
	addParameters(learner);
}

COfflinePolicyEvaluation::~COfflinePolicyEvaluation()
{
	
}
		
void COfflinePolicyEvaluation::evaluatePolicy()
{
	CActionSet *actions = stepHistory->getActions();
	CStep *step = new CStep(stepHistory->getStateProperties(), modifiers, actions);
	CActionDataSet *dataSet = new CActionDataSet(actions);
	
	resetLearnData();
	learner->newEpisode();
	
	int maxEvaluationEpisodes = (int) getParameter("PolicyEvaluationMaxOfflineEpisodes");
	double treshold = getParameter("PolicyEvaluationTreshold");
	
	for (int j = 0; j < maxEvaluationEpisodes; j ++)
	{
		learnData->getWeights(oldWeights);
	
		printf("Offline Policy Evaluation, Episode %d, %d Steps for Evaluation\n", j, stepHistory->getNumSteps());
		
		double dur1 = 0;
		double dur2 = 0;
		
		
		timespec t1;
		timespec t2;
		timespec t3;

		
		for (int i = 0; i < stepHistory->getNumSteps(); i++)
		{
			clock_gettime(CLOCK_REALTIME, &t1);
//		
			stepHistory->getStep(i, step);
//			
			clock_gettime(CLOCK_REALTIME, &t2);
								
			CActionData *data = step->action->getActionData();
			if (data != NULL)
			{
				data->setData(step->actionData->getActionData(step->action));
			}
			learner->nextStep(step->oldState, step->action, step->newState);
			
			clock_gettime(CLOCK_REALTIME, &t3);
//	
//			
			dur1 += (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_nsec - t1.tv_nsec) / 1000000.0;
			dur2 += (t3.tv_sec - t1.tv_sec) * 1000 + (t3.tv_nsec - t1.tv_nsec) / 1000000.0;

			
			if (i % 1000 == 0)
			{
				printf("%d Steps\n", i);
				printf("Time for step %f, time for learners %f\n", dur1, dur2);

				dur1 = 0;
				dur2 = 0;
				
			}
			
		}
		double difference = getWeightDifference(oldWeights);
			
		if (difference < treshold)
		{
			printf("PE: FINISHED Update Difference in Weight Vector after %d Episodes: %f\n",j, difference);
			break;
		}
		printf("PE: Update Difference in Weight Vector after %d Episode: %f\n", j, difference);
	}
	
	CActionSet::iterator it = actions->begin();
	
	for (;it != actions->end(); it ++)
	{
		(*it)->getActionData()->setData(dataSet->getActionData(*it));
	}
	delete step;
	delete dataSet;
}
*/

COfflineEpisodePolicyEvaluation::COfflineEpisodePolicyEvaluation(CEpisodeHistory *l_episodeHistory, CSemiMDPRewardListener *l_learner, CGradientUpdateFunction *learnData, std::list<CStateModifier *> *l_modifiers, int l_maxEvaluationEpisodes) : CPolicyEvaluationGradientFunction(learnData, 0.01, l_maxEvaluationEpisodes)
{
	resetData = false;
	learner = l_learner;
	
	episodeHistory = l_episodeHistory;
	rewardLogger = NULL;
	
	modifiers = l_modifiers;
	
	
	addParameters(learner);
	
	policy = NULL;	
}

COfflineEpisodePolicyEvaluation::COfflineEpisodePolicyEvaluation(CEpisodeHistory *l_episodeHistory, CRewardHistory *l_rewardLogger, CSemiMDPRewardListener *l_learner, CGradientUpdateFunction *learnData, std::list<CStateModifier *> *l_modifiers, int l_maxEvaluationEpisodes) : CPolicyEvaluationGradientFunction(learnData, 0.01, l_maxEvaluationEpisodes)
{
	resetData = false;
	learner = l_learner;
		
	episodeHistory = l_episodeHistory;
	rewardLogger = l_rewardLogger;
	
	modifiers = l_modifiers;
	
	
	addParameters(learner);
	
	policy = NULL;	
}

COfflineEpisodePolicyEvaluation::~COfflineEpisodePolicyEvaluation()
{
	
}

void COfflineEpisodePolicyEvaluation::setBatchLearningPolicy(CBatchLearningPolicy *l_policy)
{
	policy = l_policy;
}

		
void COfflineEpisodePolicyEvaluation::evaluatePolicy(int maxEvaluationEpisodes)
{
	CActionSet *actions = episodeHistory->getActions();
	CStep *step = new CStep(episodeHistory->getStateProperties(), episodeHistory->getStateModifiers(), actions);
	CActionDataSet *dataSet = new CActionDataSet(actions);
	CActionDataSet *nextDataSet = new CActionDataSet(actions);
	
	resetLearnData();
	
	learner->newEpisode();
	startNewEpisode();
	
	double treshold = getParameter("PolicyEvaluationTreshold");
	
	static int count = episodeHistory->getNumEpisodes() - 1;

	printf("Offline Policy Evaluation, Episodes %d (%d), beginning with Episode %d\n", episodeHistory->getNumEpisodes(), rewardLogger->getNumEpisodes(), count);	

	printf("OfflineEvaluation: Episode has %d (%d), State Modifiers, Episodes: %d\n", episodeHistory->getStateModifiers()->size(), step->oldState->getStateModifiers()->size(), maxEvaluationEpisodes);
	
	int resetCount = 0;

	for (int j = 0; j < maxEvaluationEpisodes; j ++)
	{
		learnData->getWeights(oldWeights);

		if (count < 0 || count >= episodeHistory->getNumEpisodes())
		{		
			count = episodeHistory->getNumEpisodes() - 1;
			resetCount ++;

			if (resetCount > 10)
			{
				break;
			}
		}
		
		if (count < 0)
		{
			break;
		}

		CEpisode *episode = episodeHistory->getEpisode(count);
		
		CRewardEpisode *rewardEpisode = NULL;
		if (rewardLogger)
		{
			rewardEpisode = rewardLogger->getEpisode(count);
		}
		
		learner->newEpisode();
		startNewEpisode();
		
		for (int i = 0; i < episode->getNumSteps(); i++)
		{
			episode->getStep(i, step);
								
			CActionData *data = step->action->getActionData();
			
			if (policy != NULL)
			{
				CAction *nextAction;
				if (i < episode->getNumSteps() - 1)
				{
					nextAction = episode->getAction(i + 1, nextDataSet);
				}
				else
				{
					nextAction = NULL;
				}			
				policy->setAction(nextAction, nextDataSet->getActionData(nextAction));
			}
			
			if (data != NULL)
			{
				data->setData(step->actionData->getActionData(step->action));
			}
			if (rewardLogger)
			{
				double reward = rewardEpisode->getReward(i);
				
				if (isnan(reward))
				{
					printf("Reward is nan !! %d %d %d %d %d\n", count, i, episode->getNumSteps(), rewardEpisode->getNumRewards(), rewardLogger->getNumEpisodes());
					reward = rewardEpisode->getReward(i);
					assert(false);
				}
				
				learner->nextStep(step->oldState, step->action, rewardEpisode->getReward(i), step->newState);
			}
			else
			{
				learner->nextStep(step->oldState, step->action, step->newState);
			}
			sendNextStep(step->oldState, step->action, step->newState);
			
		}
	
		count --;
				
		if (count % 100 == 0)
		{
			double difference = getWeightDifference(oldWeights);
			
			if (difference < treshold)
			{
				printf("PE: FINISHED Update Difference in Weight Vector after %d Episodes: %f\n", count, difference);
				break;
			}
			printf("PE: Update Difference in Weight Vector after %d Episode: %f\n", count, difference);
		}
	}
	
	CActionSet::iterator it = actions->begin();
	
	for (;it != actions->end(); it ++)
	{
		(*it)->getActionData()->setData(dataSet->getActionData(*it));
	}
	delete step;
	delete dataSet;
	delete nextDataSet;
}

/*
CLSTDOfflinePolicyEvaluation::CLSTDOfflinePolicyEvaluation(CStepHistory *stepHistory, CLSTDLambda *learner, CGradientVFunction *learnData,  std::list<CStateModifier *> *l_modifiers) : COfflinePolicyEvaluation(stepHistory, learner, learnData, l_modifiers, 1)
{
	lstdLearner = learner;
}

CLSTDOfflinePolicyEvaluation::~CLSTDOfflinePolicyEvaluation()
{
}

void CLSTDOfflinePolicyEvaluation::resetLearnData()
{
	COfflinePolicyEvaluation::resetLearnData();
	if (resetData)
	{
		lstdLearner->resetData();
	}
}

double CLSTDOfflinePolicyEvaluation::getWeightDifference(double *oldWeights)
{
	lstdLearner->doOptimization();
	return COfflinePolicyEvaluation::getWeightDifference(oldWeights);
}
*/

 
CLSTDOfflineEpisodePolicyEvaluation::CLSTDOfflineEpisodePolicyEvaluation(CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger, CLSTDLambda *learner, CGradientVFunction *learnData,  std::list<CStateModifier *> *l_modifiers, int episodes) : COfflineEpisodePolicyEvaluation(episodeHistory, rewardLogger, learner, learnData, l_modifiers, episodes)
{
	lstdLearner = learner;
}

CLSTDOfflineEpisodePolicyEvaluation::~CLSTDOfflineEpisodePolicyEvaluation()
{
}

void CLSTDOfflineEpisodePolicyEvaluation::resetLearnData()
{
	COfflineEpisodePolicyEvaluation::resetLearnData();
	if (resetData)
	{
		lstdLearner->resetData();
	}
}

double CLSTDOfflineEpisodePolicyEvaluation::getWeightDifference(double *oldWeights)
{
	lstdLearner->doOptimization();
	return COfflineEpisodePolicyEvaluation::getWeightDifference(oldWeights);
}

CDataCollector::CDataCollector()
{
}

CDataCollector::~CDataCollector()
{
}
		
CDataCollectorFromAgentLogger::CDataCollectorFromAgentLogger(CAgent *l_agent, CAgentLogger *l_logger, CRewardLogger *l_rewardLogger, int numEpisodes, int numSteps)
{
	logger = l_logger;
	agent = l_agent;
	sender = agent;	

	rewardLogger = l_rewardLogger;
	
	addParameter("DataCollectorNumEpisodes", numEpisodes);
	addParameter("DataCollectorNumCollections", 1);
	
	addParameter("DataCollectorNumSteps", numSteps);
	addParameter("DataCollectorResetEpisodes", 0.0);

	numCollections = 0;

	unknownDataQFunctions = new std::list<CUnknownDataQFunction *>;
	
}


CDataCollectorFromAgentLogger::~CDataCollectorFromAgentLogger()
{
	delete unknownDataQFunctions;
}

void CDataCollectorFromAgentLogger::setController(CAgentController *l_controller)
{
	controller = l_controller;
}

void CDataCollectorFromAgentLogger::addUnknownDataFunction(CUnknownDataQFunction *unknownDataQFunction)
{
	unknownDataQFunctions->push_back(unknownDataQFunction);
}
	
void CDataCollectorFromAgentLogger::collectData()
{
	bool bReset = getParameter("DataCollectorResetEpisodes") > 0.5;
	int numEpisodes = (int) getParameter("DataCollectorNumEpisodes");
	int numSteps = (int) getParameter("DataCollectorNumSteps");
	int numColl = (int) getParameter("DataCollectorNumCollections");
	
	
	numCollections ++;

	if (numCollections % numColl == 0)
	{
		if (bReset)
		{
			logger->resetData();
			rewardLogger->resetData();
		}
		
		sender->addSemiMDPListener(logger);
		sender->addSemiMDPListener(rewardLogger);

		CAgentController *tempController = sender->getController();

		if (controller)
		{
			sender->setController( controller);
		}

		for (int i = 0; i < numEpisodes; i++)
		{
			agent->startNewEpisode();
			agent->doControllerEpisode(1, numSteps);
		}
		agent->startNewEpisode();
		sender->removeSemiMDPListener(logger);
		sender->removeSemiMDPListener(rewardLogger);

		sender->setController( tempController);

		std::list<CUnknownDataQFunction *>::iterator it = unknownDataQFunctions->begin();

		for (; it != unknownDataQFunctions->end(); it ++)
		{
			(*it)->recalculateTrees();
		}
	}
	
}

void CDataCollectorFromAgentLogger::setSemiMDPSender(CSemiMarkovDecisionProcess *l_sender)
{
	sender = l_sender;	
}

CPolicyIteration::CPolicyIteration(CLearnDataObject *l_policyFunction, CLearnDataObject *l_evaluationFunction, CPolicyEvaluation *l_evaluation, CDataCollector *l_collector)
{
	policyFunction = l_policyFunction;
	evaluationFunction = l_evaluationFunction;

	
	evaluation = l_evaluation;
	collector = l_collector;
	
	addParameters(evaluation);
	addParameters(collector);
	
	addParameter("PolicyIterationInitSteps", 0.0);
}

CPolicyIteration::~CPolicyIteration()
{
}

void CPolicyIteration::doPolicyIterationStep()
{
	evaluation->evaluatePolicy();
	
	evaluationFunction->copy(policyFunction);
		
	if (collector)
	{
		collector->collectData();
	}
	

}

void CPolicyIteration::initPolicyIteration()
{
	
	evaluation->resetLearnData();
	int numInitSteps = (int) getParameter("PolicyIterationInitSteps");
	
	printf("Initializing Policy Iteration\n");
	evaluation->evaluatePolicy(numInitSteps);
}


CPolicyIterationNewFeatures::CPolicyIterationNewFeatures(CLearnDataObject *policyFunction, CLearnDataObject *evaluationFunction, CPolicyEvaluation *evaluation, CNewFeatureCalculatorDataGenerator *l_newFeatureCalculator, CDataCollector *collector) : CPolicyIteration(policyFunction, evaluationFunction, evaluation, collector)
{
	newFeatureCalculator = l_newFeatureCalculator;
}

CPolicyIterationNewFeatures::~CPolicyIterationNewFeatures()
{
}

void CPolicyIterationNewFeatures::doPolicyIterationStep()
{
	newFeatureCalculator->calculateNewFeatures();

	CPolicyIteration::doPolicyIterationStep();
}

void CPolicyIterationNewFeatures::initPolicyIteration()
{
	newFeatureCalculator->calculateNewFeatures();

	CPolicyIteration::doPolicyIterationStep();
}

/*
CValueGradientCalculator::CValueGradientCalculator(CEpisodeHistory *l_episodeHistory, CRewardHistory *l_rewardLogger, CResidualFunction *l_residual, CResidualGradientFunction *l_gradient)
{
	episodeHistory = l_episodeHistory;
	rewardLogger = l_rewardLogger;

	residual = l_residual;
	residualGradientFunction = l_gradient;

	rewardFunction = NULL;
	
	addParameters(residual);
	addParameters(residualGradientFunction);
}

CValueGradientCalculator::CValueGradientCalculator(CEpisodeHistory *l_episodeHistory, CRewardFunction *l_rewardFunction, CResidualFunction *l_residual, CResidualGradientFunction *l_gradient)
{
	episodeHistory = l_episodeHistory;
	rewardLogger = NULL;

	residual = l_residual;
	residualGradientFunction = l_gradient;

	rewardFunction = l_rewardFunction;

}

CValueGradientCalculator::~CValueGradientCalculator()
{
}

void CValueGradientCalculator::getGradient(CFeatureList *gradient)
{
	gradient->clear();
	
	CActionSet *actions = episodeHistory->getActions();
	CStep *step = new CStep(episodeHistory->getStateProperties(), episodeHistory->getStateModifiers(), episodeHistory->getActions());
	CActionDataSet *dataSet = new CActionDataSet(actions);
	CActionDataSet *nextDataSet = new CActionDataSet(actions);
			
	printf("Residual Gradient Calculation, Episodes %d\n", episodeHistory->getNumEpisodes());	

	double error = 0;
	int steps = 0;
	
	CFeatureList *oldGradient = new CFeatureList();
	CFeatureList *newGradient = new CFeatureList();
	CFeatureList *residualGradient = new CFeatureList();
	
	for (int j = 0; j < episodeHistory->getNumEpisodes(); j ++)
	{
		double dur1 = 0;
		double dur2 = 0;
		
//		timespec t1;
//		timespec t2;
//		timespec t3;

		
		CEpisode *episode = episodeHistory->getEpisode(j);
		
		CRewardEpisode *rewardEpisode = NULL;
		if (rewardLogger)
		{
			rewardEpisode = rewardLogger->getEpisode(j);
		}
		
		for (int i = 0; i < episode->getNumSteps(); i++)
		{
			steps ++;
			
//			clock_gettime(CLOCK_REALTIME, &t1);
			//		
			episode->getStep(i, step);
			//			
//			clock_gettime(CLOCK_REALTIME, &t2);
								
			CActionData *data = step->action->getActionData();
			
			if (policy != NULL)
			{
				CAction *nextAction;
				if (i < episode->getNumSteps() - 1)
				{
					nextAction = episode->getAction(i + 1, nextDataSet);
				}
				else
				{
					nextAction = NULL;
				}			
				policy->setAction(nextAction, nextDataSet->getActionData(nextAction));
			}
			
			if (data != NULL)
			{
				data->setData(step->actionData->getActionData(step->action));
			}
			double reward = 0;
			if (rewardLogger)
			{
				reward = rewardEpisode->getReward(i);
			}
			else
			{
				reward = rewardFunction->getReward(step->oldState, step->action, step->newState);
			}
			
			double oldV = getValue(step->oldState, step->action);
			
			double newV = 0;
			
			oldGradient->clear();
			newGradient->clear();
			residualGradient->clear();
			
			getValueGradient(step->oldState, step->action, oldGradient);
			
			CAction *nextAction;
			
			
			if (!step->newState->isResetState())
			{
				if (estimationPolicy)
				{
					nextAction = estimationPolicy->getNextAction(step->newState, nextDataSet);
					CActionData *data = nextAction->getActionData();
			
					if (data != NULL)
					{
						data->setData(nextDataSet->getActionData(nextAction));
					}	
				}
				
				newV = getValue(step->newState, nextAction);
				getValueGradient(step->newState, nextAction, newGradient);
			}
			
			double residualError = residual->getResidual(oldV, reward, step->action->getDuration(), newV);
			DebugPrint('b', "Residual Error: %f, %f %f %f", residualError, reward, oldV,  newV);
			residualGradientFunction->getResidualGradient(oldGradient, newGradient, step->action->getDuration(), residualGradient);
			
			gradient->add(residualGradient, residualError);						
		}
	
		//printf("Time for Episode %d: step %f, time for learners %f\n", j, dur1, dur2);
								
		if (j % 100 == 0)
		{
			printf("Residual Gradient Calculation... Episode %d\n", j);
		}
	}
	
	CActionSet::iterator it = actions->begin();
	
	for (;it != actions->end(); it ++)
	{
		(*it)->getActionData()->setData(dataSet->getActionData(*it));
	}
	delete step;
	delete dataSet;
	delete nextDataSet;
	
	delete oldGradient;
	delete newGradient;
	delete residualGradient;
}

double CValueGradientCalculator::getFunctionValue()
{
	CActionSet *actions = episodeHistory->getActions();
	CStep *step = new CStep(episodeHistory->getStateProperties(), episodeHistory->getStateModifiers(), episodeHistory->getActions());
		
	CActionDataSet *dataSet = new CActionDataSet(actions);
	CActionDataSet *nextDataSet = new CActionDataSet(actions);
			
	printf("Residual Error Calculation, Episodes %d\n", episodeHistory->getNumEpisodes());	

	double error = 0;
	int steps = 0;
	for (int j = 0; j < episodeHistory->getNumEpisodes(); j ++)
	{
		double dur1 = 0;
		double dur2 = 0;
		
//		timespec t1;
//		timespec t2;
//		timespec t3;

		
		CEpisode *episode = episodeHistory->getEpisode(j);
		
		CRewardEpisode *rewardEpisode = NULL;
		if (rewardLogger)
		{
			rewardEpisode = rewardLogger->getEpisode(j);
		}
				
		for (int i = 0; i < episode->getNumSteps(); i++)
		{
			steps ++;
			
//			clock_gettime(CLOCK_REALTIME, &t1);
			//		
			episode->getStep(i, step);
			//			
//			clock_gettime(CLOCK_REALTIME, &t2);
								
			
			
			if (policy != NULL)
			{
				CAction *nextAction;
				if (i < episode->getNumSteps() - 1)
				{
					nextAction = episode->getAction(i + 1, nextDataSet);
				}
				else
				{
					nextAction = NULL;
				}			
				policy->setAction(nextAction, nextDataSet->getActionData(nextAction));
			}
			
			CActionData *data = step->action->getActionData();
			
			if (data != NULL)
			{
				data->setData(step->actionData->getActionData(step->action));
			}
			
			double reward = 0;
			if (rewardLogger)
			{
				reward = rewardEpisode->getReward(i);
			}
			else
			{
				reward = rewardFunction->getReward(step->oldState, step->action, step->newState);
			}
			
			double oldV = getValue(step->oldState, step->action);
			
			double newV = 0;
			
			CAction *nextAction = NULL;
			
			
			
			if (!step->newState->isResetState())
			{
				if (estimationPolicy)
				{
					nextAction = estimationPolicy->getNextAction(step->newState, nextDataSet);
				}
				else
				{
					if (i < episode->getNumSteps() - 1)
					{
						nextAction = episode->getAction(i + 1, nextDataSet);
					}
				}
			
				if (nextAction != NULL)
				{	
					CActionData *data = nextAction->getActionData();
			
					if (data != NULL)
					{
						data->setData(nextDataSet->getActionData(nextAction));
					}
				}
			
				newV = getValue(step->newState, nextAction);
			}
			
			error += pow(residual->getResidual(oldV, reward, step->action->getDuration(), newV), 2.0);
						
		}
	
		//printf("Time for Episode %d: step %f, time for learners %f\n", j, dur1, dur2);		
		
	}
	
	CActionSet::iterator it = actions->begin();
	
	for (;it != actions->end(); it ++)
	{
		(*it)->getActionData()->setData(dataSet->getActionData(*it));
	}
	delete step;
	delete dataSet;
	delete nextDataSet;
	
	error = error / steps / 2.0;
	
	printf("Average Bellman Error: %f\n", error);
	
	return error;
}

void CValueGradientCalculator::setBatchPolicy(CBatchLearningPolicy *l_policy)
{
	policy = l_policy;
}

void CValueGradientCalculator::setEstimationPolicy(CAgentController *l_estimationPolicy)
{
	estimationPolicy = l_estimationPolicy;
}

CVResidualGradientCalculator::CVResidualGradientCalculator(CGradientVFunction *l_vFunction, CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger, CResidualFunction *residual, CResidualGradientFunction *gradient) : CValueGradientCalculator(episodeHistory, rewardLogger, residual, gradient)
{
	vFunction = l_vFunction;
}
		
CVResidualGradientCalculator::CVResidualGradientCalculator(CGradientVFunction *l_vFunction, CEpisodeHistory *episodeHistory, CRewardFunction *rewardFunction, CResidualFunction *residual, CResidualGradientFunction *gradient) : CValueGradientCalculator(episodeHistory, rewardFunction, residual, gradient)
{
	vFunction = l_vFunction;
}

double CVResidualGradientCalculator::getValue(CStateCollection *state, CAction *action)
{
	return vFunction->getValue(state);
}
		

void CVResidualGradientCalculator::getValueGradient(CStateCollection *state, CAction *action, CFeatureList *gradient)
{
	vFunction->getGradient(state, gradient);
}

CVResidualGradientCalculator::~CVResidualGradientCalculator()
{
}
	

CQResidualGradientCalculator::CQResidualGradientCalculator(CGradientQFunction *l_qFunction, CAgentController *l_estimationPolicy, CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger, CResidualFunction *residual, CResidualGradientFunction *gradient) : CValueGradientCalculator(episodeHistory, rewardLogger, residual, gradient)
{
	qFunction = l_qFunction;
	estimationPolicy = l_estimationPolicy;
}

CQResidualGradientCalculator::CQResidualGradientCalculator(CGradientQFunction *l_qFunction, CAgentController *l_estimationPolicy, CEpisodeHistory *episodeHistory, CRewardFunction *rewardFunction, CResidualFunction *residual, CResidualGradientFunction *gradient) : CValueGradientCalculator(episodeHistory, rewardFunction, residual, gradient)
{
	qFunction = l_qFunction;
	estimationPolicy = l_estimationPolicy;
}

CQResidualGradientCalculator::~CQResidualGradientCalculator()
{
}

double CQResidualGradientCalculator::getValue(CStateCollection *state, CAction *action)
{
	return qFunction->getValue(state, action);
}
		
void CQResidualGradientCalculator::getValueGradient(CStateCollection *state, CAction *action, CFeatureList *gradient)
{
	qFunction->getGradient(state, action, action->getActionData(), gradient);
}


CResidualGradientBatchLearner::CResidualGradientBatchLearner(CGradientLearner *l_gradientLearner, CGradientUpdateFunction *learnData, double treshold, int maxEvaluations ) : CPolicyEvaluationGradientFunction(learnData, treshold, maxEvaluations)
{
	gradientLearner = l_gradientLearner;
	addParameters(gradientLearner);
	
	resetData = false;
}
		 
CResidualGradientBatchLearner::~CResidualGradientBatchLearner()
{
}
		
void CResidualGradientBatchLearner::evaluatePolicy(int numEvaluations)
{
	gradientLearner->doOptimization((int) getParameter("PolicyEvaluationMaxEpisodes"));
}

void CResidualGradientBatchLearner::resetLearnData()
{
	CPolicyEvaluation::resetLearnData();
	
	gradientLearner->resetOptimization();
}
*/
void CBatchDataGenerator::generateInputData(CEpisodeHistory *episodeHistory)
{
	CActionSet *actions = episodeHistory->getActions();
	CStep *step = new CStep(episodeHistory->getStateProperties(), episodeHistory->getStateModifiers(), episodeHistory->getActions());
		
	CActionDataSet *dataSet = new CActionDataSet(actions);

	for (int j = 0; j < episodeHistory->getNumEpisodes(); j ++)
	{
		CEpisode *episode = episodeHistory->getEpisode(j);

		for (int i = 0; i < episode->getNumSteps(); i++)
		{
			episode->getStep(i, step);
			
			addInput(step->oldState, step->action, 0.0);
		}
	}
	
	CActionSet::iterator it = actions->begin();
	
	for (;it != actions->end(); it ++)
	{
		(*it)->getActionData()->setData(dataSet->getActionData(*it));
	}
	delete step;
	delete dataSet;
}


CBatchVDataGenerator::CBatchVDataGenerator(CSupervisedLearner *l_learner, int numDim)
{
	
	learner = l_learner;
	vFunction = NULL;

	addParameters(learner);

	weightedLearner = NULL;

	init(numDim);
}
	
CBatchVDataGenerator::CBatchVDataGenerator(CAbstractVFunction *l_vFunction, CSupervisedLearner *l_learner)
{
	vFunction = l_vFunction;
	learner = l_learner;
	weightedLearner = NULL;
	
	int numDim = vFunction->getStateProperties()->getNumContinuousStates() +vFunction->getStateProperties()->getNumDiscreteStates() ;

	addParameters(learner);

	init(numDim);
}

CBatchVDataGenerator::CBatchVDataGenerator(CAbstractVFunction *l_vFunction, CSupervisedWeightedLearner *l_learner)
{
	vFunction = l_vFunction;
	learner = NULL;
	weightedLearner = l_learner;

	int numDim = vFunction->getStateProperties()->getNumContinuousStates() +vFunction->getStateProperties()->getNumDiscreteStates() ;


	addParameters(weightedLearner);

	init(numDim);
}

void CBatchVDataGenerator::init(int numDim)
{
	inputData = new CDataSet(numDim);
	outputData = new CDataSet1D();
	weightingData = new CDataSet1D();

	buffVector = new ColumnVector(numDim);
}


CBatchVDataGenerator::~CBatchVDataGenerator()
{
	delete inputData;
	delete outputData;
	delete weightingData;

	delete buffVector;
}
	
void CBatchVDataGenerator::addInput(CStateCollection *stateCol, CAction *, double output, double weighting)
{
	if (weighting > 0.001)
	{
		CStateProperties *properties = vFunction->getStateProperties();
		CState *state = stateCol->getState(properties);
		
		int offset = 0;
		
		for (unsigned int i = 0; i < properties->getNumDiscreteStates(); i ++)
		{
			buffVector->element(i + offset) = state->getDiscreteState( i);
		}
	
		offset += properties->getNumDiscreteStates();
		for (unsigned int i = 0; i < properties->getNumContinuousStates(); i ++)
		{
			buffVector->element(i + offset) = state->getContinuousState( i);
		}
	
		offset += properties->getNumContinuousStates();
	
		inputData->addInput(buffVector);
		outputData->push_back(output);
		weightingData->push_back(weighting);

	}
	else
	{
	}
}

void CBatchVDataGenerator::trainFA()
{
	if (learner)
	{
		learner->learnFA(inputData, outputData);
	}
	else
	{
		weightedLearner->learnWeightedFA(inputData, outputData, weightingData);
	}
}

void CBatchVDataGenerator::resetPolicyEvaluation()
{
	inputData->clear();
	outputData->clear();
	weightingData->clear();
	if (learner)
	{
		learner->resetLearner();
	}
	if (weightedLearner)
	{
		weightedLearner->resetLearner();
	}
}

double CBatchVDataGenerator::getValue(CStateCollection *state, CAction *)
{
	return vFunction->getValue(state);
}


CDataSet *CBatchVDataGenerator::getInputData()
{
	return inputData;
}

CDataSet1D *CBatchVDataGenerator::getOutputData()
{
	return outputData;
}

CDataSet1D *CBatchVDataGenerator::getWeighting()
{
	return weightingData;
}

CBatchCAQDataGenerator::CBatchCAQDataGenerator(CStateProperties *l_properties, CContinuousActionQFunction *l_qFunction, CSupervisedLearner *learner) : CBatchVDataGenerator(learner, properties->getNumContinuousStates() + l_qFunction->getContinuousActionObject()->getNumDimensions() + properties->getNumDiscreteStates())
{
	qFunction = l_qFunction;
	properties = l_properties;
	
}

CBatchCAQDataGenerator::~CBatchCAQDataGenerator()
{
	
}
	
void CBatchCAQDataGenerator::addInput(CStateCollection *stateCol, CAction *action, double output)
{
	CContinuousActionData *data = dynamic_cast<CContinuousAction *>(action)->getContinuousActionData();	

	int offset = 0;
	CState *state = stateCol->getState( properties);
	for (unsigned int i = 0; i < properties->getNumDiscreteStates(); i ++)
	{
		buffVector->element(i + offset) = state->getDiscreteState( i);
	}

	offset += properties->getNumDiscreteStates();
	for (unsigned int i = 0; i < properties->getNumContinuousStates(); i ++)
	{
		buffVector->element(i + offset) = state->getContinuousState( i);
	}

	offset += properties->getNumContinuousStates();

	for (int i = 0; i < data->nrows(); i ++)
	{
		buffVector->element(i + offset) = data->element(i);
	}		

	inputData->addInput(buffVector);
	outputData->push_back(output);
}
	
double CBatchCAQDataGenerator::getValue(CStateCollection *state, CAction *action)
{
	return qFunction->getValue(state, action);
}

CBatchQDataGenerator::CBatchQDataGenerator(CQFunction *l_qFunction, CSupervisedQFunctionLearner *l_learner, CStateProperties *inputState)
{
	init(l_qFunction, NULL, inputState);

	qFunction = l_qFunction;
	learner = l_learner;

	if (learner)
	{
		addParameters(learner);
	}
	weightedLearner = NULL;
}

CBatchQDataGenerator::CBatchQDataGenerator(CQFunction *l_qFunction, CSupervisedQFunctionWeightedLearner *l_learner, CStateProperties *inputState)
{
	init(l_qFunction, NULL, inputState);

	qFunction = l_qFunction;
	weightedLearner = l_learner;

	if (weightedLearner)
	{
		addParameters(weightedLearner);
	}
	learner = NULL;
}

CBatchQDataGenerator::CBatchQDataGenerator(CActionSet *l_actions, CStateProperties *l_properties)
{
	init(NULL, l_actions, l_properties);

	learner = NULL;
	weightedLearner = NULL;
	qFunction = NULL;
}

void CBatchQDataGenerator::init(CQFunction *l_qFunction, CActionSet *l_actions, CStateProperties *l_properties)
{
	inputMap = new std::map<CAction *, CDataSet *>;
	outputMap = new std::map<CAction *, CDataSet1D *>;

	weightedMap = new std::map<CAction *, CDataSet1D *>;
	buffVectorMap = new std::map<CAction *, ColumnVector *>;
	
	properties = l_properties;

	actions = l_actions;
	qFunction = l_qFunction;

	if (actions == NULL)
	{
		actions = l_qFunction->getActions();
	}
	
	CActionSet::iterator it = actions->begin();

	for (; it != actions->end(); it ++)
	{
		int dim = 0;
		if (properties == NULL)
		{		
			CStateProperties *properties = qFunction->getVFunction(*it)->getStateProperties();
			dim = properties->getNumContinuousStates() + properties->getNumDiscreteStates();
		}
		else
		{
			dim = properties->getNumContinuousStates() + properties->getNumDiscreteStates();
		}
		(*inputMap)[(*it)] = new CDataSet(dim);
		(*outputMap)[(*it)] = new CDataSet1D();
		(*weightedMap)[*it] = new CDataSet1D();
		(*buffVectorMap)[(*it)] = new ColumnVector(dim);
	}

}

CBatchQDataGenerator::~CBatchQDataGenerator()
{
	CActionSet::iterator it = actions->begin();

	for (; it != actions->end(); it ++)
	{
		delete (*inputMap)[(*it)];
		delete (*outputMap)[(*it)];
		delete (*weightedMap)[*it];
		delete (*buffVectorMap)[(*it)];

	}

	delete inputMap;
	delete outputMap;
	delete weightedMap;
	delete buffVectorMap;
}


	
void CBatchQDataGenerator::addInput(CStateCollection *stateCol, CAction *action, double output, double weighting)
{
	if (weighting > 0.001)
	{
		
		CStateProperties *inputState = properties;
		if (properties == NULL)
		{
			CAbstractVFunction *vFunction = qFunction->getVFunction(action);
			inputState = vFunction->getStateProperties();
		}
		
		CState *state = stateCol->getState(inputState);
	
		int offset = 0;
	
		ColumnVector *buffVector = (*buffVectorMap)[action];
		
		for (unsigned int i = 0; i < inputState->getNumDiscreteStates(); i ++)
		{
			buffVector->element(i + offset) = state->getDiscreteState( i);
		}
	
		offset += inputState->getNumDiscreteStates();
		for (unsigned int i = 0; i < inputState->getNumContinuousStates(); i ++)
		{
			buffVector->element(i + offset) = state->getContinuousState( i);
		}
	
		offset += inputState->getNumContinuousStates();
			
		CDataSet *inputData = (*inputMap)[action];
		CDataSet1D *outputData = (*outputMap)[action];
		CDataSet1D *weightData = (*weightedMap)[action];
	
		inputData->addInput(buffVector);
		outputData->push_back(output);
		weightData->push_back(weighting);

		//printf("Added Input to action ");
	}
}

void CBatchQDataGenerator::trainFA()
{
	CActionSet::iterator it = actions->begin();
	
	for (int i = 0; it != actions->end(); it ++, i++)
	{
		printf("Action %d : %d Examples...", i, (*inputMap)[*it]->size());
		if (learner)
		{
			learner->learnQFunction(*it, (*inputMap)[*it],  (*outputMap)[*it]);
		}
		if (weightedLearner)
		{
			weightedLearner->learnQFunction(*it, (*inputMap)[*it],  (*outputMap)[*it], (*weightedMap)[*it]);
			
		}

	}
}

CStateProperties *CBatchQDataGenerator::getStateProperties(CAction *action)
{
	if (properties == NULL)
	{
		CAbstractVFunction *vFunction = qFunction->getVFunction(action);
		properties = vFunction->getStateProperties();
		return properties;
	}
	else
	{
		return properties;
	}
}

void CBatchQDataGenerator::resetPolicyEvaluation()
{
// 
	CActionSet::iterator it = actions->begin();
		
	for (; it != actions->end(); it ++)
	{
		(*inputMap)[(*it)]->clear();
		(*outputMap)[(*it)]->clear();
		(*weightedMap)[(*it)]->clear();
	}
	if (learner)
	{
		learner->resetLearner();
	}
	if (weightedLearner)
	{
		weightedLearner->resetLearner();
	}
}

double CBatchQDataGenerator::getValue(CStateCollection *state, CAction *action)
{
	return qFunction->getValue(state, action);
}

CDataSet *CBatchQDataGenerator::getInputData(CAction *action)
{
	return (*inputMap)[action];
}

CDataSet1D *CBatchQDataGenerator::getOutputData(CAction *action)
{
	return (*outputMap)[action];
}


CFittedIteration::CFittedIteration(CEpisodeHistory *l_episodeHistory, CRewardHistory *l_rewardLogger, CBatchDataGenerator *l_generator) : CPolicyEvaluation()
{
	rewardLogger = l_rewardLogger;
	episodeHistory = l_episodeHistory;

	estimationPolicy = NULL;

	addParameter("DiscountFactor", 0.95);
	addParameter("ResidualNearestNeighborDistance", 0.0);
	addParameter("UseResidualAlgorithm", 0.0);

		
	dataCollector = NULL;
	dataGenerator = l_generator;

	addParameters(dataGenerator);

	useResidualAlgorithm = 0;
	initialPolicyEvaluation = NULL;

	actorLearner = NULL;
}

CFittedIteration::~CFittedIteration()
{
}

void CFittedIteration::setDataCollector(CDataCollector *l_dataCollector)
{
	dataCollector = l_dataCollector;
	addParameters(dataCollector);
}

void CFittedIteration::setActorLearner(CPolicyEvaluation *l_actorLearner)
{
	actorLearner = l_actorLearner;
}

void CFittedIteration::evaluatePolicy()
{
	CPolicyEvaluation::evaluatePolicy();

	if (actorLearner)
	{
		actorLearner->evaluatePolicy();
	}
}

void CFittedIteration::evaluatePolicy(int trials)
{
	for (int i = 0; i < trials; i ++)
	{
		doEvaluationTrial();
	}
}

void CFittedIteration::addResidualInput(CStep *, CAction *, double , double , double , CAction *, double )
{
}

CBatchDataGenerator *CFittedIteration::createTrainingsData()
{
	CActionSet *actions = episodeHistory->getActions();
	CStep *step = new CStep(episodeHistory->getStateProperties(), episodeHistory->getStateModifiers(), episodeHistory->getActions());

	printf("FittetIteration: Episode has %d State Modifiers\n", episodeHistory->getStateModifiers()->size());
		
	CActionDataSet *dataSet = new CActionDataSet(actions);
	CActionDataSet *nextDataSet = new CActionDataSet(actions);
			
	printf("Tree Regression Value Calculation, Episodes %d\n", episodeHistory->getNumEpisodes());	

	int steps = 0;
	
	double discountFactor = getParameter("DiscountFactor");
	double nearestNeighborDistance = getParameter("ResidualNearestNeighborDistance");

	dataGenerator->resetPolicyEvaluation();
	
	for (int j = 0; j < episodeHistory->getNumEpisodes(); j ++)
	{
		CEpisode *episode = episodeHistory->getEpisode(j);
		
		CRewardEpisode *rewardEpisode = NULL;
		if (rewardLogger)
		{
			rewardEpisode = rewardLogger->getEpisode(j);
		}
				
		for (int i = 0; i < episode->getNumSteps(); i++)
		{
			steps ++;
			
			episode->getStep(i, step);
			
			double reward = 0;
			reward = rewardEpisode->getReward(i);
									
			double newV = 0;
			
			CAction *nextAction = NULL;

			CAction *nextEpisodeAction = NULL;
			double nextReward = 0.0;
			
			if (!step->newState->isResetState())
			{

				if (i < episode->getNumSteps() - 1)
				{
					nextEpisodeAction = episode->getAction(i + 1, nextDataSet);
					nextReward = rewardEpisode->getReward(i +1);
				}
				if (estimationPolicy)
				{
					nextAction = estimationPolicy->getNextAction(step->newState, nextDataSet);
				}
				else
				{
					nextAction = nextEpisodeAction;
				}
			
				if (nextAction != NULL)
				{	
					CActionData *data = nextAction->getActionData();
			
					if (data != NULL)
					{
						data->setData(nextDataSet->getActionData(nextAction));
					}
				}
			
				newV = getValue(step->newState, nextAction);
			}
			double V = reward + discountFactor * newV;
			if (nextAction == NULL)
			{
				addResidualInput(step, nextAction, V, newV, nearestNeighborDistance, nextEpisodeAction, reward);
			}
			else
			{
				addResidualInput(step, nextAction, V, newV, nearestNeighborDistance, nextEpisodeAction, nextReward);
			
			}
			
			double weighting = getWeighting(step->oldState, step->action);
			
			dataGenerator->addInput(step->oldState, step->action, V, weighting);
			
		}
	}
	printf("Finished Creating Training-set\n");


	CActionSet::iterator it = actions->begin();
	
	for (;it != actions->end(); it ++)
	{
		(*it)->getActionData()->setData(dataSet->getActionData(*it));
	}
	delete step;
	delete dataSet;
	delete nextDataSet;

	return dataGenerator;
}

void CFittedIteration::onParametersChanged()
{
	CPolicyEvaluation::onParametersChanged();

	useResidualAlgorithm = (int) getParameter("UseResidualAlgorithm");
}

double CFittedIteration::getWeighting(CStateCollection *, CAction *)
{
	return 1.0;
}

double CFittedIteration::getValue(CStateCollection *state, CAction *action)
{
	return dataGenerator->getValue(state, action);
}

void CFittedIteration::doEvaluationTrial()
{
	createTrainingsData();

	dataGenerator->trainFA();

	if (dataCollector != NULL)
	{
		dataCollector->collectData();
	}
}

void CFittedIteration::setInitialPolicyEvaluation(CPolicyEvaluation *l_initialPolicyEvaluation)
{
	initialPolicyEvaluation = l_initialPolicyEvaluation;	
}

void CFittedIteration::resetLearnData()
{
	if (initialPolicyEvaluation)
	{
		initialPolicyEvaluation->evaluatePolicy();
	}

	if (actorLearner)
	{
		actorLearner->resetLearnData();
	}
}


CFittedCAQIteration::CFittedCAQIteration(CContinuousActionQFunction *l_qFunction, CStateProperties *l_properties, CAgentController *l_estimationPolicy, CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger, CSupervisedLearner *l_learner) : CFittedIteration(episodeHistory, rewardLogger, new CBatchCAQDataGenerator(l_properties, l_qFunction, l_learner))
{
	estimationPolicy = l_estimationPolicy;
}

CFittedCAQIteration::~CFittedCAQIteration()
{
	delete dataGenerator;
}

CFittedVIteration::CFittedVIteration(CAbstractVFunction *l_vFunction,  CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger, CSupervisedLearner *l_learner) : CFittedIteration(episodeHistory, rewardLogger, new CBatchVDataGenerator(l_vFunction, l_learner))
{
	estimationPolicy = NULL;

	actionProbabilities = NULL;
	availableActions = NULL;
}

CFittedVIteration::CFittedVIteration(CAbstractVFunction *l_vFunction,  CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger, CSupervisedWeightedLearner *l_learner, CStochasticPolicy *l_estimationPolicy) : CFittedIteration(episodeHistory, rewardLogger, new CBatchVDataGenerator(l_vFunction, l_learner))
{
	estimationPolicy = l_estimationPolicy;

	actionProbabilities = new double[estimationPolicy->getActions()->size()];
	availableActions = new CActionSet();
}

CFittedVIteration::~CFittedVIteration()
{
	delete dataGenerator;

	if (actionProbabilities)
	{
		delete [] actionProbabilities;
		delete availableActions;
	}
}

double CFittedVIteration::getWeighting(CStateCollection *state, CAction *action)
{
	if (estimationPolicy == NULL)
	{
		return 1.0;
	}
	else
	{
		availableActions->clear();
		
	 	
		estimationPolicy->getActions()->getAvailableActions(availableActions, state);

		estimationPolicy->getActionProbabilities(state, availableActions, actionProbabilities);
		
		int index = availableActions->getIndex(action);
		assert(index >= 0);
		
		return actionProbabilities[index];
	}
}

CFittedQIteration::CFittedQIteration(CQFunction *l_qFunction, CAgentController *l_estimationPolicy, CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger, CSupervisedQFunctionLearner *l_learner, CStateProperties *l_residualProperties) : CFittedIteration(episodeHistory, rewardLogger, new CBatchQDataGenerator(l_qFunction, l_learner))
{
	inputDatas = new std::map<CAction *, CDataSet *>;
	outputDatas = new std::map<CAction *, CDataSet1D *>;
	kdTrees = new std::map<CAction *, CKDTree *>;
	nearestNeighbors = new std::map<CAction *, CKNearestNeighbors *>;
	dataPreProc = new std::map<CAction *, CDataPreprocessor *>;
	neighborsList = new std::list<int>;

	estimationPolicy = l_estimationPolicy;
	residualProperties = l_residualProperties;

	CStateProperties *l_prop = residualProperties; 

	if (l_prop == NULL)
	{	
		l_prop = l_qFunction->getVFunction(0)->getStateProperties();
	}

	buffState = new CState(l_prop);

	kNN = 1;
}

CFittedQIteration::CFittedQIteration(CQFunction *l_qFunction, CStateProperties *inputState, CAgentController *l_estimationPolicy, CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger, CSupervisedQFunctionLearner *l_learner, CStateProperties *l_residualProperties) : CFittedIteration(episodeHistory, rewardLogger, new CBatchQDataGenerator(l_qFunction, l_learner, inputState))
{
	inputDatas = new std::map<CAction *, CDataSet *>;
	outputDatas = new std::map<CAction *, CDataSet1D *>;

	kdTrees = new std::map<CAction *, CKDTree *>;
	nearestNeighbors = new std::map<CAction *, CKNearestNeighbors *>;
	dataPreProc = new std::map<CAction *, CDataPreprocessor *>;
	neighborsList = new std::list<int>;
	residualProperties = l_residualProperties;

	estimationPolicy = l_estimationPolicy;

	CStateProperties *l_prop = residualProperties; 

	if (l_prop == NULL)
	{
		l_prop = l_qFunction->getVFunction(0)->getStateProperties();
	}

	buffState = new CState(l_prop);

	kNN = 1;
}


CFittedQIteration::~CFittedQIteration()
{
	delete dataGenerator;
	CActionSet *actions = episodeHistory->getActions();

	CActionSet::iterator it = actions->begin();

	for (; it != actions->end(); it ++)
	{
		if ((*inputDatas)[*it] != NULL)
		{
			delete (*inputDatas)[*it];
			delete (*outputDatas)[*it];
			
			delete (*kdTrees)[*it];
			delete (*nearestNeighbors)[*it];
			delete (*dataPreProc)[*it];
			
		}
	}
	delete inputDatas;
	delete outputDatas;
	delete kdTrees;
	delete nearestNeighbors;
	delete dataPreProc;
	
	delete neighborsList;

	delete buffState;
}


void CFittedQIteration::addResidualInput(CStep *step, CAction *nextAction, double , double newV, double nearestNeighborDistance, CAction *, double  )
{

	if (useResidualAlgorithm > 0 && nextAction != NULL && (*nearestNeighbors)[nextAction] != NULL)
	{
		CStateProperties *l_prop = residualProperties; 

		if (l_prop == NULL)
		{
			CBatchQDataGenerator *qGenerator = dynamic_cast<CBatchQDataGenerator *>(dataGenerator);
			
			l_prop = qGenerator->getStateProperties(nextAction);
		}
		
		CState *state = step->newState->getState(l_prop);
		
		(*dataPreProc)[nextAction]->preprocessInput(state, buffState);	

		neighborsList->clear();
		(*nearestNeighbors)[nextAction]->getNearestNeighbors(state, neighborsList);

						
		ColumnVector *nearestNeighbor = (*(*inputDatas)[nextAction])[*(neighborsList->begin())];
						
		double distance = buffState->getDistance(nearestNeighbor);	

		if (distance > nearestNeighborDistance && (useResidualAlgorithm  == 1 || useResidualAlgorithm  == 3))
		{
			dataGenerator->addInput( step->newState, nextAction, newV);
		}
		else
		{
			if (nextAction == step->action && useResidualAlgorithm < 3)
			{
				CState *oldState = step->oldState->getState(l_prop);

				(*dataPreProc)[nextAction]->preprocessInput(oldState, buffState);
				double distanceOldState = buffState->getDistance(nearestNeighbor);

				if (distanceOldState < distance + nearestNeighborDistance / 3)
				{
					dataGenerator->addInput( step->newState, nextAction, newV);
				}
			}
		}
	}
}

void CFittedQIteration::doEvaluationTrial()
{
	CActionSet *actions = episodeHistory->getActions();

	CActionSet::iterator it = actions->begin();

	for (; it != actions->end(); it ++)
	{
		if ((*inputDatas)[*it] != NULL)
		{
			delete (*inputDatas)[*it];
			delete (*outputDatas)[*it];
			delete (*kdTrees)[*it];
			delete (*nearestNeighbors)[*it];
			delete (*dataPreProc)[*it];
			(*inputDatas)[*it] = NULL;
		}
		CDataSet *dataSet = dynamic_cast<CBatchQDataGenerator *>(dataGenerator)->getInputData(*it);
		CDataSet1D *outputSet = dynamic_cast<CBatchQDataGenerator *>(dataGenerator)->getOutputData(*it);

		if (dataSet != NULL && dataSet->size() > 0 )
		{
			(*inputDatas)[*it] = new CDataSet(*dataSet);
			(*outputDatas)[*it] = new CDataSet1D(*outputSet);
			
			(*dataPreProc)[*it] = new CMeanStdPreprocessor(dataSet);
			(*dataPreProc)[*it]->preprocessDataSet((*inputDatas)[*it]);

			(*kdTrees)[*it] = new CKDTree((*inputDatas)[*it], 1);
			(*kdTrees)[*it]->setPreprocessor((*dataPreProc)[*it]);
			(*nearestNeighbors)[*it] = new CKNearestNeighbors((*kdTrees)[*it], (*inputDatas)[*it], kNN);
		}
	}
	CFittedIteration::doEvaluationTrial();
}



CFittedQNewFeatureCalculator::CFittedQNewFeatureCalculator(CQFunction *l_qFunction, CQFunction *l_qFunctionPolicy, CStateProperties *inputState, CAgentController *estimationPolicy, CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger, CNewFeatureCalculator *l_newFeatCalc) : CFittedQIteration(l_qFunction, inputState, estimationPolicy, episodeHistory, rewardLogger, NULL)
{
	qFunction = l_qFunction;
	qFunctionPolicy = l_qFunctionPolicy;

	estimationCalculator = new std::map<CAction *, CFeatureCalculator *>;
	policyCalculator = new std::map<CAction *, CFeatureCalculator *>;

	newFeatureCalc = l_newFeatCalc;
	agent = NULL;	
}


	
CFittedQNewFeatureCalculator::~CFittedQNewFeatureCalculator()
{
	clearCalculators();

	delete estimationCalculator;
	delete policyCalculator;
}

void CFittedQNewFeatureCalculator::clearCalculators()
{
	
	CActionSet *actions = estimationPolicy->getActions();	

	CActionSet::iterator it = actions->begin();
	
	for (; it != actions->end(); it ++)
	{
		CFeatureVFunction *vFunction = dynamic_cast<CFeatureVFunction *>( qFunction->getVFunction(*it));
		vFunction->setFeatureCalculator(NULL);
		
		vFunction = dynamic_cast<CFeatureVFunction *>( qFunctionPolicy->getVFunction(*it));
		vFunction->setFeatureCalculator(NULL);
		
		if ((*estimationCalculator)[*it])
		{
			episodeHistory->removeStateModifier((*estimationCalculator)[*it]);
			
			if (agent)
			{
				agent->removeStateModifier((*estimationCalculator)[*it]);
			}
		
			delete (*estimationCalculator)[*it];
		}
		episodeHistory->removeStateModifier((*policyCalculator)[*it]);
			
		if (agent)
		{
			agent->removeStateModifier((*policyCalculator)[*it]);
		}


		delete (*policyCalculator)[*it];
	}
	estimationCalculator->clear();
	policyCalculator->clear();
}


void CFittedQNewFeatureCalculator::resetData()
{
	clearCalculators();	
}

void CFittedQNewFeatureCalculator::calculateNewFeatures()
{
	createTrainingsData();

	CActionSet *actions = estimationPolicy->getActions();	

	CActionSet::iterator it = actions->begin();
	
	CBatchQDataGenerator *qDataGenerator = dynamic_cast<CBatchQDataGenerator *>(dataGenerator);

	for (; it != actions->end(); it ++)
	{
		CDataSet *dataSet = qDataGenerator->getInputData(*it);

		CDataSet1D *outputDataSet = qDataGenerator->getOutputData(*it);

		CFeatureVFunction *vFunction = dynamic_cast<CFeatureVFunction *>( qFunction->getVFunction(*it));

		(*estimationCalculator)[*it] = newFeatureCalc->getFeatureCalculator(vFunction, dataSet, outputDataSet);

		vFunction->setFeatureCalculator((*estimationCalculator)[*it]);		

		episodeHistory->addStateModifier( (*estimationCalculator)[*it]);
		
		if (agent)
		{
			agent->addStateModifier((*estimationCalculator)[*it]);
		}
	}
	printf("Episode has %d State Modifiers\n", episodeHistory->getStateModifiers()->size());
}

void CFittedQNewFeatureCalculator::swapValueFunctions()
{

	CActionSet *actions = estimationPolicy->getActions();	

	CActionSet::iterator it = actions->begin();
	
	for (; it != actions->end(); it ++)
	{
		printf("Swap V-Func: %d Modifier\n", episodeHistory->getStateModifiers()->size());
		if ((*policyCalculator)[*it] != NULL)
		{
			printf("Removing State Modifier\n");
			episodeHistory->removeStateModifier((*policyCalculator)[*it]);
			
			if (agent)
			{
				agent->removeStateModifier((*policyCalculator)[*it]);
			}
		}
		printf("Swap V-Func 2: %d Modifier\n", episodeHistory->getStateModifiers()->size());
		
		(*policyCalculator)[*it] = (*estimationCalculator)[*it];

		CFeatureVFunction *vFunction = dynamic_cast<CFeatureVFunction *>( qFunctionPolicy->getVFunction(*it));

		vFunction->setFeatureCalculator((*policyCalculator)[*it]);

		(*estimationCalculator)[*it] = NULL;
	}
}

void CFittedQNewFeatureCalculator::setStateModifiersObject(CStateModifiersObject *l_agent)
{
	agent = l_agent;
}

CContinuousDynamicProgramming::CContinuousDynamicProgramming(CActionSet *allActions, CSamplingBasedTransitionModel *l_transModel)
{
	transModel = l_transModel;
	
	availableActions = new CActionSet();
	actionValues = new double[allActions->size()];
	actionProbabilities = new double[allActions->size()];

	addParameter("DiscountFactor", 0.95);

	addParameters( transModel);
	
	numIteration = 0;
}

CContinuousDynamicProgramming::~CContinuousDynamicProgramming()
{
	delete availableActions;
	delete [] actionValues;
	delete [] actionProbabilities;
}

void CContinuousDynamicProgramming::evaluatePolicy(int numEvaluations)
{
	for (int k = 0; k < numEvaluations; k ++)
	{
		numIteration ++;
		double gamma = getParameter("DiscountFactor");
		
		printf("DynamicProgramming Evaluation %d (%d)\n", numIteration, k);
		resetDynamicProgramming();
		for (int i = 0; i < transModel->getNumStates(); i ++)
		{
			availableActions->clear();
	
			std::map<CAction *, CSampleTransition *> * transitions = transModel->getTransitions(i);
			
			std::map<CAction *, CSampleTransition *>::iterator it = transitions->begin();
	
			for (int j = 0; it != transitions->end(); it ++, j++)
			{
				availableActions->push_back((*it).first);
	
				CSampleTransition *sampleTrans = (*it).second;
	
				CState *nextState = sampleTrans->state;
	
				double value = sampleTrans->reward;
				
				if (!nextState->isResetState())
				{
					value += gamma  * getValue(nextState, sampleTrans->availableActions);	
				}
				actionValues[j] = value;
			}
			updateOutputs(i, availableActions, actionValues);

			if (i % 5000 == 0)
			{
				printf("Finished State %d\n", i);
			}
		}
		learn();
	}
}

double CContinuousDynamicProgramming::getValueFromDistribution(CActionSet *availableActions, double *actionValues, CActionDistribution *distribution)
{
	memcpy(actionProbabilities, actionValues, sizeof(double) * availableActions->size());

	distribution->getDistribution(NULL, availableActions, actionProbabilities);

	double sum = 0;
	for (unsigned int i = 0; i < availableActions->size(); i ++)
	{
		sum += actionValues[i] * actionProbabilities[i];
	}
	return sum;
}

void CContinuousDynamicProgramming::resetLearnData()
{
	numIteration = 0;
	resetDynamicProgramming();
}

CContinuousDynamicVProgramming::CContinuousDynamicVProgramming(CActionSet *allActions, CActionDistribution *l_distribution, CSamplingBasedTransitionModel *transModel, CAbstractVFunction *l_vFunction, CSupervisedLearner *l_learner) : CContinuousDynamicProgramming(allActions, transModel)
{
	distribution = l_distribution;

	vFunction = l_vFunction;
	learner = l_learner;

	outputValues = new CDataSet1D();

	addParameters( learner);
}

CContinuousDynamicVProgramming::~CContinuousDynamicVProgramming()
{
	delete outputValues;
}

double CContinuousDynamicVProgramming::getValue(CState *state, CActionSet *) 
{
	return vFunction->getValue(state);
}

void CContinuousDynamicVProgramming::updateOutputs(int , CActionSet *availableActions, double *actionValues)
{
	double value = getValueFromDistribution(availableActions, actionValues, distribution);

	outputValues->push_back(value);
}

void CContinuousDynamicVProgramming::learn()
{
	learner->learnFA(transModel->getStateList(), outputValues);
}

void CContinuousDynamicVProgramming::resetDynamicProgramming()
{
	outputValues->clear();
}


CContinuousDynamicQProgramming::CContinuousDynamicQProgramming(CActionSet *allActions, CActionDistribution *l_distribution, CSamplingBasedTransitionModel *transModel, CAbstractQFunction *l_qFunction, CSupervisedQFunctionLearner *l_learner) : CContinuousDynamicProgramming(allActions, transModel)
{
	qFunction = l_qFunction;
	learner = l_learner;

	actionValues2 = new double[allActions->size()];

	distribution = l_distribution;

	outputValues = new std::map<CAction *, CDataSet1D *>;

	inputValues = new std::map<CAction *, CDataSet *>;

	CActionSet::iterator it = allActions->begin();
	for (; it != allActions->end(); it ++)
	{
		(*outputValues)[*it] =  new CDataSet1D();
		(*inputValues)[*it] =  new CDataSet(transModel->getStateList()->getNumDimensions());
	}


	addParameters( learner);
}

CContinuousDynamicQProgramming::~CContinuousDynamicQProgramming()
{
	std::map<CAction *, CDataSet1D *>::iterator it = outputValues->begin();
	for (; it != outputValues->end(); it ++)
	{
		delete (*it).second;
		delete (*inputValues)[(*it).first];
	}

	delete outputValues;
	delete inputValues;

	delete [] actionValues2;
}

double CContinuousDynamicQProgramming::getValue(CState *state, CActionSet *availableActions) 
{
	qFunction->getActionValues(state, availableActions, actionValues2);
	return getValueFromDistribution( availableActions, actionValues2, distribution);
}

void CContinuousDynamicQProgramming::updateOutputs(int index, CActionSet *availableActions, double *actionValues)
{
	CActionSet::iterator it = availableActions->begin();

	for (int i = 0; it != availableActions->end(); it ++, i++)
	{
		(*outputValues)[*it]->push_back(actionValues[i]);
		(*inputValues)[*it]->addInput((*transModel->getStateList())[index]);
;	}
}

void CContinuousDynamicQProgramming::learn()
{
	std::map<CAction *, CDataSet1D *>::iterator it = outputValues->begin();
	for (int i = 0; it != outputValues->end(); it ++, i++)
	{
		printf("%d\n", i);
		learner->learnQFunction( (*it).first, (*inputValues)[(*it).first],		(*outputValues)[(*it).first]);
	}
}

void CContinuousDynamicQProgramming::resetDynamicProgramming()
{
	std::map<CAction *, CDataSet1D *>::iterator it = outputValues->begin();
	for (; it != outputValues->end(); it ++)
	{
		(*inputValues)[(*it).first]->clear();
		(*outputValues)[(*it).first]->clear();
	}
}

CContinuousMCQEvaluation::CContinuousMCQEvaluation(CActionSet *allActions, CActionDistribution *distribution, CSamplingBasedTransitionModel *transModel, CPolicySameStateEvaluator *l_evaluator, CSupervisedQFunctionLearner *learner) : CContinuousDynamicQProgramming(allActions, distribution, transModel, NULL, learner)
{
	evaluator = l_evaluator;
}

CContinuousMCQEvaluation::~CContinuousMCQEvaluation()
{

}

double CContinuousMCQEvaluation::getValue(CState *state, CActionSet *)
{
	return evaluator->getValueForState(state, numIteration + 1);
}



CGraphDynamicProgramming::CGraphDynamicProgramming(CSamplingBasedGraph *l_transModel)
{
	transModel = l_transModel;

	outputValues = new CDataSet1D();

	addParameter("InitNodeValue", -50);
	
	addParameters(transModel);

	resetGraph = true;
}

CGraphDynamicProgramming::~CGraphDynamicProgramming()
{
	delete outputValues;
}

void CGraphDynamicProgramming::evaluatePolicy(int numEvaluations)
{
	double init = getParameter("InitNodeValue");

	
	if (numEvaluations > 0)
	{
		while (outputValues->size() < transModel->getNumStates())
		{
			outputValues->push_back(init);
		}
	}

	for (int i = 0; i < numEvaluations; i ++)
	{
		printf("Beginning %d Iteration: Output Mean %f\n", i, outputValues->getMean(NULL));
		for (int j = 0; j < transModel->getNumStates(); j ++)
		{
//			double oldVal = (*outputValues)[j];
			double newVal = 0.0;

			getMaxTransition(j, newVal);


			(*outputValues)[j] = newVal; 
		}
		printf("Finished %d Iteration: Output Mean %f\n", i, outputValues->getMean(NULL));
	}
}

void CGraphDynamicProgramming::resetLearnData()
{
	outputValues->clear();
	double init = getParameter("InitNodeValue");

	printf("Init Graph Iteration\n");

	if (resetGraph)
	{
		CContinuousStateList *stateList = transModel->getStateList();
		stateList->resetData();
		printf("Creating Graph...(%d Nodes)\n", 	transModel->getStateList()->size());
		transModel->createTransitions();
		printf("Done\n");
		transModel->getStateList()->initNearestNeighborSearch();
		transModel->getStateList()->getKDTree()->createLeavesArray();
	}
	
	for (int i = 0; i < transModel->getNumStates(); i++)
	{
		outputValues->push_back(init);
	}

}

double CGraphDynamicProgramming::getValue(int node)
{
	return (*outputValues)[node];
}

double CGraphDynamicProgramming::getValue(ColumnVector *input)
{
	int node = 0;
	double distance = 0.0;

	transModel->getStateList()->getNearestNeighbor(input, node, distance);

	return (*outputValues)[node];
}

void CGraphDynamicProgramming::getNearestNode(ColumnVector *input, int &node, double &distance)
{
	transModel->getStateList()->getNearestNeighbor(input, node, distance);
}

CDataSet1D *CGraphDynamicProgramming::getOutputValues()
{
	return outputValues;
}

CGraphTransition *CGraphDynamicProgramming::getMaxTransition(int index, double &maxValue, CKDRectangle *range)
{

	std::list<CGraphTransition *> *transitions = transModel->getTransitions(index);
	std::list<CGraphTransition *>::iterator it = transitions->begin();

	maxValue = 0;
	CGraphTransition *edge = NULL;

	for (int k = 0; it != transitions->end(); it ++, k ++)
	{
		CContinuousStateList *stateList = transModel->getStateList();

		double distance = 0.0;

		bool newPoint = ((*it)->newStateIndex > 0) && (fabs((*stateList)[index]->element(2) - (*stateList)[(*it)->newStateIndex]->element(2)) > 0.001);

		if (range != NULL && newPoint)
		{
			//printf("TargetRange: %f\n", range->getMinValue(3));

			//cout << (*stateList)[(*it)->newStateIndex]->t();
		
			distance = range->getDistanceToPoint((*stateList)[(*it)->newStateIndex]);
			//printf("Distance: %f\n", distance);
		}	

		
		if (range == NULL || distance <= 0.00001)
		{
		
			double tempVal = (*it)->getReward();

		
			if ((*it)->newStateIndex >= 0)
			{
				tempVal += (*it)->discountFactor * (*outputValues)[(*it)->newStateIndex];
			}
	
			if (k == 0 || tempVal > maxValue)
			{
				maxValue = tempVal;
				edge = *it;
			}
		}
	}
	if (transitions->size() == 0)
	{
		maxValue = (*outputValues)[index];
	}
	return edge;
}

CSamplingBasedGraph *CGraphDynamicProgramming::getGraph()
{
	return transModel;
}

void CGraphDynamicProgramming::saveCSV(string filename, DataSubset *nodeSubset)
{
	string nodeFilename = filename + "_Nodes.data";
	string edgeFilename = filename + "_Edges.data";

	FILE *nodeFILE = fopen(nodeFilename.c_str(), "w");
	FILE *edgeFILE = fopen(edgeFilename.c_str(), "w");
	
	DataSubset::iterator it = nodeSubset->begin();
	CDataSet *dataSet = transModel->getStateList();

	std::map<int, int> indexMap;

	for (int j = 0;it != nodeSubset->end(); it ++, j++)
	{
		indexMap[*it] = j;
	}
	
	it = nodeSubset->begin();
	for (int j = 0;it != nodeSubset->end(); it ++, j++)
	{
		// Save node
		ColumnVector *node = (*dataSet)[*it];
		for (int i = 0; i < node->nrows(); i ++)
		{
			fprintf(nodeFILE, "%f ", node->element(i));
		}
		fprintf(nodeFILE, "%f\n", (*outputValues)[*it]);

		//Save edges

		std::list<CGraphTransition *> *transitions = transModel->getTransitions(*it);
		
		std::list<CGraphTransition *>::iterator itTrans = transitions->begin();

		for (; itTrans != transitions->end(); itTrans ++)
		{
			int newStateIndex = (*itTrans)->newStateIndex;
			if (newStateIndex >= 0)
			{
				newStateIndex = indexMap[newStateIndex];
			}
			if ((*itTrans)->action->isType(CONTINUOUSACTION))
			{
				CContinuousActionData *data = dynamic_cast<CContinuousActionData *>((*itTrans)->actionData);
				fprintf(edgeFILE, "%d %d %f %f %f\n", j, newStateIndex, (*itTrans)->getReward(), (*itTrans)->discountFactor, data->element(0));
			}
			else
			{
				fprintf(edgeFILE, "%d %d %f %f 0\n", j, newStateIndex, (*itTrans)->getReward(), (*itTrans)->discountFactor);
			}
		}
	}

	fclose(nodeFILE);
	fclose(edgeFILE);
}

CGraphAdaptiveTargetDynamicProgramming::CGraphAdaptiveTargetDynamicProgramming(CAdaptiveTargetGraph *graph) : CGraphDynamicProgramming(graph)
{
	adaptiveTargetGraph = graph;
	currentTarget = NULL;

	targetMap = new std::map<CGraphTarget *, CDataSet1D *>();
	targets = new std::list<CGraphTarget *>();

	delete outputValues;
	outputValues = NULL;
}

CGraphAdaptiveTargetDynamicProgramming::~CGraphAdaptiveTargetDynamicProgramming()
{
	delete targets;

	std::map<CGraphTarget *, CDataSet1D *>::iterator it = targetMap->begin();

	for (; it != targetMap->end(); it ++)
	{
		delete (*it).second;
	}
	delete targetMap;
}

CGraphTransition *CGraphAdaptiveTargetDynamicProgramming::getMaxTransition(int index, double &maxValue, CKDRectangle *range)
{
	std::list<CGraphTransition *> *transitions = transModel->getTransitions(index);
	std::list<CGraphTransition *>::iterator it = transitions->begin();

	maxValue = 0;
	CGraphTransition *edge = NULL;

	for (int k = 0; it != transitions->end(); it ++, k ++)
	{
		CContinuousStateList *stateList = transModel->getStateList();

		CGraphTransitionAdaptiveTarget *adaptiveTrans = dynamic_cast<CGraphTransitionAdaptiveTarget *>(*it);

		double tempVal = adaptiveTrans->getReward(currentTarget);

		
		if ((*it)->newStateIndex >= 0)
		{

			CDataSet1D *targetNodeValues = NULL;
			if (adaptiveTrans->isFinished(currentTarget))
			{
				CGraphTarget *nextTarget = currentTarget->getNextTarget();

				if (nextTarget != NULL)
				{
					targetNodeValues = (*targetMap)[nextTarget];
				}	
			}
			else
			{
				targetNodeValues = (*targetMap)[currentTarget];
			}
				
			if (targetNodeValues)
			{
				tempVal += (*it)->discountFactor * (*targetNodeValues)[(*it)->newStateIndex];
			}
		}
	
		if (k == 0 || tempVal > maxValue)
		{
			maxValue = tempVal;
			edge = *it;
		}
	}

	if (transitions->size() == 0)
	{
		CDataSet1D *targetNodeValues = (*targetMap)[currentTarget];
		maxValue = (*targetNodeValues)[index];
	}
	return edge;
}

void CGraphAdaptiveTargetDynamicProgramming::addTarget(CGraphTarget *target)
{
	currentTarget = target;
	targets->push_back(target);

	outputValues = new CDataSet1D();

	adaptiveTargetGraph->addTarget(target);

	(*targetMap)[target] = getOutputValues();
	evaluatePolicy(getParameter("PolicyEvaluationMaxEpisodes"));
	

//	outputValues = new CDataSet1D();	
}

CGraphTarget * CGraphAdaptiveTargetDynamicProgramming::getTargetForState(CStateCollection *state)
{
	std::list<CGraphTarget *>::iterator it = targets->begin();
	for (; it != targets->end(); it ++)
	{
		CGraphTarget *target = *it;

		if (target->isTargetForState(state))
		{
			return target;
		}
	}
	return NULL;
}
	
void CGraphAdaptiveTargetDynamicProgramming::setCurrentTarget(CGraphTarget *target)
{
	currentTarget = target;
	adaptiveTargetGraph->setCurrentTarget(target);

	outputValues = (*targetMap)[target];
}

int CGraphAdaptiveTargetDynamicProgramming::getNumTargets()
{
	return targets->size();
}

CGraphTarget *CGraphAdaptiveTargetDynamicProgramming::getTarget(int index)
{
	std::list<CGraphTarget *>::iterator it = targets->begin();

	for (int i = 0; it != targets->end() && i < index; i++, it ++);
	
	return (*it);
}

void CGraphAdaptiveTargetDynamicProgramming::resetLearnData()
{
	double init = getParameter("InitNodeValue");

	
	if (resetGraph)
	{
		printf("Init Graph Iteration\n");

		CContinuousStateList *stateList = transModel->getStateList();
		stateList->resetData();
		printf("Creating Graph...(%d Nodes)\n", 	transModel->getStateList()->size());
		transModel->createTransitions();
		printf("Done\n");
		transModel->getStateList()->initNearestNeighborSearch();
		transModel->getStateList()->getKDTree()->createLeavesArray();
	}
	
	
	
}
