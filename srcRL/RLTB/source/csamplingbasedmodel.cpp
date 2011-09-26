#include "csamplingbasedmodel.h"

#include "caction.h"
#include "cepisodehistory.h"
#include "ctransitionfunction.h"
#include "crewardmodel.h"
#include "cstatemodifier.h"
#include "cstate.h"
#include "cstatecollection.h"
#include "cepisode.h"
#include "cnearestneighbor.h"
#include "ckdtrees.h"
#include "cbatchlearning.h"
#include "cevaluator.h"
#include "csupervisedlearner.h"
#include "ccontinuousactions.h"
#include "cinputdata.h"

CSampleTransition::CSampleTransition(CState *l_state, CActionSet *l_availableActions, double l_reward, CActionData *l_actionData)
{
	state = new CState(l_state);
	reward = l_reward;

	availableActions = new CActionSet();
	availableActions->add( l_availableActions);

	actionData = l_actionData;

}
		
CSampleTransition::~CSampleTransition()
{
	delete state;
	delete availableActions;
	
	if (actionData)
	{
		delete actionData;
	}
}

CContinuousStateList::CContinuousStateList(CStateProperties *properties) : CDataSet(properties->getNumContinuousStates()), CStateObject(properties)
{
	initLogger = NULL;
	
	kdTree = NULL;
	nearestNeighbor = NULL;
	rangeSearch = NULL;

	initNearestNeighborSearch();

	addParameter("StateListMinimumDistance", 0.0);
}

CContinuousStateList::~CContinuousStateList()
{
	if (kdTree)
	{
		delete kdTree;
		delete nearestNeighbor;
		delete rangeSearch;
	}

}

void CContinuousStateList::disableNearestNeighborSearch()
{
	if (kdTree)
	{
		delete kdTree;
		delete nearestNeighbor;
		delete rangeSearch;
	}
	kdTree = NULL;
	nearestNeighbor = NULL;
	rangeSearch = NULL;
}

void CContinuousStateList::initNearestNeighborSearch()
{
	if (kdTree)
	{
		delete kdTree;
		delete nearestNeighbor;
		delete rangeSearch;
	}
	
	
	
	kdTree = new CKDTree(this, 1);
	nearestNeighbor = new CKNearestNeighbors(kdTree, this, 1);	
	rangeSearch = new CRangeSearch(kdTree, this);
}

bool CContinuousStateList::isMember(ColumnVector *point)
{
	int nN = 0;
	double distance = 2;
	nearestNeighbor->getNearestNeighborDistance(point,nN, distance);

	return distance == 0;
}


void CContinuousStateList::getNearestNeighbor(ColumnVector *point, int &index, double &distance)
{
	nearestNeighbor->getNearestNeighborDistance(point,index, distance);
}

void CContinuousStateList::getSamplesInRange(CKDRectangle *rectangle, DataSubset *subset)
{
	rangeSearch->getSamplesInRange(rectangle, subset);
}

void CContinuousStateList::nextStep(CStateCollection *oldState, CAction *, CStateCollection *)
{
	CState *state = oldState->getState(properties);
	
	addState(state, getParameter("StateListMinimumDistance"));
}

void CContinuousStateList::loadData(FILE *stream)
{
	int states = 0;
	int dimensions = 0;
	fscanf(stream, "ContinuousStateList: %d States, %d Dimensions\n", &states, &dimensions);

	for (int i = 0; i < states; i ++)
	{
		ColumnVector state(dimensions);

		for (int j = 0; j < state.nrows(); j ++)
		{
			double buf = 0.0;
			fscanf(stream, "%lf ", &buf);
			
			state.element(j) = buf;
		}
		fscanf(stream, "\n");
		addInput(&state);
	}
	printf("Continuous State List: Loaded %d States\n", size());
}

void CContinuousStateList::saveData(FILE *stream)
{
	fprintf(stream, "ContinuousStateList: %d States, %d Dimensions\n", size(), getNumDimensions());

	for (int i = 0; i < size(); i ++)
	{
		ColumnVector *state = (*this)[i];

		for (int j = 0; j < state->nrows(); j ++)
		{
			fprintf(stream, "%f ", state->element(j));
		}
		fprintf(stream, "\n");
	}
}


int CContinuousStateList::addState(CState *state, double minDist)
{
	int nN = 0;
	double distance = 2;

	if (minDist < 0)
	{
		minDist = getParameter("StateListMinimumDistance");
	}

	if (size() > 0 && kdTree)
	{
		nearestNeighbor->getNearestNeighborDistance(state,nN, distance);
	}

	if (distance <= minDist)
	{
		return nN;
	}
	else
	{
		addInput(state);
		if (kdTree)
		{
			kdTree->addNewInput(size() - 1);
			//printf("Growing KD Tree: %d Examples %d Depth %d Leaves %d Samples\n", size(), kdTree->getDepth(), kdTree->getNumLeaves());

		}
		return size() - 1;
	}
}

	
void CContinuousStateList::createStateList(CEpisodeHistory *history,  bool useInitLogger)
{
	if (useInitLogger)
	{
		initLogger = history;
	}
	else
	{
		CStep step(history->getStateProperties(), history->getStateModifiers(), history->getActions());

		printf("Creating Sample Transition Model ...\n");

		for (int i = 0; i < history->getNumEpisodes(); i ++)
		{
			CEpisode *episode = history->getEpisode(i);

			

			for (int j = 0; j < episode->getNumSteps(); j ++)
			{
				episode->getStep(j, &step);

				nextStep(step.oldState, step.action, step.newState);			
			}
			
		}
		printf("Finished : %d States\n", size());
	}
}

void CContinuousStateList::resetData()
{
	CDataSet::clear();
	if (kdTree)
	{
		initNearestNeighborSearch();
	}
	if (initLogger)
	{	
		createStateList(initLogger);
	}
	
}

void CSamplingBasedTransitionModel::clearTransitions()
{
	std::map<int, Transitions *>::iterator it = transitions->begin();
	
	for (;it != transitions->end(); it ++)
	{
		Transitions::iterator it2 = (*it).second->begin();
	
		for (;it2 != (*it).second->end(); it2 ++)
		{
			delete (*it2).second;
		}
		delete (*it).second;
	}
	transitions->clear();
	
}

CSamplingBasedTransitionModel::CSamplingBasedTransitionModel(CStateProperties *properties, CStateProperties *l_targetProperties, CActionSet *action, CRewardFunction *reward) : CActionObject(action),  CSemiMDPRewardListener(reward), CStateObject(properties)
{
	transitions = new std::map<int, Transitions *>;
	targetProperties = l_targetProperties;
	

	stateList = new CContinuousStateList(properties);

}
	
CSamplingBasedTransitionModel::~CSamplingBasedTransitionModel()
{
	clearTransitions();
	delete transitions;
	delete stateList;
}

int CSamplingBasedTransitionModel::getNumStates()
{
	return stateList->size();
}

void CSamplingBasedTransitionModel::createStateList(CEpisodeHistory *history, CRewardLogger *rewardLogger, bool useInitLogger)
{
	if (useInitLogger)
	{
		initLogger = history;
		initRewardLogger = rewardLogger;
	}
	else
	{
		CStep step(history->getStateProperties(), history->getStateModifiers(), history->getActions());

		printf("Creating Sample Transition Model ...\n");

		for (int i = 0; i < history->getNumEpisodes(); i ++)
		{
			CEpisode *episode = history->getEpisode(i);

			CRewardEpisode *rewardEpisode = NULL;

			if (rewardLogger)
			{
				rewardEpisode = rewardLogger->getEpisode(i);
			}

			for (int j = 0; j < episode->getNumSteps(); j ++)
			{
				episode->getStep(j, &step);

				if (rewardEpisode)
				{
					nextStep(step.oldState, step.action, rewardEpisode->getReward(j), step.newState);

				}
				else
				{
					nextStep(step.oldState, step.action, step.newState);			
				}
			}
			
		}
		printf("Finished : %d States\n", getNumStates());
	}
}

void CSamplingBasedTransitionModel::addTransition(int index, CAction *action, CStateCollection *state, double reward)
{
	assert((unsigned int) index <= transitions->size());
	
	CActionSet availableActions;
	actions->getAvailableActions(&availableActions, state);
	
	Transitions *transSamples = NULL;
	if ((unsigned int) index < transitions->size())
	{
		transSamples = (*transitions)[index];
	}
	else
	{
		transSamples = new Transitions();
		(*transitions)[index] = transSamples;
	}
	CSampleTransition *trans = new CSampleTransition(state->getState(targetProperties), &availableActions, reward);
	
	if ((*transSamples)[action] != NULL)
	{
		delete (*transSamples)[action];
	}
	(*transSamples)[action] = trans;

	
}

void CSamplingBasedTransitionModel::nextStep(CStateCollection *oldState, CAction *action, CStateCollection *newState)
{
	double reward = rewardFunction->getReward(oldState, action, newState);
	nextStep(oldState, action, reward, newState);
}

void CSamplingBasedTransitionModel::nextStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *newState)
{
	int index = stateList->addState( oldState->getState(properties));

	addTransition(index, action, newState, reward);
}

void CSamplingBasedTransitionModel::loadData(FILE *)
{
}

void CSamplingBasedTransitionModel::saveData(FILE *)
{
}


void CSamplingBasedTransitionModel::resetData()
{
	clearTransitions();
	stateList->resetData();
}

std::map<CAction *, CSampleTransition *> *CSamplingBasedTransitionModel::getTransitions(int index)
{
	return (*transitions)[index];
}

CContinuousStateList *CSamplingBasedTransitionModel::getStateList()
{
	return stateList;
}

CSamplingBasedTransitionModelFromTransitionFunction::CSamplingBasedTransitionModelFromTransitionFunction(CStateProperties *properties, CStateProperties *targetProperties, CActionSet *l_allActions, CTransitionFunction *l_transitionFunction, CRewardFunction *rewardFunction, std::list<CStateModifier *> *stateModifiers, CRewardFunction *l_rewardPrediction) : CSamplingBasedTransitionModel(properties, targetProperties, l_allActions, rewardFunction) 
{
	transitionFunction = l_transitionFunction;
	predictState = new CStateCollectionImpl(transitionFunction->getStateProperties(), stateModifiers);

	rewardPrediction = l_rewardPrediction;
	availableActions = new CActionSet();

	allActions = l_allActions;
}

CSamplingBasedTransitionModelFromTransitionFunction::~CSamplingBasedTransitionModelFromTransitionFunction()
{
	delete predictState;
	delete availableActions;
}


void CSamplingBasedTransitionModelFromTransitionFunction::nextStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *newState)
{
	int index = stateList->addState( oldState->getState(properties));

	addTransition(index, action, newState, reward);

	Transitions *sampleTransitions = (*transitions)[index];

	availableActions->clear();
	allActions->getAvailableActions(availableActions, oldState);

	CActionSet::iterator it = availableActions->begin();

	for (; it != availableActions->end(); it ++)
	{
		if ((*sampleTransitions)[*it] == NULL)
		{
			transitionFunction->transitionFunction(oldState->getState(transitionFunction->getStateProperties()), *it, predictState->getState());

			predictState->newModelState();
			predictState->setResetState(transitionFunction->isResetState(predictState->getState()));
			
			double reward = 0;
			if (rewardPrediction)
			{
				reward = rewardPrediction->getReward(oldState, *it, predictState);
			}
			else
			{
				reward = rewardFunction->getReward(oldState, *it, predictState);
			}

			addTransition(index, *it, predictState, reward);
		}
	}
}

CGraphTransition::CGraphTransition(int l_newStateIndex, double l_reward, double l_discountFactor, CAction *l_action, CActionData *l_actionData)
{
	newStateIndex = l_newStateIndex;
	reward = l_reward;

	action = l_action;
	actionData = l_actionData;

	discountFactor = l_discountFactor;
}

CGraphTransition::~CGraphTransition()
{
	if (actionData)
	{
		delete actionData;
	}
}

double CGraphTransition::getReward()
{
	return reward;
}


	
void CSamplingBasedGraph::clearTransitions()
{
	std::map<int, Transitions *>::iterator it = transitions->begin();
	
	for (;it != transitions->end(); it ++)
	{
		Transitions::iterator it2 = (*it).second->begin();
	
		for (;it2 != (*it).second->end(); it2 ++)
		{
			delete (*it2);
		}
		delete (*it).second;
	}
	transitions->clear();
	
	for (unsigned int i = 0; i < stateList->size(); i ++)
	{
		(*transitions)[i] = new Transitions();
	}

	numTransitions =  0;
}

void CSamplingBasedGraph::addTransition(int index, int newIndex, CAction *action,  CActionData *actionData, double reward, double discountFactor)
{
	Transitions *transSamples = NULL;
	transSamples = (*transitions)[index];
	
	CGraphTransition *trans = new CGraphTransition( newIndex, reward, discountFactor, action, actionData);

	transSamples->push_back(trans);	

	numTransitions ++;
}

CSamplingBasedGraph::CSamplingBasedGraph(CContinuousStateList *l_stateList, CActionSet *actions) : CActionObject(actions), CStateObject(l_stateList->getStateProperties())
{
	transitions = new std::map<int, Transitions *>;

	stateList = l_stateList;

	clearTransitions();

	addParameters(stateList);

	numTransitions =  0;

		
}

CSamplingBasedGraph::~CSamplingBasedGraph()
{
	std::map<int, Transitions *>::iterator it = transitions->begin();
	
	for (;it != transitions->end(); it ++)
	{
		Transitions::iterator it2 = (*it).second->begin();
	
		for (;it2 != (*it).second->end(); it2 ++)
		{
			delete (*it2);
		}
		delete (*it).second;
	}
	delete transitions;

}

void CSamplingBasedGraph::loadData(FILE *stream)
{
	stateList->loadData(stream);
	clearTransitions();

	fscanf(stream, "Continuous Graph\n");
	int gesTransitions = 0;

	for (unsigned int i = 0; i < stateList->size(); i ++)
	{
		Transitions *transitions = getTransitions(i);
		int buf = 0;
		int numTransitions = 0;

		fscanf(stream, "Transitions for State %d: %d\n", &buf, &numTransitions);
		

		for (int j = 0; j < numTransitions; j ++)
		{
			int newStateIndex = 0;
			double reward = 0.0;
			double discountFactor = 0.0;
			
			CAction *action = NULL;
			CActionData *actionData = NULL;
			int actionIndex = -1;
	
			fscanf(stream, "%d %lf %lf %d ", &newStateIndex, &reward, &discountFactor, &actionIndex);

			if (actionIndex >= 0)
			{
				action = actions->get(actionIndex);
				
				actionData = action->getNewActionData();

				if (actionData != NULL)
				{
					actionData->loadASCII(stream);	
				}
			}
			addTransition(i, newStateIndex, action, actionData, reward, discountFactor);

			gesTransitions ++;
		}
		fscanf(stream, "\n");
		assert(numTransitions == getTransitions(i)->size());
	}
	printf("Loaded %d Transitions\n");
}

void CSamplingBasedGraph::saveData(FILE *stream)
{
	stateList->saveData(stream);
	
	fprintf(stream, "Continuous Graph\n");
	for (unsigned int i = 0; i < stateList->size(); i ++)
	{
		Transitions *transitions = getTransitions(i);
		fprintf(stream, "Transitions for State %d: %d\n", i, transitions->size());
		Transitions::iterator it = transitions->begin();

		for (; it != transitions->end(); it ++)
		{
			int actionIndex = -1;
	
			fprintf(stream, "%d %f %f ", (*it)->newStateIndex, (*it)->reward, (*it)->discountFactor);
			if ((*it)->action != NULL)
			{
				fprintf(stream, "%d ", actions->getIndex((*it)->action));
			}
			else
			{
				fprintf(stream, "-1 ");
			}
			
			if ((*it)->actionData)
			{
				(*it)->actionData->saveASCII(stream);
			}
		}
		fprintf(stream, "\n");
	}
}

void CSamplingBasedGraph::resetData()
{
	clearTransitions();
}

int CSamplingBasedGraph::getNumStates()
{
	return stateList->size();
}

std::list<CGraphTransition *> *CSamplingBasedGraph::getTransitions(int index)
{
	return (*transitions)[index];
}

CContinuousStateList *CSamplingBasedGraph::getStateList()
{
	return stateList;
}

void CSamplingBasedGraph::getConnectedNodes(int node, DataSubset *subset)
{
	subset->insert(node);

	Transitions *transitions = getTransitions(node);

	Transitions::iterator it = transitions->begin();
	for (; it != transitions->end(); it ++)
	{
		int newNode = (*it)->newStateIndex;

		if (newNode >= 0)
		{
			if (subset->find(newNode) == subset->end())
			{
				getConnectedNodes(newNode, subset);	
			}
		}
	}
}


void CSamplingBasedGraph::createTransitions()
{
	clearTransitions();
	for (int i = 0; i < stateList->size(); i ++)
	{
		addTransitions(i, false);
	}

	printf("Added %d Transitions to the graph\n", numTransitions);
}

void CSamplingBasedGraph::addState(CState *state)
{
	int oldSize = stateList->size();
	int index = stateList->addState(state);

	if (oldSize < stateList->size())
	{
		(*transitions)[oldSize] = new Transitions();
		addTransitions(oldSize, true);
	}

}

void CSamplingBasedGraph::addTransitions(int node, bool newNode)
{
	DataSubset elementList;

	if (isFinalNode(node))
	{
		addFinalTransition(node); 
	}
	else
	{
		elementList.clear();
		getNeighboredNodes(node, &elementList);
	
		DataSubset::iterator it = elementList.begin();
	
		/*if (elementList.size() > 0)
		{
			printf("Found %d States for State %d ", elementList.size(), i);
		}
		else
		{
			printf("Error: NO States found in the neighborhood of %d!!\n", i);
		}*/
	
	
		for (int m = 0; it != elementList.end(); it ++, m ++)
		{
			if (*it != node)
			{
				calculateTransition(node, *it);
				if (newNode)
				{
					calculateTransition(*it, node);	
				}
			}	
		}

		if (getTransitions(node)->size() == 0)
		{
			printf("No transitions found for node %d\n", node);
			ColumnVector *nodeVector = (*stateList)[node];
			cout << nodeVector->t();
		}
	}
}


CGraphTarget::CGraphTarget(CGraphTarget *l_nextTarget)
{
	nextTarget = l_nextTarget;
}

CGraphTarget::~CGraphTarget()
{
}


CGraphTransitionAdaptiveTarget::CGraphTransitionAdaptiveTarget(int newStateIndex, double reward, double discountFactor,  CAction *action, CActionData *actionData, CGraphTarget **l_currentTarget) : CGraphTransition(newStateIndex, reward, discountFactor, action, actionData)
{
	targetReward = new std::map<CGraphTarget *, double>();
	targetReached = new std::map<CGraphTarget *, bool>();
	
	currentTarget = l_currentTarget;
}

CGraphTransitionAdaptiveTarget::~CGraphTransitionAdaptiveTarget()
{
	delete targetReward;
	delete targetReached;
}

double CGraphTransitionAdaptiveTarget::getReward()
{
	return getReward(*currentTarget);
}

double CGraphTransitionAdaptiveTarget::getReward(CGraphTarget *target)
{
	std::map<CGraphTarget *, double>::iterator it = targetReward->find(target);

	if (it != targetReward->end())
	{
		//printf("Reward: %f, Target Reward: %f\n", reward, (*it).second);
		return reward + (*it).second;
	}
	else
	{
		//printf("Reward: %f\n", reward);
		return reward;
	}
}


bool CGraphTransitionAdaptiveTarget::isFinished(CGraphTarget *target)
{
	std::map<CGraphTarget *, bool>::iterator it = targetReached->find(target);

	if (it != targetReached->end())
	{
		return (*it).second;
	}
	else
	{
		return false;
	}
}

void CGraphTransitionAdaptiveTarget::addTarget(CGraphTarget *target, double reward, bool isFinished)
{
	(*targetReward)[target] = reward;
	(*targetReached)[target] = isFinished;
}


CAdaptiveTargetGraph::CAdaptiveTargetGraph(CContinuousStateList *stateList, CActionSet *actions) : CSamplingBasedGraph(stateList, actions)
{
	targetList = new std::list<CGraphTarget *>();
	currentTarget = NULL;
}

CAdaptiveTargetGraph::~CAdaptiveTargetGraph()
{
	delete targetList;	
}

void CAdaptiveTargetGraph::setCurrentTarget(CGraphTarget *target)
{
	currentTarget = target;
}

void CAdaptiveTargetGraph::addTransition(int index, int newIndex, CAction *action,  CActionData *actionData, double reward, double discountFactor)
{
	Transitions *transSamples = NULL;
	transSamples = (*transitions)[index];
	
	CGraphTransition *trans = new CGraphTransitionAdaptiveTarget( newIndex, reward, discountFactor, action, actionData, &currentTarget);

	transSamples->push_back(trans);	

	numTransitions ++;
}

void CAdaptiveTargetGraph::addTarget(CGraphTarget *target)
{
	targetList->push_back(target);

	
	for (int i = 0; i < stateList->size(); i ++)
	{
		addTargetForNode(target, i);
	}
	
}

void CAdaptiveTargetGraph::addTargetForNode(CGraphTarget *target, int i)
{
	ColumnVector *node = (*stateList)[i];

	int numFinished = 0;
	int numCanditates = 0;

	if (target->isFinishedCanditate(node))
	{	
		Transitions *transitions = getTransitions(i);
	
		Transitions::iterator it = transitions->begin();
	
		numCanditates ++;
					
		for (; it != transitions->end(); it ++)
			{
			CGraphTransitionAdaptiveTarget *targetTrans = dynamic_cast<CGraphTransitionAdaptiveTarget *>(*it);
	
			ColumnVector *newNode = (*stateList)[targetTrans->newStateIndex];

			double reward = 0.0;
			bool finished = target->isFinished(node, newNode, reward);

			if (finished || fabs(reward) > 0.001)
			{
				targetTrans->addTarget(target, reward, finished);
					
				
				if (finished)
				{
					numFinished ++;
//					printf("End Node %d: %f %f\n", targetTrans->newStateIndex, newNode->element(0), newNode->element(1));
				}
			}
		}
		//printf("%d Canditates, %d Final Transitions\n", numCanditates, numFinished);
	}
}

void CAdaptiveTargetGraph::addState(CState *addState)
{
	int oldSize = stateList->size();

	CSamplingBasedGraph::addState(addState);

	if (oldSize < stateList->size())
	{
		std::list<CGraphTarget *>::iterator it = targetList->begin();

		for (; it != targetList->end(); it ++)
		{
			addTargetForNode(*it, oldSize);
		}
	}

}


CGraphController::CGraphController(CActionSet *actionSet, CGraphDynamicProgramming *l_graph) : CAgentController(actionSet)
{
	graph = l_graph;
}
	
CGraphController::~CGraphController()
{
}

CAction *CGraphController::getNextAction(CStateCollection *state, CActionDataSet *dataSet)
{
	CState *inputState = state->getState(graph->getGraph()->getStateProperties());

	int nearestNeighbor = -1;
	double distance = -1.0;
	double maxValue = 0.0;

	graph->getNearestNode(inputState, nearestNeighbor, distance);

	CGraphTransition *transition = graph->getMaxTransition(nearestNeighbor, maxValue); 

	//assert(transition != NULL);

	if (transition == NULL)
	{
		return actions->get(0);
	}

	if (dataSet && transition->actionData)
	{
		dataSet->setActionData(transition->action, transition->actionData);
	}
	return transition->action;
}



CAdaptiveTargetGraphController::CAdaptiveTargetGraphController(CActionSet *actionSet, CGraphAdaptiveTargetDynamicProgramming *l_adaptiveGraph) : CAgentController(actionSet)
{
	adaptiveGraph = l_adaptiveGraph;
}
	
CAdaptiveTargetGraphController::~CAdaptiveTargetGraphController()
{
}

CAction *CAdaptiveTargetGraphController::getNextAction(CStateCollection *state, CActionDataSet *dataSet)
{
	
	CGraphTarget *target = adaptiveGraph->getTargetForState(state);

	
	adaptiveGraph->setCurrentTarget(target);


	int index = -1;
	for (int i = 0; i < adaptiveGraph->getNumTargets(); i ++)	
	{
		if (target == adaptiveGraph->getTarget(i))
		{
			index = i;
			break;
		}
	}
	printf("Controller Target: %d\n", index);	

	CState *inputState = state->getState(adaptiveGraph->getGraph()->getStateProperties());


	if (target == NULL)
	{
		printf("No target found for state ");
		state->getState()->saveASCII(stdout);
		printf("\n");
		
	}

	

	int nearestNeighbor = -1;
	double distance = -1.0;
	double maxValue = 0.0;

	adaptiveGraph->getNearestNode(inputState, nearestNeighbor, distance);

	CGraphTransition *transition = adaptiveGraph->getMaxTransition(nearestNeighbor, maxValue); 

	//assert(transition != NULL);

	if (transition == NULL)
	{
		printf("No graph transition found... Using random action: %d\n", nearestNeighbor);
		ColumnVector *node = (*adaptiveGraph->getGraph()->getStateList())[nearestNeighbor];

		cout << node->t();

		inputState->saveASCII(stdout);
		printf("\n");
		return actions->get(0);
	}

	if (dataSet && transition->actionData)
	{
		dataSet->setActionData(transition->action, transition->actionData);
	}
	return transition->action;
}


CGraphDebugger::CGraphDebugger(CGraphDynamicProgramming *l_graph, CRewardFunction *reward, CStateModifier *l_hcState) : CSemiMDPRewardListener(reward)
{
	graph = l_graph;
	
	realRewardSum = 0.0;
	graphRewardSum = 0.0;

	hcState = l_hcState;

	step = 0;
}

void CGraphDebugger::nextStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *newState)
{
	

	CState *inputState = oldState->getState(graph->getGraph()->getStateProperties());

	int nearestNeighbor = -1;
	double distance = -1.0;
	double maxValue = 0.0;

	graph->getNearestNode(inputState, nearestNeighbor, distance);

	CKDRectangle range(4);

	
	/*double nextTarget = oldState->getState(hcState)->getContinuousState(14) * 100;
	range.setMinValue(3, nextTarget - 0.0001);
	range.setMaxValue(3, nextTarget + 0.0001);
	*/

	CGraphTransition *transition = graph->getMaxTransition(nearestNeighbor, maxValue);//, &range); 
	
	realRewardSum += reward;

	if (transition == NULL)
	{
		printf("No transition found\n");
		return;
	}

	double graphReward = transition->getReward();
	graphRewardSum += graphReward;
	
	CState *newGraphState = newState->getState(graph->getGraph()->getStateProperties());

	ColumnVector *newNode = (* graph->getGraph()->getStateList())[transition->newStateIndex];

	ColumnVector distVec = *newGraphState - *newNode;

	

	if (fabs(graphReward - reward) > 0.001 || distVec.norm_Frobenius() > 0.0001 || (*graph->getOutputValues())[nearestNeighbor] < -10 || true)
	{
		printf("%d : Transition from Node %d to %d, target: %d\n", step, nearestNeighbor, transition->newStateIndex, oldState->getState()->getDiscreteState(0));
		printf("Real reward: %f, Graph reward: %f, Graph Value: %f (%f %f)\n", reward, graphReward, (*graph->getOutputValues())[nearestNeighbor], realRewardSum, graphRewardSum);	
	
		CContinuousActionData *data = dynamic_cast<CContinuousActionData *>(action->getActionData());
	
		CContinuousActionData *graphData = dynamic_cast<CContinuousActionData *>(transition->actionData);
	
	
		if (graphData)
		{
			printf("RealAction: %f %f, Graph Action: %f %f\n", data->element(0), data->element(3), graphData->element(0), graphData->element(3));
		}
	
		oldState->getState(graph->getGraph()->getStateProperties())->saveASCII(stdout);
		printf("\n");
	
		newState->getState(graph->getGraph()->getStateProperties())->saveASCII(stdout);
		printf("\n");
	
		cout << (*graph->getGraph()->getStateList())[nearestNeighbor]->t();
		
		if (transition->newStateIndex >= 0)
		{
			cout << (*graph->getGraph()->getStateList())[transition->newStateIndex]->t();
		}
		else
		{
			cout << "End State\n";
		}
	}

	step ++;
}

void CGraphDebugger::newEpisode()
{
	printf("Real reward Sum %f, Graph %f\n", realRewardSum, graphRewardSum);
	
	realRewardSum = 0.0;
	graphRewardSum = 0.0;

	step = 0;
}


CGraphValueFromValueFunctionCalculator::CGraphValueFromValueFunctionCalculator(CGraphDynamicProgramming *l_graph, CSupervisedLearner *l_learner, CPolicyEvaluator *l_evaluator)
{
	graph = l_graph;
	learner = l_learner;

	evaluator = l_evaluator;

}

CGraphValueFromValueFunctionCalculator::~CGraphValueFromValueFunctionCalculator()
{

}

double CGraphValueFromValueFunctionCalculator::evaluate()
{
	learner->learnFA(graph->getGraph()->getStateList(), graph->getOutputValues());

	double value = evaluator->evaluatePolicy();

	printf("Learned Value Function: %f\n", value);

	return value;
}


