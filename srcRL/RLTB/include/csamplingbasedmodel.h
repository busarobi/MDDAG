#ifndef C_SAMPLINGBASED_MODEL
#define C_SAMPLINGBASED_MODEL


#include "cagentlistener.h"
#include "cagentcontroller.h"
#include "clearndataobject.h"
#include "cinputdata.h"
#include "cbaseobjects.h"
#include "cparameters.h"
#include "cevaluator.h"

class CEpisodeHistory;
class CAction;
class CTransitionFunction;
class CRewardLogger;
class CActionSet;
class CStateModifier;
class CKDTree;
class CKNearestNeighbors;
class CRangeSearch;
class CKDRectangle;

class CContinuousStateList : public CDataSet, public CSemiMDPListener, public CLearnDataObject, public CStateObject 
{
protected:
	CEpisodeHistory *initLogger;
	
	CKNearestNeighbors *nearestNeighbor;
	CRangeSearch *rangeSearch;

	CKDTree *kdTree;
	

public:
	

	CContinuousStateList(CStateProperties *properties);
	virtual ~CContinuousStateList();

	virtual void nextStep(CStateCollection *oldState, CAction *action, CStateCollection *newState);
	
	virtual int addState(CState *state, double minDist = -1.0);	

	virtual void createStateList(CEpisodeHistory *history, bool useInitLogger = false);

	virtual void resetData();

	virtual void loadData(FILE *);
	virtual void saveData(FILE *);

	void initNearestNeighborSearch();
	void disableNearestNeighborSearch();

	virtual bool isMember(ColumnVector *point);
	virtual void getNearestNeighbor(ColumnVector *point, int &index, double &distance);

	virtual void getSamplesInRange(CKDRectangle *rectangle, DataSubset *subset);

	virtual CKDTree *getKDTree() {return kdTree;};

};



class CSampleTransition
{
	public:
		CSampleTransition(CState *state, CActionSet *availableActions, double reward, CActionData *actionData = NULL);

		virtual ~CSampleTransition();	

		CState *state;
		CActionSet *availableActions;
		double reward;
		CActionData *actionData;
};

class CSamplingBasedTransitionModel : public CActionObject, CSemiMDPRewardListener, public CLearnDataObject, public CStateObject  
{
protected:
	CEpisodeHistory *initLogger;
	CRewardLogger *initRewardLogger;

	typedef std::map<CAction *, CSampleTransition *> Transitions;

	CRewardFunction *rewardFunction;
	CStateProperties *targetProperties;		

	std::map<int, Transitions *> *transitions;
	
	CContinuousStateList *stateList;
		
	void clearTransitions();
	void addTransition(int index, CAction *action, CStateCollection *state, double reward);
public:
	CSamplingBasedTransitionModel(CStateProperties *properties, CStateProperties *targetProperties, CActionSet *actions, CRewardFunction *rewardFunction);
	virtual ~CSamplingBasedTransitionModel();

	virtual void nextStep(CStateCollection *oldState, CAction *action, CStateCollection *newState);
	
	virtual void nextStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *newState);	

	virtual void resetData();
	virtual void loadData(FILE *);
	virtual void saveData(FILE *);

	int getNumStates();

	std::map<CAction *, CSampleTransition *> *getTransitions(int index);

	virtual void createStateList(CEpisodeHistory *history, CRewardLogger *logger, bool useInitLogger = false);

	CContinuousStateList *getStateList();
};

class CGraphTransition
{
public:
	CGraphTransition(int newStateIndex, double reward, double discountFactor,  CAction *action, CActionData *actionData = NULL);

	virtual ~CGraphTransition();	

	int newStateIndex;
	double reward;
	CActionData *actionData;
	CAction *action;

	double discountFactor;

	virtual double getReward();
};


class CSamplingBasedGraph : public CActionObject, public CStateObject, public CLearnDataObject
{
protected:
	
	typedef std::list<CGraphTransition *> Transitions;

	std::map<int, Transitions *> *transitions;
	
	CContinuousStateList *stateList;
		
	void clearTransitions();
	virtual void addTransition(int index, int newIndex, CAction *action,  CActionData *actionData, double reward, double discountFactor);

	int numTransitions;
public:
	CSamplingBasedGraph(CContinuousStateList *stateList, CActionSet *actions);
	virtual ~CSamplingBasedGraph();

	virtual void resetData();

	int getNumStates();

	std::list<CGraphTransition *> *getTransitions(int index);

	CContinuousStateList *getStateList();

	virtual void loadData(FILE *stream);
	virtual void saveData(FILE *stream);

	virtual void getConnectedNodes(int node, DataSubset *subset);

	virtual void createTransitions(); 

	virtual bool calculateTransition(int startNode, int endNode) = 0;
	virtual void getNeighboredNodes(int node, DataSubset *elementList) = 0;

	virtual bool isFinalNode(int node) = 0;
	virtual void addFinalTransition(int node) = 0;

	virtual void addState(CState *addState);
	virtual void addTransitions(int node, bool newNode = false);
};



class CGraphTarget
{
protected:
	CGraphTarget *nextTarget;
public:
	CGraphTarget(CGraphTarget *nextTarget);
	virtual ~CGraphTarget();

	virtual bool isFinishedCanditate(ColumnVector *node) = 0;
	virtual bool isFinished(ColumnVector *oldNode, ColumnVector *newNode, double &reward) = 0;

	virtual bool isTargetForState(CStateCollection *state) = 0;

	CGraphTarget *getNextTarget() {return nextTarget;};
};

class CGraphTransitionAdaptiveTarget : public CGraphTransition
{
protected:
	std::map<CGraphTarget *, double> *targetReward;
	std::map<CGraphTarget *, bool> *targetReached;
	
	CGraphTarget **currentTarget;
public:
	CGraphTransitionAdaptiveTarget(int newStateIndex, double reward, double discountFactor,  CAction *action, CActionData *actionData, CGraphTarget **currentTarget);

	virtual ~CGraphTransitionAdaptiveTarget();

	virtual double getReward();

	virtual double getReward(CGraphTarget *target);
	virtual bool isFinished(CGraphTarget *target);
	virtual void addTarget(CGraphTarget *target, double reward, bool isFinished);
};

class CAdaptiveTargetGraph : public CSamplingBasedGraph
{
protected:
	std::list<CGraphTarget *> *targetList;	
	CGraphTarget *currentTarget;

	virtual void addTransition(int index, int newIndex, CAction *action,  CActionData *actionData, double reward, double discountFactor);

	virtual void addTargetForNode(CGraphTarget *target, int node);
public:
	CAdaptiveTargetGraph(CContinuousStateList *stateList, CActionSet *actions);
	virtual ~CAdaptiveTargetGraph();

	void setCurrentTarget(CGraphTarget *target);
	virtual void addTarget(CGraphTarget *target);
	

	virtual void addState(CState *addState);
};




class CGraphDynamicProgramming;

class CGraphController : public CAgentController
{
protected:
	CGraphDynamicProgramming *graph;
public:
	CGraphController(CActionSet *actionSet, CGraphDynamicProgramming *graph);
	virtual ~CGraphController();

	virtual CAction *getNextAction(CStateCollection *state, CActionDataSet *dataSet);
};

class CGraphAdaptiveTargetDynamicProgramming;

class CAdaptiveTargetGraphController : public CAgentController
{
protected:
	CGraphAdaptiveTargetDynamicProgramming *adaptiveGraph;

public:
	CAdaptiveTargetGraphController(CActionSet *actionSet, CGraphAdaptiveTargetDynamicProgramming *adaptiveGraph);

	virtual ~CAdaptiveTargetGraphController();
	
	virtual CAction *getNextAction(CStateCollection *state, CActionDataSet *dataSet);
};


class CPolicyEvaluator;
class CSupervisedLearner;

class CGraphValueFromValueFunctionCalculator : public CEvaluator
{
protected:
	CSupervisedLearner *learner;
	CGraphDynamicProgramming *graph;
	
	CPolicyEvaluator *evaluator;
public:
	CGraphValueFromValueFunctionCalculator(CGraphDynamicProgramming *l_graph, CSupervisedLearner *learner, CPolicyEvaluator *evaluator);
    	virtual ~CGraphValueFromValueFunctionCalculator();
    
   	virtual double evaluate();
};



class CStateCollectionImpl;

class CSamplingBasedTransitionModelFromTransitionFunction : public CSamplingBasedTransitionModel
{
protected:
	CTransitionFunction *transitionFunction;
	
	CActionSet *availableActions;
	CActionSet *allActions;

	CRewardFunction *rewardPrediction;

	CStateCollectionImpl *predictState;
public:
	CSamplingBasedTransitionModelFromTransitionFunction(CStateProperties *properties, CStateProperties *targetProperties, CActionSet *allActions, CTransitionFunction *transitionFunction, CRewardFunction *rewardFunction, std::list<CStateModifier *> *stateModifier, CRewardFunction *predictReward);

	virtual ~CSamplingBasedTransitionModelFromTransitionFunction();

	virtual void nextStep(CStateCollection *oldState, CAction *action, double reward,  CStateCollection *newState);	
};


class CGraphDebugger : public CSemiMDPRewardListener
{
protected:
	CGraphDynamicProgramming *graph;

	double realRewardSum;
	double graphRewardSum;

	CStateModifier *hcState;

	int step;
public:
	CGraphDebugger(CGraphDynamicProgramming *graph, CRewardFunction *reward, CStateModifier *hcState);

	virtual void nextStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *newState);

	virtual void newEpisode();

};

#endif

