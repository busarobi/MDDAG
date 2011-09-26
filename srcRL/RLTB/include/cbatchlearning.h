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


#ifndef C_BATCHLEARNING__
#define C_BATCHLEARNING__


#include "cagentcontroller.h"
#include "cparameters.h"
#include "cagent.h"

#include "newmat/newmat.h"

class CRewardHistory;
class CEpisodeHistory;
class CRewardLogger;
class CAgentLogger;

class CContinuousActionQFunction;
class CPolicySameStateEvaluator;
class CGradientUpdateFunction;
class CSemiMDPListener;
class CLSTDLambda;
class CSemiMDPRewardListener;

class CFeatureCalculator;
class CFeatureVFunction;
class CDataSet;
class CDataSet1D;
class CDataSubset;

class CAbstractVFunction;
class CGradientVFunction;
class CQFunction; 
	
class CSupervisedLearner;
class CSupervisedWeightedLearner;
class CSupervisedQFunctionLearner;
class CSupervisedQFunctionWeightedLearner;

class CStochasticPolicy;

class CLearnDataObject;

class CContinuousActionQFunction;
class CStateProperties;

class CKDTree;
class CKNearestNeighbors;
class CDataPreprocessor;

class CSamplingBasedTransitionModel;
class CSamplingBasedGraph;
class CActionDistribution;

class CAction;
class CActionSet;

class CState;
class CStateCollection;
class CStateProperties;

class CStep;
class CNewFeatureCalculator;
class CAbstractQFunction;


class CBatchLearningPolicy : public CDeterministicController
{
	protected:
		CActionDataSet *actionDataSet;
		CAction *nextAction;
	public:
		CBatchLearningPolicy(CActionSet *actions);
		~CBatchLearningPolicy();
		
		virtual CAction *getNextAction(CStateCollection *state, CActionDataSet *actionData = NULL);
		
		virtual void setAction(CAction *action, CActionData *data);
};


class CPolicyEvaluation : public CParameterObject
{
	protected:
		
	public:
		CPolicyEvaluation(int maxEvaluations = 100);
		virtual ~CPolicyEvaluation();
		
		virtual void evaluatePolicy();
		virtual void evaluatePolicy(int numEvaluations) = 0;

		virtual void resetLearnData() {};
};

class CPolicyEvaluationGradientFunction : public CPolicyEvaluation
{
protected:
		CGradientUpdateFunction *learnData;
		double *oldWeights;
		
		
		virtual double getWeightDifference(double *oldWeights);
	public:
		bool resetData;
	
		CPolicyEvaluationGradientFunction(CGradientUpdateFunction *learnData, double treshold = 0.1, int maxEvaluations = 100);
		virtual ~CPolicyEvaluationGradientFunction();
		
		
		virtual void evaluatePolicy(int numEvaluations) = 0;
		
		virtual void resetLearnData();
};

class COnlinePolicyEvaluation : public CPolicyEvaluationGradientFunction
{
	protected:
		CAgent *agent;
		CSemiMDPListener *learner;

		CSemiMDPSender *semiMDPSender;
	public:
	
		COnlinePolicyEvaluation(CAgent *agent, CSemiMDPListener *learner, CGradientUpdateFunction *learnData, int maxEvaluationEpisodes, int numSteps, int checkWeightsPerEpisode);
		virtual ~COnlinePolicyEvaluation();
		
		virtual void evaluatePolicy(int numEvaluations);
		
		void setSemiMDPSender(CSemiMDPSender *semiMDPSender);
};

class CLSTDOnlinePolicyEvaluation : public COnlinePolicyEvaluation
{
	protected:
		CLSTDLambda *lstdLearner;
		
		virtual double getWeightDifference(double *oldWeights);
	public:

		CLSTDOnlinePolicyEvaluation(CAgent *agent, CLSTDLambda *learner, CGradientUpdateFunction *learnData, int maxEvaluationSteps, int nSteps);
		virtual ~CLSTDOnlinePolicyEvaluation();

		virtual void resetLearnData();
		
};

/*
class COfflinePolicyEvaluation : public CPolicyEvaluation
{
	protected:
		CSemiMDPListener *learner;
		CStepHistory *stepHistory;
		
		std::list<CStateModifier *> *modifiers;
	public:

		COfflinePolicyEvaluation(CStepHistory *stepHistory, CSemiMDPListener *learner, CGradientUpdateFunction *learnData, std::list<CStateModifier *> *l_modifiers, int maxEvaluationEpisodes);
		virtual ~COfflinePolicyEvaluation();
		
		
		virtual void evaluatePolicy(int numEvaluations);
};

class CLSTDOfflinePolicyEvaluation : public COfflinePolicyEvaluation
{
	protected:
		CLSTDLambda *lstdLearner;
		
		virtual double getWeightDifference(double *oldWeights);
	public:

		CLSTDOfflinePolicyEvaluation(CStepHistory *stepHistory, CLSTDLambda *learner, CGradientVFunction *learnData, std::list<CStateModifier *> *l_modifiers);
		virtual ~CLSTDOfflinePolicyEvaluation();

		virtual void resetLearnData();
};*/

class COfflineEpisodePolicyEvaluation : public CPolicyEvaluationGradientFunction, public CSemiMDPSender
{
	protected:
		CSemiMDPRewardListener *learner;
		CEpisodeHistory *episodeHistory;
		CRewardHistory *rewardLogger;
		
		std::list<CStateModifier *> *modifiers;
		
		CBatchLearningPolicy *policy;
	public:


		COfflineEpisodePolicyEvaluation(CEpisodeHistory *episodeHistory, CSemiMDPRewardListener *learner, CGradientUpdateFunction *learnData, std::list<CStateModifier *> *l_modifiers, int maxEvaluationEpisodes);
		COfflineEpisodePolicyEvaluation(CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger, CSemiMDPRewardListener *learner, CGradientUpdateFunction *learnData, std::list<CStateModifier *> *l_modifiers, int maxEvaluationEpisodes);
		virtual ~COfflineEpisodePolicyEvaluation();

		void setBatchLearningPolicy(CBatchLearningPolicy *l_policy);
		
		virtual void evaluatePolicy(int numEvaluations);
};


class CLSTDOfflineEpisodePolicyEvaluation : public COfflineEpisodePolicyEvaluation
{
	protected:
		CLSTDLambda *lstdLearner;
		
		virtual double getWeightDifference(double *oldWeights);
	public:

		CLSTDOfflineEpisodePolicyEvaluation(CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger, CLSTDLambda *learner, CGradientVFunction *learnData, std::list<CStateModifier *> *l_modifiers, int episodes);
		virtual ~CLSTDOfflineEpisodePolicyEvaluation();

		virtual void resetLearnData();
};


class CDataCollector : public CParameterObject
{
	protected:
	
	public:
		CDataCollector();
		virtual ~CDataCollector();
		
		virtual void collectData() = 0;

};

class CUnknownDataQFunction;

class CDataCollectorFromAgentLogger : public CDataCollector
{
protected:
	CAgentLogger *logger;
	CRewardLogger *rewardLogger;
	CAgent *agent;

	int numCollections;
	CSemiMarkovDecisionProcess *sender;

	std::list<CUnknownDataQFunction *> *unknownDataQFunctions;
	CAgentController *controller;
public:
	CDataCollectorFromAgentLogger(CAgent *agent, CAgentLogger *logger, CRewardLogger *rewardLogger, int numEpisodes, int numSteps);
	virtual ~CDataCollectorFromAgentLogger();
	
	virtual void collectData();

	virtual void setController(CAgentController *controller);
	virtual void addUnknownDataFunction(CUnknownDataQFunction *unknownDataQFunctions);	

	void setSemiMDPSender(CSemiMarkovDecisionProcess *sender);
};



class CPolicyIteration : virtual public CParameterObject
{
protected:
	CLearnDataObject *policyFunction;
	CLearnDataObject *evaluationFunction;
	
	CPolicyEvaluation *evaluation;
	CDataCollector *collector;
public:
	
	CPolicyIteration(CLearnDataObject *policyFunction, CLearnDataObject *evaluationFunction, CPolicyEvaluation *evaluation, CDataCollector *collector = NULL);
	virtual ~CPolicyIteration();

	virtual void doPolicyIterationStep();
	virtual void initPolicyIteration();
};



class CNewFeatureCalculatorDataGenerator 
{
protected:

public:
	virtual ~CNewFeatureCalculatorDataGenerator() {}; 
	
	virtual void initFeatures() {calculateNewFeatures();};
	virtual void calculateNewFeatures() = 0;

	virtual void swapValueFunctions() = 0;

	virtual void resetData() {};
};

class CPolicyIterationNewFeatures : public CPolicyIteration
{
protected:
        CNewFeatureCalculatorDataGenerator *newFeatureCalculator;
public:

	CPolicyIterationNewFeatures(CLearnDataObject *policyFunction, CLearnDataObject *evaluationFunction, CPolicyEvaluation *evaluation, CNewFeatureCalculatorDataGenerator *newFeatureCalculator, CDataCollector *collector = NULL);
	virtual ~CPolicyIterationNewFeatures();

	virtual void doPolicyIterationStep();
	virtual void initPolicyIteration();
};

/*
class CResidualFunction;
class CResidualGradientFunction;

class CValueGradientCalculator : public CGradientCalculator
{
	protected:
		CResidualFunction *residual;
		CResidualGradientFunction *residualGradientFunction;
		
		CEpisodeHistory *episodeHistory;
		CRewardHistory *rewardLogger;
		CRewardFunction *rewardFunction;
		
		virtual double getValue(CStateCollection *state, CAction *action) = 0;
		virtual void getValueGradient(CStateCollection *state, CAction *action, CFeatureList *gradient) = 0;
			
		CBatchLearningPolicy *policy;
		CAgentController *estimationPolicy;
	public:
		CValueGradientCalculator(CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger, CResidualFunction *residual, CResidualGradientFunction *gradient);
		CValueGradientCalculator(CEpisodeHistory *episodeHistory, CRewardFunction *rewardFunction, CResidualFunction *residual, CResidualGradientFunction *gradient);
		virtual ~CValueGradientCalculator();

		virtual void getGradient(CFeatureList *gradient);
		virtual double getFunctionValue();
	
		virtual void resetGradientCalculator() {};
		
		virtual void setBatchPolicy(CBatchLearningPolicy *policy);
		virtual void setEstimationPolicy(CAgentController *estimationPolicy);
};


class CVResidualGradientCalculator : public CValueGradientCalculator
{
	protected:
		CGradientVFunction *vFunction;
		
		virtual double getValue(CStateCollection *state, CAction *action);
		virtual void getValueGradient(CStateCollection *state, CAction *action, CFeatureList *gradient);
		
		
	public:
		CVResidualGradientCalculator(CGradientVFunction *vFunction, CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger, CResidualFunction *residual, CResidualGradientFunction *gradient);
		CVResidualGradientCalculator(CGradientVFunction *vFunction, CEpisodeHistory *episodeHistory, CRewardFunction *rewardFunction, CResidualFunction *residual, CResidualGradientFunction *gradient);
		virtual ~CVResidualGradientCalculator();
};


class CQResidualGradientCalculator : public CValueGradientCalculator
{
	protected:
		CGradientQFunction *qFunction;
		
		
		virtual double getValue(CStateCollection *state, CAction *action);
		virtual void getValueGradient(CStateCollection *state, CAction *action, CFeatureList *gradient);
		
		
	public:
		CQResidualGradientCalculator(CGradientQFunction *qFunction, CAgentController *estimationPolicy, CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger, CResidualFunction *residual, CResidualGradientFunction *gradient);
		CQResidualGradientCalculator(CGradientQFunction *qFunction, CAgentController *estimationPolicy, CEpisodeHistory *episodeHistory, CRewardFunction *rewardFunction, CResidualFunction *residual, CResidualGradientFunction *gradient);
		virtual ~CQResidualGradientCalculator();
};

class CResidualGradientBatchLearner : public CPolicyEvaluationGradientFunction
{
	protected:
		CGradientLearner *gradientLearner;
	public:
		CResidualGradientBatchLearner(CGradientLearner *gradientLearner, CGradientUpdateFunction *learnData, double treshold = 0.1, int maxEvaluations = 10);
		
		virtual ~CResidualGradientBatchLearner();
		
		virtual void evaluatePolicy(int numEvaluations);
		virtual void resetLearnData();
};
*/
class CBatchDataGenerator : public CParameterObject
{
	protected:
	public:
		CBatchDataGenerator() {};
		virtual ~CBatchDataGenerator() {};
	
		virtual void addInput(CStateCollection *state, CAction *action, double output, double weighting = 1.0) = 0;

		virtual void trainFA() = 0;
		virtual void resetPolicyEvaluation() = 0;

		virtual double getValue(CStateCollection *state, CAction *action) = 0;

		void generateInputData(CEpisodeHistory *logger);
};

class CBatchVDataGenerator : public CBatchDataGenerator
{
protected:
	CDataSet *inputData;
	CDataSet1D *outputData;
	CDataSet1D *weightingData;
	ColumnVector *buffVector;
	
	CAbstractVFunction *vFunction;
	
	CSupervisedLearner *learner;
	CSupervisedWeightedLearner *weightedLearner;

	CBatchVDataGenerator(CSupervisedLearner *learner, int inputDim);
	
	
public:
	CBatchVDataGenerator(CAbstractVFunction *vFunction, CSupervisedLearner *learner);
	CBatchVDataGenerator(CAbstractVFunction *vFunction, CSupervisedWeightedLearner *learner);
	virtual ~CBatchVDataGenerator();

	virtual void init(int numDim);	

	virtual void addInput(CStateCollection *state, CAction *action, double output, double weighting = 1.0);

	virtual void trainFA();
	virtual void resetPolicyEvaluation();	

	virtual double getValue(CStateCollection *state, CAction *action);

	virtual CDataSet *getInputData();
	virtual CDataSet1D *getOutputData();
	virtual CDataSet1D *getWeighting();
};



class CBatchCAQDataGenerator : public CBatchVDataGenerator
{
protected:
	CContinuousActionQFunction *qFunction;
	CStateProperties *properties;

	
public:
	CBatchCAQDataGenerator(CStateProperties *properties, CContinuousActionQFunction *qFunction, CSupervisedLearner *learner);
	virtual ~CBatchCAQDataGenerator();
	
	virtual void addInput(CStateCollection *state, CAction *action, double output);
	
	virtual double getValue(CStateCollection *state, CAction *action);
};


class CBatchQDataGenerator : public CBatchDataGenerator
{
protected:
	std::map<CAction *, CDataSet *> *inputMap;
	std::map<CAction *, CDataSet1D *> *outputMap;
	std::map<CAction *, ColumnVector *> *buffVectorMap;
	std::map<CAction *, CDataSet1D *> *weightedMap;
		

	CQFunction *qFunction;
	CStateProperties *properties;
	CActionSet *actions;	

	CSupervisedQFunctionLearner *learner;
	CSupervisedQFunctionWeightedLearner *weightedLearner;
public:
	CBatchQDataGenerator(CQFunction *qFunction, CSupervisedQFunctionLearner *learner, CStateProperties *inputState = NULL);
	CBatchQDataGenerator(CQFunction *qFunction, CSupervisedQFunctionWeightedLearner *learner, CStateProperties *inputState = NULL);
	CBatchQDataGenerator(CActionSet *actions, CStateProperties *properties);

	void init(CQFunction *qFunction, CActionSet *actions, CStateProperties *properties);

	virtual ~CBatchQDataGenerator();
	
	virtual void addInput(CStateCollection *state, CAction *action, double output, double weighting = 1.0);

	virtual void trainFA();
	virtual void resetPolicyEvaluation();	

	virtual double getValue(CStateCollection *state, CAction *action);

	CDataSet *getInputData(CAction *action);
	CDataSet1D *getOutputData(CAction *action);

	CStateProperties *getStateProperties(CAction *action);
};



class CFittedIteration : public CPolicyEvaluation
{
protected:
	CAgentController *estimationPolicy;
	CBatchDataGenerator *dataGenerator;
	
	CEpisodeHistory *episodeHistory;
	CRewardHistory *rewardLogger;

	CDataCollector *dataCollector;
	CPolicyEvaluation *actorLearner;

	virtual void addResidualInput(CStep *step, CAction *action, double oldV, double newV, double nearestNeighborDistance, CAction *nextHistoryActon = NULL, double nextReward = 0.0);
	
	CPolicyEvaluation *initialPolicyEvaluation;

	virtual double getWeighting(CStateCollection *state, CAction *action);

	virtual double getValue(CStateCollection *state, CAction *action);

	int useResidualAlgorithm;

	virtual void onParametersChanged();
public:
	

	CFittedIteration(CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger, CBatchDataGenerator *dataGenerator);

	virtual ~CFittedIteration();

	

	virtual void doEvaluationTrial();
	virtual void evaluatePolicy(int trials);

	virtual CBatchDataGenerator *createTrainingsData();

	virtual void setDataCollector(CDataCollector *dataCollector);

	virtual void setInitialPolicyEvaluation(CPolicyEvaluation *initialPolicyEvaluation);

	virtual void resetLearnData();

	void setActorLearner(CPolicyEvaluation *actorLearner);
	virtual void evaluatePolicy();
};

class CFittedCAQIteration : public CFittedIteration
{
protected:
	
	
public:
	CFittedCAQIteration(CContinuousActionQFunction *qFunction, CStateProperties *properties, CAgentController *estimationPolicy, CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger, CSupervisedLearner *learner);

	virtual ~CFittedCAQIteration();
};


class CFittedVIteration : public CFittedIteration
{
protected:
	CStochasticPolicy *estimationPolicy;

	virtual double getWeighting(CStateCollection *state, CAction *action);

	double *actionProbabilities;
	CActionSet *availableActions;
public:
	CFittedVIteration(CAbstractVFunction *vFunction, CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger, CSupervisedLearner *learner);


	CFittedVIteration(CAbstractVFunction *vFunction, CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger, CSupervisedWeightedLearner *learner, CStochasticPolicy *estimationPolicy);

	virtual ~CFittedVIteration();

	
};

class CFittedQIteration : public CFittedIteration
{
protected:
	std::map<CAction *, CDataSet *> *inputDatas;
	std::map<CAction *, CDataSet1D *> *outputDatas;
	std::map<CAction *, CKDTree *> *kdTrees;
	std::map<CAction *, CKNearestNeighbors *> *nearestNeighbors;
	std::map<CAction *, CDataPreprocessor *> *dataPreProc;

	std::list<int> *neighborsList;

	CStateProperties *residualProperties;

	int kNN;

	CState *buffState;

	virtual void addResidualInput(CStep *step, CAction *action, double oldV, double newV, double nearestNeighborDistance, CAction *nextHistoryActon = NULL, double nextReward = 0.0);
	
public:
	CFittedQIteration(CQFunction *qFunction, CAgentController *estimationPolicy, CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger, CSupervisedQFunctionLearner *learner, CStateProperties *residualProperties = NULL);

	CFittedQIteration(CQFunction *qFunction, CStateProperties *inputState, CAgentController *estimationPolicy, CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger, CSupervisedQFunctionLearner *learner, CStateProperties *residualProperties = NULL);


	virtual ~CFittedQIteration();

	virtual void doEvaluationTrial();
};



class CFittedQNewFeatureCalculator : public CFittedQIteration, public CNewFeatureCalculatorDataGenerator
{
protected:
	CQFunction *qFunction;
	CQFunction *qFunctionPolicy;

	std::map<CAction *, CFeatureCalculator *> *policyCalculator;
	std::map<CAction *, CFeatureCalculator *> *estimationCalculator;

	CNewFeatureCalculator *newFeatureCalc;
	CStateModifiersObject *agent;
public:
	CFittedQNewFeatureCalculator(CQFunction *qFunction, CQFunction *qFunctionPolicy, CStateProperties *inputState, CAgentController *estimationPolicy, CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger, CNewFeatureCalculator *newFeatCalc);
	
	virtual ~CFittedQNewFeatureCalculator();

	virtual void calculateNewFeatures();

	virtual void swapValueFunctions();

	void clearCalculators();
	virtual void resetData();

	void setStateModifiersObject(CStateModifiersObject *agent);

};

class CActionDistribution;

class CContinuousDynamicProgramming : public CPolicyEvaluation
{
protected:
	CSamplingBasedTransitionModel *transModel;
	
	CActionSet *availableActions;
	

	double *actionValues;
	double *actionProbabilities;

	int numIteration;
public:
	CContinuousDynamicProgramming(CActionSet *allActions, CSamplingBasedTransitionModel *transModel);

	virtual ~CContinuousDynamicProgramming();

	virtual void evaluatePolicy(int numEvaluations);

	virtual double getValueFromDistribution(CActionSet *availableActions, double *actionValues, CActionDistribution *distribution); 

	virtual double getValue(CState *state, CActionSet *availableActions) = 0;
	virtual void updateOutputs(int index, CActionSet *availableActions, double *actionValues) = 0;

	virtual void learn() = 0;

	virtual void resetLearnData();
	virtual void resetDynamicProgramming() = 0;
};

class CContinuousDynamicVProgramming : public CContinuousDynamicProgramming
{
protected:
	CDataSet1D *outputValues;
	CActionDistribution *distribution;

	CSupervisedLearner *learner;
	CAbstractVFunction *vFunction;
public:
	CContinuousDynamicVProgramming(CActionSet *allActions, CActionDistribution *distribution, CSamplingBasedTransitionModel *transModel, CAbstractVFunction *vFunction, CSupervisedLearner *learner);

	virtual ~CContinuousDynamicVProgramming();

	virtual double getValue(CState *state, CActionSet *availableActions);
	virtual void updateOutputs(int index, CActionSet *availableActions, double *actionValues);

	virtual void learn();

	virtual void resetDynamicProgramming();
};

class CContinuousDynamicQProgramming : public CContinuousDynamicProgramming
{
protected:
	std::map<CAction *, CDataSet1D *> *outputValues;
	std::map<CAction *, CDataSet *> *inputValues;
	
	CSupervisedQFunctionLearner *learner;
	CAbstractQFunction *qFunction;

	CActionDistribution *distribution;
	double *actionValues2;
public:
	CContinuousDynamicQProgramming(CActionSet *allActions, CActionDistribution *distribution, CSamplingBasedTransitionModel *transModel, CAbstractQFunction *vFunction, CSupervisedQFunctionLearner *learner);

	virtual ~CContinuousDynamicQProgramming();

	virtual double getValue(CState *state, CActionSet *availableActions);
	virtual void updateOutputs(int index, CActionSet *availableActions, double *actionValues);

	virtual void learn();

	virtual void resetDynamicProgramming();
};

class CContinuousMCQEvaluation : public CContinuousDynamicQProgramming
{
protected:
	CPolicySameStateEvaluator *evaluator;
public:
	CContinuousMCQEvaluation(CActionSet *allActions, CActionDistribution *distribution, CSamplingBasedTransitionModel *transModel, CPolicySameStateEvaluator *evaluator, CSupervisedQFunctionLearner *learner);

	virtual ~CContinuousMCQEvaluation();

	virtual double getValue(CState *state, CActionSet *availableActions);
};


class CGraphTransition;
class DataSubset;
class CKDRectangle;

class CGraphDynamicProgramming : public CPolicyEvaluation
{
protected:
	CSamplingBasedGraph *transModel;
	CDataSet1D *outputValues;
	
	
public:
	bool resetGraph;

	CGraphDynamicProgramming(CSamplingBasedGraph *transModel);

	virtual ~CGraphDynamicProgramming();

	virtual void evaluatePolicy(int numEvaluations);

	virtual void resetLearnData();

	virtual double getValue(int node);
	virtual double getValue(ColumnVector *input);

	virtual CGraphTransition *getMaxTransition(int index, double &maxValue, CKDRectangle *range = NULL); 

	virtual void getNearestNode(ColumnVector *input, int &node, double &distance);

	CSamplingBasedGraph *getGraph();

	CDataSet1D *getOutputValues();

	virtual void saveCSV(string filename, DataSubset *nodeSubset);
};

class CGraphTarget;
class CAdaptiveTargetGraph;

class CGraphAdaptiveTargetDynamicProgramming : public CGraphDynamicProgramming
{
protected:
	std::map<CGraphTarget *, CDataSet1D *> *targetMap;
	std::list<CGraphTarget *> *targets;	

	CGraphTarget *currentTarget;
	CAdaptiveTargetGraph *adaptiveTargetGraph;
public:
	CGraphAdaptiveTargetDynamicProgramming(CAdaptiveTargetGraph *graph);

	~CGraphAdaptiveTargetDynamicProgramming();

	virtual CGraphTransition *getMaxTransition(int index, double &maxValue, CKDRectangle *range = NULL); 

	virtual void addTarget(CGraphTarget *target);
	virtual CGraphTarget *getTargetForState(CStateCollection *state);
	
	virtual void setCurrentTarget(CGraphTarget *target);

	int getNumTargets();
	CGraphTarget *getTarget(int index);

	virtual void resetLearnData();
};



#endif
 
