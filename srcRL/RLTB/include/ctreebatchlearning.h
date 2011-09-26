#ifndef C_TREEBATCHLEARNING__H
#define C_TREEBATCHLEARNING__H

#include "cparameters.h"
#include "csupervisedlearner.h"
#include "cqfunction.h"
#include "cstatemodifier.h"

class CDataSet;
class CDataSet1D;
class CDataSubset;
class CDataPreprocessor;
class CRegressionForest;

class CRegressionTreeFunction;

class CStateProperties;
class CFeatureVFunction;

class ActionSet;
class CAction;
class CActionData;

class CEpisodeHistory;

class CStateProperties;
class CState;
class CStateCollection;

class CBatchQDataGenerator;
class CKDTree;
class CKNearestNeighbors;

class CRegressionTreeVFunction;


class CExtraRegressionForestTrainer : virtual public CParameterObject
{
	public:
		CExtraRegressionForestTrainer(int numTrees, int K, int n_min, double treshold);
		virtual ~CExtraRegressionForestTrainer();

		virtual CRegressionForest * getNewTree(CDataSet *input, CDataSet1D *output, CDataSet1D *weightData);
};


class CExtraRegressionForestLearner : public CExtraRegressionForestTrainer, public CSupervisedLearner, public CSupervisedWeightedLearner
{
	protected:
		CRegressionTreeFunction *treeFunction;

	public:
		CExtraRegressionForestLearner(CRegressionTreeFunction *treeFunction, int numTrees, int K, int n_min, double treshold);
		virtual ~CExtraRegressionForestLearner();

		virtual void learnFA(CDataSet *input, CDataSet1D *output);
		virtual void learnWeightedFA(CDataSet *input, CDataSet1D *output, CDataSet1D *weightData);

		virtual void resetLearner();
};

class CExtraRegressionForestFeatureLearner : public CNewFeatureCalculator, public CExtraRegressionForestTrainer
{
	protected:
		CStateProperties *originalState;

	public:
		CExtraRegressionForestFeatureLearner(CStateProperties *originalState, int numTrees, int K, int n_min, double treshold);
		virtual ~CExtraRegressionForestFeatureLearner();

		virtual CFeatureCalculator * getFeatureCalculator(CFeatureVFunction *vFunction, CDataSet *inputData, CDataSet1D *outputData);
};


class CExtraLinearRegressionModelForestLearner : public CSupervisedLearner
{
	protected:
		CRegressionTreeFunction *treeFunction;

	public:
		CExtraLinearRegressionModelForestLearner(CRegressionTreeFunction *treeFunction, int numTrees, int K, int n_min, double treshold, int t1, int t2, int t3);
		virtual ~CExtraLinearRegressionModelForestLearner();

		virtual void learnFA(CDataSet *input, CDataSet1D *output);
};

class CRBFForestLearner : public CSupervisedLearner
{
	protected:
		CRegressionTreeFunction *treeFunction;

	public:
		CRBFForestLearner(CRegressionTreeFunction *treeFunction, int numTrees, int kNN, int K, int n_min, double treshold, double varMult, double minVar);
		virtual ~CRBFForestLearner();

		virtual void learnFA(CDataSet *input, CDataSet1D *output);
};


class CLocalLinearLearner : public CSupervisedLearner
{
	protected:
		CRegressionTreeFunction *treeFunction;

		CDataSet *inputData;
		CDataSet1D *outputData;

		CDataPreprocessor *preprocessor;
	public:
		CLocalLinearLearner(CRegressionTreeFunction *treeFunction, int kNN, int degree);
		virtual ~CLocalLinearLearner();

		virtual void learnFA(CDataSet *input, CDataSet1D *output);
};


class CLocalRBFLearner : public CSupervisedLearner
{
	protected:
		CRegressionTreeFunction *treeFunction;

		CDataSet *inputData;
		CDataSet1D *outputData;

		CDataPreprocessor *preprocessor;
	public:
		CLocalRBFLearner(CRegressionTreeFunction *treeFunction, int kNN, double varMult);
		virtual ~CLocalRBFLearner();

		virtual void learnFA(CDataSet *input, CDataSet1D *output);
};

class CUnknownDataQFunction : public CAbstractQFunction
{
protected:
	CStateProperties *properties;
	std::map<CAction *, CKDTree *> *treeMap;
	std::map<CAction *, CKNearestNeighbors *> *nnMap;

//	std::map<CAction *, ColumnVector *> *bufferMap;
	std::map<CAction *, CDataPreprocessor *> *preMap;

	CEpisodeHistory *logger;

	CBatchQDataGenerator *dataGenerator;
	ColumnVector *distVector;

	void clearMaps();
public:

	CUnknownDataQFunction(CActionSet *actions, CEpisodeHistory *logger, CStateProperties *properties, double factor);

	virtual ~CUnknownDataQFunction();

	virtual double getValue(CStateCollection *state, CAction *action, CActionData *data = NULL);

	void recalculateTrees();

	virtual double getUnknownDataValue(ColumnVector *distances);

	virtual void onParametersChanged();

	virtual void resetData();
};


class CUnknownDataQFunctionFromLocalRBFRegression : public CAbstractQFunction
{
protected:
	std::map<CAction *, CRegressionTreeVFunction *> *regressionMap;
public:
	bool recalculateFactors;

	CUnknownDataQFunctionFromLocalRBFRegression(CActionSet *actions, std::map<CAction *, CRegressionTreeVFunction *> *regressionMap, double factor);

	virtual ~CUnknownDataQFunctionFromLocalRBFRegression();

	virtual double getValue(CStateCollection *state, CAction *action, CActionData *data = NULL);
};



/*
class CTreeBatchPolicyEvaluation : public CPolicyEvaluation
{
protected:
	CAgentController *estimationPolicy;	
	
	CTreeTrainer *treeTrainer;

	CEpisodeHistory *episodeHistory;
	CRewardHistory *rewardLogger;

	CDataCollector *dataCollector;

	virtual double getValue(CStateCollection *state, CAction *action) = 0;
	virtual void addInput(CStateCollection *state, CAction *action, double output) = 0;

	virtual void trainTree() = 0;
	virtual void resetPolicyEvaluation() = 0;
public:
	CTreeBatchPolicyEvaluation(CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger, CTreeTrainer *treeTrainer);

	virtual ~CTreeBatchPolicyEvaluation();

	virtual void doEvaluationTrial();
	virtual void evaluatePolicy(int trials);

	virtual void setDataCollector(CDataCollector *dataCollector);
};

class CCAQTreeBatchPolicyEvaluation : public CTreeBatchPolicyEvaluation
{
protected:
	CDataSet *inputData;
	CDataSet1D *outputData;

	ColumnVector *buffVector;

	CRegressionTreeQFunction *qFunction;

	virtual double getValue(CStateCollection *state, CAction *action);
	virtual void addInput(CStateCollection *state, CAction *action, double output);

	virtual void trainTree();
	virtual void resetPolicyEvaluation();
public:
	CCAQTreeBatchPolicyEvaluation(CRegressionTreeQFunction *qFunction, CAgentController *estimationPolicy, CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger, CTreeTrainer *treeTrainer);

	virtual ~CCAQTreeBatchPolicyEvaluation();
};


class CQTreeBatchPolicyEvaluation : public CTreeBatchPolicyEvaluation
{
protected:
	ColumnVector *buffVector;

	CQFunction *qFunction;

	std::map<CAction *, CRegressionTreeFunction *> *functionMap;
	std::map<CAction *, CDataSet *> *inputMap;
	std::map<CAction *, CDataSet1D *> *outputMap;

	virtual double getValue(CStateCollection *state, CAction *action);
	virtual void addInput(CStateCollection *state, CAction *action, double output);

	virtual void trainTree();
	virtual void resetPolicyEvaluation();
public:
	CQTreeBatchPolicyEvaluation(CQFunction *qFunction, std::map<CAction *, CRegressionTreeFunction *> *functionMap, CAgentController *estimationPolicy, CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger, CTreeTrainer *treeTrainer);

	virtual ~CQTreeBatchPolicyEvaluation();
};
*/
#endif

