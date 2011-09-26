//
// C++ Implementation: ctreebatchlearning
//
// Description: 
//
//
// Author: Neumann Gerhard <gerhard@tu-graz.ac.at>, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//

#include "ctreebatchlearning.h"
#include "cforest.h"
#include "crbftrees.h"
#include "clocalregression.h"
#include "cmodeltree.h"
#include "cbatchlearning.h"
#include "cstatecollection.h"
#include "ctreevfunction.h"
#include "crewardmodel.h"
#include "ckdtrees.h"
#include "cnearestneighbor.h"


CExtraRegressionForestTrainer::CExtraRegressionForestTrainer(int numTrees, int K, int n_min, double treshold)
{
	addParameter("NumRegressionTrees", numTrees);
	addParameter("ExtraTreesK", K);
	addParameter("ExtraTreesMaxSamplesPerLeaf", n_min);
	addParameter("ExtraTreesMaxOutputVarianceTreshold", treshold);
}

CExtraRegressionForestTrainer::~CExtraRegressionForestTrainer()
{
}

CRegressionForest *CExtraRegressionForestTrainer::getNewTree( CDataSet *input, CDataSet1D *output, CDataSet1D *weightData)
{
	
	
	int numTrees = (int) getParameter("NumRegressionTrees");
	int K = (int) getParameter("ExtraTreesK");
	int n_min = (int) getParameter("ExtraTreesMaxSamplesPerLeaf");
	double treshold = getParameter("ExtraTreesMaxOutputVarianceTreshold");

	printf("Training Forest (%d trees, %f, %f) with %d (%d) Inputs...\n", numTrees, output->getMean(NULL), output->getVariance(NULL), input->size(), output->size());
	CExtraTreeRegressionForest *extraTree = new CExtraTreeRegressionForest( numTrees, input, output,  K, n_min, treshold, weightData);
	
	
	printf("Average Depth : %f, NumLeaves: %f\n", extraTree->getAverageDepth(), extraTree->getAverageNumLeaves());

	return extraTree;
}

CExtraRegressionForestLearner::CExtraRegressionForestLearner(CRegressionTreeFunction *l_treeFunction, int numTrees, int K, int n_min, double treshold) : CExtraRegressionForestTrainer( numTrees, K, n_min, treshold)
{
	treeFunction = l_treeFunction;	
}
	
CExtraRegressionForestLearner::~CExtraRegressionForestLearner()
{
}
		
void CExtraRegressionForestLearner::learnFA(CDataSet *inputData, CDataSet1D *outputData)
{
	learnWeightedFA(inputData, outputData, NULL);	
}

void CExtraRegressionForestLearner::learnWeightedFA(CDataSet *inputData, CDataSet1D *outputData, CDataSet1D *weightData)
{
	CMapping<double> *tree = treeFunction->getTree();

	if (tree != NULL)
	{
		delete tree;
	}
	CRegressionForest *extraTree = getNewTree(inputData, outputData, weightData);
	treeFunction->setTree( extraTree);
}

void CExtraRegressionForestLearner::resetLearner()
{
/*	CMapping<double> *tree = treeFunction->getTree();

	if (tree != NULL)
	{
		delete tree;
	}
	treeFunction->setTree(NULL);*/
}

/*
CExtraRegressionForestQFunctionLearner::CExtraRegressionForestQFunctionLearner(std::map<CAction *, CRegressionTreeFunction *> *l_functionMap, int numTrees, int K, int n_min, double treshold) : CExtraRegressionForestTrainer( numTrees, K, n_min, treshold)
{
	functionMap = l_functionMap;
}
	
CExtraRegressionForestQFunctionLearner::~CExtraRegressionForestQFunctionLearner()
{
}
	
void CExtraRegressionForestQFunctionLearner::learnQFunction(CAction *action, CDataSet *inputData, CDataSet1D *outputData)
{
	CRegressionTreeFunction *treeFunction = (*functionMap)[action];
	CMapping<double> *tree = treeFunction->getTree();

	if (tree != NULL)
	{
		delete tree;
	}
	CRegressionForest *extraTree = getNewTree(inputData, outputData);
	treeFunction->setTree( extraTree);
}

CExtraRegressionForestFeatureLearner::CExtraRegressionForestFeatureLearner(CStateProperties *l_originalState, int numTrees, int K, int n_min, double treshold) : CExtraRegressionForestTrainer(numTrees, K, n_min, treshold)
{
	originalState = l_originalState;
}
*/


		

CExtraRegressionForestFeatureLearner::~CExtraRegressionForestFeatureLearner()
{
}

CFeatureCalculator *CExtraRegressionForestFeatureLearner::getFeatureCalculator(CFeatureVFunction *vFunction, CDataSet *inputData, CDataSet1D *outputData)
{
	CRegressionForest *forest = getNewTree(inputData, outputData, NULL);
	
	CForestFeatureCalculator<double> *featCalc = new CForestFeatureCalculator<double>(forest);

	featCalc->setOriginalState( originalState);

	CFeatureVRegressionTreeFunction *vRegFunction = dynamic_cast<CFeatureVRegressionTreeFunction *>(vFunction);

	vRegFunction->setForest( forest, featCalc);

	return featCalc;
}


CRBFForestLearner::CRBFForestLearner(CRegressionTreeFunction *l_treeFunction, int numTrees, int kNN, int K, int n_min, double treshold, double varMult, double minVar)
{
	treeFunction = l_treeFunction;

	addParameter("NumRegressionTrees", numTrees);
	addParameter("NumNearestNeighbors", kNN);
	addParameter("ExtraTreesK", K);
	addParameter("ExtraTreesMaxSamplesPerLeaf", n_min);
	addParameter("ExtraTreesMaxOutputVarianceTreshold", treshold);
	addParameter("RBFTreeVarianceMultiplier", varMult);
	addParameter("RBFTreeMinVariance", minVar);
}

CRBFForestLearner::~CRBFForestLearner()
{

}

void CRBFForestLearner::learnFA(CDataSet *input, CDataSet1D *output)
{
	CMapping<double> *tree = treeFunction->getTree();

	if (tree != NULL)
	{
		delete tree;
	}

	
	int numTrees = (int) getParameter("NumRegressionTrees");
	int K = (int) getParameter("ExtraTreesK");
	int kNN = (int) getParameter("NumNearestNeighbors");
	int n_min = (int) getParameter("ExtraTreesMaxSamplesPerLeaf");
	double treshold = getParameter("ExtraTreesMaxOutputVarianceTreshold");

	double varMult = getParameter("RBFTreeVarianceMultiplier");
	double minVar = getParameter("RBFTreeMinVariance");

	printf("Training Tree with %d Inputs...", input->size());

	ColumnVector varMultiplier(treeFunction->getNumDimensions());
	varMultiplier = varMult;

	ColumnVector minVariance(treeFunction->getNumDimensions());
	minVariance = minVar;


	CRBFExtraRegressionForest *forest = new CRBFExtraRegressionForest(numTrees, kNN, input, output, K, n_min, treshold, &varMultiplier, &minVariance);
	
	treeFunction->setTree(forest);
	printf("Average Depth : %f, NumLeaves: %f\n", forest->getAverageDepth(), forest->getAverageNumLeaves());
}


CExtraLinearRegressionModelForestLearner::CExtraLinearRegressionModelForestLearner(CRegressionTreeFunction *l_treeFunction, int numTrees, int K, int n_min, double treshold, int t1, int t2, int t3)
{
	addParameter("NumRegressionTrees", numTrees);
	addParameter("ExtraTreesK", K);
	addParameter("ExtraTreesMaxSamplesPerLeaf", n_min);
	addParameter("ExtraTreesMaxOutputVarianceTreshold", treshold);
	
	addParameter("LinearRegressionModelTreshold1", t1);
	addParameter("LinearRegressionModelTreshold2", t2);
	addParameter("LinearRegressionModelTreshold3", t3);
	addParameter("LinearRegressionModelLambda", 0.01);

	treeFunction = l_treeFunction;
}

CExtraLinearRegressionModelForestLearner::~CExtraLinearRegressionModelForestLearner()
{
}

void CExtraLinearRegressionModelForestLearner::learnFA(CDataSet *input, CDataSet1D *output)
{
	CMapping<double> *tree = treeFunction->getTree();

	if (tree)
	{
		delete tree;
	}	

	int numTrees = (int) getParameter("NumRegressionTrees");
	int K = (int) getParameter("ExtraTreesK");
	int n_min = (int) getParameter("ExtraTreesMaxSamplesPerLeaf");
	double treshold = getParameter("ExtraTreesMaxOutputVarianceTreshold");
	double lambda = getParameter("LinearRegressionModelLambda");


	int t1 = (int)  getParameter("LinearRegressionModelTreshold1");
	int t2 = (int) getParameter("LinearRegressionModelTreshold2");
	int t3 = (int) getParameter("LinearRegressionModelTreshold3");


	printf("Training Model Tree with %d Inputs...", input->size());

	CRegressionMultiMapping *multiMapping = new CRegressionMultiMapping(numTrees, input->getNumDimensions());	
	multiMapping->deleteMappings = true;

	for (int i = 0; i < numTrees; i ++)
	{
		multiMapping->addMapping(i, new CExtraLinearRegressionModelTree(input, output, K, n_min, treshold, t1, t2, t3, lambda));
	}
	printf("done\n");

	treeFunction->setTree(multiMapping);
}

CLocalRBFLearner::CLocalRBFLearner(CRegressionTreeFunction *l_treeFunction, int kNN, double varMult)
{
	treeFunction = l_treeFunction;

	addParameter("NumNearestNeighbors", kNN);
	addParameter("LocalRBFVarianceMultiplier", varMult);
	
	inputData = NULL;
	outputData = NULL;
}

CLocalRBFLearner::~CLocalRBFLearner()
{
	CMapping<double> *tree = treeFunction->getTree();

	if (tree != NULL)
	{
		delete tree;
	}
	
	if (inputData != NULL)
	{
		delete inputData;
		delete outputData;
	}

}

void CLocalRBFLearner::learnFA(CDataSet *input, CDataSet1D *output)
{
	CMapping<double> *tree = treeFunction->getTree();

	if (tree != NULL)
	{
		delete tree;
	}
	if (inputData != NULL)
	{
		delete inputData;
		delete outputData;
		
		delete preprocessor;
	}
	
	inputData = new CDataSet(*input);
	outputData = new CDataSet1D(*output);

	preprocessor = new CMeanStdPreprocessor(inputData);
	
	preprocessor->preprocessDataSet(inputData);
	
	int kNN = (int) getParameter("NumNearestNeighbors");
	
	double varMult = getParameter("LocalRBFVarianceMultiplier");
	
	ColumnVector sigma(treeFunction->getNumDimensions());
	
	sigma = varMult;
	

	CLocalRBFRegression *regression = new CLocalRBFRegression(inputData, outputData, kNN, &sigma);
	regression->setPreprocessor(preprocessor);	

	printf("Finished Building KD Tree\n");	

	treeFunction->setTree(regression);
}

CLocalLinearLearner::CLocalLinearLearner(CRegressionTreeFunction *l_treeFunction, int kNN, int degree)
{
	treeFunction = l_treeFunction;

	addParameter("NumNearestNeighbors", kNN);
	addParameter("LocalLinearRegressionDegree", degree);
	addParameter("LocalLinearRegressionLambda", 0.01);
	
	inputData = NULL;
	outputData = NULL;
}

CLocalLinearLearner::~CLocalLinearLearner()
{
	CMapping<double> *tree = treeFunction->getTree();

	if (tree != NULL)
	{
		delete tree;
	}
	
	if (inputData != NULL)
	{
		delete inputData;
		delete outputData;
	}
}


void CLocalLinearLearner::learnFA(CDataSet *input, CDataSet1D *output)
{
	CMapping<double> *tree = treeFunction->getTree();

	if (tree != NULL)
	{
		delete tree;
	}
	if (inputData != NULL)
	{
		delete inputData;
		delete outputData;
		
		delete preprocessor;
	}
	
	inputData = new CDataSet(*input);
	outputData = new CDataSet1D(*output);

	preprocessor = new CMeanStdPreprocessor(inputData);
	
	preprocessor->preprocessDataSet(inputData);
	
	int kNN = (int) getParameter("NumNearestNeighbors");
	
	int degree = (int) getParameter("LocalLinearRegressionDegree");
	double lambda =	getParameter("LocalLinearRegressionLambda");
	
	CLocalLinearRegression *regression = new CLocalLinearRegression(inputData, outputData, kNN, degree, lambda);
	regression->setPreprocessor(preprocessor);	

	printf("Finished Building KD Tree\n");	

	treeFunction->setTree(regression);
}


CUnknownDataQFunction::CUnknownDataQFunction(CActionSet *actions, CEpisodeHistory *l_logger, CStateProperties *l_properties, double factor) : CAbstractQFunction(actions)
{
	int kNN = 3;

	addParameter("UnknownDataFactor", factor);
	addParameter("UnknownDataNearestNeighbors", kNN);
	addParameter("UnknownDataVarianceMultiplier", 0.05);

	logger = l_logger;
	properties = l_properties;

	dataGenerator = NULL;

	treeMap = new std::map<CAction *, CKDTree *>;
	nnMap = new std::map<CAction *, CKNearestNeighbors *>;
	preMap = new std::map<CAction *, CDataPreprocessor *>;


	distVector = new ColumnVector(kNN);
	
	recalculateTrees();
}

CUnknownDataQFunction::~CUnknownDataQFunction()
{
	clearMaps();

	delete treeMap;
	delete nnMap;
	delete preMap;

	delete distVector;
}

void CUnknownDataQFunction::clearMaps()
{
	if (dataGenerator)
	{
		delete dataGenerator;
	
		CActionSet::iterator it = actions->begin();
		for (; it != actions->end(); it ++)
		{
			delete (*treeMap)[*it];
	
			delete (*nnMap)[*it];

			delete (*preMap)[*it];
		}
		treeMap->clear();
		nnMap->clear();
		preMap->clear();
	}
}

void CUnknownDataQFunction::onParametersChanged()
{
	CAbstractQFunction::onParametersChanged();
	
	delete distVector;
	distVector = new ColumnVector((int) getParameter("UnknownDataNearestNeighbors"));
}

double CUnknownDataQFunction::getValue(CStateCollection *stateCol, CAction *action, CActionData *)
{
	double varMult = getParameter("UnknownDataVarianceMultiplier");
	
	double punishment = getParameter("UnknownDataFactor");

	if (fabs(punishment) <= 0.0001)	
	{
		return 0;
	}

	if (varMult < 0.000001)
	{
		return 0;
	} 

	CKNearestNeighbors *nearestN = (*nnMap)[action];
	CDataSet *dataSet = dataGenerator->getInputData(action);

	CState *state = stateCol->getState( properties);

	//cout << "State: " << state->t();	

	std::list<int> subset;

	int kNN = (int) getParameter("UnknownDataNearestNeighbors");
	
	
	ColumnVector distToState(dataSet->getNumDimensions());

	nearestN->getNearestNeighbors(state, &subset, kNN, distVector);

	if (distVector->element(0) > varMult)
	{
		return punishment;
	}
	return 0;
	/*double value = getUnknownDataValue(distVector);
	
	return value * getParameter("UnknownDataFactor");*/
}

double CUnknownDataQFunction::getUnknownDataValue(ColumnVector *distVector)
{
	double varMult = getParameter("UnknownDataVarianceMultiplier");	
	
	double value = 0.0;

	//printf("Unknown Data Punishment: ");	

	double factor = 1.0;

	for (int i = 0; i < distVector->nrows(); i ++)
	{
	//	printf("(%f) ", distVector->element(i));	
		if (distVector->element(i) < varMult / 2)
		{
			factor = factor / 4;
		}

		if (distVector->element(i) == 0.0)
		{
			break;
		}

		if (distVector->element(i) > varMult)
		{
			value += pow((distVector->element(i) - varMult)  / varMult, 2.0) * factor;
		}
		else
		{
			value += (distVector->element(i) - varMult) * factor * 0.2;
		}
		factor = factor * 0.25;
	}
	//printf("\nValue: %f\n", value);
	return value;
}


void CUnknownDataQFunction::recalculateTrees()
{
	clearMaps();

	dataGenerator = new CBatchQDataGenerator(actions, properties);

	dataGenerator->generateInputData(logger);

	int kNN = (int) getParameter("UnknownDataNearestNeighbors");

	CActionSet::iterator it = actions->begin();
	for (; it != actions->end(); it ++)
	{
		CDataSet *dataSet = dataGenerator->getInputData( *it);
		
		CDataPreprocessor *preProc = new CMeanStdPreprocessor(dataSet);
		
		(*preMap)[*it] = preProc;
		preProc->preprocessDataSet( dataSet);

		(*treeMap)[*it] = new CKDTree(dataSet, 1);
		(*treeMap)[*it]->setPreprocessor(preProc);
		

		(*nnMap)[*it] = new CKNearestNeighbors((*treeMap)[*it], dataSet, kNN);	
	}
}

void CUnknownDataQFunction::resetData()
{
	recalculateTrees();
}

CUnknownDataQFunctionFromLocalRBFRegression::CUnknownDataQFunctionFromLocalRBFRegression(CActionSet *actions, std::map<CAction *, CRegressionTreeVFunction *> *l_regressionMap, double factor) : CAbstractQFunction(actions)
{
	regressionMap = l_regressionMap;
	addParameter("UnknownDataFactor", factor);
	addParameter("UnknownDataNearestNeighbors", 3);

	recalculateFactors = false;
}

CUnknownDataQFunctionFromLocalRBFRegression::~CUnknownDataQFunctionFromLocalRBFRegression()
{
}

double CUnknownDataQFunctionFromLocalRBFRegression::getValue(CStateCollection *stateCol, CAction *action, CActionData *)
{
	CRegressionTreeVFunction *vFunction = (*regressionMap)[action];
	CState *state = stateCol->getState(vFunction->getStateProperties()); 

	double unknownDataValue = 0;
	int unknownNNs = (int) getParameter("UnknownDataNearestNeighbors");
	double factor = getParameter("UnknownDataFactor");

	if (vFunction->getTree() != NULL)
	{
		CLocalRBFRegression *localRegression = dynamic_cast<CLocalRBFRegression *>(vFunction->getTree());

		if (recalculateFactors)
		{
			localRegression->getOutputValue( state);
		}		
		ColumnVector *factors = localRegression->getLastRBFFactors();
		
		for (int i = 0; i < unknownNNs; i ++)
		{
			unknownDataValue += (1 - factors->element(i));
		}
	}
	return unknownDataValue * factor;
}



/*
CTreeBatchPolicyEvaluation::CTreeBatchPolicyEvaluation(CEpisodeHistory *l_episodeHistory, CRewardHistory *l_rewardLogger, CTreeTrainer *l_treeTrainer) : CPolicyEvaluation()
{
	treeTrainer = l_treeTrainer;
	rewardLogger = l_rewardLogger;
	episodeHistory = l_episodeHistory;

	estimationPolicy = NULL;

	addParameter("DiscountFactor", 0.95);
	addParameters(treeTrainer);
	
	dataCollector = NULL;
}

CTreeBatchPolicyEvaluation::~CTreeBatchPolicyEvaluation()
{
}

void CTreeBatchPolicyEvaluation::setDataCollector(CDataCollector *l_dataCollector)
{
	dataCollector = l_dataCollector;
	addParameters(dataCollector);
}

void CTreeBatchPolicyEvaluation::evaluatePolicy(int trials)
{
	for (int i = 0; i < trials; i ++)
	{
		doEvaluationTrial();
	}
}

void CTreeBatchPolicyEvaluation::doEvaluationTrial()
{
	CActionSet *actions = episodeHistory->getActions();
	CStep *step = new CStep(episodeHistory->getStateProperties(), episodeHistory->getStateModifiers(), episodeHistory->getActions());
		
	CActionDataSet *dataSet = new CActionDataSet(actions);
	CActionDataSet *nextDataSet = new CActionDataSet(actions);
			
	printf("Tree Regression Value Calculation, Episodes %d\n", episodeHistory->getNumEpisodes());	

	double error = 0;
	int steps = 0;
	
	double discountFactor = getParameter("DiscountFactor");

	resetPolicyEvaluation();

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
								
			
			double reward = 0;
			reward = rewardEpisode->getReward(i);
									
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
			
			double V = reward + discountFactor * newV;
			
			addInput(step->oldState, step->action, V);
			
		}
	
		//printf("Time for Episode %d: step %f, time for learners %f\n", j, dur1, dur2);		
		
	}
	printf("Finished Creating Training-set\n");

	trainTree();

	CActionSet::iterator it = actions->begin();
	
	for (;it != actions->end(); it ++)
	{
		(*it)->getActionData()->setData(dataSet->getActionData(*it));
	}
	delete step;
	delete dataSet;
	delete nextDataSet;

	if (dataCollector != NULL)
	{
		dataCollector->collectData();
	}
}


double CCAQTreeBatchPolicyEvaluation::getValue(CStateCollection *state, CAction *action)
{
	return qFunction->getValue(state, action);
}

void CCAQTreeBatchPolicyEvaluation::addInput(CStateCollection *state, CAction *action, double output)
{
	qFunction->getInputData(state, action, buffVector);
	inputData->addInput(buffVector);
	outputData->push_back(output);
}

void CCAQTreeBatchPolicyEvaluation::trainTree()
{
	printf("Continuous Action...");
	treeTrainer->setNewTree(qFunction, inputData, outputData);
}

void CCAQTreeBatchPolicyEvaluation::resetPolicyEvaluation()
{
	inputData->clear();
	outputData->clear();
}

CCAQTreeBatchPolicyEvaluation::CCAQTreeBatchPolicyEvaluation(CRegressionTreeQFunction *l_qFunction, CAgentController *l_estimationPolicy, CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger, CTreeTrainer *treeTrainer) : CTreeBatchPolicyEvaluation(episodeHistory, rewardLogger, treeTrainer)
{
	inputData = new CDataSet(qFunction->getNumDimensions());
	outputData = new CDataSet1D();

	qFunction = l_qFunction;

	buffVector = new ColumnVector(qFunction->getNumDimensions());
	estimationPolicy = l_estimationPolicy;
}

CCAQTreeBatchPolicyEvaluation::~CCAQTreeBatchPolicyEvaluation()
{
	delete inputData;
	delete outputData;

	delete buffVector;
}

double CQTreeBatchPolicyEvaluation::getValue(CStateCollection *state, CAction *action)
{
	return qFunction->getValue(state, action);
}

void CQTreeBatchPolicyEvaluation::addInput(CStateCollection *state, CAction *action, double output)
{
	CRegressionTreeFunction *treeFunction = (*functionMap)[action];
	
	treeFunction->getInputData(state, action, buffVector);
	
	CDataSet *inputData = (*inputMap)[action];
	CDataSet1D *outputData = (*outputMap)[action];

	inputData->addInput(buffVector);
	outputData->push_back(output);
}

void CQTreeBatchPolicyEvaluation::trainTree()
{
	std::map<CAction *, CRegressionTreeFunction *>::iterator it = functionMap->begin();
	
	for (int i = 0; it != functionMap->end(); it ++, i++)
	{
		printf("Action %d...", i);
		treeTrainer->setNewTree((*it).second, (*inputMap)[(*it).first], (*outputMap)[(*it).first]);
	}
}

void CQTreeBatchPolicyEvaluation::resetPolicyEvaluation()
{
	std::map<CAction *, CRegressionTreeFunction *>::iterator it = functionMap->begin();
	
	for (; it != functionMap->end(); it ++)
	{
		(*inputMap)[(*it).first]->clear();
		(*outputMap)[(*it).first]->clear();
	}
}

CQTreeBatchPolicyEvaluation::CQTreeBatchPolicyEvaluation(CQFunction *l_qFunction, std::map<CAction *, CRegressionTreeFunction *> *l_functionMap, CAgentController *l_estimationPolicy, CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger, CTreeTrainer *treeTrainer) : CTreeBatchPolicyEvaluation(episodeHistory, rewardLogger, treeTrainer)
{
	inputMap = new std::map<CAction *, CDataSet *>;
	outputMap = new std::map<CAction *, CDataSet1D *>;

	qFunction = l_qFunction;
	functionMap = l_functionMap;

	estimationPolicy = l_estimationPolicy;

	std::map<CAction *, CRegressionTreeFunction *>::iterator it = functionMap->begin();
	
 	buffVector = new ColumnVector((*it).second->getNumDimensions());
	
	for (; it != functionMap->end(); it ++)
	{
		(*inputMap)[(*it).first] = new CDataSet((*it).second->getNumDimensions());
		(*outputMap)[(*it).first] = new CDataSet1D();
	}
}

CQTreeBatchPolicyEvaluation::~CQTreeBatchPolicyEvaluation()
{
	std::map<CAction *, CRegressionTreeFunction *>::iterator it = functionMap->begin();
	
	for (; it != functionMap->end(); it ++)
	{
		delete (*inputMap)[(*it).first];
		delete (*outputMap)[(*it).first];
	}

	delete inputMap;
	delete outputMap;

	delete buffVector;
}*/
