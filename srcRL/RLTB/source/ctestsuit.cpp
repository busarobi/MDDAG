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

#include <time.h>
#include <stdio.h>
#include <map>

#include "cpolicies.h"
#include "cagent.h"
#include "ril_debug.h"
#include "ctdlearner.h"
#include "clinearfafeaturecalculator.h"
#include "cvfunctionlearner.h"
#include "crewardmodel.h"
#include "ccontinuoustime.h"
#include "ctestsuit.h"
#include "cvfunction.h"
#include "cparameters.h"
#include "cagentlogger.h"
#include "cexploration.h"
#include "cbatchlearning.h"
#include "cresiduals.h"
#include "canalyzer.h"
#include "cevaluator.h"
#include "cpolicygradient.h"
#include "cvfunctionfromqfunction.h"
#include "csamplingbasedmodel.h"

#include "cstate.h"
#include "cstateproperties.h"
#include "cstatecollection.h"

#include <math.h>

#include <iostream>
#include <sstream>
#include <string>

CTestSuiteEvaluatorLogger::CTestSuiteEvaluatorLogger(string l_outputDirectory) : outputDirectory(l_outputDirectory)
{
	nEpisodesBeforeEvaluate = 1;
}

void CTestSuiteEvaluatorLogger::setOutputDirectory(string l_outputDirectory)
{
	outputDirectory = string(l_outputDirectory);
}

void CTestSuiteEvaluatorLogger::startNewEvaluation(string , CParameters *, int )
{
}

CTestSuiteLoggerFromEvaluator::CTestSuiteLoggerFromEvaluator(string outputDirectory, string l_outputFileName, CEvaluator *l_evaluator) : CTestSuiteEvaluatorLogger(outputDirectory), outputFileName(l_outputFileName) 
{
	evaluator = l_evaluator;
}
	
void CTestSuiteLoggerFromEvaluator::startNewEvaluation(string evaluationDirectory, CParameters *, int trial)
{
	stringstream sstream;
	
	sstream << evaluationDirectory  << outputDirectory << "/" << outputFileName << "_" << trial << ".data";
	string filename = sstream.str();
	
	FILE *file = fopen(filename.c_str(), "w");
	if (file == NULL)
	{
		printf("Couldn't create file %s ... please create the directory !!\n", filename.c_str());
	}
	
	fprintf(file, "\n");
	fclose(file);
}

double CTestSuiteLoggerFromEvaluator::evaluateValue(string evaluationDir, int trial, int numEpisode)
{
	stringstream sstream;
	
	sstream << evaluationDir  << outputDirectory << "/" << outputFileName << "_" << trial << ".data";
	string filename = sstream.str();
	
	double value = evaluator->evaluate();
	
	FILE *file = fopen(filename.c_str(), "a");
	fprintf(file, "%d %f\n", numEpisode, value);
	fclose(file);
	return value;
}

void CTestSuiteLoggerFromEvaluator::evaluate(string evaluationDir, int trial, int numEpisodes)
{
	evaluateValue(evaluationDir, trial, numEpisodes);
}

		

CMatlabEpisodeOutputLogger::CMatlabEpisodeOutputLogger(CAgent *l_agent, CRewardFunction *l_rewardFunction, CStateProperties *l_modifier, CActionSet *l_actions, int nEpisodes, int nSteps) : CTestSuiteEvaluatorLogger("Episodes")
{
	agent = l_agent;

	actions = l_actions;
	modifier = l_modifier;	
	
	this->nEpisodes = nEpisodes;
	this->nSteps = nSteps;
	
	rewardFunction = l_rewardFunction;
}

CMatlabEpisodeOutputLogger::~CMatlabEpisodeOutputLogger()
{
}
		
void CMatlabEpisodeOutputLogger::evaluate(string evaluationDir, int trial, int numEpisode)
{
	
	int numEval = numEpisode / nEpisodesBeforeEvaluate;
	printf("Episode Ouput numEval %d\n", numEval);
	
	
	stringstream sstream;
	
	sstream << evaluationDir << outputDirectory << "/" << "Trial" << trial << "/Episodes" << "_" << numEval << ".data";
	string filename = sstream.str();
	
			
	FILE *file = fopen(filename.c_str(), "w");
	
	if (file == NULL)
	{
		printf("Couldn't create file %s ... please create the directory !!\n", filename.c_str());
	}
	
	
	CEpisodeMatlabOutput *output = new 	CEpisodeMatlabOutput(modifier, rewardFunction, actions, file); 
	
	output->nEpisodes = numEpisode;
	agent->addSemiMDPListener(output);
	
	for (int i = 0; i < nEpisodes; i ++)
	{
		agent->startNewEpisode();
		agent->doControllerEpisode(1, nSteps);
	}
	agent->removeSemiMDPListener(output);

	delete output;
	fclose(file);
	
}

void CMatlabEpisodeOutputLogger::startNewEvaluation(string evaluationDirectory, CParameters *, int trial)
{
	stringstream sstream;
	
	sstream << "mkdir " << evaluationDirectory << outputDirectory << "/" << "Trial" << trial;
	string scall = sstream.str();
	system(scall.c_str());
}

CMatlabVAnalyzerLogger::CMatlabVAnalyzerLogger(CAbstractVFunction *l_vFunction, CFeatureCalculator *featCalc, CErrorSender *l_vLearner, CStateList *l_states, int l_dim1, int l_dim2, int l_part1, int l_part2, std::list<CStateModifier *> *l_modifiers) : CTestSuiteEvaluatorLogger("Analyzer")
{
	modifiers = l_modifiers;

	vFunction = l_vFunction;
	vLearner = l_vLearner;
	
	visitCounter = new CFeatureVFunction(featCalc);
	averageError = new CFeatureVFunction(featCalc);
	averageVariance = new CFeatureVFunction(featCalc);

	visitCounterLearner = new CVisitStateCounter(visitCounter);
		
	errorLearner = new CVAverageTDErrorLearner(averageError, 0.99);
	varianceLearner = new CVAverageTDVarianceLearner(averageVariance, 0.99);

	vLearner->addErrorListener(errorLearner);
	vLearner->addErrorListener(varianceLearner);

	dim1 = l_dim1;
	dim2 = l_dim2;		

	part1 = l_part1;
	part2 = l_part2;
	
	states = l_states;
}


CMatlabVAnalyzerLogger::~CMatlabVAnalyzerLogger()
{
	delete visitCounter;
	delete averageError;
	delete averageVariance;

	delete visitCounterLearner;
		
	delete errorLearner;
	delete varianceLearner;
	
	vLearner->removeErrorListener(errorLearner);
	vLearner->removeErrorListener(varianceLearner);
}
		
void CMatlabVAnalyzerLogger::evaluate(string evaluationDir, int trial, int numEpisode)
{
	int numEval = numEpisode / nEpisodesBeforeEvaluate;
	
	CStateProperties *properties = states->getStateProperties();
	CState modelState(properties);
	
	CVFunctionAnalyzer *analyzer = new CVFunctionAnalyzer(vFunction, properties, modifiers);
	
	for (unsigned int j = 0; j < states->getNumStates(); j ++)
	{
		states->getState(j, &modelState);
		
		char filename[200];
				
		sprintf(filename, "%s%s/Trial%d/vFunction_State%d_%d.data", evaluationDir.c_str(), outputDirectory.c_str(), trial, j, numEval);

		analyzer->save2DValues(filename, &modelState, dim1, part1, dim2, part2);

		sprintf(filename, "%s%s/Trial%d/StateVisits_State%d_%d.data", evaluationDir.c_str(), outputDirectory.c_str(), trial, j, numEval);

		analyzer->setVFunction(visitCounter);
		analyzer->save2DValues(filename, &modelState, dim1, part1, dim2, part2);

		sprintf(filename, "%s%s/Trial%d/AverageError_State%d_%d.data", evaluationDir.c_str(), outputDirectory.c_str(), trial, j, numEval);

		analyzer->setVFunction(averageError);
		analyzer->save2DValues(filename, &modelState, dim1, part1, dim2, part2);

		sprintf(filename, "%s%s/Trial%d/AverageVariance_State%d_%d.data", evaluationDir.c_str(), outputDirectory.c_str(), trial, j, numEval);

		analyzer->setVFunction(averageVariance);
		analyzer->save2DValues(filename, &modelState, dim1, part1, dim2, part2);
	}
	delete analyzer;
}

void CMatlabVAnalyzerLogger::startNewEvaluation(string evaluationDirectory, CParameters *, int trial)
{
	visitCounter->resetData();
	averageError->resetData();
	averageVariance->resetData();	
	
	stringstream sstream;
	
	sstream << "mkdir " << evaluationDirectory << outputDirectory << "/" << "Trial" << trial;
	string scall = sstream.str();
	system(scall.c_str());
}

void CMatlabVAnalyzerLogger::addListenersToAgent(CSemiMDPSender *agent)
{
	agent->addSemiMDPListener(visitCounterLearner);
	
}

void CMatlabVAnalyzerLogger::removeListenersToAgent(CSemiMDPSender *agent)
{
	agent->removeSemiMDPListener(visitCounterLearner);
}

CMatlabQAnalyzerLogger::CMatlabQAnalyzerLogger(CFeatureQFunction *l_qFunction, CFeatureCalculator *featCalc, CErrorSender *l_vLearner, CStateList *l_states, int l_dim1, int l_dim2, int l_part1, int l_part2, std::list<CStateModifier *> *l_modifiers) : CMatlabVAnalyzerLogger(new COptimalVFunctionFromQFunction(l_qFunction, featCalc), featCalc, l_vLearner, l_states, l_dim1, l_dim2, l_part1, l_part2, l_modifiers)
{
	qFunction = l_qFunction;
	saVisits = new CFeatureQFunction(qFunction->getActions(), featCalc);
		
	visitStateActionCounterLearner = new CVisitStateActionCounter(saVisits);
}

CMatlabQAnalyzerLogger::CMatlabQAnalyzerLogger(CFeatureVFunction *vFunction, CFeatureQFunction *l_qFunction, CFeatureCalculator *featCalc, CErrorSender *l_vLearner, CStateList *l_states, int l_dim1, int l_dim2, int l_part1, int l_part2, std::list<CStateModifier *> *l_modifiers) : CMatlabVAnalyzerLogger(vFunction, featCalc, l_vLearner, l_states, l_dim1, l_dim2, l_part1, l_part2, l_modifiers)
{
	qFunction = l_qFunction;
	saVisits = new CFeatureQFunction(qFunction->getActions(), featCalc);
		
	visitStateActionCounterLearner = new CVisitStateActionCounter(saVisits);
}


CMatlabQAnalyzerLogger::~CMatlabQAnalyzerLogger()
{
	delete vFunction;
	delete saVisits;
	delete visitStateActionCounterLearner;
}
		
void CMatlabQAnalyzerLogger::evaluate(string evaluationDir, int trial, int numEpisode)
{
	CMatlabVAnalyzerLogger::evaluate(evaluationDir, trial, numEpisode);
	
	int numEval = numEpisode / nEpisodesBeforeEvaluate;
	
	CStateProperties *properties = states->getStateProperties();
	CState modelState(properties);
	
	CQFunctionAnalyzer *analyzer = new CQFunctionAnalyzer(qFunction, properties, modifiers);
	
	for (unsigned int j = 0; j < states->getNumStates(); j ++)
	{
		states->getState(j, &modelState);
		
		char filename[200];
				
		sprintf(filename, "%s%s/Trial%d/qFunction_State%d_%d.data", evaluationDir.c_str(), outputDirectory.c_str(), trial, j, numEval);

		analyzer->save2DValues(filename, qFunction->getActions(), &modelState, dim1, part1, dim2, part2);

		sprintf(filename, "%s%s/Trial%d/StateActionVisits_State%d_%d.data", evaluationDir.c_str(), outputDirectory.c_str(), trial, j, numEval);

		analyzer->setQFunction(saVisits);
		analyzer->save2DValues(filename, qFunction->getActions(), &modelState, dim1, part1, dim2, part2);
	
	}
	saVisits->saveDataToFile("Analyzer/debug.data");
	qFunction->saveDataToFile("Analyzer/debug1.data");
	vFunction->saveDataToFile("Analyzer/debug2.data");
	visitCounter->saveDataToFile("Analyzer/debug3.data");
	delete analyzer;
}

void CMatlabQAnalyzerLogger::startNewEvaluation(string evaluationDirectory, CParameters *parameters, int trial)
{
	saVisits->resetData();
	CMatlabVAnalyzerLogger::startNewEvaluation(evaluationDirectory, parameters, trial);
}

void CMatlabQAnalyzerLogger::addListenersToAgent(CSemiMDPSender *agent)
{
	CMatlabVAnalyzerLogger::addListenersToAgent(agent);
	agent->addSemiMDPListener(visitStateActionCounterLearner);
	
}

void CMatlabQAnalyzerLogger::removeListenersToAgent(CSemiMDPSender *agent)
{
	CMatlabVAnalyzerLogger::removeListenersToAgent(agent);
	agent->removeSemiMDPListener(visitStateActionCounterLearner);
}

CTestSuite::CTestSuite(CAgent *agent, CAgentController *controller, CLearnDataObject *learnDataObject, char *l_testSuiteName) :testSuiteName(l_testSuiteName)
{
	this->agent = agent;
	this->learnDataObjects = new std::list<CLearnDataObject *>();
	saveLearnData = new std::map<CLearnDataObject *, bool>();
	
	paramCalculators = new std::list<CAdaptiveParameterCalculator *>();
	
	if (learnDataObject)
	{
		addLearnDataObject(learnDataObject, true);
	}

	addParameters(controller);

	this->controller = controller;
	this->evaluationController = controller;
}

CTestSuite::CTestSuite(CAgent *agent, CAgentController *controller, CAgentController *evaluationController, CLearnDataObject *learnDataObject, char *l_testSuiteName) : testSuiteName(l_testSuiteName)
{
	this->agent = agent;
	this->learnDataObjects = new std::list<CLearnDataObject *>();
	saveLearnData = new std::map<CLearnDataObject *, bool>();

	paramCalculators = new std::list<CAdaptiveParameterCalculator *>();

	if (learnDataObject)
	{
		addLearnDataObject(learnDataObject, true);
	}

	addParameters(controller);
	addParameters(evaluationController);

	

	this->controller = controller;
	this->evaluationController = evaluationController;
}

CTestSuite::~CTestSuite()
{
	delete learnDataObjects;
	delete saveLearnData;
	delete paramCalculators;
}

void CTestSuite::deleteObjects()
{
	if (evaluationController != controller)
	{
		delete evaluationController;
	}
	delete controller;

	std::list<CLearnDataObject *>::iterator it = learnDataObjects->begin();
	for (; it != learnDataObjects->end(); it ++)
	{
		delete *it;
	}	
}


void CTestSuite::addParamCalculator(CAdaptiveParameterCalculator *paramCalculator)
{
	addParameters(paramCalculator);
	paramCalculators->push_back(paramCalculator);
	
}

void CTestSuite::resetParamCalculators()
{
	std::list<CAdaptiveParameterCalculator *>::iterator it = paramCalculators->begin();
	for (; it != paramCalculators->end(); it ++)
	{
		(*it)->resetCalculator();
	}	

}



CAgentController *CTestSuite::getEvaluationController()
{
	return evaluationController;
}

void CTestSuite::setEvaluationController(CAgentController *l_evaluationController)
{
	this->evaluationController = l_evaluationController;
}


void CTestSuite::saveLearnedData(FILE *stream)
{
	std::list<CLearnDataObject *>::iterator it = learnDataObjects->begin();

	for (; it != learnDataObjects->end(); it ++)
	{
		if ((*saveLearnData)[*it])
		{
			(*it)->saveData(stream);
		}
	}
}

void CTestSuite::loadLearnedData(FILE *stream)
{
	std::list<CLearnDataObject *>::iterator it = learnDataObjects->begin();

	for (; it != learnDataObjects->end(); it ++)
	{
		if ((*saveLearnData)[*it])
		{
			(*it)->loadData(stream);
		}
	}	
}

void CTestSuite::resetLearnedData()
{
	std::list<CLearnDataObject *>::iterator it = learnDataObjects->begin();


	for (; it != learnDataObjects->end(); it ++)
	{
		(*it)->resetData();
	}

	resetParamCalculators();
}

string CTestSuite::getTestSuiteName()
{
	return testSuiteName;
}

void CTestSuite::setTestSuiteName(string name)
{
	testSuiteName = name;
}


void CTestSuite::addLearnDataObject(CLearnDataObject *learnDataObject, bool l_saveLearnData)
{
	learnDataObjects->push_back(learnDataObject);
	(*saveLearnData)[learnDataObject] = l_saveLearnData;
}

CAgentController *CTestSuite::getController()
{
	return controller;
}

void CTestSuite::setController(CAgentController *controller)
{
	this->controller = controller;
}


CListenerTestSuite::CListenerTestSuite(CAgent *agent, CSemiMDPListener *learner, CAgentController *controller, CLearnDataObject *vFunction, char *testSuiteName) : CTestSuite(agent, controller, vFunction, testSuiteName)
{
	this->learnerObjects = new std::list<CSemiMDPListener *>();
	addToAgent = new std::map<CSemiMDPListener *, CSemiMarkovDecisionProcess *>();

	if (learner != NULL)
	{
		learnerObjects->push_back(learner);
		addParameters(learner);
	}

}

CListenerTestSuite::CListenerTestSuite(CAgent *agent, CSemiMDPListener *learner, CAgentController *controller, CAgentController *evaluationController, CLearnDataObject *vFunction, char *testSuiteName) : CTestSuite(agent, controller, evaluationController, vFunction, testSuiteName)
{
	this->learnerObjects = new std::list<CSemiMDPListener *>();
	addToAgent = new std::map<CSemiMDPListener *, CSemiMarkovDecisionProcess *>();

	if (learner != NULL)
	{
		learnerObjects->push_back(learner);
		addParameters(learner);
	}
}

CListenerTestSuite::~CListenerTestSuite()
{
	delete learnerObjects;
	delete addToAgent;
}

void CListenerTestSuite::deleteObjects()
{
	std::list<CSemiMDPListener *>::iterator it = learnerObjects->begin();
	for (;it != learnerObjects->end(); it ++)
	{
		delete *it;
	}
}

void CListenerTestSuite::addLearnerObject(CSemiMDPListener *listener, bool addParams,bool addBack, CSemiMarkovDecisionProcess *addAgent)
{
	if (addParams)
	{
		addParameters(listener);
	}
	if (addBack)
	{
		learnerObjects->push_back(listener);
		(*addToAgent)[listener] = addAgent;
	}
	else
	{
		learnerObjects->push_front(listener);
		(*addToAgent)[listener] = addAgent;
	}

}

void CListenerTestSuite::addLearnersToAgent()
{
	std::list<CSemiMDPListener *>::iterator it = learnerObjects->begin();

	for (; it != learnerObjects->end(); it ++)
	{
		(*it)->enabled = true;
		if ((*addToAgent)[(*it)] == NULL)
		{
			agent->addSemiMDPListener(*it);
		}
		else
		{
			(*addToAgent)[(*it)]->addSemiMDPListener(*it);
		}
	}
}

void CListenerTestSuite::removeLearnersFromAgent()
{
	std::list<CSemiMDPListener *>::iterator it = learnerObjects->begin();

	for (; it != learnerObjects->end(); it ++)
	{
		if ((*addToAgent)[(*it)] == NULL)
		{
			agent->removeSemiMDPListener(*it);
		}
		else
		{
			(*addToAgent)[(*it)]->removeSemiMDPListener(*it);
		}
		//(*it)->enabled = false;
	}
}

void CListenerTestSuite::learn(int nEpisodes, int nStepsPerEpisode)
{
	addLearnersToAgent();
	agent->setController(controller);
	for (int i = 0; i < nEpisodes; i ++)
	{
		agent->startNewEpisode();
		agent->doControllerEpisode(1, nStepsPerEpisode);
	}
	removeLearnersFromAgent();
}

CPolicyEvaluationTestSuite::CPolicyEvaluationTestSuite(CAgent *agent, CPolicyEvaluation *learner, CAgentController *controller, CLearnDataObject *vFunction, char *testSuiteName) : CTestSuite(agent, controller, vFunction, testSuiteName)
{
	evaluation = learner;
	addParameters(evaluation);
}

CPolicyEvaluationTestSuite::~CPolicyEvaluationTestSuite()
{
}

void CPolicyEvaluationTestSuite::learn(int, int)
{
	agent->setController(controller);
	evaluation->evaluatePolicy();
}

void CPolicyEvaluationTestSuite::resetLearnedData()
{
	CTestSuite::resetLearnedData();
	
	evaluation->resetLearnData();
	
	
}

CPolicyIterationTestSuite::CPolicyIterationTestSuite(CAgent *agent, CPolicyIteration *learner, CAgentController *controller, CLearnDataObject *vFunction, char *testSuiteName) : CTestSuite(agent, controller, vFunction, testSuiteName)
{
	policyIteration = learner;
	addParameters(policyIteration);
}

CPolicyIterationTestSuite::~CPolicyIterationTestSuite()
{
}

void CPolicyIterationTestSuite::learn(int, int)
{
	agent->setController(controller);
	policyIteration->doPolicyIterationStep();
}

void CPolicyIterationTestSuite::resetLearnedData()
{
	CTestSuite::resetLearnedData();
	policyIteration->initPolicyIteration();
}
						  
CPolicyGradientTestSuite::CPolicyGradientTestSuite(CAgent *agent, CGradientLearner *learner, CAgentController *controller, CLearnDataObject *vFunction, char *testSuiteName, int maxGradientUpdates) : CTestSuite(agent, controller, vFunction, testSuiteName)
{
	this->learner = learner;
	addParameters(learner);

	addParameter("MaxGradientUpdates", maxGradientUpdates);
}


CPolicyGradientTestSuite::CPolicyGradientTestSuite(CAgent *agent, CGradientLearner *learner, CAgentController *controller, CAgentController *evaluationController, CLearnDataObject *vFunction, char *testSuiteName, int maxGradientUpdates) : CTestSuite(agent, controller, evaluationController, vFunction, testSuiteName)
{
	this->learner = learner;
	addParameters(learner);
	addParameter("MaxGradientUpdates", maxGradientUpdates);
}
	
CPolicyGradientTestSuite::~CPolicyGradientTestSuite()
{

}

void CPolicyGradientTestSuite::deleteObjects()
{
	if (learner)
	{
		delete learner;
	}
}


void CPolicyGradientTestSuite::learn(int , int )
{
	agent->setController(controller);
	learner->doOptimization((int) getParameter("MaxGradientUpdates"));
}

void CPolicyGradientTestSuite::resetLearnedData()
{
	CTestSuite::resetLearnedData();
	
	learner->resetOptimization();
}

CTestSuiteCollection::CTestSuiteCollection()
{
	testSuiteMap = new std::map<string, CTestSuite *>();
	objectsToDelete = new std::list<void *>();
}

CTestSuiteCollection::~CTestSuiteCollection()
{
	std::map<string, CTestSuite *>::iterator it = testSuiteMap->begin();
	for (; it != testSuiteMap->end(); it ++)
	{
		delete (*it).second;
	}
		
	deleteObjects();
	delete objectsToDelete;
	delete testSuiteMap;
}

void CTestSuiteCollection::addObjectToDelete(void *object)
{
	objectsToDelete->push_back(object);
}

void CTestSuiteCollection::deleteObjects()
{
	std::list<void *>::iterator it = objectsToDelete->begin();
	/*for (; it != objectsToDelete->end(); it ++)
	{
		delete (*it);
	}*/
	
}

void CTestSuiteCollection::addTestSuite(CTestSuite *testSuite)
{
	(*testSuiteMap)[testSuite->getTestSuiteName()] = testSuite;
}

void CTestSuiteCollection::removeTestSuite(CTestSuite *testSuite)
{
	(*testSuiteMap)[testSuite->getTestSuiteName()] = NULL;
}

int CTestSuiteCollection::getNumTestSuites()
{
	return testSuiteMap->size();
}

CTestSuite *CTestSuiteCollection::getTestSuite(string testSuiteName)
{
	return (*testSuiteMap)[testSuiteName];
}

CTestSuite *CTestSuiteCollection::getTestSuite(int index)
{
	CTestSuite *testSuite = NULL;
	std::map<string, CTestSuite *>::iterator it = testSuiteMap->begin();

	for (int i = 0; it != testSuiteMap->end(); it ++, i ++)
	{
		if ( i == index)
		{
			testSuite = (*it).second;
			break;
		}
	}
	return testSuite;
}

void CTestSuiteCollection::removeAllTestSuites()
{
	std::map<string, CTestSuite *>::iterator it = testSuiteMap->begin();

	for (int i = 0; it != testSuiteMap->end(); it ++, i++)
	{
		CTestSuite *testSuite = (*it).second;
		delete testSuite;
	}
	testSuiteMap->clear();
}

/*
CTestSuite::CTestSuite(int Type)
{
	this->Type = Type;
	properties = new CTestSuiteProperties();

	properties->addParameter("Alpha", 0.2);

	controller = NULL;
	learner = NULL;
}

CTestSuite::~CTestSuite()
{
	delete properties;
}

void CTestSuite::addLearner(CAgent *agent)
{
	agent->addSemiMDPListener(learner);
}

void CTestSuite::removeLearner(CAgent *agent)
{
	agent->removeSemiMDPListener(learner);
}

CAgentController *CTestSuite::getController()
{
	return controller;
}


CTestSuiteProperties *CTestSuite::getTestSuiteProperties()
{
	return properties;
}
*/


CTestSuiteEvaluator::CTestSuiteEvaluator(CAgent *l_agent, string l_baseDirectory, CTestSuite *l_testSuite, int l_nTrials) : baseDirectory(l_baseDirectory)
{
	
	agent = l_agent;
	nTrials = l_nTrials;
	
	testSuite = l_testSuite;
	
	evaluators = new std::list<CTestSuiteEvaluatorLogger *>();

	trialNumber = 0;
	
	parameterList = new std::list<CParameters *>();
	evaluations = new std::map<CParameters *, EvaluationValues *>();
	
	printf("Base %s\n", baseDirectory.c_str());
	string fileName = baseDirectory + string("/") + testSuite->getTestSuiteName() + string("/EvaluationData.data");
	loadEvaluationData(fileName);
	
	addParameter("TestSuiteEvaluator", 0.0);
}

CTestSuiteEvaluator::~CTestSuiteEvaluator()
{
	std::map<CParameters *, EvaluationValues *>::iterator it = evaluations->begin();
	for (; it != evaluations->end(); it++)
	{
		delete (*it).second;
	}
	std::list<CParameters *>::iterator it2 = parameterList->begin();
	for (; it2 != parameterList->end(); it2++)
	{
		delete *it2;
	}
	
	delete evaluators;

	delete parameterList;
	
	
	delete evaluations;
}

void CTestSuiteEvaluator::evaluateParameters(CParameters *testSuiteParam)
{
	
	CParameters *targetParams = getParametersObject(testSuiteParam);
	EvaluationValues *evalValues = NULL;
	if (targetParams != NULL)
	{
		evalValues = (*evaluations)[targetParams];
	}
	else
	{
		targetParams = new CParameters(*testSuiteParam);
		targetParams->addParameters(this);
		
		parameterList->push_back(targetParams);
		
		evalValues = new EvaluationValues();
		(*evaluations)[targetParams] = evalValues;
	}
	
	printf("Evaluating Parameter set:\n");
	targetParams->saveParameters(stdout);
	printf("Found %d Trials... Need to evaluate %d Trials\n", evalValues->size(), nTrials - evalValues->size());
	
	while (evalValues->size() < nTrials)
	{
		EvaluationValue value;
		std::stringstream sstream;
		value.trialNumber = getNewTrialNumber();
				
		doEvaluationTrial(testSuite, &value);
		evalValues->push_back(value);
		string fileName = baseDirectory + string("/") + testSuite->getTestSuiteName() + string("/EvaluationData.data");
		saveEvaluationData(fileName);
		
		printf("Evaluation of Trial %d finished: Av: %f, Best %f\n", value.trialNumber, value.averageValue, value.bestValue);
	}
}
		
string CTestSuiteEvaluator::getEvaluationDirectory()
{
	return baseDirectory + string("/") +  testSuite->getTestSuiteName() + string("/");
}
	
int CTestSuiteEvaluator::getNewTrialNumber()
{
	trialNumber ++;	
	return trialNumber - 1;
}
	
string CTestSuiteEvaluator::getLearnDataFileName(int trialNumber)
{

	char filename[500];
	sprintf(filename, "%s/%s/LearnData/Trial_%d.data", baseDirectory.c_str(), testSuite->getTestSuiteName().c_str(), trialNumber);
	return string(filename);
}

void CTestSuiteEvaluator::loadEvaluationData(string fileName)
{
	FILE *parameterFile = fopen(fileName.c_str(), "r");
	
	if (parameterFile == NULL)
	{
		printf("Parameter File not found %s\n", fileName.c_str());
		return;
	}
	int gesTrials = 0;
	int tmp = 0;
	while (!feof(parameterFile))
	{
		char buffer[256];

		int results = fscanf(parameterFile, "%s\n", buffer);
		while (results == 1 && strcmp(buffer, "<testsuiteevaluation>") != 0 && !feof(parameterFile)) 
		{
			results =  fscanf(parameterFile, "%s\n", buffer);
		}

		if (feof(parameterFile))
		{
			break;
		}

		CParameters *parameters = new CParameters();
		parameters->loadParametersXML(parameterFile);

		std::list<CParameters *>::iterator it = parameterList->begin();
	
		for (; it != parameterList->end(); it ++)
		{
			if ( *(*it) == *parameters)
			{
				printf("Loading Evaluation File ... Duplicate Parameters !!\n");
				parameters->saveParameters(stdout);
			}
		}

		if (it == parameterList->end())
		{
			int nTrials = 0;
			int results = fscanf(parameterFile, "<evaluationdata nTrials = %d>\n", &nTrials);

			if (feof(parameterFile) || results != 1)
			{
				delete parameters;
				break;
			}

			parameterList->push_back(parameters);
			EvaluationValues *values = new EvaluationValues();
			(*evaluations)[parameters] = values;
			

			
			

			gesTrials += nTrials; 

			for (int i = 0; i < nTrials; i ++)
			{
				EvaluationValue value;

				tmp = fscanf(parameterFile, "<trial number = %d >\n", &(value.trialNumber));
				tmp = fscanf(parameterFile, "Average Value = %lf, Best Value = %lf\n", &(value.averageValue), &(value.bestValue));
				tmp = fscanf(parameterFile, "EvaluationTime = %lf\n",  &(value.evaluationTime));
				char date[200];
				char buffDate[50];

				date[0] = '\0';				
				tmp = fscanf(parameterFile, "EvaluationDate = %s\n",  buffDate);
				do
				{
					sprintf(date, "%s %s", date, buffDate);
					tmp = fscanf(parameterFile, "%s\n",  buffDate);
					//printf("Part of Date:_%s_\n", buffDate);
				}
				while (strcmp(buffDate, "</trial>") != 0);
				
				value.evaluationDate = string(date);
				
				//fscanf(parameterFile, "</trial>\n");
				
				printf("Loaded trial %d : Average Value %f, Best %f, time %f, date: %s\n", value.trialNumber, value.averageValue, value.bestValue, value.evaluationTime, date);
		
				if(value.trialNumber >= trialNumber)
				{
					trialNumber = value.trialNumber + 1;
				}
				
				values->push_back(value);
			}
			tmp = fscanf(parameterFile, "</evaluationdata>\n");
		}			
		tmp = fscanf(parameterFile, "</testsuiteevaluation>\n\n");
	}
	fclose(parameterFile);
	
	printf("Loaded %d parameter sets with %d trials (%d) !!\n", parameterList->size(), gesTrials, trialNumber);
}

void CTestSuiteEvaluator::saveEvaluationData(string filename)
{
	FILE *parameterFile = fopen(filename.c_str(), "w");
	
	if (parameterFile == NULL)
	{
		printf("Could not create File %s\n", filename.c_str()); 
	}
	
	std::list<CParameters*>::iterator it = parameterList->begin();
	
	for (;it != parameterList->end(); it ++)
	{
		fprintf(parameterFile, "<testsuiteevaluation>\n");

		CParameters *parameters = *it;
		EvaluationValues *values = (*evaluations)[parameters];
		parameters->saveParametersXML(parameterFile);		
			
		fprintf(parameterFile, "<evaluationdata nTrials = %d>\n", values->size());
			
		EvaluationValues::iterator itValues = values->begin();
		for (; itValues != values->end(); itValues ++)
		{
			fprintf(parameterFile, "<trial number = %d >\n", (*itValues).trialNumber);
			fprintf(parameterFile, "Average Value = %f, Best Value = %f\n", (*itValues).averageValue, (*itValues).bestValue);
			fprintf(parameterFile, "EvaluationTime = %lf\n",  (*itValues).evaluationTime);
			
			fprintf(parameterFile, "EvaluationDate = %s\n",  (*itValues).evaluationDate.c_str());
			fprintf(parameterFile, "</trial>\n");
							
		}
		fprintf(parameterFile, "</evaluationdata>\n");	
		fprintf(parameterFile, "</testsuiteevaluation>\n\n");
	}
	fclose(parameterFile);
}

void CTestSuiteEvaluator::saveEvaluationDataMatlab(string filename)
{

	FILE *parameterFile = fopen(filename.c_str(), "w");
	
	if (parameterFile == NULL)
	{
		printf("Could not create File %s\n", filename.c_str()); 
	}
	
	std::list<CParameters*>::iterator it = parameterList->begin();
	
	for (int nParam = 1;it != parameterList->end(); it ++, nParam ++)
	{
		for (int j = 0; j < (*it)->getNumParameters(); j ++)
		{
			fprintf(parameterFile, "parameterSet{%d}.parameters{%d} = {'%s', %f};\n", nParam, j + 1, (*it)->getParameterName(j).c_str(), (*it)->getParameterFromIndex(j));
		}	
		
		CParameters *parameters = *it;
		EvaluationValues *values = (*evaluations)[parameters];
		
		EvaluationValues::iterator itValues = values->begin();
		for (int j = 1; itValues != values->end(); itValues ++, j++)
		{
			fprintf(parameterFile, "parameterSet{%d}.trials(%d) = %d;\n", nParam, j, (*itValues).trialNumber + 1);
			
			fprintf(parameterFile, "trials{%d}.average = %f;\n", (*itValues).trialNumber + 1, (*itValues).averageValue);
			fprintf(parameterFile, "trials{%d}.bestValue = %f;\n", (*itValues).trialNumber + 1, (*itValues).bestValue);
			fprintf(parameterFile, "trials{%d}.time = %f;\n", (*itValues).trialNumber + 1, (*itValues).evaluationTime);
			fprintf(parameterFile, "trials{%d}.parameterSet = %d;\n", (*itValues).trialNumber + 1, nParam);
		}
	}
	fclose(parameterFile);
}

	
void CTestSuiteEvaluator::doEvaluationTrial(CParameters *parameters, EvaluationValue *evaluationData)
{
	testSuite->setParameters(parameters);
	testSuite->resetLearnedData();

	string learnDataFileName = getLearnDataFileName(evaluationData->trialNumber);

	time_t startTime;
	time_t endTime;
	
	printf("Beginning to Evaluate Trial number %d (%d) ... LearnDataFile %s\n", evaluationData->trialNumber, trialNumber, learnDataFileName.c_str());
	

	newEvaluationTrial(testSuite, evaluationData);
	
	std::list<CTestSuiteEvaluatorLogger *>::iterator it = evaluators->begin();
	string evaluationDirectory = getEvaluationDirectory();

	
	for (; it != evaluators->end(); it++)
	{
		(*it)->startNewEvaluation(evaluationDirectory, testSuite, evaluationData->trialNumber);
	}
	
	try
	{
		startTime = time(NULL);
		int nEpisode = 0;
		while (!isFinished(nEpisode))
		{
			nEpisode ++;			
			int actualSteps = agent->getTotalSteps();
			doEpisode(testSuite, nEpisode);

			if (agent->getEnvironmentModel()->isFailed())
			{
				printf("Finished Episode %d  (%d steps, failed : ",nEpisode,  agent->getTotalSteps() - actualSteps);
				agent->getCurrentState()->getState()->saveASCII(stdout);
				printf(")\n");
			}
			else
			{
				printf("Finished Episode %d (%d steps, succeded :", nEpisode, agent->getTotalSteps() - actualSteps);
				agent->getCurrentState()->getState()->saveASCII(stdout);
				printf(")\n");
			} 
			
			std::list<CTestSuiteEvaluatorLogger *>::iterator it = evaluators->begin();
					
			for (; it != evaluators->end(); it++)
			{
				if (nEpisode % (*it)->nEpisodesBeforeEvaluate == 0)
				{
					(*it)->evaluate(evaluationDirectory, evaluationData->trialNumber, nEpisode);
				}
			}	
				
		}
		
		it = evaluators->begin();
		for (; it != evaluators->end(); it++)
		{
			(*it)->endEvaluation();
		}
		
		
	}
	catch (CMyException *E)
	{
		printf("EXCEPTION: ");
		printf(E->getErrorMsg().c_str());
	
		evaluationData->averageValue = -100000000;
		evaluationData->bestValue = -100000000;
		
		
		for (; it != evaluators->end(); it++)
		{
			(*it)->endEvaluation();
		}
	}
	endTime = time(NULL);
	getEvaluationValue(evaluationData);
	evaluationData->evaluationTime = difftime(endTime, startTime);
		
	char *time = ctime(&endTime);
	char buffTime[50];
	sprintf(buffTime, "%s", time);
		
	buffTime[strlen(buffTime) - 1] = 0;
		
	evaluationData->evaluationDate = string(buffTime);
	FILE *dataFILE = fopen(learnDataFileName.c_str(), "w");
	if (dataFILE == NULL)
	{
		printf("Could not create file %s\n", learnDataFileName.c_str());
	}
	else
	{
		printf("Storing Learndata in file %s\n", learnDataFileName.c_str());
	}
	testSuite->saveLearnedData(dataFILE);
	fclose(dataFILE);
}

void CTestSuiteEvaluator::addPolicyEvaluator(CTestSuiteEvaluatorLogger *evaluator)
{
	evaluators->push_back(evaluator);
}

CParameters *CTestSuiteEvaluator::getParametersObject(CParameters *parameters)
{
	CParameters tempParams(*parameters);
	tempParams.addParameters(this);
	
	std::list<CParameters *>::iterator it = parameterList->begin();
	
	for (; it != parameterList->end(); it ++)
	{
		if ( *(*it) == tempParams)
		{
			break;
		}
	}
	if (it != parameterList->end())
	{
		return *it;
	}
	else
	{
		return NULL;
	}
}

double CTestSuiteEvaluator::getAverageValue(CParameters *testSuite)
{
	CParameters *parameters = getParametersObject(testSuite);
	double average = 0.0;	
	if (parameters != NULL)
	{
		EvaluationValues *values = (*evaluations)[parameters];
		EvaluationValues::iterator itVal = values->begin();
		
		for (; itVal != values->end(); itVal ++)
		{
			average += (*itVal).averageValue;
		}
		average = average / values->size();
	}
	return average;
}

double CTestSuiteEvaluator::getBestValue(CParameters *testSuite)
{
	CParameters *parameters = getParametersObject(testSuite);
	double best = 0.0;	
	if (parameters != NULL)
	{
		EvaluationValues *values = (*evaluations)[parameters];
		EvaluationValues::iterator itVal = values->begin();
		
		for (int i = 0; itVal != values->end(); itVal ++, i++)
		{
			if (i == 0 || best < (*itVal).bestValue)
			{
				best = (*itVal).bestValue;
			}
		}
	}
	return best;
}
	
EvaluationValues *CTestSuiteEvaluator::getEvaluationValues()
{
	CParameters *parameters = getParametersObject(testSuite);

	if (parameters != NULL)
	{
		return (*evaluations)[parameters];
	}
	else
	{
		return NULL;
	}
}


			
void CAverageRewardTestSuiteEvaluator::newEvaluationTrial(CTestSuite *testSuite, EvaluationValue *evaluationData)
{
	numEvals = 1;

	char filename[200];
	
	sprintf(filename, "%s/%s/Evaluation/Trial%d.data", baseDirectory.c_str(), testSuite->getTestSuiteName().c_str(), evaluationData->trialNumber);
	
	if (evaluationFile)
	{
		fclose(evaluationFile);
	}
	evaluationFile = fopen(filename, "w");
	
	if (evaluationFile == NULL)
	{
		printf("TestSuiteEvaluator : Could not create file %s ... please create directories\n", filename);
	}
	
	double val = evaluator->evaluate();
	
	printf("Evaluating Policy after %d Episodes : %f\n", 0, val);
	
	averageValue = val;
	bestValue = val;	
	
	fprintf(evaluationFile, "%d %f\n", 0, val);
}

void CAverageRewardTestSuiteEvaluator::doEpisode(CTestSuite *testSuite, int nEpisode)
{
	testSuite->learn(1, stepsLearnEpisode);
	if (nEpisode % episodesBeforeEvaluate == 0)
	{
		double val = evaluator->evaluate();
		
		averageValue += val;
		
		if (bestValue < val)
		{
			bestValue = val;
		}
		printf("Evaluating Policy after %d Episodes : %f, best: %f\n", nEpisode, val, bestValue);
		
		numEvals ++;
		fprintf(evaluationFile, "%d %f\n", nEpisode, val);
		fflush(evaluationFile);
	}
	
}

void CAverageRewardTestSuiteEvaluator::getEvaluationValue(EvaluationValue *evaluationData)
{
	evaluationData->bestValue = bestValue;
	evaluationData->averageValue = averageValue / numEvals;
}

bool CAverageRewardTestSuiteEvaluator::isFinished(unsigned int nEpisode)
{
	bool finished = (nEpisode >= totalLearnEpisodes);
	
	if (finished)
	{
		fflush(evaluationFile);
		fclose(evaluationFile);
		evaluationFile = NULL;
	}

	return finished;
}


CAverageRewardTestSuiteEvaluator::CAverageRewardTestSuiteEvaluator(CAgent *agent, string baseDirectory, CTestSuite *testSuite, CEvaluator *l_evaluator, int l_totalLearnEpisodes, int l_episodesBeforeEvaluate, int l_stepsLearnEpisode, int nTrials) : CTestSuiteEvaluator(agent, baseDirectory, testSuite, nTrials)
{
	this->episodesBeforeEvaluate = l_episodesBeforeEvaluate;
	this->totalLearnEpisodes = l_totalLearnEpisodes;
	this->stepsLearnEpisode = l_stepsLearnEpisode;

	numEvals = 0;
	
	averageValue = 0;
	bestValue = 0;
	
	this->evaluator = l_evaluator;

	evaluationFile = NULL;


	setParameter("TestSuiteEvaluator", 1.0);
	addParameter("EpisodesBeforeEvaluate", episodesBeforeEvaluate);
	addParameter("TotalLearnEpisodes", totalLearnEpisodes);
	addParameter("StepsLearnEpisode", stepsLearnEpisode);
}
	
CAverageRewardTestSuiteEvaluator::~CAverageRewardTestSuiteEvaluator()
{
	if (evaluationFile)
	{
		fflush(evaluationFile);
		fclose(evaluationFile);
		evaluationFile = NULL;
	}
}


/*
CTestSuiteEvaluator::CTestSuiteEvaluator(CAgent *agent, string ltestSuiteCollectionName, int nTrials, int numValuesPerTrial, char *homeDir)  : evaluatorDirectory(""), testSuiteCollectionName(ltestSuiteCollectionName)
{
	this->agent = agent;
	addParameter("DivergentEvaluationValue", -1000000000.0);

	this->nTrials = nTrials;

	values = new std::list<double *>();

	exception = false;

	this->numValuesPerTrial = numValuesPerTrial;
	
	char directory[256];
	
	if (homeDir == NULL)
	{
		sprintf(directory, "%s/%s/", EVALUATION_DIRECTORY, testSuiteCollectionName.c_str());
		evaluatorDirectory = string(directory);
	}
	else
	{
		evaluatorDirectory = string(homeDir);
	}
}

CTestSuiteEvaluator::~CTestSuiteEvaluator()
{
	std::list<double *>::iterator it = values->begin();
	for (; it!= values->end(); it++)
	{
		delete *it;
	}
	delete values;
}


string CTestSuiteEvaluator::getEvaluatorDirectory()
{
	char directory[256];
#ifdef WIN32
	sprintf(directory, "%s\\%s",testSuiteCollectionName.c_str(), evaluatorDirectory.c_str());
#else
	sprintf(directory, "%s", evaluatorDirectory.c_str());
#endif
	return string(directory);
}

string CTestSuiteEvaluator::getEvaluationFileName(CTestSuite *testSuite)
{
	char evaluationFileName[255];

#ifdef WIN32
	sprintf(evaluationFileName, "%s\\%s_params.txt", getEvaluatorDirectory().c_str(),testSuite->getTestSuiteName().c_str());

#else
	sprintf(evaluationFileName, "%s/%s_params.txt", getEvaluatorDirectory().c_str(),testSuite->getTestSuiteName().c_str());
#endif
	return string(evaluationFileName);
}

string CTestSuiteEvaluator::getLearnDataFileName(const char *trialFileName)
{
	char learnDataFileName[256];
	
#ifdef WIN32
		sprintf(learnDataFileName,"%s\\LearnData\\%s.data", getEvaluatorDirectory().c_str(), trialFileName);
#else
		sprintf(learnDataFileName,"%s/LearnData/%s.data", getEvaluatorDirectory().c_str(),	trialFileName);
#endif
	
	return string(learnDataFileName);
}

string CTestSuiteEvaluator::getTrialFileName(CTestSuite *testSuite)
{
	char learnDataFileName[256];
	FILE *learnDataFile = NULL;
	int learnDataFileNumber = 0;
	do 
	{
		if (learnDataFile)
		{
			fclose(learnDataFile);
		}
#ifdef WIN32
		sprintf(learnDataFileName,"%s\\LearnData\\%s_%d.data", getEvaluatorDirectory().c_str(), testSuite->getTestSuiteName().c_str(), learnDataFileNumber);
#else
		sprintf(learnDataFileName,"%s/LearnData/%s_%d.data", getEvaluatorDirectory().c_str(), testSuite->getTestSuiteName().c_str(), learnDataFileNumber);
#endif
		learnDataFile = fopen(learnDataFileName, "r");

		learnDataFileNumber ++;	
	} 
	while(learnDataFile != NULL && learnDataFileNumber< 30000);

	if (learnDataFile)
	{
		fclose(learnDataFile);
	}
	#ifdef WIN32
	sprintf(learnDataFileName,"s_%d", testSuite->getTestSuiteName().c_str(), learnDataFileNumber - 1);
	#else
	sprintf(learnDataFileName,"%s_%d", testSuite->getTestSuiteName().c_str(), learnDataFileNumber - 1);
	#endif
	
	return string(learnDataFileName);
}

void CTestSuiteEvaluator::clearValues()
{
	std::list<double *>::iterator it = values->begin();
	for (; it != values->end(); it ++)
	{
		delete *it;
	}
	values->clear();

}

void CTestSuiteEvaluator::setEvaluatorDirectory(char *homeDir)
{
	evaluatorDirectory = string(homeDir);
}

void CTestSuiteEvaluator::getXLabel(char *xLabel, int i)
{
	sprintf(xLabel, "%d", i);
}

void CTestSuiteEvaluator::saveMatlabData(CParameters *testSuite, char *outFileName, char *inFileName)
{
	checkDirectories();

	exception = false;

	clearValues();

	char trialFile[255];

	if (inFileName != NULL)
	{
		sprintf(trialFile, "%s/%s", getEvaluatorDirectory().c_str(), inFileName);
	}
	else
	{
		printf("No input file given !!!\n");
		return;
	}

	loadEvaluationData(testSuite, trialFile);

	FILE *matlabFile = fopen(outFileName, "a");
	for (int i = 0; i < numValuesPerTrial; i ++)
	{
		char xLabel[80];
		getXLabel(xLabel, i);

		fprintf(matlabFile, "%s ", xLabel);
		std::list<double *>::iterator it = values->begin();
		double sum = 0.0;
		for (; it != values->end(); it ++)
		{
			sum += (*it)[i];
			fprintf(matlabFile, " %1.4f", (*it)[i]);
		}	
		fprintf(matlabFile, " %1.4f\n", sum / values->size());
	}
	fprintf(matlabFile, "\n");

	fclose(matlabFile);
}

double CTestSuiteEvaluator::evaluateTestSuite(CTestSuite *testSuite, bool loadEvaluationTrial)
{
	checkDirectories();

	exception = false;

	clearValues();
	

	printf("Evaluating TestSuite %s with Parameters:\n", testSuite->getTestSuiteName().c_str());
	testSuite->saveParameters(stdout);

	if (loadEvaluationTrial)
	{
		loadEvaluationData(testSuite, getEvaluationFileName(testSuite).c_str());
	}
	printf("Loaded %d Trials for Evaluation (%s)!\n", values->size(), getEvaluationFileName(testSuite).c_str());

	FILE *evaluationFile = fopen(getEvaluationFileName(testSuite).c_str(), "a");
	
	if (evaluationFile == NULL)
	{
		printf("Could not create evaluation File %s\n", getEvaluationFileName(testSuite).c_str());
	}

	assert(evaluationFile);
	
	while (values->size() < nTrials && ! exception)
	{
		doEvaluationTrial(testSuite, evaluationFile, getTrialFileName(testSuite).c_str());
	}
	fclose(evaluationFile);

	double evaluationValue = 0.0;

	if(exception)
	{
		evaluationValue = getParameter("DivergentEvaluationValue");
	}
	else
	{
		evaluationValue = getEvaluationValue(values);
	}

	return evaluationValue;
}

void CTestSuiteEvaluator::checkDirectories()
{
	char evaluationFileDirectory[255];
	char checkDirSystemCall[255];

#ifdef WIN32
	sprintf(evaluationFileDirectory, "%s\\", getEvaluatorDirectory().c_str());
#else
	sprintf(evaluationFileDirectory, "%s/", getEvaluatorDirectory().c_str());
#endif
	sprintf(checkDirSystemCall, "checkdir.bat %s", evaluationFileDirectory);

	system(checkDirSystemCall);

}
 
void CTestSuiteEvaluator::loadEvaluationData(CParameters *testSuite, const char *fileName)
{

	FILE *parameterFile = fopen(fileName, "r");

	if (parameterFile == NULL)
	{
		return;
	}
	
	loadEvaluationData(testSuite, parameterFile);	
	
	fclose(parameterFile);
}

void CTestSuiteEvaluator::loadBestEvaluationData(CParameters *testSuite, const char *fileName, bool min)
{
	printf("Looking for best Parameter Setting\n");
	
	FILE *parameterFile = fopen(fileName, "r");

	if (parameterFile == NULL)
	{
		printf("No Parameter File found!!\n");
		return;
	}
	
	
	CParameters *bestParams = new CParameters();;
	bestParams->addParameters(testSuite);
	
	clearValues();
	loadEvaluationData(bestParams, parameterFile);
	
	double bestValue = getEvaluationValue(values);
	
	
	//printf("Value %f\n", bestValue);
	
	while (!feof(parameterFile) && values->size() > 0)
	{
		CParameters bufferParams;
		bufferParams.addParameters(testSuite);
	
		clearValues();
		loadEvaluationData(&bufferParams, parameterFile);
		
		if (values->size() > 0)
		{
			double value =  getEvaluationValue(values);
			
			//printf("Value %f\n", value);
		
			bool newBest = false;
			if (min)
			{
				newBest = (((bestValue < 0 || value < bestValue) && value > 0) || (value > bestValue && value < 0));
			}
			else
			{
				newBest = value > bestValue;
			}
			
			if (newBest)
			{
				bestValue = value;
				//printf("New Best Parameters...\n");
				
				delete bestParams;
				
				bestParams = new CParameters();
				bestParams->addParameters(&bufferParams);
			}
		}
	}
	
	testSuite->addParameters(bestParams);
	delete bestParams;
	fclose(parameterFile);
}

 
CTestSuiteNeededStepsEvaluator::CTestSuiteNeededStepsEvaluator(CAgent *agent, string testSuiteCollectionName, int totalLearnEpisodes, int stepsLearnEpisode, int episodesBeforeEvaluate, int nTrials, bool maxStepsSucceded) : CTestSuiteEvaluator(agent, testSuiteCollectionName, nTrials, (int) ceil((double)totalLearnEpisodes / (double) episodesBeforeEvaluate))
{
	this->totalLearnEpisodes = totalLearnEpisodes;
	this->stepsLearnEpisode = stepsLearnEpisode;
	this->episodesBeforeEvaluate = episodesBeforeEvaluate;
	this->maxStepsSucceded = maxStepsSucceded;

	succeded = new std::list<double *>();

	char evaluatorDirectoryChar[80];
	sprintf(evaluatorDirectoryChar, "ST_%d_%d", totalLearnEpisodes, stepsLearnEpisode);
	evaluatorDirectory = string(evaluatorDirectoryChar);

	nValues = (int) ceil((double)totalLearnEpisodes / (double) episodesBeforeEvaluate);
}
	
CTestSuiteNeededStepsEvaluator::~CTestSuiteNeededStepsEvaluator()
{
	std::list<double *>::iterator it = succeded->begin();
	for (; it!= succeded->end(); it++)
	{
		delete *it;
	}
	delete succeded;
}

void CTestSuiteNeededStepsEvaluator::clearValues()
{
	CTestSuiteEvaluator::clearValues();	

	std::list<double *>::iterator it = succeded->begin();
	for (; it!= succeded->end(); it++)
	{
		delete *it;
	}
	succeded->clear();
}



void CTestSuiteNeededStepsEvaluator::loadEvaluationData(CParameters *testSuite, FILE *parameterFile)
{
	bool first = true;
	
	//printf("Looking for parameters :\n");
	//testSuite->saveParameters(stdout);
	
	fpos_t filepos;
	fgetpos(parameterFile, &filepos);
	
	while (!feof(parameterFile))
	{
		char buffer[256];

		int results = fscanf(parameterFile, "%s\n", buffer);
		while (results == 1 && strcmp(buffer, "<testsuiteevaluation>") != 0 && !feof(parameterFile)) 
		{
			results =  fscanf(parameterFile, "%s\n", buffer);
		}

		if (feof(parameterFile))
		{
			break;
		}

		CParameters *parameters = new CParameters();
		parameters->loadParametersXML(parameterFile);

		if (parameters->containsParameters(testSuite))
		{
			if (first)
			{
				testSuite->addParameters(parameters);
				
				//printf("Found Parameters : \n");
				//parameters->saveParameters(stdout);
				first = false;
			}
			
			fscanf(parameterFile, "<evaluationdata>\n");
			fscanf(parameterFile, "%s\n", buffer);

//			double time = 0.0;

			int bufEpisode;
			int bufSteps;

			double *trialValues = new double[nValues];
			memset(trialValues, 0, sizeof(double) * nValues);
			values->push_back(trialValues);

			double *trialSucceded = new double[nValues];
			memset(trialSucceded, 0, sizeof(double) * nValues);
			succeded->push_back(trialSucceded);

			results = 1;
			bool bOk = true;
			int i = 0;
			while (strcmp(buffer, "</evaluationdata>") != 0 )
			{
				if (buffer[0] == '<' || results <= 0 || feof(parameterFile))
				{
					bOk = false;
					break;
				}

				sscanf(buffer, "%d,", &bufEpisode);

				int n_result = fscanf(parameterFile, "%d: %lf %lf\n", &bufSteps, &trialValues[i], &trialSucceded[i]);

				if (n_result !=  3|| trialValues[i] < getParameter("DivergentEvaluationValue") + 1)
				{
					exception = true;
				}

				results = fscanf(parameterFile, "%s\n", buffer);
				i ++;
			}
			if (bOk)
			{
				while (results == 1 && strcmp(buffer, "</testsuiteevaluation>") != 0 && !feof(parameterFile)) 
				{
					results =  fscanf(parameterFile, "%s\n", buffer);
				}
			}
			fgetpos(parameterFile, &filepos);
		}
		delete parameters;	
	}
	if (!first)
	{
		fsetpos(parameterFile, &filepos);
	}
}

void CTestSuiteNeededStepsEvaluator::doEvaluationTrial(CTestSuite *testSuite, FILE *stream, const char *learnDataFileName)
{
	fprintf(stream, "<testsuiteevaluation>\n");

	testSuite->resetLearnedData();
	testSuite->saveParametersXML(stream);

	double *trialValues = new double[nValues];
	double *trialSucceded = new double[nValues];

	memset(trialValues, 0, sizeof(double) * nValues);
	values->push_back(trialValues);


	memset(trialSucceded, 0, sizeof(double) * nValues);
	succeded->push_back(trialSucceded);

	printf("Evaluating TestSet, %d Trial\n", values->size());

	int nLearnEpisodes = 0;

//	double maxValue = 0;
//	int maxIndex = 0;

	time_t startTime = time(NULL);
	time_t endTime;
	int totalSteps = 0;

//	double average = 0.0;

	int nSucc = 0;

	try
	{
		fprintf(stream, "<evaluationdata>\n");
		for (unsigned int i = 0; i < nValues; i ++)
		{
			unsigned int nEpisodes = 0;
			
			unsigned int oldSteps = totalSteps;
			while (nEpisodes < episodesBeforeEvaluate && nLearnEpisodes < totalLearnEpisodes)
			{
				nEpisodes ++;
				nLearnEpisodes ++;

				printf("Learning Episode %d\n", nLearnEpisodes);

				int actualSteps = agent->getTotalSteps();
				testSuite->learn(1, stepsLearnEpisode);
				totalSteps += agent->getTotalSteps() - actualSteps;
				CEnvironmentModel *model = agent->getEnvironmentModel();

				

				if (model->isFailed() || (agent->getTotalSteps() - actualSteps >= stepsLearnEpisode && !maxStepsSucceded))
				{
					printf("Finished Learning (%d steps, failed : ", agent->getTotalSteps() - actualSteps);
					agent->getCurrentState()->getState()->saveASCII(stdout);
					printf(")\n");
				}
				else
				{
					printf("Finished Learning (%d steps, succeded :", agent->getTotalSteps() - actualSteps);
					agent->getCurrentState()->getState()->saveASCII(stdout);
					printf(")\n");

					trialSucceded[i] ++;
					nSucc ++;
				}

			}
			trialSucceded[i] /= nEpisodes;
			trialValues[i] = totalSteps - oldSteps;

			fprintf(stream, "%d, %d: %f %f\n", nLearnEpisodes, totalSteps, trialValues[i], trialSucceded[i]);
		}
		fprintf(stream, "</evaluationdata>\n");
		fprintf(stream, "<totalsteps> %d </totalsteps>\n", totalSteps);
		fprintf(stream, "<percentsucceded> %f </percentsucceded>\n", ((double) nSucc) / totalLearnEpisodes);


		printf("Evaluated Value: %d\n", totalSteps);
		endTime = time(NULL);
		printf("Time needed for Evaluation: %f\n", difftime(endTime, startTime));
		fprintf(stream, "<evaluationtime> %f </evaluationtime>\n", difftime(endTime, startTime)), 

		fprintf(stream, "<learndatafile> %s </learndatafile>\n", learnDataFileName);

		if (learnDataFileName)
		{
			FILE *learnDataFile = fopen(learnDataFileName, "w");

			testSuite->saveLearnedData(learnDataFile);
			fclose(learnDataFile);
		}
	}
	catch (CMyException *E)
	{
		printf(E->getErrorMsg().c_str());
		fprintf(stream, "%d, %d: %f 0.0\n", nLearnEpisodes, totalSteps, getParameter("DivergentEvaluationValue"));
		fprintf(stream, "</evaluationdata>\n");
		fprintf(stream, "<totalsteps> %f </totalsteps>\n", getParameter("DivergentEvaluationValue"));
		fprintf(stream, "<exception> %s </exception>\n", E->getErrorMsg().c_str());

		exception = true;

	}
	fprintf(stream, "</testsuiteevaluation>\n\n");
	fflush(stream);	
}

double CTestSuiteNeededStepsEvaluator::getEvaluationValue(std::list<double *> *values)
{
	std::list<double *>::iterator it = values->begin();

	printf("Found %d value lists\n", values->size());
	
	double steps = 0;
	
	

	for (; it != values->end();it++)
	{
		for (unsigned int i = 0; i < nValues; i++)
		{
			
			steps += (*it)[i];
		}
	}
	
	return steps / values->size();
}

double CTestSuiteNeededStepsEvaluator::getPercentageSucceded()
{
	std::list<double *>::iterator it = succeded->begin();

	double succ = 0;

	for (; it != succeded->end();it++)
	{
		for (unsigned int i = 0; i < nValues; i++)
		{
			succ += (*it)[i];
		}
		succ /= nValues;
	}
	succ /= succeded->size();
	
	return succ;
}



CAverageRewardTestSuiteEvaluator::CAverageRewardTestSuiteEvaluator(CAgent *agent, string testSuiteCollectionName, CTestSuitePolicyEvaluator *evaluator, int totalLearnEpisodes, int episodesBeforeEvaluate, int stepsLearnEpisode, int nTrials) : CTestSuiteEvaluator(agent, testSuiteCollectionName, nTrials, (int) ceil((double)totalLearnEpisodes / (double) episodesBeforeEvaluate))
{
	this->totalLearnEpisodes = totalLearnEpisodes;
	this->episodesBeforeEvaluate = episodesBeforeEvaluate;
	this->evaluators = new std::list<CTestSuitePolicyEvaluator *>();
	evaluators->push_back(evaluator);
	this->stepsLearnEpisode = stepsLearnEpisode;

	char evaluatorDirectoryChar[80];
	sprintf(evaluatorDirectoryChar, "AR_%d_%d_%d", totalLearnEpisodes, episodesBeforeEvaluate, stepsLearnEpisode);
	evaluatorDirectory = string(evaluatorDirectoryChar);

	nAverageRewards = (int) ceil((double)totalLearnEpisodes / (double) episodesBeforeEvaluate);

	evaluationFunction = ARCF_AVERAGE;
}

CAverageRewardTestSuiteEvaluator::~CAverageRewardTestSuiteEvaluator()
{
	delete evaluators;
}


void CAverageRewardTestSuiteEvaluator::getXLabel(char *xLabel, int i)
{
	sprintf(xLabel, "%d", i* episodesBeforeEvaluate);
}

void CAverageRewardTestSuiteEvaluator::addPolicyEvaluator(CTestSuitePolicyEvaluator *evaluator)
{
	evaluators->push_back(evaluator);
}

void CAverageRewardTestSuiteEvaluator::doEvaluationTrial(CTestSuite *testSuite, FILE *stream, const char *evaluationFileName)
{
	fprintf(stream, "<testsuiteevaluation>\n");

	testSuite->resetLearnedData();
	testSuite->saveParametersXML(stream);

	double *trialValues = new double[nAverageRewards];

	memset(trialValues, 0, sizeof(double) * nAverageRewards);
	values->push_back(trialValues);

	printf("Evaluating TestSet, %d Trial\n", values->size());

	char buf[200];
	
	const char *learnDataFileName = getLearnDataFileName(evaluationFileName).c_str();

	int nLearnEpisodes = 0;

	double maxValue = 0;
	int maxIndex = 0;

	time_t startTime = time(NULL);
	time_t endTime;
	int totalSteps = 0;

	double average = 0.0;

	CTestSuitePolicyEvaluator *evaluator = *evaluators->begin();
	
	std::list<CTestSuitePolicyEvaluator *>::iterator it = evaluators->begin();
	it++;
	
	
	for (; it != evaluators->end(); it++)
	{
		(*it)->startNewEvaluation(evaluationFileName, testSuite, totalLearnEpisodes, episodesBeforeEvaluate, stepsLearnEpisode);
	}
	
	
	try
	{
		fprintf(stream, "<evaluationdata>\n");
		for (unsigned int i = 0; i < nAverageRewards; i ++)
		{
			unsigned int nEpisodes = 0;
			

			while (nEpisodes < episodesBeforeEvaluate && nLearnEpisodes < totalLearnEpisodes)
			{
				nEpisodes ++;
				nLearnEpisodes ++;

				printf("Learning Episode %d\n", nLearnEpisodes);

				int actualSteps = agent->getTotalSteps();
				testSuite->learn(1, stepsLearnEpisode);
				totalSteps += agent->getTotalSteps() - actualSteps;
				CEnvironmentModel *model = agent->getEnvironmentModel();

				if (model->isFailed())
				{
					printf("Finished Learning (%d steps, failed : ", agent->getTotalSteps() - actualSteps);
					agent->getCurrentState()->getState()->saveASCII(stdout);
					printf(")\n");
				}
				else
				{
					printf("Finished Learning (%d steps, succeded :", agent->getTotalSteps() - actualSteps);
					agent->getCurrentState()->getState()->saveASCII(stdout);
					printf(")\n");

				}

			}

			agent->setController(testSuite->getEvaluationController());
			
			trialValues[i] = evaluator->evaluate(evaluationFileName, values->size(), i, nLearnEpisodes);

			average += trialValues[i];
			agent->setController(testSuite->getController());

			if ((i == 0) || maxValue < trialValues[i])
			{
				maxValue = trialValues[i];
				maxIndex = i;
			}

			printf("Value after %d (Steps %d) Episodes: %f", nLearnEpisodes, totalSteps, trialValues[i]);
			fprintf(stream, "%d, %d: %f ", nLearnEpisodes, totalSteps, trialValues[i]);
			if (evaluators->size() > 1)
			{
				std::list<CTestSuitePolicyEvaluator *>::iterator it = evaluators->begin();
				it++;

				for (int j = 1; it != evaluators->end(); it++, j++)
				{
					double value = (*it)->evaluate(evaluationFileName, values->size(), i, nLearnEpisodes);
					printf(" Evaluator %d: %f ", j, value);
					fprintf(stream, "%f ", value);
				}
			}
			fprintf(stream, "\n");
			printf("\n");
		}
		fprintf(stream, "</evaluationdata>\n");
		fprintf(stream, "<averagevalue> %f </averagevalue>\n", average / nAverageRewards);
		fprintf(stream, "<bestvalue> %f </bestvalue>\n", maxValue);

		printf("Evaluated Value: %f, Best Value: %f after %d LearnEpisodes\n", average / nAverageRewards, maxValue, maxIndex);
		endTime = time(NULL);
		printf("Time needed for Evaluation: %f\n", difftime(endTime, startTime));
		fprintf(stream, "<evaluationtime> %f </evaluationtime>\n", difftime(endTime, startTime)), 

		fprintf(stream, "<learndatafile> %s </learndatafile>\n", learnDataFileName);

		if (learnDataFileName)
		{
			FILE *learnDataFile = fopen(learnDataFileName, "w");

			testSuite->saveLearnedData(learnDataFile);
			fclose(learnDataFile);
		}
	}
	catch (CMyException *E)
	{
		printf(E->getErrorMsg().c_str());
		fprintf(stream, "%d, %d: %f\n", nLearnEpisodes, totalSteps, getParameter("DivergentEvaluationValue"));
		fprintf(stream, "</evaluationdata>\n");
		fprintf(stream, "<averagevalue> %f </averagevalue>\n", getParameter("DivergentEvaluationValue"));
		fprintf(stream, "<bestvalue> %f </bestvalue>\n", getParameter("DivergentEvaluationValue"));
		fprintf(stream, "<exception> %s </exception>\n", E->getErrorMsg().c_str());

		exception = true;

	}
	fprintf(stream, "</testsuiteevaluation>\n\n");
	fflush(stream);	
}

void CAverageRewardTestSuiteEvaluator::loadEvaluationData(CParameters *testSuite, FILE *parameterFile)
{ 
	bool first = true;
	
	fpos_t filepos;
	fgetpos(parameterFile, &filepos);
	
	while (!feof(parameterFile))
	{
		char buffer[256];

		int results = fscanf(parameterFile, "%s\n", buffer);
		while (results == 1 && strcmp(buffer, "<testsuiteevaluation>") != 0 && !feof(parameterFile)) 
		{
			results =  fscanf(parameterFile, "%s\n", buffer);
		}

		if (feof(parameterFile))
		{
			break;
		}

		CParameters *parameters = new CParameters();
		parameters->loadParametersXML(parameterFile);

		if (parameters->containsParameters(testSuite))
		{
			if (first)
			{
				testSuite->addParameters(parameters);
				
				//printf("Found Parameters : \n");
				//parameters->saveParameters(stdout);
				first = false;
			}
	
			fscanf(parameterFile, "<evaluationdata>\n");
			fscanf(parameterFile, "%s\n", buffer);

			//double time = 0.0;

			int bufEpisode;
			int bufSteps;

			double *trialValues = new double[nAverageRewards];
			memset(trialValues, 0, sizeof(double) * nAverageRewards);
			values->push_back(trialValues);

			results = 1;
			bool bOk = true;
			int i = 0;
			while (strcmp(buffer, "</evaluationdata>") != 0 )
			{
				if (buffer[0] == '<' || results <= 0 || feof(parameterFile))
				{
					bOk = false;
					break;
				}
				
				sscanf(buffer, "%d,", &bufEpisode);

				int n_result = fscanf(parameterFile, "%d: %lf", &bufSteps, &trialValues[i]);

				if (n_result != 2 || trialValues[i] < getParameter("DivergentEvaluationValue") + 1)
				{
					exception = true;
				}
				double dBuf = 0;
				for (int i = 1;i < evaluators->size(); i ++)
				{
					fscanf(parameterFile," %lf", &dBuf);
				}
				fscanf(parameterFile,"\n");


				results = fscanf(parameterFile, "%s\n", buffer);
				i ++;
			}
			if (bOk)
			{
				while (results == 1 && strcmp(buffer, "</testsuiteevaluation>") != 0 && !feof(parameterFile)) 
				{
					results =  fscanf(parameterFile, "%s\n", buffer);
				}
			}
			fgetpos(parameterFile, &filepos);
		}
		delete parameters;	
	}
	if (!first)
	{
		fsetpos(parameterFile, &filepos);
	}
}

double CAverageRewardTestSuiteEvaluator::evaluateTestSuite(CTestSuite *testSuite, int evaluationFunction, bool loadEvaluationTrial)
{
	int tmpEvalF = this->evaluationFunction;
	this->evaluationFunction = evaluationFunction;
	double value = evaluateTestSuite(testSuite,  loadEvaluationTrial);

	this->evaluationFunction = tmpEvalF;
	return value;
}

double CAverageRewardTestSuiteEvaluator::evaluateTestSuite(CTestSuite *testSuite, bool loadEvaluationTrial)
{
	return CTestSuiteEvaluator::evaluateTestSuite(testSuite, loadEvaluationTrial);
}

double CAverageRewardTestSuiteEvaluator::evaluateTestSuite(CTestSuite *testSuite,int evaluationFunction, bool loadEvaluationTrial)
{
	checkDirectories();
	
	exception = false;
	
	std::list<double *>::iterator it = values->begin();
	for (; it != values->end(); it ++)
	{
		delete *it;
	}
	values->clear();

	printf("Evaluating TestSuite %s with Parameters:\n", testSuite->getTestSuiteName().c_str());
	testSuite->saveParameters(stdout);

	if (loadEvaluationTrial)
	{
		loadEvaluationData(testSuite);
	}
	printf("Loaded %d Trials for Evaluation!\n", values->size());
	
	FILE *evaluationFile = fopen(getEvaluationFileName(testSuite).c_str(), "a");
	while (values->size() < nTrials && ! exception)
	{
		doEvaluationTrial(testSuite, evaluationFile, getLearnDataFileName(testSuite).c_str());
	}
	fclose(evaluationFile);

	double evaluationValue = 0.0;
	
	if(exception)
	{
		evaluationValue = getParameter("DivergentEvaluationValue");
	}
	else
	{
		evaluationValue = getEvaluationValue(values, evaluationFunction);
	}

	return evaluationValue;
}

std::list<double *> *CAverageRewardTestSuiteEvaluator::getTrialAverageRewards()
{
	return values;
}

double CAverageRewardTestSuiteEvaluator::getEvaluationValue(std::list<double *> *values)
{
	std::list<double *>::iterator it = values->begin();

	double average = 0.0;
	switch (evaluationFunction)
	{
		case ARCF_IDENTITY:
		{
			for (; it != values->end();it++)
			{
				for (unsigned int i = 0; i < nAverageRewards; i++)
				{
					average += (*it)[i];
				}

			}
			break;
		}
		case ARCF_LINEAR:
		{
			for (; it != values->end();it++)
			{
				for (unsigned int i = 0; i < nAverageRewards; i++)
				{
					average += (*it)[i] * (i + 1) / nAverageRewards;
				}

			}
			break;
		}
		case ARCF_AVERAGE:
		{
			for (; it != values->end();it++)
			{
				double l_average = 0;
				for (unsigned int i = 0; i < nAverageRewards; i++)
				{
					l_average += (*it)[i];
				}
				average += l_average / nAverageRewards;

			}
			break;
		}
		default:
		{
			for (; it != values->end();it++)
			{
				double l_average = 0;
				for (unsigned  int i = 0; i < nAverageRewards; i++)
				{
					l_average += (*it)[i];
				}
				average += l_average / nAverageRewards;

			}

		}
	}
	average = average / values->size();
	printf("FINISHED with evaluation !!!\n");
	printf("Evaluated Average Value : %f\n\n\n", average );
	return average;
}



CTestSuiteParameterCalculator::CTestSuiteParameterCalculator(CTestSuiteEvaluator *evaluator, CTestSuite *testSuite)
{
	this->evaluator = evaluator;
	this->testSuite = testSuite;
}


CTestSuiteParameterCalculator::~CTestSuiteParameterCalculator()
{
}

void  CTestSuiteParameterCalculator::setFilenamesAndDirectories(char *testSuiteCollectionDirectory)
{
	char evaluationFileDirectory[255];
	char evaluationFileName[255];
	char checkDirSystemCall[255];

	sprintf(evaluationFileDirectory, "%s/%s/", testSuiteCollectionDirectory, evaluator->getEvaluatorDirectory().c_str());

	sprintf(evaluationFileName, "%s/%s_params.txt", evaluationFileDirectory, testSuite->getTestSuiteName().c_str());

	sprintf(this->learnDataFileDirectory, "%s/LearnData/", evaluationFileDirectory);

	sprintf(checkDirSystemCall, "checkdir.bat %s", evaluationFileDirectory);

	system(checkDirSystemCall);


	evaluatedTestsuiteParameters = new std::map<CParameters *, ParameterData>();

	FILE *input = fopen(evaluationFileName, "r");
	if (input)
	{
		loadEvaluatedParameters(input);
		fclose(input);
	}

	this->parameterFile = fopen(evaluationFileName, "a");
	assert(parameterFile != NULL);
}


void CTestSuiteParameterCalculator::loadEvaluatedParameters(FILE *parameterFile)
{
	CParameters *testSuiteParameters = new CParameters(*testSuite);
	while (!feof(parameterFile))
	{
		char buffer[256];
		ParameterData data;
		//fscanf(input, "TestSuiteEvaluation: ");
	

		int results = fscanf(parameterFile, "%s\n", buffer);
		while (strcmp(buffer, "<testsuiteevaluation>") != 0 && !feof(parameterFile)) 
		{
			results =  fscanf(parameterFile, "%s\n", buffer);
		}
		
		if (feof(parameterFile))
		{
			break;
		}

		testSuite->loadParametersXML(parameterFile);
		CParameters *parameters = new CParameters(*testSuite);

		fscanf(parameterFile, "<evaluationdata>\n");
		fscanf(parameterFile, "%s\n", buffer);

		double time = 0.0;

		if (parameters->getNumParameters() > 0)
		{
			results = 1;
			bool bOk = true;
			while (strcmp(buffer, "</evaluationdata>") != 0 )
			{
				if (buffer[0] == '<' || results <= 0 || feof(parameterFile))
				{
					bOk = false;
					break;
				}
				results = fscanf(parameterFile, "%s\n", buffer);
			}
			if (bOk)
			{
				data.learnDataFile = "";

 				fscanf(parameterFile, "<evaluatedvalue> %lf </evaluatedvalue>\n", &data.value);

				fscanf(parameterFile, "%s", buffer);

				if (strcmp(buffer, "<learntime>") == 0)
				{
					fscanf(parameterFile, " %lf </learntime>\n", &time);
				}
				if (strcmp(buffer, "<learndatafile>") == 0)
				{
					results = fscanf(parameterFile, "%s </learndatafile>\n", buffer);
					if (results == 1)
					{
						data.learnDataFile = string(buffer);
					}
				}

				fscanf(parameterFile, "</testsuiteevaluation>\n");

				(*evaluatedTestsuiteParameters)[parameters] = data;

			}
			else
			{
				delete parameters;
			}
		}
		else
		{
			delete parameters;
		}
		testSuite->setParameters(testSuiteParameters);
	}
	delete testSuiteParameters;
	printf("%d Parameter-Sets Loaded\n", this->getNumEvaluatedParameters());
}*/

CGraphLogger::CGraphLogger(CStateList *l_states, CGraphDynamicProgramming *l_graph) : CTestSuiteEvaluatorLogger("Graph")
{
	graph = l_graph;
	states = l_states;
}
		
CGraphLogger::~CGraphLogger()
{

}

void CGraphLogger::evaluate(string evaluationDirectory, int trial, int numEpisodes)
{
	
	CState state(states->getStateProperties());
	
	for (unsigned int i = 0; i < states->getNumStates(); i ++)
	{
		stringstream sstream;
	
		states->getState(i, &state);
		DataSubset subset;
	
		int nearestNeighbor = -1;
		double distance = -1.0;

		graph->getNearestNode(&state, nearestNeighbor, distance);

		graph->getGraph()->getConnectedNodes(nearestNeighbor, &subset);

		sstream << evaluationDirectory << outputDirectory << "/" << "Trial" << trial << "/Graph" << numEpisodes << "_" << i;

		graph->saveCSV(sstream.str(), &subset);
		
	}
}

void CGraphLogger::startNewEvaluation(string evaluationDirectory, CParameters *parameters, int trial)
{
	stringstream sstream;
	
	sstream << "mkdir " << evaluationDirectory << outputDirectory << "/" << "Trial" << trial;
	string scall = sstream.str();
	system(scall.c_str());
}


