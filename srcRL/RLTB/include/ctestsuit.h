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


#ifndef C_TESTSuite__H
#define C_TESTSuite__H

#include <time.h>
#include <stdio.h>

#include "cparameters.h"

#include <map>
#include <string>
#include <iostream>

//using namespace std;

#define ARCF_IDENTITY 1
#define ARCF_LINEAR 2
#define ARCF_AVERAGE 3


class CEvaluator;
class CVisitStateCounter;
class CVisitStateActionCounter;

class CAgent;
class CStateProperties;
class CActionSet;
class CRewardFunction;

class CAbstractVFunction;
class CStateModifier;
class CErrorSender;
class CFeatureVFunction;
class CFeatureQFunction;
		
class CStateList;

class CVAverageTDErrorLearner;
class CVAverageTDVarianceLearner;

class CAgentController;
class CLearnDataObject;
class CAdaptiveParameterCalculator;
class CSemiMDPListener;
class CSemiMarkovDecisionProcess;	

class CPolicyEvaluation;
class CPolicyIteration;
class CGradientLearner;

class CGraphDynamicProgramming;

class CTestSuiteEvaluatorLogger
{
	protected:
		string outputDirectory;
	public:
		int nEpisodesBeforeEvaluate;
	
		CTestSuiteEvaluatorLogger(string outputDirectory); 
		virtual ~CTestSuiteEvaluatorLogger() {};
		
		void setOutputDirectory(string outputDirectory);
		
		virtual void evaluate(string evaluationDirectory, int trial, int numEpisodes) = 0;	
		virtual void startNewEvaluation(string evaluationDirectory, CParameters *parameters, int trial);
		virtual void endEvaluation() {};
};

class CTestSuiteLoggerFromEvaluator : public CTestSuiteEvaluatorLogger
{
	protected:
	 	CEvaluator *evaluator;
		string outputFileName;
	public:
		CTestSuiteLoggerFromEvaluator(string outputDirectory, string outputFileName, CEvaluator *evaluator);
		virtual ~CTestSuiteLoggerFromEvaluator() {};
		
		virtual void evaluate(string evaluationDirectory, int trial, int numEpisodes);
		virtual double evaluateValue(string evaluationDirectory, int trial, int numEpisodes);
		
		virtual void startNewEvaluation(string evaluationDirectory, CParameters *parameters, int trial);
};

class CGraphLogger : public CTestSuiteEvaluatorLogger
{
	protected:
		CStateList *states;
		CGraphDynamicProgramming *graph;


	public:
		CGraphLogger(CStateList *states, CGraphDynamicProgramming *graph);
		virtual ~CGraphLogger();

		virtual void evaluate(string evaluationDirectory, int trial, int numEpisodes);	
		virtual void startNewEvaluation(string evaluationDirectory, CParameters *parameters, int trial);
};

/*
class CAdaptiveTargetGraphLogger : public CAdaptiveTargetGraphLogger
{
	protected:
		CAdaptiveTargetGraphDynamicProgramming *graph;


	public:
		CAdaptiveTargetGraphLogger(CStateList *states, CGraphDynamicProgramming *graph);
		virtual ~CAdaptiveTargetGraphLogger();

		virtual void evaluate(string evaluationDirectory, int trial, int numEpisodes);	
		virtual void startNewEvaluation(string evaluationDirectory, CParameters *parameters, int trial);
};*/


class CMatlabEpisodeOutputLogger : public CTestSuiteEvaluatorLogger
{
	protected:
		CAgent *agent;
		int nEpisodes;
		int nSteps;	

		CStateProperties *modifier;
		CActionSet *actions;
		
		CRewardFunction *rewardFunction;
	public:
		CMatlabEpisodeOutputLogger( CAgent *agent, CRewardFunction *rewardFunction, CStateProperties *modifier, CActionSet *actions, int nEpisodes, int nSteps);
		virtual ~CMatlabEpisodeOutputLogger();
		
		virtual void evaluate(string evaluationDirectory, int trial, int numEpisodes);	
		virtual void startNewEvaluation(string evaluationDirectory, CParameters *parameters, int trial);
};

class CMatlabVAnalyzerLogger : public CTestSuiteEvaluatorLogger
{
	protected:
		CAbstractVFunction *vFunction;
		
		std::list<CStateModifier *> *modifiers;

		CErrorSender *vLearner;
	
		CFeatureVFunction *visitCounter;
		CFeatureVFunction *averageError;
		CFeatureVFunction *averageVariance;

		int dim1;
		int dim2;		

		int part1;
		int	part2;
	
		CStateList *states;

	public:
		int nTrialEvaluate;		
	
		CVisitStateCounter *visitCounterLearner;
			
		CVAverageTDErrorLearner *errorLearner;
		CVAverageTDVarianceLearner *varianceLearner;
	
		CMatlabVAnalyzerLogger(CAbstractVFunction *l_vFunction, CFeatureCalculator *featCalc, CErrorSender *l_vLearner, CStateList *l_States, int l_dim1, int l_dim2, int l_part1, int l_part2, std::list<CStateModifier *> *l_modifiers);
		virtual ~CMatlabVAnalyzerLogger();
		
		virtual void evaluate(string evaluationDirectory, int trial, int numEpisodes);	
		virtual void startNewEvaluation(string evaluationDirectory, CParameters *parameters, int trial);
		
		virtual void addListenersToAgent(CSemiMDPSender *agent);
		virtual void removeListenersToAgent(CSemiMDPSender *agent);
};

class CMatlabQAnalyzerLogger : public CMatlabVAnalyzerLogger
{
	protected:
		CFeatureQFunction *qFunction;
		CFeatureQFunction *saVisits;
		
		bool delVFunction;
	public:
		CVisitStateActionCounter *visitStateActionCounterLearner;
		
		CMatlabQAnalyzerLogger(CFeatureQFunction *l_qFunction, CFeatureCalculator *featCalc, CErrorSender *l_vLearner, CStateList *l_States, int l_dim1, int l_dim2, int l_part1, int l_part2, std::list<CStateModifier *> *l_modifiers);
		CMatlabQAnalyzerLogger(CFeatureVFunction *vFunction, CFeatureQFunction *l_qFunction, CFeatureCalculator *featCalc, CErrorSender *l_vLearner, CStateList *l_States, int l_dim1, int l_dim2, int l_part1, int l_part2, std::list<CStateModifier *> *l_modifiers);
		virtual ~CMatlabQAnalyzerLogger();
		
		virtual void evaluate(string evaluationDirectory, int trial, int numEpisodes);	
		virtual void startNewEvaluation(string evaluationDirectory, CParameters *parameters, int trial);
		
		virtual void addListenersToAgent(CSemiMDPSender *agent);
		virtual void removeListenersToAgent(CSemiMDPSender *agent);
};

class CTestSuite :  virtual public CParameterObject
{
protected:

	CAgentController *controller;
	CAgentController *evaluationController;
	std::list<CLearnDataObject *> *learnDataObjects;
	
	std::map<CLearnDataObject *, bool> *saveLearnData;
	
	std::list<CAdaptiveParameterCalculator *> *paramCalculators;

	CAgent *agent;

	string testSuiteName;
	string learnDataFileName;

public:
	CTestSuite(CAgent *agent, CAgentController *controller, CLearnDataObject *vFunction, char *testSuiteName);
	CTestSuite(CAgent *agent, CAgentController *controller, CAgentController *evaluationController, CLearnDataObject *vFunction, char *testSuiteName);
	virtual ~CTestSuite();
	
	virtual void addParamCalculator(CAdaptiveParameterCalculator *paramCalculator);
	virtual void resetParamCalculators();
	
	virtual void saveLearnedData(FILE *stream);
	virtual void loadLearnedData(FILE *stream);

	virtual void resetLearnedData();
	
	void addLearnDataObject(CLearnDataObject *learnDataObject, bool saveLearnData = true);

	virtual void learn(int nEpisodes, int nStepsPerEpisode) = 0;

	virtual CAgentController *getController();
	virtual void setController(CAgentController *controller);
	virtual CAgentController *getEvaluationController();
	virtual void setEvaluationController(CAgentController *evaluationController);

	virtual void deleteObjects();

	string getTestSuiteName();
	void setTestSuiteName(string name);
};



class CListenerTestSuite : public CTestSuite
{
protected:
	std::list<CSemiMDPListener *> *learnerObjects;
	std::map<CSemiMDPListener *, CSemiMarkovDecisionProcess *> *addToAgent;
public:
	CListenerTestSuite(CAgent *agent, CSemiMDPListener *learner, CAgentController *controller, CLearnDataObject *vFunction, char *testSuiteName);
	CListenerTestSuite(CAgent *agent, CSemiMDPListener *learner, CAgentController *controller, CAgentController *evaluationController, CLearnDataObject *vFunction, char *testSuiteName);

	virtual ~CListenerTestSuite();

	virtual void addLearnersToAgent();
	virtual void removeLearnersFromAgent();

	void addLearnerObject(CSemiMDPListener *listener, bool addParams = true, bool addBack = true, CSemiMarkovDecisionProcess *remove = NULL);

	virtual void learn(int nEpisodes, int nStepsPerEpisode);
	virtual void deleteObjects();
	
	std::list<CSemiMDPListener *> *getLearnerList() {return learnerObjects;};
};

class CPolicyEvaluation;

class CPolicyEvaluationTestSuite : public CTestSuite
{
protected:
	CPolicyEvaluation *evaluation;
public:
	CPolicyEvaluationTestSuite(CAgent *agent, CPolicyEvaluation *learner, CAgentController *controller, CLearnDataObject *vFunction, char *testSuiteName);

	virtual ~CPolicyEvaluationTestSuite();

	virtual void learn(int nEpisodes, int nStepsPerEpisode);
	
	virtual void resetLearnedData();
};

class CPolicyIteration;

class CPolicyIterationTestSuite : public CTestSuite
{
protected:
	CPolicyIteration *policyIteration;
public:
	CPolicyIterationTestSuite(CAgent *agent, CPolicyIteration *policyIteration, CAgentController *controller, CLearnDataObject *vFunction, char *testSuiteName);

	virtual ~CPolicyIterationTestSuite();

	virtual void learn(int nEpisodes, int nStepsPerEpisode);

	virtual void resetLearnedData();
};

class CPolicyGradientTestSuite : public CTestSuite
{
protected:
	CGradientLearner *learner;

public:
	CPolicyGradientTestSuite(CAgent *agent, CGradientLearner *learner, CAgentController *controller, CLearnDataObject *vFunction, char *testSuiteName, int nMaxGradientUpdates = 1);
	CPolicyGradientTestSuite(CAgent *agent, CGradientLearner *learner, CAgentController *controller, CAgentController *evaluationController, CLearnDataObject *vFunction, char *testSuiteName, int nMaxGradientUpdates = 1);

	virtual ~CPolicyGradientTestSuite();
	virtual void deleteObjects();

	virtual void learn(int nEpisodes, int nStepsPerEpisode);
	virtual void resetLearnedData();
};

class CTestSuiteCollection 
{
protected:
	std::map<string, CTestSuite *> *testSuiteMap;
	std::list<void *> *objectsToDelete;
public:
	CTestSuiteCollection();
	virtual ~CTestSuiteCollection();

	void addTestSuite(CTestSuite *testSuite);
	void removeTestSuite(CTestSuite *testSuite);

	void removeAllTestSuites();

	int getNumTestSuites();
	CTestSuite *getTestSuite(string testSuiteName);
	CTestSuite *getTestSuite(int index);
	
	void addObjectToDelete(void *object);
	void deleteObjects();
};

typedef struct 
{
	double averageValue;
	double bestValue;
	unsigned int trialNumber;
	double evaluationTime;
	string evaluationDate;
} EvaluationValue;

typedef std::list<EvaluationValue> EvaluationValues;

class CTestSuiteEvaluator : virtual public CParameterObject
{
protected:
	CAgent *agent;
	
	std::list<CTestSuiteEvaluatorLogger *> *evaluators;
	
	string baseDirectory;
	CTestSuite *testSuite;

	unsigned int nTrials;
	unsigned int trialNumber;
	
	bool exception;

	std::list<CParameters *> *parameterList;
	std::map<CParameters *, EvaluationValues *> *evaluations;

	virtual void newEvaluationTrial(CTestSuite *testSuite, EvaluationValue *evaluationData) = 0;
	virtual void doEpisode(CTestSuite *testSuite, int nEpisode) = 0;
	virtual void getEvaluationValue(EvaluationValue *evaluationData) = 0;
	virtual bool isFinished(int unsigned nEpisode) = 0;
	
	CParameters *getParametersObject(CParameters *);
	
public:
	CTestSuiteEvaluator(CAgent *agent, string baseDirectory, CTestSuite *testSuite, int nTrials);
	virtual ~CTestSuiteEvaluator();
	
	string getEvaluationDirectory();
	int getNewTrialNumber();
	
	string getLearnDataFileName(int trialNumber);

	void checkDirectories();

	virtual void loadEvaluationData(string filename);
	virtual void saveEvaluationData(string filename);
	virtual void saveEvaluationDataMatlab(string filename);
	
	
	virtual void doEvaluationTrial(CParameters *testSuite, EvaluationValue *evaluationData);
	virtual void evaluateParameters(CParameters *testSuite);

	virtual double getAverageValue(CParameters *testSuite);
	virtual double getBestValue(CParameters *testSuite);
	
	virtual EvaluationValues *getEvaluationValues();
	
	virtual void addPolicyEvaluator(CTestSuiteEvaluatorLogger *evaluator);
};

/*
class CTestSuiteNeededStepsEvaluator : public CTestSuiteEvaluator
{
protected:
	std::list<double *> *succeded;

	unsigned int totalLearnEpisodes;
	unsigned int stepsLearnEpisode;
	unsigned int nTrials;
	unsigned int episodesBeforeEvaluate;

	unsigned int nValues;

	bool maxStepsSucceded;
public: 
	CTestSuiteNeededStepsEvaluator(CAgent *agent, string testSuiteCollectionName, int totalLearnEpisodes, int stepsLearnEpisode, int episodesBeforeEvaluate, int nTrials, bool maxStepsSucceded = false);
	virtual ~CTestSuiteNeededStepsEvaluator();

	virtual void loadEvaluationData(CParameters *testSuite, FILE *file);

	virtual void doEvaluationTrial(CTestSuite *testSuiteName, FILE *evaluationFile, const char *learnDataFileName);

	virtual double getEvaluationValue(std::list<double *> *values);

	double getPercentageSucceded();

	virtual void clearValues();

};
*/

class CAverageRewardTestSuiteEvaluator : public CTestSuiteEvaluator
{
protected:
	int numEvals;
	
	double bestValue;
	double averageValue;
	
	CEvaluator *evaluator;
			
	virtual void newEvaluationTrial(CTestSuite *testSuite, EvaluationValue *evaluationData);
	virtual void doEpisode(CTestSuite *testSuite, int nEpisode);
	virtual void getEvaluationValue(EvaluationValue *evaluationData);
	virtual bool isFinished(unsigned int nEpisode);

	FILE *evaluationFile;
public:
	unsigned int episodesBeforeEvaluate;
	unsigned int totalLearnEpisodes;
	unsigned int stepsLearnEpisode;

	CAverageRewardTestSuiteEvaluator(CAgent *agent, string baseDirectory, CTestSuite *testSuite, CEvaluator *evaluator, int totalLearnEpisodes, int episodesBeforeEvaluate, int stepsLearnEpisode, int nTrials);
	virtual ~CAverageRewardTestSuiteEvaluator();
};



#endif

