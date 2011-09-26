/*
 *  AdaBoostMDPClassifier.h
 *  RLTools
 *
 *  Created by Robert Busa-Fekete on 3/3/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

/*
 Ket akcio van:
 kihagy, nincs koltsege
 osztalyoz, koltsege 1
 */



#ifndef __ADABOOST_MDP_CLASSIFIER_H
#define __ADABOOST_MDP_CLASSIFIER_H


#include "WeakLearners/BaseLearner.h"
#include "IO/InputData.h"
#include "Utils/Utils.h"
#include "IO/Serialization.h"
#include "IO/OutputInfo.h"
#include "Classifiers/AdaBoostMHClassifier.h"
#include "Classifiers/ExampleResults.h"

#include "WeakLearners/SingleStumpLearner.h" // for saveSingleStumpFeatureData

// for RL toolbox
#include "cgridworldmodel.h"
#include "cevaluator.h"
#include "cagent.h"

#include <iomanip> // for setw
#include <cmath> // for setw
#include <functional>

#include "Utils/Args.h"
#include "Defaults.h"

#include <string>
#include <cassert>

using namespace std;

namespace MultiBoost {
	
	//////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////
	
	// Forward declarations.
	class ExampleResults;
	class InputData;
	class BaseLearner;
	
	
	
	class AdaBoostMDPClassifier : public CGridWorldModel {
	public:
		AdaBoostMDPClassifier(const nor_utils::Args& args, int verbose) : _pData( NULL ), _pTestData(NULL), _args(args), _verbose(verbose), _exampleResult(NULL) , CGridWorldModel(0,0,0)
		{
		}
		
		virtual double getReward(CStateCollection *oldState, CAction *action, CStateCollection *newState);
				
		virtual void transitionFunction(CState *oldstate, CAction *action, CState *newState, CActionData *);

		void getResetState(CState *resetState);
		
		void initEpisode()
		{
			// chose an instance
			const int numClasses = _pData->getNumClasses();
			const int numExamples = _pData->getNumExamples();
			
			_numOfCalls = 0;
			_history.clear();
			_isClassified=false;
			_classifierNumber=0;
			
			// initialize Example result
			_currentInstance = (int) (rand() % numExamples );
			
			clearResult();
		}
		
		void clearResult()
		{
			const int numClasses = _pData->getNumClasses();

			if (_exampleResult) delete _exampleResult;
			_exampleResult = new ExampleResults( _currentInstance, numClasses );
		}
		
		void classifyKthWeakLearner( int wHypInd )		
		{
			
			if (_verbose>3) {
				//cout << "Classifiying: " << wHypInd << endl;
			}
			
			
			if ( wHypInd >= _numIterations ) return;						
			
			const int numClasses = _pData->getNumClasses();
			
			BaseLearner* currWeakHyp = _weakHypotheses[wHypInd];
			float alpha = currWeakHyp->getAlpha();
			
			// a reference for clarity and speed
			vector<AlphaReal>& currVotesVector = _exampleResult->getVotesVector();
			
			// for every class
			for (int l = 0; l < numClasses; ++l)
				currVotesVector[l] += alpha * currWeakHyp->classify(_pData, _currentInstance, l);
		}
		
		void printVotes()
		{
			const int numClasses = _pData->getNumClasses();
			// a reference for clarity and speed
			vector<AlphaReal>& currVotesVector = _exampleResult->getVotesVector();
			
			for (int l = 0; l < numClasses; ++l)
				cout << currVotesVector[l] << " ";
			
			cout << endl << flush;
		}
		
		
		// classification result for MDP set and for the current element for clalculating the reward
		bool currentClassifyingResult();
		
		virtual void init();
		
		int getCurrentInstance() { return _currentInstance; }
		
		int getUsedClassifierNumber() { return _classifierNumber; }
		
		virtual ~AdaBoostMDPClassifier()
		{
			delete _pData;
			delete _pTestData;

			_outputStream.close();
		}
		
		void setClassificationReward( double r ) { _classificationReward=r; }
		void setSkipReward( double r ) { _skipReward=r; }		
		void setJumpReward( double r ) { _jumpReward=r; }
		
		int getGetSizeOfDataSet() { return _pData->getNumExamples(); }
		int getGetSizeOfTestDataSet() { return _pTestData->getNumExamples(); }
		
		double classifyTestMDP();				
		bool classifyTestMDP(int i );		// ith test element
		
		
		double classifyTest();
		
		
		bool isithIsUsed( int i ) { return _history[i]; };
		int getLengthOfHistory() { return _history.size(); }
		int getWeakHypNumber() { return _weakHypotheses.size(); }
		
		void outPutStatistic( double acc, double curracc, double uc, double sumrew );
	protected:
		void loadInputData(const string& dataFileName, const string& testDataFileName, const string& shypFileName);		
		void createGridWorld();
 		
		ExampleResults*			_exampleResult;
		int						_numIterations;
		int						_verbose;		
		const nor_utils::Args&  _args;  //!< The arguments defined by the user.		
		
		InputData*				_pData;
		InputData*				_pTestData;
		
		int						_currentInstance;
		vector<BaseLearner*>	_weakHypotheses;		
		
		double					_classificationReward;
		double					_skipReward;
		double					_jumpReward;
		
		long int				_numOfCalls;
		vector<bool>			_history;
		bool					_isClassified;
		int						_classifierNumber;	
		
		ofstream				_outputStream;
	};
	
	
	class CGridWorldActionAdaBoost : public CGridWorldAction 
	{
	protected:
		int _mode; // 0 skip, 1 classify, 2 jump to the end
	public:
		CGridWorldActionAdaBoost( int mode ) : CGridWorldAction(1,0), _mode(mode) {};
		int getMode() { return _mode;}
	};
	
	class AdaBoostMDPClassifierEvaluator : public CRewardPerEpisodeCalculator
	{
	public:
		AdaBoostMDPClassifierEvaluator(CAgent *agent, CRewardFunction *rewardFunction) : CRewardPerEpisodeCalculator( agent, rewardFunction, 1000, 2000 )
		{
		}
		
		double classficationAccruacy( double& acc, double& usedClassifierAvg, char* logFileName )
		{
			double value = 0;
			
			agent->addSemiMDPListener(this);
			
			CAgentController *tempController = NULL;
			if (controller)
			{
				tempController = detController->getController();
				detController->setController(controller);	
			}
			
			AdaBoostMDPClassifier* classifier = dynamic_cast<AdaBoostMDPClassifier*>(semiMDPRewardFunction);
			const int numTestExamples = classifier->getGetSizeOfTestDataSet();
			int  correct = 0, notcorrect = 0;
			usedClassifierAvg=0;
			
			ofstream output( logFileName );
			
			cout << "Output classfication reult: " << logFileName << endl;
			
			for (int i = 0; i < numTestExamples; i ++)
			{
				//cout << i << endl;
				agent->startNewEpisode();				
				//cout << "Length of history: " << classifier->getLengthOfHistory() << endl;
				
				agent->doControllerEpisode(1, classifier->getSizeX());
				//cout << "Length of history: " << classifier->getLengthOfHistory() << endl;
				
				//cout << "Intance: " << i << '\t' << "Num of classifier: " << classifier->getUsedClassifierNumber() << endl;
				
				if ( classifier->classifyTestMDP( i ) )
				{
					correct++;
					output << "1"; 
				} else {
					notcorrect++;
					output << "0";
				}
				usedClassifierAvg += classifier->getUsedClassifierNumber();
				
				for(int j=0; j < classifier->getLengthOfHistory(); ++j )
				{
					if (classifier->isithIsUsed( j ) )
						output << " 1";
					else
						output << " 0";
				}
				output << endl;
				
				value += this->getEpisodeValue();
				
				if ((i>10)&&((i%100)==0))
					cout << i << " " << flush;
				
			}
			
			cout << endl;
			
			value /= nEpisodes;
			usedClassifierAvg /= nEpisodes;
			acc = ((double)correct/(double)numTestExamples)*100.0;
			
			output.close();
			
			agent->removeSemiMDPListener(this);
			
			if (tempController)
			{
				detController->setController(tempController);
			}
			
			return value;		
		}
	};
	
} // end of namespace MultiBoost

#endif // __ADABOOST_MDP_CLASSIFIER_H
