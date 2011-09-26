/*
 *  AdaBoostMDPClassifierContinous.h
 *  RLTools
 *
 *  Created by Robert Busa-Fekete on 3/11/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __ADABOOST_MDP_CLASS_CONTINOUS_H
#define __ADABOOST_MDP_CLASS_CONTINOUS_H

//////////////////////////////////////////////////////////////////////
// for multiboost
//////////////////////////////////////////////////////////////////////
#include "WeakLearners/BaseLearner.h"
#include "IO/InputData.h"
#include "Utils/Utils.h"
#include "IO/Serialization.h"
#include "IO/OutputInfo.h"
#include "Classifiers/AdaBoostMHClassifier.h"
#include "Classifiers/ExampleResults.h"

//////////////////////////////////////////////////////////////////////
// for RL toolbox
//////////////////////////////////////////////////////////////////////
#include "cenvironmentmodel.h"
#include "crewardfunction.h"
#include "caction.h"
#include "cdiscretizer.h"
#include "cevaluator.h"
#include "cagent.h"
#include "cdiscretizer.h"
#include "cstate.h"
#include "cstatemodifier.h"
#include "clinearfafeaturecalculator.h"
//////////////////////////////////////////////////////////////////////
// general includes
//////////////////////////////////////////////////////////////////////
#include "AdaBoostMDPClassifierAdv.h"

using namespace std;

namespace MultiBoost {
	////////////////////////////////////////////////////////////////////////////////////////////////	
	enum SuccesRewardModes {
		RT_HAMMING,
		RT_EXP
	};
	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////
	class AdaBoostMDPClassifierContinous : public	CEnvironmentModel,	public CRewardFunction
	{
	protected: 
		// 
		int						_verbose;
		int						_classNum;
		int						_classifierNumber; // number of classifier used during the episode
		vector<bool>			_classifierUsed; // store which classifier was used during the process
		
		// rewards
		double					_classificationReward;
		double					_skipReward;
		double					_jumpReward;
		double					_successReward;
		SuccesRewardModes		_succRewardMode;

		DataReader*				_data;
		
		/// internal state variables 
		ExampleResults*			_exampleResult;
		int						_currentClassifier;
		double					_currentSumAlpha;
		
		// for output info
		ofstream				_outputStream;
		
		//!< The arguments defined by the user.		
		const nor_utils::Args&		_args;  
		
		/// calculate the next state based on the action virtual 
		void doNextState(CPrimitiveAction *act);
		
		// this instance will be used in this episode
		int						_currentRandomInstance;
		
		// contain the sum of alphas
		double					_sumAlpha;
	public:
		// set randomzed element
		void setCurrentRandomIsntace( int r ) { _currentRandomInstance = r; }		
		void setRandomizedInstance() {_currentRandomInstance = (int) (rand() % _data->getNumExamples() ); }
		
		// getter setter
		int getUsedClassifierNumber() { return _classifierNumber; }
		void setClassificationReward( double r ) { _classificationReward=r; }
		void setSkipReward( double r ) { _skipReward=r; }		
		void setJumpReward( double r ) { _jumpReward=r; }
		void setSuccessReward( double r ) { _successReward=r; }
		
		int getIterNum() { return _data->getIterationNumber(); };
		int getNumClasses() { return _data->getClassNumber(); };
		int getNumExamples() { return _data->getNumExamples(); }
		void getHistory( vector<bool>& history );
		void getCurrentExmapleResult( vector<double>& result );
		
		void setCurrentDataToTrain() { _data->setCurrentDataToTrain(); }
		void setCurrentDataToTest() { _data->setCurrentDataToTest(); }		
		double getAccuracyOnCurrentDataSet(){ return _data->getAccuracyOnCurrentDataSet(); }
		
		void outPutStatistic( double acc, double curracc, double uc, double sumrew );
		
		
		
		// constructor
		AdaBoostMDPClassifierContinous(const nor_utils::Args& args, int verbose, DataReader* datareader, int classNum, int discState);
		// destructor
		virtual	~AdaBoostMDPClassifierContinous() 
		{
			_outputStream.close();
		}
		
		///returns the reward for the transition, implements the CRewardFunction interface
		virtual	double	getReward( CStateCollection	*oldState , CAction *action , CStateCollection *newState);
		
		///fetches the internal state and stores it in the state object
		virtual void getState(CState *state); ///resets the model 
		virtual	void doResetModel();		
		
		// get the discretized state space
		virtual CStateModifier* getStateSpace();
		virtual CStateModifier* getStateSpaceRBF(unsigned int partitionNumber);
		virtual CStateModifier* getStateSpaceTileCoding(unsigned int partitionNumber);
		
		// classify correctly
		bool classifyCorrectly();
		bool hasithLabelCurrentElement( int i );
	};
	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////					
	class AdaBoostMDPClassifierContinousEvaluator : public CRewardPerEpisodeCalculator
	{
	public:
		AdaBoostMDPClassifierContinousEvaluator(CAgent *agent, CRewardFunction *rewardFunction) : CRewardPerEpisodeCalculator( agent, rewardFunction, 1000, 2000 )
		{
		}
		
		double classficationAccruacy( double& acc, double& usedClassifierAvg, const char* logFileName = NULL)
		{
			double value = 0;
			
			agent->addSemiMDPListener(this);
			
			CAgentController *tempController = NULL;
			if (controller)
			{
				tempController = detController->getController();
				detController->setController(controller);	
			}
			
			AdaBoostMDPClassifierContinous* classifier = dynamic_cast<AdaBoostMDPClassifierContinous*>(semiMDPRewardFunction);
			const int numTestExamples = classifier->getNumExamples();
			const int numClasses = classifier->getNumClasses();
			int  correct = 0, notcorrect = 0;
			usedClassifierAvg=0;
			
			//ofstream output( logFileName );
			
			//cout << "Output classfication reult: " << logFileName << endl;
			ofstream output;
			vector<double> currentVotes(0);
			vector<bool> currentHistory(0);
			
			if ( logFileName )
			{
				output.open( logFileName );			
				cout << "Output classfication reult: " << logFileName << endl;
			}
			
			for (int i = 0; i < numTestExamples; i ++)
			{
				
				
				//cout << i << endl;
				agent->startNewEpisode();				
				//cout << "Length of history: " << classifier->getLengthOfHistory() << endl;
				classifier->setCurrentRandomIsntace(i);
				agent->doControllerEpisode(1, classifier->getIterNum()*2 );
				//cout << "Length of history: " << classifier->getLengthOfHistory() << endl;
				
				//cout << "Intance: " << i << '\t' << "Num of classifier: " << classifier->getUsedClassifierNumber() << endl;
				bool clRes = classifier->classifyCorrectly();				
				if (clRes ) correct++;
				else notcorrect++;
				
				usedClassifierAvg += classifier->getUsedClassifierNumber();
				value += this->getEpisodeValue();
				
				//if ((i>10)&&((i%100)==0))
				//	cout << i << " " << flush;
				if ( logFileName ) {
					output << (clRes ? "1" : "0");
					output << " ";
					
					//output << (isNeg ? "1" : "2");
					output << " ";
					classifier->getCurrentExmapleResult( currentVotes );
					classifier->getHistory( currentHistory );
					for( int l=0; l<numClasses; ++l ) output << currentVotes[l] << " ";
					//for( int i=0; i<currentHistory.size(); ++i) output << currentHistory[i] << " ";
					for( int i=0; i<currentHistory.size(); ++i) 
					{ 
						if ( currentHistory[i] )
							output << i+1 << " ";
					}
					
					output << endl << flush;
				}
				
			}
			
			cout << endl;
			
			value /= (double)numTestExamples ;
			usedClassifierAvg /= (double)numTestExamples ;
			acc = ((double)correct/(double)numTestExamples)*100.0;
			
			//output.close();
			if (logFileName) output.close();
			
			agent->removeSemiMDPListener(this);
			
			if (tempController)
			{
				detController->setController(tempController);
			}
			
			return value;		
		}
	};
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////		
	//class AdaBoostMDPClassifierContinousWitWHInd : public AdaBoostMDPClassifierContinous
	//{
	//	AdaBoostMDPClassifierContinousWitWHInd(const nor_utils::Args& args, int verbose, DataReader* datareader, int classNum );
	//	virtual ~AdaBoostMDPClassifierContinousWitWHInd() {}
	//};
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	class  AdaBoostMDPClassifierSimpleDiscreteSpace : public CAbstractStateDiscretizer
	{
	protected:
		unsigned int _stateNum;
		
	public:
		AdaBoostMDPClassifierSimpleDiscreteSpace(unsigned int stateNum) : CAbstractStateDiscretizer(stateNum), _stateNum( stateNum ) {}
		virtual ~AdaBoostMDPClassifierSimpleDiscreteSpace() {};
		
		virtual unsigned int getDiscreteStateNumber(CStateCollection *state)
		{
			int stateIndex = state->getState()->getDiscreteState(0);
			return stateIndex;			
		}
	};
	
	////////////////////////////////////////////////////////////////////////////////////////////////		
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	
	
} // end of namespace MultiBoost

#endif // __ADABOOST_MDP_CLASSIFIER_ADV_H

