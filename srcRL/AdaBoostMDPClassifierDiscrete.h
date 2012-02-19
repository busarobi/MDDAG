/*
 *  AdaBoostMDPClassifierDiscrete.h
 *  RLTools
 *
 *  Created by Robert Busa-Fekete on 3/13/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef __ADABOOST_MDP_CLASS_DISCRETE_H
#define __ADABOOST_MDP_CLASS_DISCRETE_H

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
#include "AdaBoostMDPClassifierContinous.h"
#include "AdaBoostMDPClassifierContinousBinary.h"

using namespace std;

namespace MultiBoost {
	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////
	class AdaBoostMDPClassifierDiscrete : public	CEnvironmentModel,	public CRewardFunction
	{
	protected: 
		// 
		int						_verbose;
		int						_classNum;
		int						_classifierNumber; // number of classifier used during the episode
		
		// rewards
		double					_classificationReward;
		double					_skipReward;
		double					_jumpReward;
		double					_successReward;
		
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
		
		vector<bool>			_classifierUsed; // store which classifier was used during the process
		
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
		
		void setCurrentDataToTrain() { _data->setCurrentDataToTrain(); }
		void setCurrentDataToTest() { _data->setCurrentDataToTest(); }		
		double getAccuracyOnCurrentDataSet(){ return _data->getAccuracyOnCurrentDataSet(); }
		
		void outPutStatistic( double acc, double curracc, double uc, double sumrew );
		
		// constructor
		AdaBoostMDPClassifierDiscrete(const nor_utils::Args& args, int verbose, DataReader* datareader );
		// destructor
		virtual	~AdaBoostMDPClassifierDiscrete() 
		{
			_outputStream.close();
		}
		
		///returns the reward for the transition, implements the CRewardFunction interface
		virtual	double	getReward( CStateCollection	*oldState , CAction *action , CStateCollection *newState);
		
		///fetches the internal state and stores it in the state object
		virtual void getState(CState *state); ///resets the model 
		virtual	void doResetModel();		
		
		// get the discretized state space
		virtual CAbstractStateDiscretizer* getStateSpace();

		
		// classify correctly
		bool classifyCorrectly();
		
		void outPutStatistic( BinaryResultStruct& bres );
		
		bool hasithLabelCurrentElement( int i );
		
		void getHistory( vector<bool>& history );
		void getCurrentExmapleResult( vector<double>& result );
		
	};
	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	

	
	
	template <typename T>
	class AdaBoostMDPBinaryDiscreteEvaluator : public CRewardPerEpisodeCalculator
	{
	public:
		AdaBoostMDPBinaryDiscreteEvaluator(CAgent *agent, CRewardFunction *rewardFunction ) : CRewardPerEpisodeCalculator( agent, rewardFunction, 1000, 2000 )
		{
		}
		
		void classficationAccruacy( BinaryResultStruct& binRes, const string &logFileName )
		{
			double value = 0.0;
            double negNumEval = 0.0;
            
			agent->addSemiMDPListener(this);
			
			CAgentController *tempController = NULL;
			if (controller)
			{
				tempController = detController->getController();
				detController->setController(controller);	
			}
			
			T* classifier = dynamic_cast<T*>(semiMDPRewardFunction);
			const int numTestExamples = classifier->getNumExamples();
			int  correct = 0, notcorrect = 0;
			int usedClassifierAvg=0;
			int correctP=0;
			int posNum=0;
			int correctN=0;
			int negNum=0;
			ofstream output;
			vector<double> currentVotes(0);
			vector<bool> currentHistory(0);
			
			if ( !logFileName.empty() )
			{
				output.open( logFileName.c_str() );			
				cout << "Output classfication result: " << logFileName << endl;
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
				bool isNeg = !classifier->hasithLabelCurrentElement(classifier->getPositiveLabelIndex());
				if (isNeg) // neg
				{
					negNum++;
					if (clRes ) correctN++;					
				} else {
					posNum++;
					if (clRes ) correctP++;			
				}
				
				
                double numEval = classifier->getUsedClassifierNumber();
				usedClassifierAvg += numEval;
				value += this->getEpisodeValue();
				
                if (isNeg) {
                    negNumEval += numEval;
                }
				
				if ( !logFileName.empty() ) {
					output << (clRes ? "1" : "0");
					output << " ";					
					output << (isNeg ? "2" : "1");
					output << " ";
					classifier->getCurrentExmapleResult( currentVotes );
					classifier->getHistory( currentHistory );
					output << currentVotes[classifier->getPositiveLabelIndex()] << " ";
					//for( int i=0; i<currentHistory.size(); ++i) output << currentHistory[i] << " ";
					for( int i=0; i<currentHistory.size(); ++i) 
					{ 
						if ( currentHistory[i] )
							output << i+1 << " ";
					}
					
					output << endl << flush;
				}
				
				//if ((i>10)&&((i%100)==0))
				//	cout << i << " " << flush;
				
			}
			
			cout << endl;
			
			binRes.avgReward = value/(double)numTestExamples ;
			binRes.usedClassifierAvg = (double)usedClassifierAvg/(double)numTestExamples ;
			binRes.negNumEval = (double)negNumEval/(double)negNum;
            
            binRes.acc = ((double)correct/(double)numTestExamples)*100.0;
			
			binRes.TP = (double)correctP/(double)posNum;
			binRes.TN = (double)correctN/(double)negNum;
			
			//cout << posNum << " " << negNum << endl << flush;
			if (!logFileName.empty()) output.close();
			
			agent->removeSemiMDPListener(this);
			
			if (tempController)
			{
				detController->setController(tempController);
			}
			
		}
	};
			
	////////////////////////////////////////////////////////////////////////////////////////////////		
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	
	
} // end of namespace MultiBoost

#endif // __ADABOOST_MDP_CLASSIFIER_ADV_H

