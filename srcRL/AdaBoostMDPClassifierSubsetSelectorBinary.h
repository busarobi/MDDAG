/*
 *  AdaBoostMDPClassifierSubsetSelectorBinary.h
 *  RLTools
 *
 *  Created by Robert Busa-Fekete on 3/17/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef __ADABOOST_MDP_CLASS_SUBSET_BINARY_H
#define __ADABOOST_MDP_CLASS_SUBSEt_BINARY_H

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
#include "cpolicies.h"
//////////////////////////////////////////////////////////////////////
// general includes
//////////////////////////////////////////////////////////////////////
#include "AdaBoostMDPClassifierContinousBinary.h"

using namespace std;

namespace MultiBoost {
	
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	class AdaBoostMDPClassifierSubsetSelectorBinary : public AdaBoostMDPClassifierContinous
	{
	protected:
	public:
		AdaBoostMDPClassifierSubsetSelectorBinary( const nor_utils::Args& args, int verbose, DataReader* datareader);
		virtual ~AdaBoostMDPClassifierSubsetSelectorBinary() {}
		
		double getReward(CStateCollection *oldState, CAction *action, CStateCollection *newState);
		
		///fetches the internal state and stores it in the state object
		virtual void getState(CState *state); ///resets the model 
		
		CStateModifier* getStateSpace();
		CStateModifier* getStateSpace( int divNum );
		CStateModifier* getStateSpace( int divNum, double maxVal );
		void outPutStatistic( BinaryResultStruct& bres );
	};
	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	class AdaBoostMDPClassifierSubsetSelectorBinaryEvaluator : public CRewardPerEpisodeCalculator
	{
	public:
		AdaBoostMDPClassifierSubsetSelectorBinaryEvaluator(CAgent *agent, CRewardFunction *rewardFunction ) : CRewardPerEpisodeCalculator( agent, rewardFunction, 1000, 2000 )
		{
		}
		
		void classficationAccruacy( BinaryResultStruct& binRes, const char* logFileName = NULL )
		{
			double value = 0.0;
			
			agent->addSemiMDPListener(this);
			
			CAgentController *tempController = NULL;
			if (controller)
			{
				tempController = detController->getController();
				detController->setController(controller);	
			}
			
			AdaBoostMDPClassifierSubsetSelectorBinary* classifier = dynamic_cast<AdaBoostMDPClassifierSubsetSelectorBinary*>(semiMDPRewardFunction);
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
				bool isNeg = classifier->hasithLabelCurrentElement(0);
				if (isNeg) // neg
				{
					negNum++;
					if (clRes ) correctN++;					
				} else {
					posNum++;
					if (clRes ) correctP++;			
				}
				
				
				usedClassifierAvg += classifier->getUsedClassifierNumber();
				value += this->getEpisodeValue();
				
				
				if ( logFileName ) {
					output << clRes ? "1" : "0";
					output << " ";
					output << isNeg ? "0" : "1";
					output << " ";
					classifier->getCurrentExmapleResult( currentVotes );
					classifier->getHistory( currentHistory );
					output << currentVotes[0] << " ";
					for( int i=0; i<currentHistory.size(); ++i) output << currentHistory[i] << " ";
					output << endl << flush;
				}
				
				//if ((i>10)&&((i%100)==0))
				//	cout << i << " " << flush;
				
			}
			
			cout << endl;
			
			binRes.avgReward = value/(double)numTestExamples ;
			binRes.usedClassifierAvg = (double)usedClassifierAvg/(double)numTestExamples ;
			binRes.acc = ((double)correct/(double)numTestExamples)*100.0;
			
			binRes.TP = (double)correctP/(double)posNum;
			binRes.TN = (double)correctN/(double)negNum;
			
			//cout << posNum << " " << negNum << endl << flush;
			if (logFileName) output.close();
			
			agent->removeSemiMDPListener(this);
			
			if (tempController)
			{
				detController->setController(tempController);
			}
			
		}
	};
	
	
	
} // end of namespace MultiBoost

#endif // __ADABOOST_MDP_CLASSIFIER_ADV_H

