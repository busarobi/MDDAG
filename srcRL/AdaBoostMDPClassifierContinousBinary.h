/*
 *  AdaBoostMDPClassifierContinousBinary.h
 *  RLTools
 *
 *  Created by Robert Busa-Fekete on 3/13/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __ADABOOST_MDP_CLASS_CONTINOUS_BINARY_H
#define __ADABOOST_MDP_CLASS_CONTINOUS_BINARY_H

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

#include "cadaptivesoftmaxnetwork.h"
//////////////////////////////////////////////////////////////////////
// general includes
//////////////////////////////////////////////////////////////////////
#include "AdaBoostMDPClassifierContinous.h"

using namespace std;

namespace MultiBoost {
	
	//////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////
	struct BinaryResultStruct {
		double origAcc;
		
		double acc;
		double TP;
		double TN;
		double usedClassifierAvg;
		double avgReward;
		
		int iterNumber;
	};
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	class AdaBoostMDPClassifierContinousBinary : public AdaBoostMDPClassifierContinous
	{
	protected:
	public:
		AdaBoostMDPClassifierContinousBinary( const nor_utils::Args& args, int verbose, DataReader* datareader);
		virtual ~AdaBoostMDPClassifierContinousBinary() {}
		
		double getReward(CStateCollection *oldState, CAction *action, CStateCollection *newState);
		
		///fetches the internal state and stores it in the state object
		virtual void getState(CState *state); ///resets the model 
		
		CStateModifier* getStateSpace();
		CStateModifier* getStateSpace( int divNum );
		CStateModifier* getStateSpace( int divNum, double maxVal );
		CStateModifier* getStateSpaceRBF(unsigned int partitionNumber);
		CStateModifier* getStateSpaceRBFAdaptiveCenters(unsigned int numberOfFeatures, CRBFCenterFeatureCalculator** rbfFC, CRBFCenterNetwork** rbfNW);
        CStateModifier* getStateSpaceNN();
        
		void outPutStatistic( BinaryResultStruct& bres );
	};
	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	class AdaBoostMDPClassifierContinousBinaryEvaluator : public CRewardPerEpisodeCalculator
	{
	public:
		AdaBoostMDPClassifierContinousBinaryEvaluator(CAgent *agent, CRewardFunction *rewardFunction ) : CRewardPerEpisodeCalculator( agent, rewardFunction, 1000, 2000 )
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
			
			AdaBoostMDPClassifierContinousBinary* classifier = dynamic_cast<AdaBoostMDPClassifierContinousBinary*>(semiMDPRewardFunction);
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
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////		
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	
	class CRandomDistribution : public CActionDistribution
	{
	protected:
	public:
		//	double epsilon;
		
		CRandomDistribution(double epsilon)
		{
			addParameter("EpsilonGreedy", epsilon);
		}
		
		virtual void getDistribution(CStateCollection *state, CActionSet *availableActions, double *actionValues)
		{
			unsigned int numValues = availableActions->size();
			double epsilon = getParameter("EpsilonGreedy");
			double prop = epsilon / numValues;
			double max = actionValues[0];
			int maxIndex = 0;
			
			for (unsigned int i = 0; i < numValues; i++)
			{
				if (actionValues[i] > max)
				{
					max = actionValues[i];
					maxIndex = i;
				}
				//actionValues[i] = prop;
				actionValues[i] = 1.0/numValues;
			}
			//actionValues[maxIndex] += 1 - epsilon;			
			actionValues[0] = 0.2;
			actionValues[1] = 0.7;
			actionValues[2] = 0.1;
		}
	};
	
	
} // end of namespace MultiBoost

#endif // __ADABOOST_MDP_CLASSIFIER_ADV_H

