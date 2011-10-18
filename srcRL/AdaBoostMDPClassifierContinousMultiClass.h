/*
 *  AdaBoostMDPClassifierContinousMultiClass.h
 *  RLTools
 *
 *  Created by Robert Busa-Fekete on 3/15/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef __ADABOOST_MDP_CLASS_CONTINOUS_MH_H
#define __ADABOOST_MDP_CLASS_CONTINOUS_MH_H

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

#include "cadaptivesoftmaxnetwork.h"
//////////////////////////////////////////////////////////////////////
// general includes
//////////////////////////////////////////////////////////////////////
#include "AdaBoostMDPClassifierContinous.h"

using namespace std;

namespace MultiBoost {
	
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	class AdaBoostMDPClassifierContinousMH : public AdaBoostMDPClassifierContinous
	{
	protected:
	public:
		AdaBoostMDPClassifierContinousMH( const nor_utils::Args& args, int verbose, DataReader* datareader, int classNum);
		virtual ~AdaBoostMDPClassifierContinousMH() {}				
			
		double getReward(CStateCollection *oldState, CAction *action, CStateCollection *newState);
		
		///fetches the internal state and stores it in the state object
		virtual void getState(CState *state); ///resets the model 
								
		// get the discretized state space		
		CStateModifier* getStateSpace();
		CStateModifier* getStateSpace( int divNum );
		CStateModifier* getStateSpaceExp( int divNum, int e );
		CStateModifier* getStateSpace( int divNum, double maxVal );
		CStateModifier* getStateSpaceRBF(unsigned int partitionNumber);
		CStateModifier* getStateSpaceRBFAdaptiveCenters(unsigned int numberOfFeatures, CRBFCenterFeatureCalculator** rbfFC);
        CStateModifier* getStateSpaceNN();
		
        CStateModifier* getStateSpaceForGSBNFQFunction( int numOfFeatures);
	};
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////		
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	
	
} // end of namespace MultiBoost

#endif // __ADABOOST_MDP_CLASSIFIER_ADV_H

