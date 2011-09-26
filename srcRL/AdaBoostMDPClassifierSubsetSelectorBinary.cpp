/*
 *  AdaBoostMDPClassifierSubsetSelectorBinary.cpp
 *  RLTools
 *
 *  Created by Robert Busa-Fekete on 3/17/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "AdaBoostMDPClassifierSubsetSelectorBinary.h"

#include "cstate.h"
#include "cstateproperties.h"
#include "clinearfafeaturecalculator.h"

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {
	
	
	
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	AdaBoostMDPClassifierSubsetSelectorBinary::AdaBoostMDPClassifierSubsetSelectorBinary( const nor_utils::Args& args, int verbose, DataReader* datareader )
	: AdaBoostMDPClassifierContinous( args, verbose, datareader, 1, 1 )
	{
		// set the dim of state space
		//properties->setMinValue(0, -1.0); // not needed
		//properties->setMaxValue(0,  1.0);
		
		_exampleResult = NULL;
		
		// set the dim of state space
		properties->setDiscreteStateSize(0,datareader->getIterationNumber()+1);		
		
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------		
	double AdaBoostMDPClassifierSubsetSelectorBinary::getReward(CStateCollection *oldState, CAction *action, CStateCollection *newState) {
		double rew = 0.0;
		CAdaBoostAction* gridAction = dynamic_cast<CAdaBoostAction*>(action);
		int mode = gridAction->getMode();
		
		if ( _currentClassifier < _data->getIterationNumber() )
		{
			if (mode==0)
			{			
				rew = _skipReward;
			} else if ( mode == 1 )
			{								
				rew = _classificationReward;			
			} else if ( mode == 2 )
			{
				rew = _jumpReward;			
			}				
			
		} else {		
			if (_verbose>3)
			{
				// restore somehow the history
				//cout << "Get the history(sequence of actions in this episode)" << endl;
				//cout << "Size of action history: " << _history.size() << endl;
			}
			
			if ( _data->currentClassifyingResult( _currentRandomInstance,  _exampleResult )  ) // classified correctly
			{
				failed = false;
				if (hasithLabelCurrentElement(0))//is a negative element
					rew += _successReward;// /100.0;
				else
					rew += _successReward;
			} else 
			{
				failed = true;
				//rew += -_successReward;
				rew += 0.0;
			}
		}
		return rew;
	} 
	
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------	
	void AdaBoostMDPClassifierSubsetSelectorBinary::getState(CState *state)
	{
		AdaBoostMDPClassifierContinous::getState(state);
		state->setNumActiveContinuousStates(0);
		state->setDiscreteState(0, _currentClassifier);
	}
	
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	CStateModifier* AdaBoostMDPClassifierSubsetSelectorBinary::getStateSpace()
	{
		CAbstractStateDiscretizer* disc = new AdaBoostMDPClassifierSimpleDiscreteSpace(_data->getIterationNumber()+1);
		return disc;		
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------	
	CStateModifier* AdaBoostMDPClassifierSubsetSelectorBinary::getStateSpace( int divNum )
	{
		CAbstractStateDiscretizer* disc = new AdaBoostMDPClassifierSimpleDiscreteSpace(_data->getIterationNumber()+1);
		return disc;		
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------	
	CStateModifier* AdaBoostMDPClassifierSubsetSelectorBinary::getStateSpace( int divNum, double maxVal )
	{
		
		CAbstractStateDiscretizer* disc = new AdaBoostMDPClassifierSimpleDiscreteSpace(_data->getIterationNumber()+1);
		return disc;		
	}
	
	
	
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	
	void AdaBoostMDPClassifierSubsetSelectorBinary::outPutStatistic( BinaryResultStruct& bres )
	{
		_outputStream << bres.iterNumber << " " <<  bres.origAcc << " " << bres.acc << " " << bres.usedClassifierAvg << " " << bres.avgReward << " " << bres.TP << " " << bres.TN << endl;
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
} // end of namespace MultiBoost