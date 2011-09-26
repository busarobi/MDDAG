/*
 *  AdaBoostMDPClassifierDiscrete.cpp
 *  RLTools
 *
 *  Created by Robert Busa-Fekete on 3/13/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "AdaBoostMDPClassifierDiscrete.h"


#include "cstate.h"
#include "cstateproperties.h"

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {
	
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------	
	AdaBoostMDPClassifierDiscrete::AdaBoostMDPClassifierDiscrete(const nor_utils::Args& args, int verbose, DataReader* datareader)
	: CEnvironmentModel(0,1), _args(args), _verbose(verbose), _classNum(2), _data(datareader) //CEnvironmentModel(classNum+1,classNum)
	{
		// set the dim of state space
		properties->setDiscreteStateSize(0,datareader->getIterationNumber()+1);		
		
		_exampleResult = NULL;
		
		// open result file
		string tmpFname = _args.getValue<string>("traintestmdp", 4);			
		_outputStream.open( tmpFname.c_str() );
	}
	
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	void AdaBoostMDPClassifierDiscrete::getState(CState *state)
	{
		// initializes the state object
		CEnvironmentModel::getState ( state );
		
		
		// not necessary since we do not store any additional information
		state->setNumActiveDiscreteStates(1);

		state->setDiscreteState(0, _currentClassifier);
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	void AdaBoostMDPClassifierDiscrete::doNextState(CPrimitiveAction *act)
	{
		CAdaBoostAction* action = dynamic_cast<CAdaBoostAction*>(act);
		
		int mode = action->getMode();
		//cout << mode << endl;
		if ( mode == 0 ) // skip
		{			
			_currentClassifier++;
		}
		else if (mode == 1 ) // classify
		{	
			_currentSumAlpha += _data->classifyKthWeakLearner(_currentClassifier,_currentRandomInstance,_exampleResult);
			_classifierUsed[_currentClassifier]=true;
			_classifierNumber++; 
			_currentClassifier++;						
		} else if (mode == 2 ) // jump to end			
		{
			_currentClassifier = _data->getIterationNumber();			
		}
		
		
		if ( _currentClassifier == _data->getIterationNumber() )
		{
			reset = true;
			if ( _data->currentClassifyingResult( _currentRandomInstance,  _exampleResult ) )
			{
				failed = true;
			} else {
				failed = false;
			}
		}
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	void AdaBoostMDPClassifierDiscrete::doResetModel()
	{
		//_currentRandomInstance = (int) (rand() % _data->getNumExamples() );
		_currentClassifier = 0;
		_classifierNumber = 0;				
		_currentSumAlpha = 0.0;
		
		if (_exampleResult==NULL) delete _exampleResult;
		_exampleResult = new ExampleResults(_currentRandomInstance,_classNum);		
		
		fill( _classifierUsed.begin(), _classifierUsed.end(), false );
				
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------		
	double AdaBoostMDPClassifierDiscrete::getReward(CStateCollection *oldState, CAction *action, CStateCollection *newState) {
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
	bool AdaBoostMDPClassifierDiscrete::classifyCorrectly()
	{
		return  _data->currentClassifyingResult( _currentRandomInstance,  _exampleResult );
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	void AdaBoostMDPClassifierDiscrete::outPutStatistic( double acc, double curracc, double uc, double sumrew )
	{
		_outputStream << acc << " " << curracc << " " << uc << " " << sumrew << endl;
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	CAbstractStateDiscretizer* AdaBoostMDPClassifierDiscrete::getStateSpace()
	{
		CAbstractStateDiscretizer* stateDiscretizer =  new AdaBoostMDPClassifierSimpleDiscreteSpace( _data->getIterationNumber()+1 );
		return stateDiscretizer;
	}
	
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	
	void AdaBoostMDPClassifierDiscrete::outPutStatistic( BinaryResultStruct& bres )
	{
		_outputStream << bres.iterNumber << " " <<  bres.origAcc << " " << bres.acc << " " << bres.usedClassifierAvg << " " << bres.avgReward << " " << bres.TP << " " << bres.TN << endl;
	}
	
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	bool AdaBoostMDPClassifierDiscrete::hasithLabelCurrentElement( int i )
	{
		return  _data->hasithLabel( _currentRandomInstance, i ); 
	}		
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	
	void AdaBoostMDPClassifierDiscrete::getHistory( vector<bool>& history )
	{
		history.resize( _classifierUsed.size() );
		copy( _classifierUsed.begin(), _classifierUsed.end(), history.begin() );
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------	
	void AdaBoostMDPClassifierDiscrete::getCurrentExmapleResult( vector<double>& result )
	{
		vector<AlphaReal>& currVotesVector = _exampleResult->getVotesVector();
		result.resize(currVotesVector.size());
		copy( currVotesVector.begin(), currVotesVector.end(), result.begin() );
		
		for( int i =0; i<currVotesVector.size(); ++i )
		{
			result[i] = currVotesVector[i]/_sumAlpha;
		}
		
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	
} // end of namespace MultiBoost