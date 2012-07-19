/*
 *  AdaBoostMDPClassifierContinous.cpp
 *  RLTools
 *
 *  Created by Robert Busa-Fekete on 3/11/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "AdaBoostMDPClassifierContinous.h"


#include "cstate.h"
#include "cstateproperties.h"
#include "clinearfafeaturecalculator.h"

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {
	
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------	
	AdaBoostMDPClassifierContinous::AdaBoostMDPClassifierContinous(const nor_utils::Args& args, int verbose, DataReader* datareader, int classNum, int discState)
	: CEnvironmentModel(classNum,discState), _args(args), _verbose(verbose), _classNum(classNum), _data(datareader), _incrementalReward(false), _lastReward(0.0) //CEnvironmentModel(classNum+1,classNum)
	{
		// set the dim of state space
		for( int i=0; i<_classNum;++i)
		{
			properties->setMinValue(i, 0.0);
			properties->setMaxValue(i, 1.0);
		}
		
		_exampleResult = NULL;
		
		// open result file
		string tmpFname = _args.getValue<string>("traintestmdp", 4);			
		_outputStream.open( tmpFname.c_str() );
		
		_sumAlpha = _data->getSumOfAlphas();
        
        cout << "[+] Sum of alphas: " << _sumAlpha << endl;
        
		_classifierUsed.resize(_data->getIterationNumber());
		
		if (args.hasArgument("rewards"))
		{
			double rew = args.getValue<double>("rewards", 0);
			setSuccessReward(rew); // classified correctly
			
			// the reward we incur if we use a classifier		
			rew = args.getValue<double>("rewards", 1);
			setClassificationReward( rew );
			
			rew = args.getValue<double>("rewards", 2);
			setSkipReward( rew );		
			
			setJumpReward(0.0);
		} else {
			cout << "No rewards are given!" << endl;
			exit(-1);
		}
		
		
		if (args.hasArgument("succrewartdtype"))
		{
			string succRewardMode = args.getValue<string>("succrewartdtype");
			
			if ( succRewardMode == "hamming" )
				_succRewardMode = RT_HAMMING;
			else if ( succRewardMode == "exp" )
				_succRewardMode = RT_EXP;
			else
			{
				cerr << "ERROR: Unrecognized (succrewartdtype) --succes rewards option!!" << endl;
				exit(1);
			}						
		} else {		
			_succRewardMode = RT_HAMMING;
		}
		
        if (args.hasArgument("incrementalrewardQ")) {
            _incrementalReward = true;
        }
	}
	
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	void AdaBoostMDPClassifierContinous::getState(CState *state)
	{
		// initializes the state object
		CEnvironmentModel::getState ( state );
		
		
		// not necessary since we do not store any additional information
		
		
		if (_classNum==2)
			state->setNumActiveContinuousStates(1);
		else 
			state->setNumActiveContinuousStates(_classNum);

		
		// a reference for clarity and speed
		vector<AlphaReal>& currVotesVector = _exampleResult->getVotesVector();
		
		// Set the internal state variables		
		for ( int i=0; i<_classNum; ++i) {
			
			double st = 0.0;
			//if ( !nor_utils::is_zero( _currentSumAlpha ) )
			//	st = currVotesVector[i] / _currentSumAlpha;
			st = ((currVotesVector[i] /_sumAlpha)+1)/2.0; // rescale between [0,1]
			state->setContinuousState(i, st);
		}
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	void AdaBoostMDPClassifierContinous::doNextState(CPrimitiveAction *act)
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
            double classifierOutput = _data->classifyKthWeakLearner(_currentClassifier,_currentRandomInstance,_exampleResult);
			_currentSumAlpha += classifierOutput;
			_classifierUsed[_currentClassifier]=true;
            _classifiersOutput.push_back(classifierOutput);
			_classifierNumber++; 
			_currentClassifier++;						
		} else if (mode == 2 ) // jump to end			
		{
			_currentClassifier = _data->getIterationNumber();			
		}
		
		
		if ( _currentClassifier == _data->getIterationNumber() ) // check whether there is any weak classifier
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
	void AdaBoostMDPClassifierContinous::doResetModel()
	{
		//_currentRandomInstance = (int) (rand() % _data->getNumExamples() );
		_currentClassifier = 0;
		_classifierNumber = 0;				
		_currentSumAlpha = 0.0;
		
		if (_exampleResult==NULL) 
			_exampleResult = new ExampleResults(_currentRandomInstance,_data->getClassNumber());		
		
		vector<AlphaReal>& currVotesVector = _exampleResult->getVotesVector();
		
		fill( currVotesVector.begin(), currVotesVector.end(), 0.0 );		
		fill( _classifierUsed.begin(), _classifierUsed.end(), false );
        _classifiersOutput.clear();
			
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------		
	double AdaBoostMDPClassifierContinous::getReward(CStateCollection *oldState, CAction *action, CStateCollection *newState) {
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
			
			// if clRes true then the this instance is classified correctly
			bool clRes = _data->currentClassifyingResult( _currentRandomInstance,  _exampleResult ); // classified correctly
			
			
			if (clRes)
			{
				failed = false;
				rew += _successReward;
			} else 
			{
				failed = true;
				//rew -= _successReward;
				rew += 0.0;
			}
		}
		return rew;
	} 
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	bool AdaBoostMDPClassifierContinous::classifyCorrectly()
	{
		return  _data->currentClassifyingResult( _currentRandomInstance,  _exampleResult );
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	bool AdaBoostMDPClassifierContinous::hasithLabelCurrentElement( int i )
	{
		return  _data->hasithLabel( _currentRandomInstance, i ); 
	}		
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	void AdaBoostMDPClassifierContinous::outPutStatistic( double acc, double curracc, double uc, double sumrew )
	{
		_outputStream << acc << " " << curracc << " " << uc << " " << sumrew << endl << flush;
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	CStateModifier* AdaBoostMDPClassifierContinous::getStateSpace()
	{
		// create the discretizer with the build in classes
		// create the partition arrays
		//double partitions[] = {-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,
		//						0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}; // partition for states
		
		double partitions[] = {0.2,0.5,0.75}; // partition for states
		//double partitions[] = {-0.2,0.0,0.2}; // partition for states
		//double partitions[] = {-0.75,-0.4,-0.2,0.0,0.2,0.4,0.75}; // partition for states
		
		
		CAbstractStateDiscretizer** disc = new CAbstractStateDiscretizer*[getNumClasses()];
		for( int i=0; i<getNumClasses(); ++i)
			disc[i] = new CSingleStateDiscretizer(i,3,partitions);
		//disc[i] = new CSingleStateDiscretizer(i,19,partitions);
		
		// Merge the discretizers
		CDiscreteStateOperatorAnd *andCalculator = new CDiscreteStateOperatorAnd();
		
		for( int i=0; i<getNumClasses(); ++i)
			andCalculator->addStateModifier(disc[i]);
		return andCalculator;
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	CStateModifier* AdaBoostMDPClassifierContinous::getStateSpaceRBF(unsigned int partitionNumber)
	{
		// Now we can already create our RBF network
		// Therefore we will use a CRBFFeatureCalculator, our feature calculator uses both dimensions of the model state 
		// (the angel and the angular velocity) and lays a 20x20 RBF grid over the state space. For each dimension the given sigmas are used.
		// For the calculation of useful sigmas we have to consider that the CRBFFeatureCalculator always uses the 
		// normalized state representation, so the state variables are scaled to the intervall [0,1]
		
		int numClasses = getNumClasses();
		
		unsigned int* dimensions = new unsigned int[numClasses];
		unsigned int* partitions = new unsigned int[numClasses];
		double* offsets = new double[numClasses];
		double* sigma = new double[numClasses];
		
		for(int i=0; i<numClasses; ++i )
		{
			dimensions[i]=i;
			partitions[i]=partitionNumber;
			offsets[i]=0.0;
			sigma[i]=0.025;
		}
		
		
		// Now we can create our Feature Calculator
		CStateModifier *rbfCalc = new CRBFFeatureCalculator(numClasses, dimensions, partitions, offsets, sigma);	
		return rbfCalc;
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	CStateModifier* AdaBoostMDPClassifierContinous::getStateSpaceTileCoding(unsigned int partitionNumber)
	{
		// Now we can already create our RBF network
		// Therefore we will use a CRBFFeatureCalculator, our feature calculator uses both dimensions of the model state 
		// (the angel and the angular velocity) and lays a 20x20 RBF grid over the state space. For each dimension the given sigmas are used.
		// For the calculation of useful sigmas we have to consider that the CRBFFeatureCalculator always uses the 
		// normalized state representation, so the state variables are scaled to the intervall [0,1]
		
		int numClasses = getNumClasses();
		
		unsigned int* dimensions = new unsigned int[numClasses];
		unsigned int* partitions = new unsigned int[numClasses];
		double* offsets = new double[numClasses];
		double* sigma = new double[numClasses];
		
		for(int i=0; i<numClasses; ++i )
		{
			dimensions[i]=i;
			partitions[i]=partitionNumber;
			offsets[i]=0.0;
			sigma[i]=0.025;
		}
		
		
		// Now we can create our Feature Calculator
		CStateModifier *tileCodeCalc = new CTilingFeatureCalculator(numClasses, dimensions, partitions, offsets );	
		return tileCodeCalc;
	}	
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	void AdaBoostMDPClassifierContinous::getHistory( vector<bool>& history )
	{
		history.resize( _classifierUsed.size() );
		copy( _classifierUsed.begin(), _classifierUsed.end(), history.begin() );
	}

    // -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	void AdaBoostMDPClassifierContinous::getClassifiersOutput( vector<double>& classifiersOutput )
	{
		classifiersOutput.resize( _classifiersOutput.size() );
		copy( _classifiersOutput.begin(), _classifiersOutput.end(), classifiersOutput.begin() );
	}
    // -----------------------------------------------------------------------
	// -----------------------------------------------------------------------	
	void AdaBoostMDPClassifierContinous::getCurrentExmapleResult( vector<double>& result )
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
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	
	
} // end of namespace MultiBoost