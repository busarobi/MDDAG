/*
 *  AdaBoostMDPClassifierAdv.cpp
 *  RLTools
 *
 *  Created by Robert Busa-Fekete on 3/10/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "AdaBoostMDPClassifierAdv.h"
#include "cstate.h"
#include "cstateproperties.h"

#include <math.h> // for exp

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {
	
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------	
	AdaBoostMDPClassifierAdvDiscrete::AdaBoostMDPClassifierAdvDiscrete(const nor_utils::Args& args, int verbose, DataReader* datareader, int classNum)
	: CEnvironmentModel(classNum,classNum+1), _args(args), _verbose(verbose), _classNum(classNum), _data(datareader) //CEnvironmentModel(classNum+1,classNum)
	{
		// set the dim of state space
		properties->setDiscreteStateSize(0,datareader->getIterationNumber()+1);
		for (int i=1; i<=_classNum; ++i) {
			properties->setDiscreteStateSize(i,2);
		}
		
		_exampleResult = NULL;
		
		// open result file
		string tmpFname = _args.getValue<string>("traintestmdp", 4);			
		_outputStream.open( tmpFname.c_str() );
	}
	
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	void AdaBoostMDPClassifierAdvDiscrete::getState(CState *state)
	{
		// initializes the state object
		CEnvironmentModel::getState ( state );
		
		state->setNumActiveDiscreteStates(_classNum+1);
		state->setNumActiveContinuousStates(0);
		
		
		// Set the internal state variables
		state->setDiscreteState(0, _currentClassifier);
		for ( int i=1; i<=_classNum; ++i) {
			// a reference for clarity and speed
			vector<AlphaReal>& currVotesVector = _exampleResult->getVotesVector();
			
			if (currVotesVector[i-1]>0)
				state->setDiscreteState(i, 1);
			else 
				state->setDiscreteState(i, 0); 
		}
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	void AdaBoostMDPClassifierAdvDiscrete::doNextState(CPrimitiveAction *act)
	{
		CAdaBoostAction* action = dynamic_cast<CAdaBoostAction*>(act);
		
		int mode = action->getMode();
		
		if ( mode == 0 ) // skip
		{
			_currentClassifier++;
		}
		else if (mode == 1 ) // classify
		{			
			_data->classifyKthWeakLearner(_currentClassifier,_currentRandomInstance,_exampleResult);
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
	void AdaBoostMDPClassifierAdvDiscrete::doResetModel()
	{
		//_currentRandomInstance = (int) (rand() % _data->getNumExamples() );
		_currentClassifier = 0;
		_classifierNumber = 0;				
		
		if (_exampleResult==NULL) delete _exampleResult;
		_exampleResult = new ExampleResults(_currentRandomInstance,_classNum);		
		
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------		
	double AdaBoostMDPClassifierAdvDiscrete::getReward(CStateCollection *oldState, CAction *action, CStateCollection *newState) {
		double rew = 0.0;
		CAdaBoostAction* gridAction = dynamic_cast<CAdaBoostAction*>(action);
		int x_old = oldState->getState()->getDiscreteState(0);
		int x_new = newState->getState()->getDiscreteState(0);
		int mode = gridAction->getMode();
		
		
		if ( x_new < _data->getIterationNumber() )
		{			
			if (mode==0)
			{
				rew = _skipReward;
			} else if ( mode == 1 )
			{
				_classifierNumber++;
				rew = _classificationReward;			
			} else if ( mode == 2 )
			{
				rew = _jumpReward;			
			}				
		}
		else {
			if (_verbose>3)
			{
				// restore somehow the history
				//cout << "Get the history(sequence of actions in this episode)" << endl;
				//cout << "Size of action history: " << _history.size() << endl;
			}
			
			if ( _data->currentClassifyingResult( _currentRandomInstance,  _exampleResult )  ) // classified correctly
			{
				rew += _successReward;
			} else 
			{
				rew += 0.0;
			}
		}			
		return rew;
	} 
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	DataReader::DataReader(const nor_utils::Args& args, int verbose) : _verbose(verbose), _args(args)
	{				
		string mdpTrainFileName = _args.getValue<string>("traintestmdp", 0);				
		string testFileName = _args.getValue<string>("traintestmdp", 1);				
		string shypFileName = _args.getValue<string>("traintestmdp", 3);
		_numIterations = _args.getValue<int>("traintestmdp", 2);				
		
		string tmpFname = _args.getValue<string>("traintestmdp", 4);
		
		
		
		if (_verbose > 0)
			cout << "Loading arff data for MDP learning..." << flush;
		
		// load the arff
		loadInputData(mdpTrainFileName, testFileName, shypFileName);
		
		if (_verbose > 0)
			cout << "Done." << endl << flush;
		
		
		if (_verbose > 0)
			cout << "Loading strong hypothesis..." << flush;
		
		// The class that loads the weak hypotheses
		UnSerialization us;
		
		// loads them
		us.loadHypotheses(shypFileName, _weakHypotheses, _pTrainData);			
		if (_numIterations<_weakHypotheses.size())
			_weakHypotheses.resize(_numIterations);
		
		if (_verbose > 0)
			cout << "Done." << endl << flush;			
		
		assert( _weakHypotheses.size() >= _numIterations );
		
		// calculate the sum of alphas
		vector<BaseLearner*>::iterator it;
		_sumAlphas=0.0;
		for( it = _weakHypotheses.begin(); it != _weakHypotheses.end(); ++it )
		{
			BaseLearner* currBLearner = *it;
			_sumAlphas += currBLearner->getAlpha();
		}
		
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	
	void DataReader::loadInputData(const string& dataFileName, const string& testDataFileName, const string& shypFileName)
	{
		// open file
		ifstream inFile(shypFileName.c_str());
		if (!inFile.is_open())
		{
			cerr << "ERROR: Cannot open strong hypothesis file <" << shypFileName << ">!" << endl;
			exit(1);
		}
		
		// Declares the stream tokenizer
		nor_utils::StreamTokenizer st(inFile, "<>\n\r\t");
		
		// Move until it finds the multiboost tag
		if ( !UnSerialization::seekSimpleTag(st, "multiboost") )
		{
			// no multiboost tag found: this is not the correct file!
			cerr << "ERROR: Not a valid MultiBoost Strong Hypothesis file!!" << endl;
			exit(1);
		}
		
		// Move until it finds the algo tag
		string basicLearnerName = UnSerialization::seekAndParseEnclosedValue<string>(st, "algo");
		
		// Check if the weak learner exists
		if ( !BaseLearner::RegisteredLearners().hasLearner(basicLearnerName) )
		{
			cerr << "ERROR: Weak learner <" << basicLearnerName << "> not registered!!" << endl;
			exit(1);
		}
		
		// get the training input data, and load it
		BaseLearner* baseLearner = BaseLearner::RegisteredLearners().getLearner(basicLearnerName);
		baseLearner->initLearningOptions(_args);
		_pTrainData = baseLearner->createInputData();
		
		// set the non-default arguments of the input data
		_pTrainData->initOptions(_args);
		// load the data
		_pTrainData->load(dataFileName, IT_TEST, _verbose);				
		
		
		_pTestData = baseLearner->createInputData();
		
		// set the non-default arguments of the input data
		_pTestData->initOptions(_args);
		// load the data
		_pTestData->load(testDataFileName, IT_TEST, _verbose);				
		
	}				
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	double DataReader::classifyKthWeakLearner( const int wHypInd, const int instance, ExampleResults* exampleResult )		
	{		
		if (_verbose>3) {
			//cout << "Classifiying: " << wHypInd << endl;
		}
		
		if ( wHypInd >= _numIterations ) return -1.0; // indicating error						
		
		const int numClasses = _pCurrentData->getNumClasses();
		
		BaseLearner* currWeakHyp = _weakHypotheses[wHypInd];
		float alpha = currWeakHyp->getAlpha();
		
		// a reference for clarity and speed
		vector<AlphaReal>& currVotesVector = exampleResult->getVotesVector();
		
		// for every class
		for (int l = 0; l < numClasses; ++l)
			currVotesVector[l] += alpha * currWeakHyp->classify(_pCurrentData, instance, l);
		
		return alpha;
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	bool DataReader::currentClassifyingResult( const int currentIstance, ExampleResults* exampleResult )
	{
		vector<Label>::const_iterator lIt;
		
		const vector<Label>& labels = _pCurrentData->getLabels(currentIstance);
		
		// the vote of the winning negative class
		float maxNegClass = -numeric_limits<float>::max();
		// the vote of the winning positive class
		float minPosClass = numeric_limits<float>::max();
		
		vector<AlphaReal>& currVotesVector = exampleResult->getVotesVector();
		
		for ( lIt = labels.begin(); lIt != labels.end(); ++lIt )
		{
			// get the negative winner class
			if ( lIt->y < 0 && currVotesVector[lIt->idx] > maxNegClass )
				maxNegClass = currVotesVector[lIt->idx];
			
			// get the positive winner class
			if ( lIt->y > 0 && currVotesVector[lIt->idx] < minPosClass )
				minPosClass = currVotesVector[lIt->idx];
		}
		
		
		if ( nor_utils::is_zero( minPosClass - maxNegClass )) return false;
		
		// if the vote for the worst positive label is lower than the
		// vote for the highest negative label -> error
		if (minPosClass < maxNegClass){
			return false;
		} else {
			return true;
		}
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	double DataReader::getExponentialLoss( const int currentIstance, ExampleResults* exampleResult )
	{
		double exploss = 0.0;
		
		vector<Label>::const_iterator lIt;
		
		const int numClasses = _pCurrentData->getNumClasses();
		const vector<Label>& labels = _pCurrentData->getLabels(currentIstance);
		vector<double> yfx(numClasses);
				
		vector<AlphaReal>& currVotesVector = exampleResult->getVotesVector();
		
		//cout << "Instance: " << currentIstance << " ";
		//cout <<  "Size: " << currVotesVector.size() << " Data: ";
		
		for ( lIt = labels.begin(); lIt != labels.end(); ++lIt )
		{
			//cout << currVotesVector[lIt->idx] << " ";
			yfx[lIt->idx] = currVotesVector[lIt->idx] * lIt->y;
		}							
		//cout << endl << flush;
		
		if (numClasses==2) // binary classification
		{
			exploss = exp(-yfx[0]);
		} else {			
			for ( lIt = labels.begin(); lIt != labels.end(); ++lIt )
			{
				exploss += exp(-yfx[lIt->idx]);
			}						
		}
		
		return exploss;
	}	
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	bool DataReader::hasithLabel( int currentIstance, int classIdx )
	{
		const vector<Label>& labels = _pCurrentData->getLabels(currentIstance);
		return (labels[classIdx].y>0);
	}
	
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	double DataReader::getAccuracyOnCurrentDataSet()
	{
		double acc=0.0;
		const int numClasses = _pCurrentData->getNumClasses();
		const int numExamples = _pCurrentData->getNumExamples();
		
		int correct=0;
		int incorrect=0;
		
		for( int i = 1; i < numExamples; i++ )
		{			
			ExampleResults* tmpResult = new ExampleResults( i, numClasses );			
			vector<AlphaReal>& currVotesVector = tmpResult->getVotesVector();
			
			for( int j=0; j<_weakHypotheses.size(); j++ )
			{
				
				BaseLearner* currWeakHyp = _weakHypotheses[j];
				float alpha = currWeakHyp->getAlpha();
				
				// for every class
				for (int l = 0; l < numClasses; ++l)
					currVotesVector[l] += alpha * currWeakHyp->classify(_pCurrentData, i, l);
				
			}
			
			
			vector<Label>::const_iterator lIt;
			
			const vector<Label>& labels = _pCurrentData->getLabels(i);
			
			
			// the vote of the winning negative class
			float maxNegClass = -numeric_limits<float>::max();
			// the vote of the winning positive class
			float minPosClass = numeric_limits<float>::max();
			
			
			for ( lIt = labels.begin(); lIt != labels.end(); ++lIt )
			{
				// get the negative winner class
				if ( lIt->y < 0 && currVotesVector[lIt->idx] > maxNegClass )
					maxNegClass = currVotesVector[lIt->idx];
				
				// get the positive winner class
				if ( lIt->y > 0 && currVotesVector[lIt->idx] < minPosClass )
					minPosClass = currVotesVector[lIt->idx];
			}
			
			// if the vote for the worst positive label is lower than the
			// vote for the highest negative label -> error
			if (minPosClass <= maxNegClass)
				incorrect++;
			else {
				correct++;
			}
			
		}
		
	    acc = ((double) correct / ((double) numExamples)) * 100.0;
		
		return acc;
	}
	
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	AdaBoostDiscreteState::AdaBoostDiscreteState(unsigned int iterNum, unsigned int classNum) : CAbstractStateDiscretizer((iterNum+1) * (2^classNum))
	{
		this->_iterNum = iterNum;
		this->_classNum = classNum;
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------	
	unsigned int AdaBoostDiscreteState::getDiscreteStateNumber(CStateCollection *state) {
		unsigned int discstate=0;		
		int iter = state->getState()->getDiscreteState(0);
		
		if (iter < 0 || (unsigned int)iter > _iterNum )
		{
			discstate = 0;
		}
		else
		{
			int base = 0;
			for( int i = 1; i <= _classNum; ++i )
			{
				int v = state->getState()->getDiscreteState(i);
				base += 2^(i-1) * v;				
			}
			discstate = base * (_iterNum+1) + iter;
		}
		return discstate;
	}
	
	
	
	
	
	
} // end of namespace MultiBoost