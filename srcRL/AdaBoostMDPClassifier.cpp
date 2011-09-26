/*
 *  AdaBoostMDPClassifier.cpp
 *  RLTools
 *
 *  Created by Robert Busa-Fekete on 3/4/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "AdaBoostMDPClassifier.h"
#include "cstate.h"

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------		
	double AdaBoostMDPClassifier::getReward(CStateCollection *oldState, CAction *action, CStateCollection *newState) {
		double rew = 0.0;
		if (newState->getState()->getDiscreteState(2) > oldState->getState()->getDiscreteState(2))
		{
			rew = reward_bounce;
		}
		else
		{
			CGridWorldActionAdaBoost* gridAction = dynamic_cast<CGridWorldActionAdaBoost*>(action);
			int x = newState->getState()->getDiscreteState(0);
			int y = newState->getState()->getDiscreteState(1);
			int mode = gridAction->getMode();
			
			
			if ( x != size_x - 1)
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
			}
			else {
				if (_verbose>3)
				{
					// restore somehow the history
					//cout << "Get the history(sequence of actions in this episode)" << endl;
					//cout << "Size of action history: " << _history.size() << endl;
				}
				if (! _isClassified) {	
					_classifierNumber=0;
					for (int i=0; i<_weakHypotheses.size(); ++i) {
						if (_history[i]) {
							_classifierNumber++;
							classifyKthWeakLearner(i);
						}					
					}
					_isClassified=true;
				}
				
				if ( currentClassifyingResult() ) // classified correctly
				{
					rew = reward_success;
				} else 
				{
					rew = 0.0;
				}
			}
			
		}
		return rew;
	} 
	
	
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	
	void AdaBoostMDPClassifier::transitionFunction(CState *oldstate, CAction *action, CState *newState, CActionData *)
	{	
		int pos_x = oldstate->getDiscreteState(0);
		int pos_y = oldstate->getDiscreteState(1);
		int actual_bounces = oldstate->getDiscreteState(2);
		
		CGridWorldActionAdaBoost* gridAction = dynamic_cast<CGridWorldActionAdaBoost*>(action);
		
		int mode = gridAction->getMode();
		
		
		_numOfCalls++;				
		//cout << "pos x " << pos_x << " Mode: " << mode << endl;
		//if ((_numOfCalls % 3)==0) {
			//cout << "Chosen pos x " << pos_x << " Mode: " << mode << endl;			
			
			if (mode==0)
			{
				_history.push_back(false);			
			} else if (mode==1) {
				_history.push_back(true);			
			}
			
		//}
		
		
		if ( mode == 0 ) // skip
		{
			pos_x++;
		}
		else if (mode == 1 ) // classify
		{			
			//this->classifyKthWeakLearner(pos_x);
			pos_x++;			
		} else if (mode == 2 ) // jump to end			
		{
			pos_x = size_x-1;			
		}
		
		if ( pos_x >= size_x )
		{
			pos_x = size_x-1;
			//cout << "Megint kiment a tarotmanybol ez a fos!" << endl;
		}
		
		newState->setDiscreteState(0, pos_x);
	}
	
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	
	void AdaBoostMDPClassifier::loadInputData(const string& dataFileName, const string& testDataFileName, const string& shypFileName)
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
		_pData = baseLearner->createInputData();
		
		// set the non-default arguments of the input data
		_pData->initOptions(_args);
		// load the data
		_pData->load(dataFileName, IT_TEST, _verbose);				
		
		
		_pTestData = baseLearner->createInputData();
		
		// set the non-default arguments of the input data
		_pTestData->initOptions(_args);
		// load the data
		_pTestData->load(testDataFileName, IT_TEST, _verbose);				
		
	}				
	
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	
	void AdaBoostMDPClassifier::createGridWorld() 
	{
		start_values = new std::set<char>();
		target_values = new std::set<char>();
		prohibited_values = new std::set<char>();
		grid = new std::vector<char *>();
		this->size_x = _weakHypotheses.size() + 1; // for initial and end state
		this->size_y = 1;
		initGrid();
		
		
		//addStartValue('0');
		
		start_values->insert('0');
		target_values->insert('2');
		prohibited_values->insert('3');
		
		grid->push_back(new char[size_x]);
		for (int j =0; j < size_x; ++j)
		{
			(*grid)[0][j] = '1';
		}
		
		(*grid)[0][0] = '0';
		(*grid)[0][size_x-1] = '2';
		
		parseGrid();
		
		is_allocated = true;
	}		
	
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	void AdaBoostMDPClassifier::outPutStatistic( double acc, double curracc, double uc, double sumrew )
	{
		_outputStream << acc << " " << curracc << " " << uc << " " << sumrew << endl;
	}
		
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	void AdaBoostMDPClassifier::init()
	{				
		string mdpTrainFileName = _args.getValue<string>("traintestmdp", 0);				
		string testFileName = _args.getValue<string>("traintestmdp", 1);				
		string shypFileName = _args.getValue<string>("traintestmdp", 3);
		_numIterations = _args.getValue<int>("traintestmdp", 2);				
		
		string tmpFname = _args.getValue<string>("traintestmdp", 4);
		
		_outputStream.open( tmpFname.c_str() );
		
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
		us.loadHypotheses(shypFileName, _weakHypotheses, _pData);			
		_weakHypotheses.resize(_numIterations);
		
		if (_verbose > 0)
			cout << "Done." << endl << flush;			
		
		assert( _weakHypotheses.size() >= _numIterations );
		
		if (_verbose > 0)
			cout << "Allocating grid world..." << flush;
		
		createGridWorld();
		
		if (_verbose > 0)
			cout << "Done." << endl << flush;			
		
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	
	void AdaBoostMDPClassifier::getResetState(CState *resetState)
	{
		initEpisode();
		
		if (!is_parsed)
			parseGrid();
		if (start_points->size() > 0)
		{
			int i = rand() % start_points->size();
			resetState->setDiscreteState(0, (*start_points)[i]->second);
			resetState->setDiscreteState(1, (*start_points)[i]->first);
		}
		else
		{
			resetState->setDiscreteState(0, 0);
			resetState->setDiscreteState(1, 0);
		}
		resetState->setDiscreteState(2, 0);				
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	double AdaBoostMDPClassifier::classifyTestMDP()
	{
		double acc=0.0;
		
		const int numExamples = _pTestData->getNumExamples();
		
		int correct=0;
		int incorrect=0;
		
		for( int i = 1; i < numExamples; i++ )
		{			
			if (classifyTestMDP( i ))
				correct++;
			else {
				incorrect++;
			}
			
		}
		
	    acc = ((double) correct / ((double) numExamples)) * 100.0;
		
		return acc;
	}
	
	
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	bool AdaBoostMDPClassifier::classifyTestMDP( int i )
	{
		double acc=0.0;
		const int numClasses = _pData->getNumClasses();
		const int numExamples = _pTestData->getNumExamples();
		
		
		ExampleResults* tmpResult = new ExampleResults( i, numClasses );			
		vector<AlphaReal>& currVotesVector = tmpResult->getVotesVector();
		
		for( int j=0; j<_weakHypotheses.size(); j++ )
		{
			
			if (_history[j]) {
				BaseLearner* currWeakHyp = _weakHypotheses[j];
				float alpha = currWeakHyp->getAlpha();
				
				// for every class
				for (int l = 0; l < numClasses; ++l)
					currVotesVector[l] += alpha * currWeakHyp->classify(_pTestData, i, l);
			}
		}
		
		
		vector<Label>::const_iterator lIt;
		
		const vector<Label>& labels = _pTestData->getLabels(i);
		
		
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
			return false;
		else {
			return true;
		}
		
	}
	
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	bool AdaBoostMDPClassifier::currentClassifyingResult()
	{
		vector<Label>::const_iterator lIt;
		
		const vector<Label>& labels = _pData->getLabels(_currentInstance);
		
		// the vote of the winning negative class
		float maxNegClass = -numeric_limits<float>::max();
		// the vote of the winning positive class
		float minPosClass = numeric_limits<float>::max();
		
		vector<AlphaReal>& currVotesVector = _exampleResult->getVotesVector();
		
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
			return false;
		else {
			return true;
		}
		
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	double AdaBoostMDPClassifier::classifyTest()
	{
		double acc=0.0;
		const int numClasses = _pData->getNumClasses();
		const int numExamples = _pTestData->getNumExamples();
		
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
					currVotesVector[l] += alpha * currWeakHyp->classify(_pTestData, i, l);
				
			}
			
			
			vector<Label>::const_iterator lIt;
			
			const vector<Label>& labels = _pTestData->getLabels(i);
			
			
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
	
	
} // end of namespace MultiBoost
