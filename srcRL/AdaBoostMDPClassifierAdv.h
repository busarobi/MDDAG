/*
 *  AdaBoostMDPClassifierAdv.h
 *  RLTools
 *
 *  Created by Robert Busa-Fekete on 3/10/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __ADABOOST_MDP_CLASSADV_H
#define __ADABOOST_MDP_CLASSADV_H

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
#include "bitset"

//////////////////////////////////////////////////////////////////////
// for RL toolbox
//////////////////////////////////////////////////////////////////////
#include "cenvironmentmodel.h"
#include "crewardfunction.h"
#include "caction.h"
#include "cdiscretizer.h"
//////////////////////////////////////////////////////////////////////
// general includes
//////////////////////////////////////////////////////////////////////


using namespace std;

#define MAX_NUM_OF_ITERATION 10000
typedef vector<bitset<MAX_NUM_OF_ITERATION> >	vBitSet;
typedef vector<bitset<MAX_NUM_OF_ITERATION> >*	pVBitSet;

typedef vector<vector<char> >	vVecChar;
typedef vector<vector<char> >*	pVVecChar;


namespace MultiBoost {
	
	//////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////
	
	// Forward declarations.
	class ExampleResults;
	class InputData;
	class BaseLearner;
	//////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////
	

	class DataReader {
	public:
		// constructor
		DataReader(const nor_utils::Args& args, int verbose);
		
		// upload the data
		void loadInputData(const string& dataFileName, const string& testDataFileName, const string& testDataFileName2, const string& shypFileName);

		// update example result, and return the alpha of the weak classifier used
		double classifyKthWeakLearner( const int wHypInd, const int instance, ExampleResults* exampleResult );
		
		bool currentClassifyingResult( const int currentIstance, ExampleResults* exampleResult );
		double getExponentialLoss( const int currentIstance, ExampleResults* exampleResult );
		bool hasithLabel( int currentIstance, int classIdx );
		
		// getter setters
		int getClassNumber() const { return _pCurrentData->getNumClasses(); }
		int getIterationNumber() const { return _numIterations; }
		
		int getNumExamples() const { return _pCurrentData->getNumExamples(); }
		int getTrainNumExamples() const { return _pTrainData->getNumExamples(); }
		int getTestNumExamples() const { return _pTestData->getNumExamples(); }				
		
		void setCurrentDataToTrain() { 
			_pCurrentData = _pTrainData; 
			if (_isDataStorageMatrix) _pCurrentMatrix = &_weakHypothesesMatrices[_pCurrentData];			
		}
		void setCurrentDataToTest() { 
			_pCurrentData = _pTestData; 
			if (_isDataStorageMatrix) _pCurrentMatrix = &_weakHypothesesMatrices[_pCurrentData];
		}		

        bool setCurrentDataToTest2() { 
            if (_pTestData2) {
                _pCurrentData = _pTestData2; 
                return true;
            }
            return false;
			
//			if (_isDataStorageMatrix) _pCurrentMatrix = &_weakHypothesesMatrices[_pCurrentData];
		}		
        
		double getAccuracyOnCurrentDataSet();
		
		double getSumOfAlphas() const { return _sumAlphas; }
        
        inline const NameMap& getClassMap()
		{ return _pCurrentData->getClassMap(); }

	protected:
		void calculateHypothesesMatrix();
		
		int						_verbose;		
		double					_sumAlphas;
		
		const nor_utils::Args&  _args;  //!< The arguments defined by the user.		
		int						_currentInstance;
		vector<BaseLearner*>	_weakHypotheses;		
		
		InputData*				_pCurrentData;
		InputData*				_pTrainData;
		InputData*				_pTestData;
		
        InputData*				_pTestData2;
        
		int						_numIterations;	
		
//		map< InputData*, vBitSet > _weakHypothesesMatrices;
//		pVBitSet				   _pCurrentBitset;
		map< InputData*, vVecChar > _weakHypothesesMatrices;
		pVVecChar				   _pCurrentMatrix;
		
		
		bool					_isDataStorageMatrix;
		vector< vector< AlphaReal > > _vs;
		vector< AlphaReal >			_alphas;
	};

	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	class CAdaBoostAction : public CPrimitiveAction
	{
	protected:
		int _mode; // 0 skip, 1 classify, 2 jump to the end
	public:
		CAdaBoostAction( int mode ) : CPrimitiveAction() 
		{
			_mode=mode;
		}
		int getMode() { return _mode;}
	};
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////
	class AdaBoostMDPClassifierAdvDiscrete : public	CEnvironmentModel,	public CRewardFunction
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
		
		// for output info
		ofstream				_outputStream;
		
		//!< The arguments defined by the user.		
		const nor_utils::Args&		_args;  
		
		/// calculate the next state based on the action virtual 
		void doNextState(CPrimitiveAction *act);

		// this instance will be used in this episode
		int						_currentRandomInstance;

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
		
		// constructor
		AdaBoostMDPClassifierAdvDiscrete(const nor_utils::Args& args, int verbose, DataReader* datareader, int classNum);
		// destructor
		virtual	~AdaBoostMDPClassifierAdvDiscrete() 
		{
			_outputStream.close();
		}
		
		///returns the reward for the transition, implements the CRewardFunction interface
		virtual	double	getReward( CStateCollection	*oldState , CAction *action , CStateCollection *newState);
		
		///fetches the internal state and stores it in the state object
		virtual void getState(CState *state); ///resets the model 
		virtual	void doResetModel();		
		
		// classify correctly
		bool classifyCorrectly();
	};
	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	////////////////////////////////////////////////////////////////////////////////////////////////	
	class AdaBoostDiscreteState : public CAbstractStateDiscretizer
	{
	protected:
		unsigned int _iterNum, _classNum;
		
	public:
		AdaBoostDiscreteState(unsigned int iterNum, unsigned int classNum);
		virtual ~AdaBoostDiscreteState() {};
		
		virtual unsigned int getDiscreteStateNumber(CStateCollection *state);		
	};
	
	
} // end of namespace MultiBoost

#endif // __ADABOOST_MDP_CLASSIFIER_ADV_H

