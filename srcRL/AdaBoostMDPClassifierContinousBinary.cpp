/*
 *  AdaBoostMDPClassifierContinousBinary.cpp
 *  RLTools
 *
 *  Created by Robert Busa-Fekete on 3/13/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "AdaBoostMDPClassifierContinousBinary.h"


#include "cstate.h"
#include "cstateproperties.h"
#include "clinearfafeaturecalculator.h"
#include "crbftrees.h"
#include "RBFBasedQFunction.h"
#include "RBFStateModifier.h"

#include <math.h> // for exp

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {
	
	
	
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	AdaBoostMDPClassifierContinousBinary::AdaBoostMDPClassifierContinousBinary( const nor_utils::Args& args, int verbose, DataReader* datareader )
	: AdaBoostMDPClassifierContinous( args, verbose, datareader, 1, 1 )
	{
		// set the dim of state space
		//properties->setMinValue(0, -1.0); // not needed
		//properties->setMaxValue(0,  1.0);
		
		_exampleResult = NULL;
        
        _positiveLabelIndex = 0;
        if ( args.hasArgument("positivelabel") )
		{
			args.getValue("positivelabel", 0, _positiveLabelName);
            const NameMap& namemap = datareader->getClassMap();
            _positiveLabelIndex = namemap.getIdxFromName( _positiveLabelName );

        }
		
        _failOnNegativesPenalty = _failOnPositivesPenalty = 0.0;
        if ( args.hasArgument("failpenalties") )
		{
			args.getValue("failpenalties", 0, _failOnPositivesPenalty);
            args.getValue("failpenalties", 1, _failOnNegativesPenalty);
            
            assert(_failOnNegativesPenalty <= 0 && _failOnPositivesPenalty <= 0);
        }
		
		// set the dim of state space
		properties->setDiscreteStateSize(0,datareader->getIterationNumber()+1);		
		
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------		
	double AdaBoostMDPClassifierContinousBinary::getReward(CStateCollection *oldState, CAction *action, CStateCollection *newState) {
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
                
                if (_incrementalReward) {    
                    
                    rew -= _lastReward;
                    if (_succRewardMode==RT_HAMMING)
                    {
                        if ( _data->currentClassifyingResult( _currentRandomInstance,  _exampleResult )  ) // classified correctly
                        {
                            _lastReward = _successReward;// /100.0;
                        }
                    } 
                    else if (_succRewardMode==RT_EXP)
                    {
                        double exploss;
                        if (_classifierNumber>0)
                        {
                            exploss = _data->getExponentialLoss( _currentRandomInstance,  _exampleResult );
                            _lastReward = 1/exploss;
                        }
                    }
                    
                    rew += _lastReward;
//                    _lastReward = rew;
                }
                
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
			
//            rew -= _lastReward;
			
			if (_succRewardMode==RT_HAMMING)
			{
				if ( _data->currentClassifyingResult( _currentRandomInstance,  _exampleResult )  ) // classified correctly
				{
					failed = false;
					if (hasithLabelCurrentElement(_positiveLabelIndex))
						rew += _successReward;// /100.0;
					else //is a negative element
						rew += _successReward;
				} else 
				{
					failed = true;
					//rew += -_successReward;
                    if (hasithLabelCurrentElement(_positiveLabelIndex))
                        rew += _failOnPositivesPenalty;
                    else 
                        rew += _failOnNegativesPenalty;
				}
			} else if (_succRewardMode==RT_EXP)
			{
				// since the AdaBoost minimize the margin e(-y_i f(x_i) 
				// we will maximize -1/e(y_i * f(x_i)
				double exploss;				
				if (_classifierNumber>0)
				{
					exploss = _data->getExponentialLoss( _currentRandomInstance,  _exampleResult );
					rew += 1/exploss;
				}
				else
				{
					//exploss = exp(_data->getSumOfAlphas());			
					//rew -= _successReward;			
				}

				/*
				cout << "Instance index: " << _currentRandomInstance << " ";
				bool clRes =  _data->currentClassifyingResult( _currentRandomInstance,  _exampleResult );
				if (clRes)
					cout << "[+] exploss: " << exploss << endl << flush;
				else
					cout << "[-] exploss: " << exploss << endl << flush;
				*/

				
				
			} else {
				cout << "Unknown succes reward type!!! Maybe it is not implemented! " << endl;
				exit(-1);
			}
			
		}
		return rew;
	} 
	
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------	
	void AdaBoostMDPClassifierContinousBinary::getState(CState *state)
	{
		AdaBoostMDPClassifierContinous::getState(state);
		state->setDiscreteState(0, _currentClassifier);
	}
	
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	CStateModifier* AdaBoostMDPClassifierContinousBinary::getStateSpace()
	{
		// create the discretizer with the build in classes
		// create the partition arrays
		double partitions[] = {-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,
			0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}; // partition for states
		
		//double partitions[] = {-0.5,-0.2,0.0,0.2,0.5}; // partition for states
		//double partitions[] = {-0.2,0.0,0.2}; // partition for states
		
		CAbstractStateDiscretizer** disc = new CAbstractStateDiscretizer*[2];
		
		//disc[0] = new CSingleStateDiscretizer(0,5,partitions);
		disc[0] = new CSingleStateDiscretizer(0,19,partitions);
		disc[1]= new AdaBoostMDPClassifierSimpleDiscreteSpace(_data->getIterationNumber()+1);
		
		
		// Merge the discretizers
		CDiscreteStateOperatorAnd *andCalculator = new CDiscreteStateOperatorAnd();
		
		
		andCalculator->addStateModifier(disc[0]);
		andCalculator->addStateModifier(disc[1]);
		return andCalculator;
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------	
	CStateModifier* AdaBoostMDPClassifierContinousBinary::getStateSpace( int divNum )
	{
		// create the discretizer with the build in classes
		// create the partition arrays
		double *partitions = new double[divNum-1];
        
        //TMP added *2
		double step = 1.0/(divNum);
        
//TMP        cout << "\n HEY AITCHEGGOU!!!!\n" ;
		for(int i=0;i<divNum-1;++i)
		{ 
			//cout << (i+1)*step << " " << endl << flush;
			partitions[i]= (i+1)*step; //TMP 1 wella 6
            
//TMP            cout << partitions[i] - 0.5 << " ";
		}
//TMP                cout << "\n HEY AITCHEGGOU!!!!\n" ;
		//double partitions[] = {-0.5,-0.2,0.0,0.2,0.5}; // partition for states
		//double partitions[] = {-0.2,0.0,0.2}; // partition for states
		
		CAbstractStateDiscretizer** disc = new CAbstractStateDiscretizer*[2];
		
		//disc[0] = new CSingleStateDiscretizer(0,5,partitions);
		disc[0] = new CSingleStateDiscretizer(0,divNum-1,partitions);
		disc[1]= new AdaBoostMDPClassifierSimpleDiscreteSpace(_data->getIterationNumber()+1);
		

				// Merge the discretizers
		CDiscreteStateOperatorAnd *andCalculator = new CDiscreteStateOperatorAnd();
		
		
		andCalculator->addStateModifier(disc[1]);
		andCalculator->addStateModifier(disc[0]);
		
		return andCalculator;
		
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------	
	CStateModifier* AdaBoostMDPClassifierContinousBinary::getStateSpace( int divNum, double maxVal )
	{
		// create the discretizer with the build in classes
		// create the partition arrays
		double *partitions = new double[divNum];
		double step = maxVal/divNum;
		for(int i=0;i<divNum-1;++i)
		{ 
			//cout << (i+1)*step << " " << endl << flush;
			partitions[i]= (i+1)*step; 
		}
		partitions[divNum-1]=maxVal;
		
		//double partitions[] = {-0.5,-0.2,0.0,0.2,0.5}; // partition for states
		//double partitions[] = {-0.2,0.0,0.2}; // partition for states
		
		CAbstractStateDiscretizer** disc = new CAbstractStateDiscretizer*[2];
		
		//disc[0] = new CSingleStateDiscretizer(0,5,partitions);
		disc[0] = new CSingleStateDiscretizer(0,divNum,partitions);
		disc[1]= new AdaBoostMDPClassifierSimpleDiscreteSpace(_data->getIterationNumber()+1);
		
		
		// Merge the discretizers
		CDiscreteStateOperatorAnd *andCalculator = new CDiscreteStateOperatorAnd();
		
		
		andCalculator->addStateModifier(disc[0]);
		andCalculator->addStateModifier(disc[1]);
		return andCalculator;
		
	}
    
    // -----------------------------------------------------------------------
	// -----------------------------------------------------------------------

    CStateModifier* AdaBoostMDPClassifierContinousBinary::getStateSpaceNN()
	{
        int numClasses = getNumClasses() - 1;
        
        unsigned int* dimensions = new unsigned int[numClasses];
        for (int i = 0; i < numClasses; ++i) {
            dimensions[i] = i;
        }

		CAbstractStateDiscretizer* whypIndexSpace= new AdaBoostMDPClassifierSimpleDiscreteSpace(_data->getIterationNumber()+1);
		CNeuralNetworkStateModifier* nnStateModifier = new CNeuralNetworkStateModifier(this->properties); //, dimensions, numClasses);

//        CFeatureOperatorAnd * combinedStates = new CFeatureOperatorAnd();
//        combinedStates->addStateModifier(nnStateModifier);
//        combinedStates->addStateModifier(whypIndexSpace);
		return nnStateModifier;
	}
    
    // -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	CStateModifier* AdaBoostMDPClassifierContinousBinary::getStateSpaceRBF(unsigned int partitionNumber)
	{
		// Now we can already create our RBF network
		// Therefore we will use a CRBFFeatureCalculator, our feature calculator uses both dimensions of the model state 
		// (the angel and the angular velocity) and lays a 20x20 RBF grid over the state space. For each dimension the given sigmas are used.
		// For the calculation of useful sigmas we have to consider that the CRBFFeatureCalculator always uses the 
		// normalized state representation, so the state variables are scaled to the intervall [0,1]
		
		int numClasses = getNumClasses();
		numClasses -= 1;
		unsigned int* dimensions = new unsigned int[numClasses];
		unsigned int* partitions = new unsigned int[numClasses];
		double* offsets = new double[numClasses];
		double* sigma = new double[numClasses];
		
		for(int i=0; i<numClasses; ++i )
		{
			dimensions[i]=i;
			partitions[i]=partitionNumber;
			offsets[i]=0.0;
			sigma[i]=1.0/(2.0*partitionNumber);
		}
		
		
		// Now we can create our Feature Calculator
		CStateModifier *rbfCalc = new CRBFFeatureCalculator(numClasses, dimensions, partitions, offsets, sigma);	
//		return rbfCalc;
		CAbstractStateDiscretizer* disc= new AdaBoostMDPClassifierSimpleDiscreteSpace(_data->getIterationNumber()+1);
		
		// Merge the discretizers
//		CDiscreteStateOperatorAnd *andDiscCalculator = new CDiscreteStateOperatorAnd();
		CFeatureOperatorAnd *andCalculator = new CFeatureOperatorAnd();
		
		
//		andCalculator->addStateModifier(rbfCalc);
//		andDiscCalculator->addStateModifier(disc);
   		andCalculator->addStateModifier(disc);
        andCalculator->addStateModifier(rbfCalc);
        
        andCalculator->initFeatureOperator();
        
		return andCalculator;
	}	
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------

	CStateModifier* AdaBoostMDPClassifierContinousBinary::getStateSpaceRBFAdaptiveCenters(unsigned int numberOfFeatures, CRBFCenterFeatureCalculator** rbfFC, CRBFCenterNetwork** rbfNW)
	{
		// Now we can already create our RBF network
		// Therefore we will use a CRBFFeatureCalculator, our feature calculator uses both dimensions of the model state 
		// (the angel and the angular velocity) and lays a 20x20 RBF grid over the state space. For each dimension the given sigmas are used.
		// For the calculation of useful sigmas we have to consider that the CRBFFeatureCalculator always uses the 
		// normalized state representation, so the state variables are scaled to the intervall [0,1]
		
        assert(numberOfFeatures > 0);
        
		int numClasses = getNumClasses();
		numClasses -= 1;
		
		CRBFCenterNetwork* network = new CRBFCenterNetworkSimpleSearch(numClasses);
		
        for (int j=0; j < numberOfFeatures; ++j) {
            
            ColumnVector* center = new ColumnVector(numClasses);
            ColumnVector* sigma = new ColumnVector(numClasses);
            
            //TODO: works only for binary !
            for (int i=0; i < numClasses; ++i) {
                center->element(i) = 1./numberOfFeatures * j;
                sigma->element(i) = 1. / (2*numberOfFeatures);
            }
            
            CRBFBasisFunction* rbf = new CRBFBasisFunction(center, sigma);
            network->addCenter(rbf);
        }
        
        //        CStateProperties* stateProp = new CStateProperties(numberOfFeatures, 1);
        
		//CStateProperties *stateProperties = new CStateProperties(1,0);
        //		CRBFCenterFeatureCalculator* rbffeat = new CRBFCenterFeatureCalculator(this->properties, network, numberOfFeatures);
		CRBFCenterFeatureCalculator* rbffeat = new CRBFCenterFeatureCalculator(this->properties, network, 1);
		*rbfNW = network;
        *rbfFC = rbffeat;
        
        
        //        for (int i=0; i < numberOfFeatures; ++i) {
        //            ColumnVector* center = new ColumnVector(numClasses);
        //            ColumnVector* sigma = new ColumnVector(numClasses);
        //            for (int j=0; j < numClasses; ++j) {
        //                center->element(j) = 1./numberOfFeatures + i;
        //                sigma->element(j) = 1. / (2*numberOfFeatures);
        //            }
        //            
        //            CRBFBasisFunction* rbf = new CRBFBasisFunction(center, sigma);
        //            rbffeat->addCenter(rbf);
        //        }
        
        
		//		return rbfCalc;
		CAbstractStateDiscretizer* disc= new AdaBoostMDPClassifierSimpleDiscreteSpace(_data->getIterationNumber()+1);
		
		// Merge the discretizers
		//		CDiscreteStateOperatorAnd *andDiscCalculator = new CDiscreteStateOperatorAnd();
		CFeatureOperatorAnd *andCalculator = new CFeatureOperatorAnd();
		
		
		//		andCalculator->addStateModifier(rbfCalc);
		//		andDiscCalculator->addStateModifier(disc);
   		andCalculator->addStateModifier(disc);
        andCalculator->addStateModifier(rbffeat);
        
        andCalculator->initFeatureOperator();
        
		return andCalculator;
	}
	
	
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	
	void AdaBoostMDPClassifierContinousBinary::outPutStatistic( BinaryResultStruct& bres )
	{
		_outputStream << bres.iterNumber << " " <<  bres.origAcc << " " << bres.acc << " " << bres.usedClassifierAvg << " " << bres.avgReward << " " << bres.TP << " " << bres.TN << endl;
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------	
	CStateModifier* AdaBoostMDPClassifierContinousBinary::getStateSpaceForRBFQFunction(int numOfFeatures)
	{
		int numClasses = getNumClasses();
		CStateModifier* retVal = new RBFStateModifier(numOfFeatures, numClasses-1, _data->getIterationNumber()+1 );
		return retVal;
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	
} // end of namespace MultiBoost