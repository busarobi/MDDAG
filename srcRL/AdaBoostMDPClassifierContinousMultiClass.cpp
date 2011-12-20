/*
 *  AdaBoostMDPClassifierContinousMultiClass.cpp
 *  RLTools
 *
 *  Created by Robert Busa-Fekete on 3/15/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "AdaBoostMDPClassifierContinousMultiClass.h"
#include "RBFBasedQFunction.h"
#include "RBFStateModifier.h"

#include "cstate.h"
#include "cstateproperties.h"

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {
	
	
	
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	AdaBoostMDPClassifierContinousMH::AdaBoostMDPClassifierContinousMH( const nor_utils::Args& args, int verbose, DataReader* datareader, int classNum )
	: AdaBoostMDPClassifierContinous( args, verbose, datareader, classNum, 1 )
	{
		// set the dim of state space
		//properties->setMinValue(0, .0);
		//properties->setMaxValue(0,  1.0);
		
		_exampleResult = NULL;
		
		// set the dim of state space
		properties->setDiscreteStateSize(0,datareader->getIterationNumber()+1);		
		
	}
	
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------	
	void AdaBoostMDPClassifierContinousMH::getState(CState *state)
	{
		AdaBoostMDPClassifierContinous::getState(state);
		state->setDiscreteState(0, _currentClassifier);
	}

	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------		
	double AdaBoostMDPClassifierContinousMH::getReward(CStateCollection *oldState, CAction *action, CStateCollection *newState) {
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
//                    rew = _lastReward;
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
			
			// useful only for "incremental reward" mode
//            rew -= _lastReward;
            
			if (_succRewardMode==RT_HAMMING)
			{
				if ( _data->currentClassifyingResult( _currentRandomInstance,  _exampleResult )  ) // classified correctly
				{
					failed = false;
					rew += _successReward;// /100.0;
				} else 
				{
					failed = true;
					//rew += -_successReward;
					rew += 0.0;
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
	CStateModifier* AdaBoostMDPClassifierContinousMH::getStateSpaceRBF(unsigned int partitionNumber)
	{
		// Now we can already create our RBF network
		// Therefore we will use a CRBFFeatureCalculator, our feature calculator uses both dimensions of the model state 
		// (the angel and the angular velocity) and lays a 20x20 RBF grid over the state space. For each dimension the given sigmas are used.
		// For the calculation of useful sigmas we have to consider that the CRBFFeatureCalculator always uses the 
		// normalized state representation, so the state variables are scaled to the intervall [0,1]
		
		int numClasses = getNumClasses();
		
		unsigned int* dimensions = new unsigned int[numClasses+1];
		unsigned int* partitions = new unsigned int[numClasses+1];
		double* offsets = new double[numClasses+1];
		double* sigma = new double[numClasses+1];
		
		for(int i=0; i<numClasses; ++i )
		{
			dimensions[i]=i;
			partitions[i]=partitionNumber;
			offsets[i]=0.0;
			sigma[i]=0.025;
		}
		
		
		// Now we can create our Feature Calculator
		CRBFFeatureCalculator *rbfCalc = new CRBFFeatureCalculator(numClasses, dimensions, partitions, offsets, sigma);	
		CAbstractStateDiscretizer* disc= new AdaBoostMDPClassifierSimpleDiscreteSpace(_data->getIterationNumber()+1);
		
		
		
		//CFeatureOperatorOr *andCalculator = new CFeatureOperatorOr();				
		//CDiscreteStateOperatorAnd *andCalculator = new CDiscreteStateOperatorAnd();
		//andCalculator->addStateModifier(rbfCalc);
		//andCalculator->addStateModifier(disc);		
		//return andCalculator;
		return NULL;
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	CStateModifier* AdaBoostMDPClassifierContinousMH::getStateSpace()
	{
		return NULL;
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	CStateModifier* AdaBoostMDPClassifierContinousMH::getStateSpaceNN()
	{
		return NULL;
	}	
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	CStateModifier* AdaBoostMDPClassifierContinousMH::getStateSpace( int divNum )
	{
		// create the discretizer with the build in classes
		// create the partition arrays
		int numClasses = getNumClasses();
		double *partitions = new double[divNum-1];
		double step = 1.0/divNum;
		for(int i=0;i<divNum-1;++i)
		{ 
			//cout << (i+1)*step << " " << endl << flush;
			partitions[i]= (i+1)*step; 
		}
		//double partitions[] = {-0.5,-0.2,0.0,0.2,0.5}; // partition for states
		//double partitions[] = {-0.2,0.0,0.2}; // partition for states
		
		CAbstractStateDiscretizer** disc = new CAbstractStateDiscretizer*[numClasses+1];
		
		//disc[0] = new CSingleStateDiscretizer(0,5,partitions);		
		disc[0]= new AdaBoostMDPClassifierSimpleDiscreteSpace(_data->getIterationNumber()+1);
		for(int l=0;l<numClasses;++l) disc[l+1] = new CSingleStateDiscretizer(0,divNum-1,partitions);
		
		// Merge the discretizers
		CDiscreteStateOperatorAnd *andCalculator = new CDiscreteStateOperatorAnd();
		
		for(int l=0;l<=numClasses;++l) andCalculator->addStateModifier(disc[l]);
		
		
		return andCalculator;
		
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------	
	
	CStateModifier* AdaBoostMDPClassifierContinousMH::getStateSpaceExp( int divNum, int e )
	{
		// create the discretizer with the build in classes
		// create the partition arrays
		int numClasses = getNumClasses();
		double *partitions = new double[divNum-1];
		for(int i=1;i<divNum;++i)
		{			
			partitions[i-1] = pow(i,e)/pow(divNum,e);
			partitions[i-1] = (partitions[i-1]/2.0);
			cout << partitions[i-1] << " ";
		}
		cout << endl << flush;
		
		double* realPartitions = new double[(divNum-1)*2+1];
		realPartitions[divNum-1]=0.5;
		for(int i=0;i<divNum-1;++i)
		{
			realPartitions[i]= 0.5 - partitions[divNum-2-i];
		}
		for(int i=0;i<divNum-1;++i)
		{
			realPartitions[divNum+i]= 0.5 + partitions[i];
		}
		
		for(int i=0; i<(divNum-1)*2+1; ++i)
		{
			cout << realPartitions[i] << " ";			
		}
		cout << endl;
		
		//double partitions[] = {-0.5,-0.2,0.0,0.2,0.5}; // partition for states
		//double partitions[] = {-0.2,0.0,0.2}; // partition for states
		
		CAbstractStateDiscretizer** disc = new CAbstractStateDiscretizer*[numClasses+1];
		
		//disc[0] = new CSingleStateDiscretizer(0,5,partitions);		
		disc[0]= new AdaBoostMDPClassifierSimpleDiscreteSpace(_data->getIterationNumber()+1);
		for(int l=0;l<numClasses;++l) disc[l+1] = new CSingleStateDiscretizer(0,(divNum-1)*2+1,realPartitions);
		
		// Merge the discretizers
		CDiscreteStateOperatorAnd *andCalculator = new CDiscreteStateOperatorAnd();
		
		for(int l=0;l<=numClasses;++l) andCalculator->addStateModifier(disc[l]);
		
		
		return andCalculator;
		
	}
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------	
    
    CStateModifier* AdaBoostMDPClassifierContinousMH::getStateSpaceForGSBNFQFunction( int numOfFeatures){
        int numClasses = getNumClasses();
		CStateModifier* retVal = new RBFStateModifier(numOfFeatures, numClasses, _data->getIterationNumber()+1 );
		return retVal;
        
    }

    // -----------------------------------------------------------------------
	// -----------------------------------------------------------------------	

} // end of namespace MultiBoost