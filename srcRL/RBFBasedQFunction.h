/*
 *  RBFBasedQFunction.h
 *  RLTools
 *
 *  Created by Robert Busa-Fekete on 10/5/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef __RBFBASEDQFUNCTION_H
#define __RBFBASEDQFUNCTION_H

#include "cqfunction.h"
#include "cqetraces.h"
#include "cgradientfunction.h"
#include "cstatemodifier.h"
#include "RBFStateModifier.h"
#include "RBFQETraces.h"
#include "AdaBoostMDPClassifierAdv.h"

class RBF {
protected:
	double _mean;
	double _sigma;
	double _alpha;
	
public:
	RBF() : _mean(0), _sigma(0), _alpha(0) {}
	virtual ~RBF() {}
	
	virtual double getMean() { return _mean; }
	virtual double getSigma() { return _sigma; }
	virtual double getAlpha() { return _alpha; }
	
	virtual void setMean( double m ) { _mean=m; }
	virtual void setSigma( double s ) { _sigma = s; }
	virtual void setAlpha( double a ) { _alpha = a; }
	
	virtual double getValue( double x ) 
	{ 
		double retVal = _alpha * exp( - (( x-_mean )*( x-_mean ))/(2*_sigma*_sigma));
		return retVal;
	}
};


class RBFBasedQFunctionBinary : public CAbstractQFunction
{
protected:
	vector< map< CAction*, vector<RBF> > >	_rbfs;
	int _featureNumber;	
	CActionSet* _actions;
	int _numberOfActions;
public:
	RBFBasedQFunctionBinary(CActionSet *actions, CStateModifier* statemodifier ) : CAbstractQFunction(actions)
	{
		// the statemodifier must be RBFStateModifier
		RBFStateModifier* smodifier = dynamic_cast<RBFStateModifier*>( statemodifier );
		
		const int iterationNumber = smodifier->getNumOfIterations();
		const int featureNumber = smodifier->getNumOfRBFsPerIteration();
		const int numOfClasses = smodifier->getNumOfClasses();
		
		_featureNumber = featureNumber;
		
		assert(numOfClasses==1);
		
		_actions = actions;
		_numberOfActions = actions->size();
		
		_rbfs.resize( iterationNumber );
		for( int i=0; i<iterationNumber; ++i)
		{
			CActionSet::iterator it=(*actions).begin();
			for(;it!=(*actions).end(); ++it )
			{				
				vector<RBF> tmpVec(_featureNumber);
				_rbfs[i].insert(pair<CAction*, vector<RBF> >(*it, tmpVec) );
			}
		}
	}
		
	virtual ~RBFBasedQFunctionBinary() {}
	
	/// Interface for getting a Q-Value
	virtual double getValue(CStateCollection *state, CAction *action, CActionData *data = NULL) 
	{
		CState* currState = state->getState();
		
		int currIter = currState->getDiscreteState(0);
		double margin = currState->getContinuousState(0);
		
		double retVal = 0.0;
		
		for( int i=0; i<_featureNumber; ++i )
		{
			retVal += _rbfs[currIter][action][i].getValue(margin);  // lehet, hogy eggyet hozza kell adni a currIter-hez
		}		
		return retVal;
	}

	virtual void updateValue(CStateCollection *state, CAction *action, double td, CActionData * = NULL)
	{
		CState* currState = state->getState();
		
		int currIter = currState->getDiscreteState(0);
		double margin = currState->getContinuousState(0);
		
	}
	
	CAbstractQETraces* getStandardETraces()
	{
		return new RBFQETraces(this);
	}
	
	virtual void saveQTable( const char* fname )
	{
		FILE* outFile = fopen( fname, "w" );
		
		for (CActionSet::iterator it=_actions->begin(); it != _actions->end(); ++it)
		{
			fprintf(outFile,"%d ", dynamic_cast<CAdaBoostAction>(*it)->getMode() );
		}
		fprintf(outFile,"\n");
		
		for(int i=0; i<_rbfs.size(); ++i)
		{
			for (CActionSet::iterator it=_actions->begin(); it != _actions->end(); ++it)
			{
			}
		}
				
		fclose( outFile );
	}
	
};

#endif