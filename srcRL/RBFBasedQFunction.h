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
//#include "RBFQETraces.h"
#include "AdaBoostMDPClassifierAdv.h"
#include "cfeaturefunction.h"

class RBFQETraces;

class RBF {
protected:
	double _mean;
	double _sigma;
	double _alpha;
	string _ID;
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
		double retVal = _alpha * getActivationFactor(x);
		return retVal;
	}
    
    virtual double getActivationFactor( double x ) 
	{ 
		double retVal = exp( - (( x-_mean )*( x-_mean ))/(2*_sigma*_sigma));
		return retVal;
	}
    
	virtual string& getID() { return _ID;}
	virtual void setID( const string ID ) { _ID = ID; }
};


class RBFBasedQFunctionBinary : public CAbstractQFunction // CAbstractQFunction
{
protected:	
	map< CAction*, vector<vector<RBF> > > 	_rbfs;
	int _featureNumber;	
	CActionSet* _actions;
	int _numberOfActions;
    int _numberOfIterations;
    
    double _muAlpha;
    double _muMean;
    double _muSigma;
public:
	// constructor
	RBFBasedQFunctionBinary(CActionSet *actions, CStateModifier* statemodifier );
    
	// initilizes the parameters of RBFs
    void uniformInit(double* init=NULL);
    
	
	// getters and setters
    virtual double getMuAlpha() { return _muAlpha; }
	virtual double getMuMean() { return _muMean; }
	virtual double getMuSigma() { return _muSigma; }
	
	virtual void setMuMean( double m ) { _muMean=m; }
	virtual void setMuSigma( double s ) { _muSigma = s; }
	virtual void setMuAlpha( double a ) { _muAlpha = a; }
    
		
	virtual ~RBFBasedQFunctionBinary() {}
        
	
	/// Interface for getting a Q-Value
	virtual double getValue(CStateCollection *state, CAction *action, CActionData *data = NULL);

	
	// update the RBF-s
	virtual void updateValue(CStateCollection *state, CAction *action, double td, vector<vector<double> >& eTraces);

    
	//clacluates the gradient
    virtual void getGradient(CStateCollection *state, CAction *action, vector<vector<double> >& gradient);
    virtual void getGradient(double margin, int currIter, CAction *action, vector<vector<double> >& gradient, bool isNorm = false );
	
    	
	// creates an Etrace object
	CAbstractQETraces* getStandardETraces();
    
	
	// IO
	virtual void saveQTable( const char* fname );
    void saveActionValueTable(FILE* stream);
    void saveActionTable(FILE* stream);
};

#endif