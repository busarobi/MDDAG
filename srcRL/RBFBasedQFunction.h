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


#include "ArrayBasedQFunctionBinary.h"
#include "cqfunction.h"
#include "cqetraces.h"
#include "cgradientfunction.h"
#include "cstatemodifier.h"
#include "RBFStateModifier.h"
//#include "RBFQETraces.h"
#include "AdaBoostMDPClassifierAdv.h"
#include "cfeaturefunction.h"

class RBFQETraces;


//---------------------------------------------------------------------
//---------------------------------------------------------------------
class RBFBasedQFunctionBinary : public CAbstractQFunction // CAbstractQFunction
{
protected:	
	map< CAction*, vector<vector<RBF> > > 	_rbfs;
	int _featureNumber;	
	CActionSet* _actions;
	int _numberOfActions;
    int _numberOfIterations;
    
public:
	// constructor
	RBFBasedQFunctionBinary(CActionSet *actions, CStateModifier* statemodifier );
    
	// initilizes the parameters of RBFs
    virtual void uniformInit(double* init=NULL);
    			
	virtual ~RBFBasedQFunctionBinary() {}
        
	
	/// Interface for getting a Q-Value
	virtual double getValue(CStateCollection *state, CAction *action, CActionData *data = NULL);

	
	// update the RBF-s
	virtual void updateValue(CStateCollection *state, CAction *action, double td, vector<vector<double> >& eTraces);

    
	//clacluates the gradient
    virtual void getGradient(CStateCollection *state, CAction *action, vector<vector<double> >& gradient);
    virtual void getGradient(double margin, int currIter, CAction *action, vector<vector<double> >& gradient );
	
    	
	// creates an Etrace object
	CAbstractQETraces* getStandardETraces();
    
	
	// IO
	virtual void saveQTable( const char* fname );
    void saveActionValueTable(FILE* stream);
    void saveActionTable(FILE* stream);
};



//---------------------------------------------------------------------
//---------------------------------------------------------------------


#endif