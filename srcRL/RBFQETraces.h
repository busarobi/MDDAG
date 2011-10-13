/*
 *  RBFQETraces.h
 *  RLTools
 *
 *  Created by Robert Busa-Fekete on 10/5/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef __RBFQETRACES_H
#define __RBFQETRACES_H

//#define RBFDEB

#include "cqetraces.h"
#include "AdaBoostMDPClassifierAdv.h"
//#include "RBFBasedQFunction.h"
#include <vector>
#include <list>

class RBFBasedQFunctionBinary;
using namespace std;


typedef vector<vector<double> > OneIterETrace;
typedef list<OneIterETrace > CustomETrace;
typedef CustomETrace::iterator ETIterator;
typedef CustomETrace::reverse_iterator ETReverseIterator;
//TODO: this structure supposes that we never come back to a state. It's fine for now.
class RBFQETraces : public CAbstractQETraces 
{
protected:
	list<double> _margins;
	list<int>	 _iters;
	
	list<CAction*> _actions;
    CustomETrace _eTraces; //strange ETrace struture, due to our peculiar state representation
	CStateCollection* _currentState;
public:

	// constructor
	RBFQETraces(CAbstractQFunction *qFunction) : CAbstractQETraces(qFunction) {
        addParameter("ETraceTreshold", 0.001);
    }
    
	virtual ~RBFQETraces() {};
	

	/// Interface function for reseting the ETraces
	virtual void resetETraces();
	/// Interface function for updating the ETraces
	/**
	 I.e. all stored ETraces-factors get multiplied by lambda * gamma, gamma is taken from the Q-Function. 
	 If the action is a multistep action, the factor lambda * gamma is exponentiated with the duration.
	 */
	virtual void updateETraces(CAction *action, CActionData *data = NULL);

	/// Interface function for adding a State-Action pair with the given factor to the ETraces
	virtual void addETrace(CStateCollection *state, CAction *action, double factor = 1.0, CActionData *data = NULL);

	/// Interface function for updating the Q-Values of all State-Action Pairs in the ETraces
	virtual void updateQFunction(double td); 
};


#endif