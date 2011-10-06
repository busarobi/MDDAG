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

#include "cqetraces.h"


class RBFQETraces : public CAbstractQETraces 
{
protected:
public:

public:
	RBFQETraces(CAbstractQFunction *qFunction) : CAbstractQETraces(qFunction) {}
	virtual ~RBFQETraces() {};
	
	/// Interface function for reseting the ETraces
	virtual void resetETraces() {}
	/// Interface function for updating the ETraces
	/**
	 I.e. all stored ETraces-factors get multiplied by lambda * gamma, gamma is taken from the Q-Function. 
	 If the action is a multistep action, the factor lambda * gamma is exponentiated with the duration.
	 */
	virtual void updateETraces(CAction *action, CActionData *data = NULL) {}
	/// Interface function for adding a State-Action pair with the given factor to the ETraces
	virtual void addETrace(CStateCollection *State, CAction *action, double factor = 1.0, CActionData *data = NULL) {};
	
	/// Interface function for updating the Q-Values of all State-Action Pairs in the ETraces
	virtual void updateQFunction(double td) {}
	
	/// sets the Parameter "Lambda"
	virtual void setLambda(double lambda) {}
	/// returns the Parameter "Lambda"
	virtual double getLambda() { return 0.0; }
	
	/// sets the parameter "ReplacingETraces"
	virtual void setReplacingETraces(bool bReplace) {}
	/// returns the parameter "ReplacingETraces"
	virtual bool getReplacingETraces() { return true;} 
	
	
	
};

#endif