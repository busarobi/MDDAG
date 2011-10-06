/*
 *  RBFStateModifier.h
 *  RLTools
 *
 *  Created by Robert Busa-Fekete on 10/5/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __RBFSTATEMODIFIER_H
#define __RBFSTATEMODIFIER_H

#include "cstate.h"

// ennek siman csak at kell masolnia az allapotot
class RBFStateModifier : public CStateModifier
{
protected:
	int _numOfIteration;
	int _numOfRBFsPerIteration;
	int _numOfClasses;
public:
	
	RBFStateModifier( int numOfRBFsPerIteration, int numOfClasses, int numOfIteration ) : CStateModifier(numOfClasses,1) // 1 continous for the margin and discrete for the iteration number
	{
		_numOfIteration = numOfIteration;
		_numOfRBFsPerIteration = numOfRBFsPerIteration;
		_numOfClasses = numOfClasses;
	}
	
	virtual ~RBFStateModifier() {}
	
	int getNumOfIterations() { return _numOfIteration; }
	int getNumOfRBFsPerIteration() { return _numOfRBFsPerIteration; }
	int getNumOfClasses() { return _numOfClasses; }
	
		
	virtual void getModifiedState(CStateCollection *originalState, CState *modifiedState)
	{
		CState* orig = originalState->getState();
		modifiedState->setState(orig);
	}
};


#endif