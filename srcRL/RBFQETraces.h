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
#include "AdaBoostMDPClassifierAdv.h"
#include <vector>

using namespace std;

class RBFQETraces : public CAbstractQETraces 
{
protected:
	vector<CStateCollection*> _states;
	vector<CAction*> _actions;

public:
	RBFQETraces(CAbstractQFunction *qFunction) : CAbstractQETraces(qFunction) {}
	virtual ~RBFQETraces() {};
	
	/// Interface function for reseting the ETraces
	virtual void resetETraces() 
	{
//		for(int i=0; i< _states.size(); ++i )
//		{
//			CState* tmpState = _states[i];
//			if (tmpState) delete tmpState;
//		}
		_states.clear();

//		for(int i=0; i< _actions.size(); ++i )
//		{
//			CAction* tmpAction = _actions[i];
//			if (tmpAction) delete tmpAction;
//		}
		_actions.clear();				
	}
	/// Interface function for updating the ETraces
	/**
	 I.e. all stored ETraces-factors get multiplied by lambda * gamma, gamma is taken from the Q-Function. 
	 If the action is a multistep action, the factor lambda * gamma is exponentiated with the duration.
	 */
	virtual void updateETraces(CAction *action, CActionData *data = NULL) 
	{
		
	}
	/// Interface function for adding a State-Action pair with the given factor to the ETraces
	virtual void addETrace(CStateCollection *state, CAction *action, double factor = 1.0, CActionData *data = NULL) 
	{
		//CState* currentState = static_cast<CState*>(state)->clone();
		_states.push_back( state );
		
		//MultiBoost::CAdaBoostAction* currentAction = NULL;
		//int mode = dynamic_cast<MultiBoost::CAdaBoostAction*>(action)->getMode();
		//currentAction = new MultiBoost::CAdaBoostAction( mode );
		_actions.push_back( action );
	}
	
	/// Interface function for updating the Q-Values of all State-Action Pairs in the ETraces
	virtual void updateQFunction(double td) 
	{
		vector<CStateCollection*>::reverse_iterator invitState = _states.rbegin();
		vector<CAction*>::reverse_iterator invitAction = _actions.rbegin();
		for (; invitState != _states.rend(); ++invitState, ++invitAction )
		{
			CStateCollectionnt* currentState = *invitState;
			CAction* currentAction = *invitAction;
			qFunction->updateValue(currentState, currentAction, td, NULL);
		}
	}	
};

#endif