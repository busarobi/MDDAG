// Copyright (C) 2003
// Gerhard Neumann (gneumann@gmx.net)
// Stephan Neumann (sneumann@gmx.net) 
//                
// This file is part of RL Toolbox.
// http://www.igi.tugraz.at/ril_toolbox
//
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products
//    derived from this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
// IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
// NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef CENVIRONMENTMODEL_H
#define CENVIRONMENTMODEL_H

#include "cbaseobjects.h"

class CStateCollectionImpl;
class CPrimitiveAction;

/// The environment in which the agent can act
/** The class CEnvironmentModel represents the Environment of the agent. The environment manages the internal state of the agent. The agent tells the environment which
primitive action to execute (nextState(...)), and the environment changes its internal states depending on the action. How the internal state is represented in the environment can be choosen freely by the programmer. The agent can always fetch the current internal State of the environment, normally this is done every time after an action has been executed. We will call the state comming from the model "model state". This state should contain any information you have about your current internal state, no matter how much information that is. The model state is anyway usually not used for learning, so you don't have to look after the dimensionality of your model state.
The most relevant functions for the user are doNextState, getState and doResetModel.
- The doNextState(CPrimitiveAction *):  
Here you have to add the internal state transitions of the model. To get further access to the specified action you have to cast the primitve action object to your own action class. Attention: Always use dynamic_cast for this cast!! For more details see the description of CPrimitiveAction.  To indicate that the model has to be reseted after this
step you have to set the reset flag, to indicate that the episode failed, you set the failed flag. If failed is set, reset gets 
automatically set by the nextState method. 
- The getState(CState *state) function allows the agent to fetch the current state. The internal state variables has to be written in
the "state" object. 
- doResetModel(): This function is called every time the agent wants to start a new episode. Here the user should reset the internal model variables (for example to a fixed start place or to random values). 
The state properties of the environment object is created at the beginning with the parameters comming from the constuctor and is refered as the model state. Always use the properties of the model if you need the model state from a state collection.
When the agent wants to execute an action, it calles nextState. This methods calles again the user-interface doNextState. If after the call failed is set, reset gets set too. If reset is set before the call of doNextState the model gets reseted (by calling resetModel()) and the reset and failed flag are resetted, so you don't have to care about the handling of the flags except if an episode has failed or has to be resetted.
*/
class CEnvironmentModel : public CStateObject
{
	protected:
/// reset flag of the model
		bool reset;
/// failed flag of the model
		bool failed;
/** The doNextState(CPrimitiveActionFunction) function is invoked by the nextState function which is again called by the agent. 
Here you have to add the internal state transitions of the model. To indicate that the model has to be reseted after this
step you have to set the reset flag, to indicate that the episode failed, you set the failed flag. If failed is set, reset gets 
automatically set by the nextState method. */
	
		virtual void doNextState(CPrimitiveAction *action) = 0 ;
/**This function is called by the resetModel() method. Here you should reset the internal model variables. */

		virtual void doResetModel() {};

/// protected constructor, only for subclasses
		CEnvironmentModel(CStateProperties *properties);
	public:
/// Creates an Environment Model with a new properties object. The properties object is created with the parameters given to the constructor.
/**
@param continuousStates Number of continuous states of the model state
@param discreteStates Number of discrete states of the model state
*/
		CEnvironmentModel(int continuousStates, int discreteStates);
		
		virtual ~CEnvironmentModel();
/// returns reset
		bool isReset() {return reset;};
/// returns failed
		bool isFailed() {return failed;};

/// Writes the internal state variables in the state object. 
/** This method is usually called by getState(CStateCollectionImpl *). 
The user has to override this function for his specific model and write the internal state of the model in the state object. The state object ´has always the properties of the model state.*/
		virtual void getState(CState *state);
/**The getState(CStateCollection *stateCol) function enables the environment to add more than one state to the 
agents state collection. Per default this function only calls the getState(CState *) function with the model-state 
object from the state-collection as variable, so the model-state of the collection is always up to date. It also tells the 
statecollection that there is a new modelstate, so the collection marks all modified states as obsolete, so that they have to be calculated again, 
once needed.*/
		virtual void getState(CStateCollectionImpl *stateCollection);

/// called by agent to reset the model
/** The function only calls the model-specific function doResetModel() and resets the flags reset and failed to false.
*/
		virtual void resetModel();

/// called by agent to execute the next action
/** It just calles the method doNextState. If after the call failed is set, reset gets set too. If reset is set before the call of doNextState
the model gets reseted (by calling resetModel())
*/
		virtual void nextState(CPrimitiveAction *action);				
};



#endif



