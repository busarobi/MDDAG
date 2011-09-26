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

#ifndef C_REWARDMODEL_H
#define C_REWARDMODEL_H

#include <map>
#include <vector>

#include "crewardfunction.h"
#include "clearndataobject.h"
#include "cbaseobjects.h"
#include "cutility.h"
#include "cagentlistener.h"

class CAbstractFeatureStochasticEstimatedModel;

class CFeatureStateRewardFunction : public CFeatureRewardFunction
{
protected:
	std::map<int, double> *rewards;
public:
	CFeatureStateRewardFunction(CStateProperties *discretizer);
	virtual ~CFeatureStateRewardFunction();

	virtual double getReward(int oldState, CAction *action, int newState);
	virtual double getReward(int state);

	virtual void setReward(int state, double reward);
};


/**
For model based learning you need an reward function which assigns a reward for transitions of feature indices, not for state objects.
But what happens if you don't have a reward function for features to yours disposal, if you just have normal reward function (e.g. for the model state)?
You can also estimate the reward you will get for a transition, this is done by CFeatureRewardModel. Therefore it stores the reward already got when the same transition occurred and the visits of the transition.
so it can calculate the mean reward. For the visits of the transition, a estimated model can also be used to spare memory. Since the reward model must learn from the training trial it has to be added to the
agent's listener list. Since the reward model implements the CFeatureRewardFunction interface it can also be used as normal reward function.
semi MDP support hans't been added by now.

@see CFeatureRewardFunction
*/

class CFeatureRewardModel : public CFeatureRewardFunction, public CSemiMDPRewardListener, public CActionObject, public CLearnDataObject
{
protected:
	/// Table of the rewards, summed up during the whole training trial, for a transition
	CMyArray2D<CFeatureMap *> *rewardTable;
	/// Table of the Transition visits, so the reward can be calculated by the mean (sum rewards/sum visits)
	/** Only used if no extern estimated model is assigned
	*/
	CMyArray2D<CFeatureMap *> *visitTable;

/// Used for calculating the visits, so no visit table is needed if used.
	CAbstractFeatureStochasticEstimatedModel *model;

	bool bExternVisitSparse;

/// Returns the transition visits of the specified state
/**
Returns either the visits from the visit table, or, if an estimated model is assigned, the visits can also be retrieved by the model
*/
	double getTransitionVisits(int oldState, int action, int newState);

public:
/// Creates a reward model, which uses an estimated model for the calculation of the transition visits.
/** It is very important that the estimated model contains the same transitions as the reward table, because otherwise a division by zero would occur.
So the estimated model has to be added before the reward model to the listener.*/
	CFeatureRewardModel(CActionSet *actions, CRewardFunction *function, CAbstractFeatureStochasticEstimatedModel *model, CStateModifier *discretizer);
/// Creates a reward model, which has to use an own visit table for the visits.
	CFeatureRewardModel(CActionSet *actions, CRewardFunction *function, CStateModifier *discretizer);
	virtual ~CFeatureRewardModel();

///Returns the reward for a specific discrete state transition
/**
Calculates the mean reward from that transition, i.e. sum rewards/sum visits
*/
	virtual double getReward(int oldState, CAction *action, int newState);
	virtual void nextStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *newState);

/// Saves the reward model
/** Saves the reward Table and the visit table if it is used*/
	virtual void saveData(FILE *stream);
///
/** Loads the reward Table and the visit table if no estimated model was assigned to the constructor*/
	virtual void loadData(FILE *stream);

	virtual void resetData();
};

class CFeatureStateRewardModel : public CFeatureRewardFunction, public CSemiMDPRewardListener, public CLearnDataObject
{
protected:
	double *rewards;
	double *visits;

	double rewardMean;
	int numRewards;

public:
	/// Creates a reward model, which uses an estimated model for the calculation of the transition visits.
	/** It is very important that the estimated model contains the same transitions as the reward table, because otherwise a division by zero would occur.
	So the estimated model has to be added before the reward model to the listener.*/
	CFeatureStateRewardModel(CRewardFunction *function, CStateModifier *discretizer);
	/// Creates a reward model, which has to use an own visit table for the visits.
	virtual ~CFeatureStateRewardModel();

	/// Calls getReward(int oldState, CAction *action, int newState) with the features from the state as argument
	/** The state has to be a feature state, this state gets decomposed into his features, the reward for the features
	is calculated and summed up wheighted with the feature factors.
	*/
	virtual double getReward(CState *oldState, CAction *action, CState *newState);
	///Returns the reward for a specific discrete state transition
	/**
	Calculates the mean reward from that transition, i.e. sum rewards/sum visits
	*/
	virtual double getReward(int oldState, CAction *action, int newState);
	virtual double getReward(int newState);

	virtual void nextStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *newState);

	/// Saves the reward model
	/** Saves the reward Table and the visit table if it is used*/
	virtual void saveData(FILE *stream);
	///
	/** Loads the reward Table and the visit table if no estimated model was assigned to the constructor*/
	virtual void loadData(FILE *stream);

	virtual void resetData();
};

///Logs the reward during an Episode
/**
Maintains a double array for logging the reward of each step. Can be used in combination with
a normal episode to log the whole State-Action-Reward-State Tuples.
*/

class CRewardEpisode : public CSemiMDPRewardListener
{
protected:
/// array for the rewards
	std::vector<double> *rewards;
public:
	CRewardEpisode(CRewardFunction *rewardFunction);
	virtual ~CRewardEpisode();

/// Stores the current reward at the and of the rewards vector
	virtual void nextStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *newState);
/// clears the reward vector	
	virtual void newEpisode();

	int getNumRewards();
/// returns the requested reward from the rewards vector.
	double getReward(int index);

	double getMeanReward();
	double getLastStepsMeanReward(int Steps);

	double getSummedReward(double gamma = 1.0);


	virtual void saveBIN(FILE *stream);
	virtual void saveData(FILE *stream);
	virtual void loadBIN(FILE *stream);
	virtual void loadData(FILE *stream);
};

class CRewardHistory
{
	protected:

	public:
		CRewardHistory() {};
		virtual ~CRewardHistory(){};

		virtual CRewardEpisode* getEpisode(int index) = 0;
		virtual int getNumEpisodes() = 0;
};

class CRewardHistorySubset : public CRewardHistory
{
	protected:
		CRewardHistory *episodes;
		std::vector<int> *indices;
	public:
		CRewardHistorySubset(CRewardHistory *episodes, std::vector<int> *indices);
		virtual ~CRewardHistorySubset();

		/// returns the number of episodes
		virtual int getNumEpisodes();
		/// returns a specific episode
		virtual CRewardEpisode* getEpisode(int index);
};

class CRewardLogger : public CSemiMDPRewardListener, public CLearnDataObject, public CRewardHistory
{
protected:
	/// autosave file name
    char filename[512];
	/// autosave file
    FILE* file;

	char loadFileName[512];

	/// number of episodes to hold in memory, if set to -1 all episodes are held in memory
	int holdMemory;

	/// list of the episodes
	std::list<CRewardEpisode *> *episodes;
	/// pointer to the currentepisode
	CRewardEpisode *currentEpisode;
	
	void init();
public:
	
	CRewardLogger(CRewardFunction *rewardFunction, char* autoSavefile, int holdMemory);
	CRewardLogger(CRewardFunction *rewardFunction);
	CRewardLogger(char *loadFile, CRewardFunction *rewardFunction);

	virtual ~CRewardLogger();

	virtual void nextStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState);
	virtual void newEpisode();

	void setAutoSaveFile(char *filename);
	virtual void saveBIN(FILE *stream);
	
	virtual void saveData(FILE *stream);
	void loadBIN(FILE *stream, int episodes = -1);
	virtual void loadData(FILE *stream, int episodes = -1);
	virtual void loadData(FILE *stream);

	virtual int getNumEpisodes();
	
	virtual CRewardEpisode* getCurrentEpisode();
	virtual CRewardEpisode* getEpisode(int index);

	void clearAutoSaveFile();
	void setLoadDataFile(char *loadData);

	virtual void resetData();
};

/// Reward Function for Behaviours
/**
Very often the reward of a behaviour consists of the summud up rewards from the primitiv actions which were executed during
the behaviour was activ. The class CSemiMDPLastNRewardFunction does this reward calculation. It is a subclass of CRewardEpisode, so it mantains a
reward array containing all rewards from the past. The function getReward(CStateCollection *oldState, CAction *action, CStateCollection *newState).
For an action of duration d, the function calculates the reward of the transition by sum_{i=0}^{d-1} gamma^i * r(N-i). The discount factor is needed
since the reward from the past has to be weakened by this factor.
<p>
Since the object has to have access to the past "primitiv" rewards, it has to be added to the listener list of the agent. 
*/

class CSemiMDPLastNRewardFunction : public CRewardFunction, public CRewardEpisode
{
protected:
/// The discount factor 
	double gamma;
public:
/// Creates the reward function with the discount factor gamma.
	CSemiMDPLastNRewardFunction(CRewardFunction *rewardFunction, double gamma);
	virtual ~CSemiMDPLastNRewardFunction();
/// Calculates the reward for extended actions.
/**
For an action of duration d, the function the reward is calculated by sum_{i=0}^{d-1} gamma^i * r(N-i). The discount factor is needed
since the reward from the past has to be weakened by this factor.
*/
	virtual double getReward(CStateCollection *oldState, CAction *action, CStateCollection *newState);
};

#endif // REWARDMODEL_H

