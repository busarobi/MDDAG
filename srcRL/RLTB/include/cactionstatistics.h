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

#ifndef C_ACTIONSTATISTIC_H
#define C_ACTIONSTATISTIC_H


#include "ril_debug.h"

class CAction;
class CAgentController;

/// Actionstatistics - for comparison of policies and agent controllers
/** Actionstatistics are used together with a @see CMultiController to
incoorporate prior knowledge and / or compare different policies
*/
class CActionStatistics
{
public:
	CActionStatistics();
	virtual ~CActionStatistics(){};
    /// propbability for this action
	double probability;
	/// how many actions promise the same reward 
	int equal;
	/// how many actions pronise the more reward. if > 0 this action is not greedy
	int superior;
	/// the policy / agent controller which proposed this action
    CAgentController *owner;
	/// the action itself
	CAction *action;

	/// clones Actionstatistics
	void copy(CActionStatistics *stat);
	/// initializes Actionstatistics to zero
	void reset();
};

/// baseclass for all ActionStatisticComparators, used by @see CMultiControllerGreedyPolicy
class CActionStatisticsComparator
{
public:
	virtual ~CActionStatisticsComparator() {};
	/// compare 2 different actionstatistics return values: -1: first < second; 0 first = second; 1 first > second
	virtual int compare(CActionStatistics *first, CActionStatistics *second) = 0;
};


/// Comparator for CMulticontrollerGreedyPolicy prefers best action
class CGreedyASComparator : public CActionStatisticsComparator
{
public:
	virtual ~CGreedyASComparator() {};
	virtual int compare(CActionStatistics *first, CActionStatistics *second);
};

/// Comparator for CMulticontrollerGreedyPolicy prefers epsilon-greedy action over best action
class CPEGreedyASComparator : public CActionStatisticsComparator
{
public:
	virtual ~CPEGreedyASComparator() {};
	virtual int compare(CActionStatistics *first, CActionStatistics *second);
};

/// Comparator for CMulticontrollerGreedyPolicy prefers actions of a specific owner
class CPOASComparator: public CActionStatisticsComparator
{
private:
	CAgentController* owner;

public:
	CPOASComparator(CAgentController *owner) {this->owner = owner;};
	virtual ~CPOASComparator() {};

	void setOwner(CAgentController *owner) {this->owner = owner;};
	CAgentController *getOwner() {return this->owner;};
	virtual int compare(CActionStatistics *first, CActionStatistics *second);
};
#endif

