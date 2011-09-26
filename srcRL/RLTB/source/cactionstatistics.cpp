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

#include "cactionstatistics.h"
#include "ril_debug.h"
#include <stdlib.h>
#include <memory.h>


CActionStatistics::CActionStatistics()
{
	this->probability = 0.0;
	this->equal = 0;
	this->superior = 0;
	this->owner = NULL;
	this->action = NULL;
}

void CActionStatistics::copy(CActionStatistics *stat)
{
	memcpy(this, stat, sizeof(CActionStatistics));
}

void CActionStatistics::reset()
{
	this->probability = 0.0;
	this->equal = 0;
	this->superior = 0;
}


int CGreedyASComparator::compare(CActionStatistics *first, CActionStatistics *second)
{
	if (first->probability  > second->probability) return 1;
	else if (first->probability < second->probability) return -1;
	else
	{
		if (first->superior > second->superior) return 1;
		else if (first->superior < second->superior) return -1;
		else
		{
			if (first->equal < second->equal) return 1;
			else if (first->equal > second->equal) return -1;
			else return 0;
		}
	}
}


int CPEGreedyASComparator::compare(CActionStatistics *first, CActionStatistics *second)
{
	if (first->superior == second->superior)
	{
		if (first->probability > second->probability) return 1;
		else if (first->probability < second->probability) return -1;
		else
		{
			if (first->equal < second->equal) return 1;
			else if (first->equal > second->equal) return -1;
			else return 0;
		}
	}
	else
	{
		if (first->superior > second->superior) return 1;
		else return -1;
	}
}


int CPOASComparator::compare(CActionStatistics *first, CActionStatistics *second)
{
	if (first->owner == second->owner) return 0;
	else
	{
		if (first->owner == this->owner) return 1;
		else if (second->owner == this->owner) return -1;
		else return 0;
	}
}
