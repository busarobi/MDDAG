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

#include <time.h>
#include <stdio.h>

#include "cparameters.h"
#include "cutility.h"
#include "ril_debug.h"

#include <math.h>
#include <assert.h>



CParameters::CParameters()
{
	parameters = new std::map<string, double>();
	isAdaptive = new std::map<string, bool>();
}

CParameters::CParameters(CParameters &copy)
{
	parameters = new std::map<string, double>();
	isAdaptive = new std::map<string, bool>();
	addParameters(&copy);
}


CParameters::~CParameters()
{
	delete isAdaptive;
	delete parameters;
}

void CParameters::setAsAdaptiveParameter(string parameter, bool adaptive)
{
	(*isAdaptive)[parameter] = adaptive;
}

bool CParameters::isAdaptiveParameter(string parameter)
{
	return (*isAdaptive)[parameter];
}

void CParameters::loadParametersStream(istream *stream)
{
	char description[50];
	char line[250];
	int numParams;
	(*stream) >> description;
	(*stream) >> numParams;

	double value = 0.0;
	for (int i = 0; i < numParams; i ++)
	{
		stream->getline(line, 250);
		sscanf(line, "%s : %lf\n", description, &value);
		addParameter(description,value);
	}
}

void CParameters::saveParametersStream(ostream *stream)
{
	(*stream) << "Parameters " << parameters->size() << endl;

	std::map<string, double>::iterator it;

	for (it = parameters->begin(); it != parameters->end(); it ++)
	{
		(*stream) << (*it).first.c_str() << " : " << (*it).second << endl;
	}
}


void CParameters::loadParameters(FILE *stream)
{
	char description[50];
	int numParams;

	char buffer[80];
	//fscanf(stream, "%s %d\n", buffer, &numParams);

	int result = fscanf(stream, "%s %d\n", buffer, &numParams);

	double value = 0.0;
	
	if (result == 2)
	{
		for (int i = 0; i < numParams; i ++)
		{
			fscanf(stream, "%s : %lf\n", description, &value);
			
			if (getParameterIndex(description) >= 0)
			{
				setParameter(description, value);
			}
			else
			{
				addParameter(description,value);
			}
		}
	}
}

void CParameters::saveParameters(FILE *stream)
{
	fprintf(stream, "Parameters %d\n", parameters->size());

	std::map<string, double>::iterator it;

	for (it = parameters->begin(); it != parameters->end(); it ++)
	{
		fprintf(stream, "%s : %f\n",(*it).first.c_str(), (*it).second);
	}
}

void CParameters::saveParametersXML(FILE *stream)
{
	fprintf(stream, "<Parameters>\n");

	std::map<string, double>::iterator it;

	for (it = parameters->begin(); it != parameters->end(); it ++)
	{
		fprintf(stream, "<%s> %f </%s>\n",(*it).first.c_str(), (*it).second, (*it).first.c_str());
	}

	fprintf(stream, "</Parameters>\n");
}

void CParameters::loadParametersXML(FILE *stream)
{
	char buffer1[256];
	char paramName[256];
	double value = 0.0;

	fscanf(stream, "<Parameters>\n");

	std::map<string, double>::iterator it;

	fscanf(stream, "%s", buffer1);
	while (strcmp(buffer1, "</Parameters>") != 0)
	{
		if (fscanf(stream, " %lf </%s>\n", &value, paramName) != 2)
		{
			break;
		}
		paramName[strlen(paramName) - 1] = '\0';
		if (getParameterIndex(paramName) >= 0)
		{
			setParameter(paramName, value);
		}
		else
		{
			addParameter(paramName,value);
		}fscanf(stream, "%s", buffer1);
	}
	fscanf(stream, "\n");
}



void CParameters::addParameter(string name, double value)
{
	std::map<string, double>::iterator it;

	it = parameters->find(name);

	if (it == parameters->end())
	{
		(*parameters)[name] = value;
		(*isAdaptive)[name] = false;
	}
}

void CParameters::removeParameter(string name)
{
	std::map<string, double>::iterator it = parameters->find(name);

	if (it != parameters->end())
	{
		parameters->erase(it);
	}
}

void CParameters::addParameters(CParameters *lparameters)
{
	std::map<string, double>::iterator it;

	
	for (int i = 0; i < lparameters->getNumParameters(); i ++)
	{
		string name = lparameters->getParameterName(i);
		it = parameters->find(name);

		if (it == parameters->end())
		{
			(*parameters)[name] = lparameters->getParameter(name);
			(*isAdaptive)[name] = lparameters->isAdaptiveParameter(name);
		}
	}
}


double CParameters::getParameter(string name)
{
	if (parameters->find(name) != parameters->end())
	{
//		printf("Getting Paramter %s: %f\n", name.c_str(),(*parameters->find(name)).second);
		return (*parameters->find(name)).second;
	}
	else
	{
		printf("Getting unknown Parameter %s, Abort!!\n", name.c_str());
		assert(false);
		return -1;
	}
}

void CParameters::setParameter(string name, double value)
{
	std::map<string, double>::iterator it;

	it = parameters->find(name);

	if (it != parameters->end())
	{
		(*it).second = value;
	}
	else
	{
		printf("Setting Parameter, Warning : Unknown Parameter %s (Value %f)\n", name.c_str(), value);
	}
}

void CParameters::setParameters(CParameters *lparameters)
{
	std::map<string, double>::iterator it = parameters->begin();

	for (; it != parameters->end(); it ++)
	{
		if (lparameters->getParameterIndex((*it).first) >= 0)
		{
			setParameter((*it).first, lparameters->getParameter((*it).first));
		}
	}
}


double CParameters::getParameterFromIndex(unsigned int index)
{
	return getParameter(getParameterName(index));
}

string CParameters::getParameterName(unsigned int index)
{
	std::map<string, double>::iterator it;

	assert(index < parameters->size());

	unsigned int i = 0;
	for (it = parameters->begin(); i < index; it ++, i++);

	return (*it).first;
}

void CParameters::setParameterWithIndex(unsigned int index, double value)
{
	std::map<string, double>::iterator it;

	unsigned int i = 0;
	for (it = parameters->begin(); it != parameters->end(),i < index; it ++, i++);

	if (it != parameters->end())
	{
		setParameter((*it).first, value);
	}
}

int CParameters::getParameterIndex(string name)
{
	std::map<string, double>::iterator it = parameters->begin();

	int i = 0;
	while (it != parameters->end() && name != (*it).first)
	{
		it++;
		i++;
	}

	if (it != parameters->end())
	{
		return i;
	}
	else
	{
		return -1;
	}
}

bool CParameters::containsParameters(CParameters *parametersObject)
{

	if (parametersObject->getNumParameters() > getNumParameters())
	{
		return false;
	}

	for (int i = 0 ;i < parametersObject->getNumParameters(); i++)
	{
		string name =  parametersObject->getParameterName(i);
		
		if (getParameterIndex(name) < 0)
		{
			return false;
		}
		if (fabs(getParameter(name) - parametersObject->getParameter(name)) > 0.00001)
		{
			//printf("Different Param Values for Parameter %s, %f %f\n", (*it).first.c_str(), (*it).second, parameterObject.getParameter((*it).first));
			return false;
		}
	}
	return true;
}


int CParameters::getNumParameters()
{
	return parameters->size();
}


bool CParameters::operator == (CParameters &parameterObject)
{
	std::map<string, double>::iterator it = parameters->begin();

	if (parameterObject.getNumParameters() != getNumParameters())
	{
		return false;
	}

	for (int i = 0 ;it != parameters->end(); it++, i++)
	{
		if (parameterObject.getParameterIndex((*it).first) >= 0)
		{
			if (!isAdaptiveParameter((*it).first) &&  !parameterObject.isAdaptiveParameter((*it).first) && fabs((*it).second - parameterObject.getParameter((*it).first)) > 0.00001)
			{
				//printf("Different Param Values for Parameter %s, %f %f\n", (*it).first.c_str(), (*it).second, parameterObject.getParameter((*it).first));
				return false;
			}
		}
		else
		{
			return false;
		}
	}
	return true;
}



bool CParameters::operator < (CParameters &compareParams)
{
	std::map<string, double>::iterator it = parameters->begin();
	while (it != parameters->end() && ((*it).second == compareParams.getParameter((*it).first) || isAdaptiveParameter((*it).first) || compareParams.isAdaptiveParameter((*it).first)))
	{
		it ++;
	}
	if (it != parameters->end())
	{
		return (*it).second < compareParams.getParameter((*it).first);
	}
	else 
	{
		return false;
	}
}


CParameterObject::CParameterObject()
{
	parameterObjects = new std::list<paramPair>();
//	adaptiveParameters = new std::map<string, CAdaptiveParameterCalculator *>();

}

CParameterObject::~CParameterObject()
{
	delete parameterObjects;
//	delete adaptiveParameters;
}

/*void CParameterObject::addAdaptiveParameter(string name, CAdaptiveParameterCalculator *paramCalc)
{
	string paramName = name;
	(*adaptiveParameters)[name] = paramCalc;

	std::list<paramPair>::iterator it = parameterObjects->begin();

	for (;it != parameterObjects->end(); it ++)
	{
		string prefix = (*it).second;

		if (prefix == paramName.substr(0, prefix.length()))
		{
			paramName = paramName.substr(prefix.length());
			if ((*it).first->getParameterIndex(paramName) >= 0)
			{
				(*it).first->addAdaptiveParameter(paramName, paramCalc);
			}
		}
	}
    addParameters(paramCalc, name);
}

void CParameterObject::removeAdaptiveParameter(string paramName)
{
	std::map<string, CAdaptiveParameterCalculator *>::iterator it = adaptiveParameters->find(paramName);

	if (it != adaptiveParameters->end())
	{
		adaptiveParameters->erase(it);
	}
}*/

bool CParameterObject::operator == (CParameters &parameterObject)
{
	std::map<string, double>::iterator it = parameters->begin();

	if (parameterObject.getNumParameters() != getNumParameters())
	{
		return false;
	}

	for (int i = 0 ;it != parameters->end(); it++, i++)
	{
		if (parameterObject.getParameterIndex((*it).first) >= 0)
		{
			//if (adaptiveParameters->find((*it).first) == adaptiveParameters->end() && fabs((*it).second - parameterObject.getParameter((*it).first)) > 0.00001)
			if (fabs((*it).second - parameterObject.getParameter((*it).first)) > 0.00001)
			{
				//printf("Different Param Values for Parameter %s, %f %f\n", (*it).first.c_str(), (*it).second, parameterObject.getParameter((*it).first));
				return false;
			}
		}
		else
		{
			return false;
		}
	}
	return true;
}


void CParameterObject::setParameter(string name, double value)
{
	CParameters::setParameter(name, value);
	parametersChanged();
}

void CParameterObject::setParameters(CParameters *parameters)
{
	CParameters::setParameters(parameters);
	parametersChanged();
}

void CParameterObject::parametersChanged()
{
	std::list<paramPair>::iterator it = parameterObjects->begin();

	for (;it != parameterObjects->end(); it ++)
	{
		string prefix = (*it).second;
		for (int i = 0; i < getNumParameters(); i++)
		{
			string paramName = getParameterName(i);
			double value = getParameter(paramName);

			if (prefix == paramName.substr(0, prefix.length()))
			{
				paramName = paramName.substr(prefix.length());
				if ((*it).first->getParameterIndex(paramName) >= 0 && fabs((*it).first->getParameter(paramName) - value) > 0.00001)
				{
					(*it).first->setParameter(paramName, value);
				}
			}
		}
	}
	onParametersChanged();
}

void CParameterObject::addParameters(CParameterObject *lparameters, string prefix)
{
	std::map<string, double>::iterator it = parameters->begin();

	for (int i = 0; i < lparameters->getNumParameters(); i ++)
	{
		string name = prefix + lparameters->getParameterName(i);
		it = parameters->find(name);

		if (it == parameters->end()) 
		{
			(*parameters)[name] = lparameters->getParameterFromIndex(i);
		}
	} 
	parameterObjects->push_back(paramPair(lparameters, prefix));
}

double CParameterObject::getParameter(string name)
{
	//std::map<string, CAdaptiveParameterCalculator *>::iterator it = adaptiveParameters->find(name);

	/*if (it != adaptiveParameters->end())
	{
		return (*it).second->getParameterValue();
	}
	else
	{
		return CParameters::getParameter(name);
	}*/
	return CParameters::getParameter(name);
}

/*
void CParameterObject::resetParameterCalculators()
{
	std::map<string, CAdaptiveParameterCalculator *>::iterator it = adaptiveParameters->begin();

	for (; it != adaptiveParameters->end(); it ++)
	{
		(*it).second->resetCalculator();
	}

	std::list<paramPair>::iterator it2 = parameterObjects->begin();

	for (;it2 != parameterObjects->end(); it2 ++)
	{
		(*it2).first->resetParameterCalculators();
	}
}*/




CAdaptiveParameterCalculator::CAdaptiveParameterCalculator(CParameters *l_targetObject, string l_targetParameter,int functionKind) : targetParameter(l_targetParameter)
{
	this->functionKind = functionKind;

	this->targetObject = l_targetObject;

	addParameter("APFunction", (double) functionKind);

	targetObject->setAsAdaptiveParameter(targetParameter, true);

}

CAdaptiveParameterCalculator::~CAdaptiveParameterCalculator()
{
}

void CAdaptiveParameterCalculator::onParametersChanged()
{
	CParameterObject::onParametersChanged();
	functionKind = my_round(getParameter("APFunction"));
}

void CAdaptiveParameterCalculator::setParameterValue(double value)
{
	targetObject->setParameter(targetParameter, value);
}

CAdaptiveParameterBoundedValuesCalculator::CAdaptiveParameterBoundedValuesCalculator(CParameters *l_targetObject, string l_targetParameter, int functionKind, double paramOffset, double paramScale, double targetMin, double targetMax) : CAdaptiveParameterCalculator(l_targetObject, l_targetParameter, functionKind)
{
	this->targetMin = targetMin;
	this->targetMax = targetMax;
	this->paramOffset = paramOffset;
	this->paramScale = paramScale;

	addParameter("APTargetMin", targetMin);
	addParameter("APTargetMax", targetMax);
	addParameter("APParamOffset", paramOffset);
	addParameter("APParamScale", paramScale);
	addParameter("APInvertTargetFunction", 0.0);
	addParameter("APTargetScale", 1.0);

	invertTarget = false;
	targetScale = 1.0;
	
}

CAdaptiveParameterBoundedValuesCalculator::~CAdaptiveParameterBoundedValuesCalculator()
{
}

void CAdaptiveParameterBoundedValuesCalculator::onParametersChanged()
{
	CAdaptiveParameterCalculator::onParametersChanged();

	targetMin = getParameter("APTargetMin");
	targetMax = getParameter("APTargetMax");
	targetScale = getParameter("APTargetScale");
	paramScale = getParameter("APParamScale");
	paramOffset = getParameter("APParamOffset");
	invertTarget = getParameter("APInvertTargetFunction") > 0.5;
}


void CAdaptiveParameterBoundedValuesCalculator::setParameterValue(double targetValue)
{
	double functionValue = 0.0;
	double functionArgument = (targetValue - targetMin) / (targetMax - targetMin);

	if (functionArgument < 0)
	{
		functionArgument = 0;
	}
	else
	{
		if (functionArgument > 1.0)
		{
			functionArgument = 1.0;
		}
	}


	switch (functionKind)
	{
	case LINEAR:
		{
			functionValue = functionArgument;
			break;
		}
	case SQUARE:
		{
			functionValue = pow(functionArgument, 2);
			break;
		}
	case LOG:
		{
			assert(functionArgument >= 0);
			functionValue = log(functionArgument * targetScale + 1.0) / log(1.0 + targetScale);
			break;
		}
	case FRACT:
		{
			assert(functionArgument > 0);
			functionValue = (1.0 / (functionArgument * targetScale + 1.0) - 1.0 / (targetScale + 1.0)) * (1.0 + targetScale) /targetScale;
		}
	case FRACTSQUARE:
		{
			assert(functionArgument > 0);
			double powScale = pow(targetScale, (double) 2.0);
			functionValue = (1.0 / (pow(functionArgument, 2) * powScale + 1.0) - 1.0 / (powScale + 1.0)) * (1.0 + powScale) /powScale;
			break;
		}
	case FRACTLOG :
		{
			assert(functionArgument >= 0);
			double offset = 1.0 / (1.0 + log(1.0 + targetScale));
			functionValue = (1.0 / (1.0 + log(functionArgument * targetScale + 1.0)) - offset) / (1 - offset);
			break;
		}
	default:
		{
			functionValue = functionArgument;
			printf("Unknown Adaptive Parameter Function Value %d, linear used instead\n", functionKind);
		}
	}
	if (invertTarget)
	{
		functionValue = 1 - functionValue;
	}
	
	CAdaptiveParameterCalculator::setParameterValue(paramOffset + paramScale * functionValue);
}

CAdaptiveParameterUnBoundedValuesCalculator::CAdaptiveParameterUnBoundedValuesCalculator(CParameters *l_targetObject, string l_targetParameter, int functionKind, double param0, double paramScale, double target0, double targetScale) : CAdaptiveParameterCalculator(l_targetObject, l_targetParameter, functionKind)
{
	addParameter("APTargetOffset", target0);
	addParameter("APTargetScale", targetScale);
	addParameter("APParamOffset", param0);
	addParameter("APParamScale", paramScale);

	addParameter("APParamLimit", 0.0);

	
	paramLimit = 0.0;
	targetOffset = target0;
	this->targetScale = targetScale;
}

CAdaptiveParameterUnBoundedValuesCalculator::~CAdaptiveParameterUnBoundedValuesCalculator()
{
}

void CAdaptiveParameterUnBoundedValuesCalculator::onParametersChanged()
{
	CAdaptiveParameterCalculator::onParametersChanged();

	targetOffset = getParameter("APTargetOffset");
	targetScale = getParameter("APTargetScale");
	paramOffset = getParameter("APParamOffset");
	paramScale = getParameter("APParamScale");
	paramLimit = getParameter("APParamLimit");
}


void CAdaptiveParameterUnBoundedValuesCalculator::setParameterValue(double targetValue)
{
	double functionValue = 0.0;
	double functionArgument = targetOffset + targetValue * targetScale;


	switch (functionKind)
	{
	case LINEAR:
		{
			functionValue = functionArgument;
			break;
		}
	case SQUARE:
		{
			functionValue = pow(functionArgument, 2);
			break;
		}
	case LOG:
		{
			assert(functionArgument >= 0);
			functionValue = log(functionArgument + 1.0);
			break;
		}
	case FRACT:
		{
			assert(functionArgument > 0);
			functionValue = (1.0 / (functionArgument + 1.0));
			break;
		}
	case FRACTSQUARE:
		{
			assert(functionArgument > 0);
			functionValue = 1.0 / (pow(functionArgument, 2) + 1.0);
			break;
		}
	case FRACTLOG :
		{
			assert(functionArgument >= 0);

			functionValue = 1.0 / (1.0 + log(functionArgument + 1.0));
			break;
		}
	case EXP : 
		{
			functionValue = exp(functionArgument);
			break;
		}
	default:
		{
			functionValue = functionArgument;
			printf("Unknown Adaptive Parameter Function Value %d, linear used instead\n", functionKind);
		}
	}

	double paramValue = paramOffset + paramScale * functionValue;
	
//	if (paramScale < 0)
//	{
//		if (paramValue < paramLimit)
//		{
//			paramValue = paramLimit;
//		}
//	}
//	else
//	{
//		if (paramValue > paramLimit)
//		{
//			paramValue = paramLimit;
//		}
//	}
	
	//printf("Setting parameter to %f %f %f %f\n", paramValue, functionValue, functionArgument, targetValue);
	CAdaptiveParameterCalculator::setParameterValue(paramValue);
}
