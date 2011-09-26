//
// C++ Interface: cdatafactory
//
// Description: 
//
//
// Author: Neumann Gerhard <gerhard@tu-graz.ac.at>, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//

#ifndef CDATAFACTORY__H
#define CDATAFACTORY__H

#include "ctrees.h"
#include <newmat/newmat.h>

class CDataSet1D;
class CDataSet;

class CRegressionFactory : public CTreeDataFactory<double>
{
	protected:
		CDataSet1D *outputData;
		CDataSet1D *weightData;
	public:
		CRegressionFactory(CDataSet1D *outputData, CDataSet1D *weightData);
		virtual ~CRegressionFactory();
		
		virtual double createTreeData(DataSubset *dataSubset, int numLeaves);
};

class CSubsetFactory : public CTreeDataFactory<DataSubset *>
{
	protected:
		

	public:
		virtual ~CSubsetFactory() {};

		virtual DataSubset *createTreeData(DataSubset *dataSubset, int numLeaves) ;
		virtual void deleteData(DataSubset *dataSet);
};
  

class CVectorQuantizationFactory : public CTreeDataFactory<ColumnVector *>
{
	protected:
		CDataSet *inputData;
	public:
		CVectorQuantizationFactory(CDataSet *inputData);
		virtual ~CVectorQuantizationFactory();
				
		
		virtual ColumnVector *createTreeData(DataSubset *dataSubset, int numLeaves) ;
		virtual void deleteData(ColumnVector *dataSet);
};

class CLeafIndexFactory : public  CTreeDataFactory<int>
{
	virtual int createTreeData(DataSubset *dataSubset, int numLeaves);
};

#endif
