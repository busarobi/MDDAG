//
// C++ Interface: cinputdata
//
// Description: 
//
//
// Author: Neumann Gerhard <gerhard@tu-graz.ac.at>, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//

#ifndef C_INPUTDATASET__H
#define C_INPUTDATASET__H

#include <newmat/newmat.h>
#include <list>
#include <vector>
#include <set>
#include <stdio.h>

class CDataSet;

class CDataPreprocessor 
{
	protected:
		
	public:
		virtual ~CDataPreprocessor() {};

		virtual void preprocessInput(ColumnVector *input, ColumnVector *preInput) = 0;
	

		void preprocessDataSet(CDataSet *dataSet);
		
};

class CMeanStdPreprocessor : public CDataPreprocessor
{
	protected:
		ColumnVector *mean;
		ColumnVector *std;

	public:
		CMeanStdPreprocessor(ColumnVector *mean, ColumnVector *std);
		CMeanStdPreprocessor(CDataSet *dataSet);

		virtual ~CMeanStdPreprocessor();

		virtual void preprocessInput(ColumnVector *input, ColumnVector *preInput);

		void setMean(ColumnVector *mean);
		void setStd(ColumnVector *std);
};


template <typename OutputValue> class CMapping
{
	protected:
		CDataPreprocessor *preprocessor;
		ColumnVector *buffVector;

		virtual OutputValue doGetOutputValue(ColumnVector *vector) = 0;

		int numDim;

	public:

		CMapping(int numDim);

		virtual ~CMapping();
		
		
		virtual OutputValue getOutputValue(ColumnVector *vector);	

		virtual void saveASCII(FILE *) {};

		void setPreprocessor(CDataPreprocessor *preprocessor);
		CDataPreprocessor *getPreprocessor() {return preprocessor;};

		int getNumDimensions() {return numDim;};

		ColumnVector *getPreprocessedInput(ColumnVector *input);
};

template <typename OutputValue> CMapping<OutputValue>::CMapping(int l_numDim)
{
	numDim = l_numDim;
	buffVector = new ColumnVector(numDim);
	preprocessor = NULL;
}

template <typename OutputValue> CMapping<OutputValue>::~CMapping()
{
	delete buffVector;
}


template <typename OutputValue> ColumnVector * CMapping<OutputValue>::getPreprocessedInput(ColumnVector *input)
{
	if (preprocessor)
	{
		preprocessor->preprocessInput(input, buffVector);
	}
	else
	{
		*buffVector = *input;
	}
	return buffVector;
}


template <typename OutputValue> OutputValue CMapping<OutputValue>::getOutputValue(ColumnVector *vector)
{
	return doGetOutputValue( getPreprocessedInput( vector));
}

template <typename OutputValue> void CMapping<OutputValue>::setPreprocessor(CDataPreprocessor *l_preprocessor)
{
	preprocessor = l_preprocessor; 
}

class  DataSubset : public std::set<int>
{
public:
	DataSubset() {};
	virtual ~DataSubset(){};

	void addElements(std::list<int> *subsetList);
};

class  CDataSet : public std::vector<ColumnVector *>
{
	protected: 
		int numDimensions;

		ColumnVector *buffVector1;
		ColumnVector *buffVector2;
	public:
		CDataSet(int numDimensions);
		CDataSet(CDataSet &dataset);
		virtual ~CDataSet();
		
		int getNumDimensions();
		virtual void addInput(ColumnVector *input);
		
		void saveCSV(FILE *stream);
		void loadCSV(FILE *stream);
		
		virtual void getSubSet(DataSubset *subSet, CDataSet *newSet);

		virtual void clear();

		double getVarianceNorm(DataSubset *dataSubset);
		void getVariance(DataSubset *dataSubset, ColumnVector *variance);
		
		void getMean(DataSubset *dataSubset,  ColumnVector *mean);
};


class CDataSet1D : public std::vector<double>
{
	public:
		CDataSet1D(CDataSet1D &dataset);
		CDataSet1D();

		void loadCSV(FILE *stream);
		void saveCSV(FILE *stream);

		double getVariance(DataSubset *dataSubset, CDataSet1D *weight = NULL);
		double getMean(DataSubset *dataSubset, CDataSet1D *weighting = NULL);
};




#endif
