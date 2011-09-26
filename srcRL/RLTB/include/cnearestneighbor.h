#ifndef C_NEAREST_NEIGHBOR__H
#define C_NEAREST_NEIGHBOR__H

#include "ctrees.h"
#include "cinputdata.h"
#include "newmat/newmatio.h"

#include <iostream>
#include <limits>

class CKDRectangle 
{
	protected:
		ColumnVector *minValues;
		ColumnVector *maxValues;
	public:
		CKDRectangle(int numDim);
		CKDRectangle(CKDRectangle &rectangle);

		ColumnVector *getMinVector();
		ColumnVector *getMaxVector();

		virtual ~CKDRectangle();		

		void setMaxValue(int dim, double value);
		void setMinValue(int dim, double value);

		double getMinValue(int dim);
		double getMaxValue(int dim);

		double getDistanceToPoint(ColumnVector *point);

		bool intersects(CKDRectangle *rectangle);
};

template<typename DataElement, typename TreeData> class CKNearestNeighborsTreeData
{
	protected:
		double *distList;
		DataElement *elementList;

		int K;

		CTree<TreeData> *tree;

		void getNearestNeighborsElements(ColumnVector *point, CTreeElement<TreeData> *element, CKDRectangle *rectangle);

		
		virtual void addAndSortDataElements(DataElement element, double distance);
		virtual void addDataElements(ColumnVector *point, CLeaf<TreeData> *leaf, CKDRectangle *rectangle) = 0;
	public:
		CKNearestNeighborsTreeData(CTree<TreeData> *tree, int K);
		virtual ~CKNearestNeighborsTreeData();

		void getNearestNeighbors(ColumnVector *point,  std::list<DataElement> *elementList, int K = -1, ColumnVector *distances = NULL);
		
		void getNearestNeighborDistance(ColumnVector *input, DataElement &nearestNeighbor, double &distance);
};

template<typename DataElement, typename TreeData> void CKNearestNeighborsTreeData<DataElement, TreeData>::getNearestNeighborsElements(ColumnVector *point,CTreeElement<TreeData> *element, CKDRectangle *rectangle)
{
	if (element->isLeaf())
	{
		//printf("Adding Leaf with %f Distance\n", rectangle->getDistanceToPoint( point));
		CLeaf<TreeData> *leaf = dynamic_cast<CLeaf<TreeData> *>(element);
		addDataElements(point, leaf, rectangle);
	}
	else
	{
		CNode<TreeData> *node = dynamic_cast<CNode<TreeData> *>(element);	
		C1DSplittingCondition *split = dynamic_cast<C1DSplittingCondition *>(node->getSplittingCondition());

		double minValue = rectangle->getMinValue(split->getDimension());
		double maxValue = rectangle->getMaxValue(split->getDimension());

		double leftDist = 0;
		double rightDist = 0;
		// set value for left branch
		rectangle->setMaxValue(split->getDimension(), split->getTreshold());
		leftDist = rectangle->getDistanceToPoint(point);
		
		rectangle->setMaxValue(split->getDimension(), maxValue);
		rectangle->setMinValue(split->getDimension(), split->getTreshold());

		rightDist = rectangle->getDistanceToPoint(point);	
	
		//printf("LeftDist: %f, RightDist: %f...", leftDist, rightDist);	
		/*
		for (int i = 0; i < K; i ++)
		{
			if (distList[i] < numeric_limits<double>::max())
			{
				printf("%f ", distList[i]);
			}
		}
		printf("\n");*/

		if (rightDist < leftDist)
		{
			// point is nearer to the right branch
			if (distList[0] == numeric_limits<double>::max() || rightDist < distList[0])
			{
				// search NN in right branch
				//printf("Going Right...\n");
				getNearestNeighborsElements(point,node->getRightElement(), rectangle);

				if (distList[0] == numeric_limits<double>::max() || leftDist < distList[0])
				{
				//	printf("Going Left (2)...\n");
					// also search NN in left branch
					rectangle->setMaxValue(split->getDimension(),  split->getTreshold());
					rectangle->setMinValue(split->getDimension(), minValue);
			
					getNearestNeighborsElements(point,node->getLeftElement(), rectangle);
		
				}
			}
		}
		else
		{
			// point is nearer to the right branch
			if (distList[0] == numeric_limits<double>::max() || leftDist < distList[0])
			{
				//printf("Going Left...\n");
				rectangle->setMaxValue(split->getDimension(), split->getTreshold());
				rectangle->setMinValue(split->getDimension(), minValue);
				
				// search NN in left branch
				getNearestNeighborsElements(point,node->getLeftElement(), rectangle);
	
				if (distList[0] == numeric_limits<double>::max() || rightDist < distList[0])
				{
					//printf("Going Right (2)...\n");
					// also search NN in right branch
					rectangle->setMaxValue(split->getDimension(), maxValue);
					rectangle->setMinValue(split->getDimension(), split->getTreshold());
				
					getNearestNeighborsElements(point,node->getRightElement(), rectangle);
				}
			}
		}
		rectangle->setMaxValue(split->getDimension(), maxValue);
		rectangle->setMinValue(split->getDimension(), minValue); 
	}
}
	
template<typename DataElement, typename TreeData> void CKNearestNeighborsTreeData<DataElement, TreeData>::addAndSortDataElements(DataElement data, double distance)
{
	int i = 0;
	while (i < K && distList[i] > distance)
	{
		i ++;
	} 
	i --;	

	
	if (i >= 0)
	{
		for (int j = 0; j < i; j ++)
		{
			distList[j] = distList[j + 1];
			elementList[j] = elementList[j + 1];
		}
		distList[i] = distance;
		elementList[i] = data;
	}	

	/*printf("Adding Element with distance %f\n", distance);
	for (int i = 0; i < K; i ++)
	{
		if (distList[i] < numeric_limits<double>::max())
		{
			printf("%f ", distList[i]);
		}
	}
	printf("\n");*/
}


template<typename DataElement, typename TreeData> CKNearestNeighborsTreeData<DataElement, TreeData>::CKNearestNeighborsTreeData(CTree<TreeData> *l_tree, int l_K)
{
	tree = l_tree;
	K = l_K;
	distList = new double[K];
	elementList = new DataElement[K];
}

template<typename DataElement, typename TreeData> CKNearestNeighborsTreeData<DataElement, TreeData>::~CKNearestNeighborsTreeData()
{
	delete [] distList;
	delete [] elementList;
}

template<typename DataElement, typename TreeData> void CKNearestNeighborsTreeData<DataElement, TreeData>::getNearestNeighbors(ColumnVector *point, std::list<DataElement> *l_elementList, int l_K, ColumnVector *distances)
{
	int tempK = K;
	if (l_K > 0)
	{
		K = l_K;
	}

	for (int i = 0; i < K; i++)
	{
		distList[i] = numeric_limits<double>::max();
	}

	ColumnVector *input = tree->getPreprocessedInput(point);
	
	CKDRectangle rectangle(tree->getNumDimensions());
	
	//printf("Starting NN search: %f for point: ", rectangle.getDistanceToPoint(point));
	//cout << point->t() << endl;	
	//cout << input->t() << endl;		

	getNearestNeighborsElements(input, tree->getRoot(), &rectangle);
	
	l_elementList->clear();
	
	for (int i = 0; i < K; i ++)
	{
		if (distances)
		{
			distances->element( K - 1 - i) = distList[i];
		}
		if (distList[i] < numeric_limits<double>::max())
		{
			l_elementList->push_front(elementList[i]);
		}
	}

	K = tempK;	
}

template<typename DataElement, typename TreeData> void CKNearestNeighborsTreeData<DataElement, TreeData>::getNearestNeighborDistance(ColumnVector *point, DataElement &nearestNeighbor, double &distance)
{
	int tempK = K;
	K = 1;
	

	for (int i = 0; i < K; i++)
	{
		distList[i] = numeric_limits<double>::max();
	}

	ColumnVector *input = tree->getPreprocessedInput(point);
	
	CKDRectangle rectangle(tree->getNumDimensions());
	
	//printf("Starting NN search: %f for point: ", rectangle.getDistanceToPoint(point));
	//cout << point->t() << endl;	
	//cout << input->t() << endl;		

	getNearestNeighborsElements(input, tree->getRoot(), &rectangle);
	
	K = tempK;	

	nearestNeighbor = elementList[0];
	distance = distList[0];

}



class CKNearestNeighbors : public  CKNearestNeighborsTreeData<int, DataSubset *>
{
	protected:
		CDataSet *inputSet;	
		ColumnVector *buffVector;

		virtual void addDataElements(ColumnVector *point, CLeaf<DataSubset *> *leaf, CKDRectangle *rectangle);
		
	public:
		CKNearestNeighbors(CTree<DataSubset *> *tree, CDataSet *inputSet, int K);
		virtual ~CKNearestNeighbors();

		CDataSet *getInputSet() {return inputSet;};
};



template<typename TreeData> class CKNearestLeaves : public  CKNearestNeighborsTreeData<int, TreeData>
{
	protected:
		
		virtual void addDataElements(ColumnVector *point, CLeaf<TreeData> *leaf, CKDRectangle *rectangle);
		
	public:
		CKNearestLeaves(CTree<TreeData> *tree, int K);
		virtual ~CKNearestLeaves();
};



template<typename TreeData> void CKNearestLeaves<TreeData>::addDataElements(ColumnVector *point, CLeaf<TreeData> *leaf, CKDRectangle *rectangle) 
{
	addAndSortDataElements(leaf->getLeafNumber(), rectangle->getDistanceToPoint(point));
}
		

template<typename TreeData> CKNearestLeaves<TreeData>::CKNearestLeaves(CTree<TreeData> *tree, int K) : CKNearestNeighborsTreeData<int, TreeData>(tree, K)
{

}

template<typename TreeData> CKNearestLeaves<TreeData>::~CKNearestLeaves()
{
}

class CRangeSearch
{
	protected:
		DataSubset *elementList;

		CTree<DataSubset *> *tree;
		CDataSet *inputSet;

		void getSamplesInRangeElements( CKDRectangle *range, CTreeElement<DataSubset *> *element, CKDRectangle *rectangle);

		
		virtual void addAndSortDataElements(int element);
		virtual void addDataElements(CKDRectangle *range, CLeaf<DataSubset *> *leaf, CKDRectangle *leafRectangle);
	public:
		CRangeSearch(CTree<DataSubset *> *tree, CDataSet *l_inputSet);
		virtual ~CRangeSearch();

		void getSamplesInRange(CKDRectangle *range, DataSubset *elementList);
		
};



#endif

