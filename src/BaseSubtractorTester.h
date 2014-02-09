#pragma once
#include <vector>
#include "BaseSubtractor.h"

class BaseSubtractorTester
{
	public:
		/// Default constructor
		BaseSubtractorTester();

		/// Method to test results of the algithm
		/// @param out_precision_array is an array of precision values
		/// @param out_recall_array is an array of recall array
		virtual void Test(std::vector<float> &out_precision_array,
			 	std::vector<float> &out_recall_array) = 0;

		/// Method to set the subtractor for testing
		/// @param in_subtractor is the subtractor for testing
		void SetSubtractor(BaseSubtractor *in_subtractor);

	protected:
		/// Background subtractor
		BaseSubtractor *m_subtractor;
};
