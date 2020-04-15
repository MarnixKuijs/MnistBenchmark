#include "NeuralNetwork.h"
#include <gsl/span>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdio>
#include <charconv>
#include <numeric>
#include <chrono>

int main()
{
	auto neuralNetwork = new NeuralNetwork<784, 100, 10>{0.3f};
	std::ifstream trainingData{ DATA_LOCATION"/MnistTrain.csv" };
	std::vector<std::array<float, 10>> trainingTargets{};
	std::vector<std::array<float, 784>> trainingImages{};

	for (std::string line; std::getline(trainingData, line);)
	{
		std::stringstream data{ line };

		//Get target 
		size_t label = static_cast<size_t>(data.get()) - 48;

		std::array<float, 10> target{};
		std::fill(std::begin(target), std::end(target), 0.01f);
		target[label] = 0.99f;

		trainingTargets.push_back(target);

		//Throw away comma
		data.get();

		size_t i{};
		std::array<float, 784> pixelValues;
		for (std::string number; std::getline(data, number, ',');)
		{
			
			std::from_chars(number.c_str(), number.c_str() + number.length(), pixelValues[i]);
			pixelValues[i] = pixelValues[i] / 255.0f * 0.99f + 0.01f;
			++i;
		}

		trainingImages.push_back(pixelValues);
	}

	puts("Start training");
	std::chrono::high_resolution_clock::time_point trainingBegin = std::chrono::high_resolution_clock::now();
	for (size_t i{}; i < trainingTargets.size(); ++i)
	{
		Train(*neuralNetwork, { trainingImages[i] }, { trainingTargets[i] });
	}
	std::chrono::high_resolution_clock::time_point trainingEnd = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> learningDuration = std::chrono::duration_cast<std::chrono::duration<double>>(trainingEnd - trainingBegin);
	printf("Training done in: %lf seconds \n", learningDuration.count());

	std::ifstream testData{ DATA_LOCATION"/MnistTest.csv" };
	std::vector<size_t> correctLables{};
	std::vector<std::array<float, 784>> testImages{};

	for (std::string line; std::getline(testData, line);)
	{
		std::stringstream data{ line };

		correctLables.push_back(static_cast<size_t>(data.get()) - 48);

		//Throw away comma
		data.get();

		size_t i{};
		std::array<float, 784> pixelValues;
		for (std::string number; std::getline(data, number, ',');)
		{

			std::from_chars(number.c_str(), number.c_str() + number.length(), pixelValues[i]);
			pixelValues[i] = pixelValues[i] / 255.0f * 0.99f + 0.01f;
			++i;
		}
		testImages.push_back(pixelValues);
	}


	puts("Testing Starts");

	std::vector<uint32_t> scores{};
	std::chrono::high_resolution_clock::time_point testingBegin = std::chrono::high_resolution_clock::now();
	for (size_t i{}; i < correctLables.size(); ++i)
	{
		std::array<float, 10> result = Query(*neuralNetwork, { testImages[i] });

		size_t label = static_cast<size_t>(std::max_element(std::begin(result), std::end(result)) - std::begin(result));

		if (label == correctLables[i])
		{
			scores.push_back(1);
		}
		else
		{
			scores.push_back(0);
		}
	}
	std::chrono::high_resolution_clock::time_point testingEnd = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> testingDuration = std::chrono::duration_cast<std::chrono::duration<double>>(testingEnd - testingBegin);
	printf("Testing done in: %lf seconds \n", testingDuration.count());

	uint32_t value{};

	for (auto score : scores)
	{
		value += score;
	}

	printf("Accuracy: %f%% \n", static_cast<float>(value) / static_cast<float>(scores.size()) * 100.0f);

	return 0;
}
