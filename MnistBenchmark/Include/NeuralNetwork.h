#pragma once
#include <array>
#include <cstdint>
#include <random>
#include <cmath>
#include <gsl/span>

static std::default_random_engine defaultRandomEngine{ std::random_device{}() };

template<size_t NumInputNodes, size_t NumHiddenNodes, size_t NumOutputNodes>
struct NeuralNetwork
{
public:
	NeuralNetwork(float learningRate) : learningRate{ learningRate }
	{
		static std::normal_distribution<float> normalDistributionHidden{ 0.0, 1 / std::sqrtf(static_cast<float>(NumHiddenNodes)) };
		static std::normal_distribution<float> normalDistributionOutput{ 0.0, 1 / std::sqrtf(static_cast<float>(NumOutputNodes)) };

		for (auto& row : inputWeights)
		{
			for (auto& entry : row)
			{
				entry = normalDistributionHidden(defaultRandomEngine);
			}
		}

		for (auto& row : outputWeights)
		{
			for (auto& entry : row)
			{
				entry = normalDistributionOutput(defaultRandomEngine);
			}
		}
	}

	NeuralNetwork(const NeuralNetwork& other) = delete;
	NeuralNetwork& operator=(const NeuralNetwork& other) = delete;
	NeuralNetwork(NeuralNetwork&& other) = delete;
	NeuralNetwork& operator=(NeuralNetwork&& other) = delete;
	~NeuralNetwork() = default;

	constexpr static size_t numInputNodes = NumInputNodes;
	constexpr static size_t numHiddenNodes = NumHiddenNodes;
	constexpr static size_t numOutputNodes = NumOutputNodes;

	float learningRate;
	std::array<std::array<float, NumInputNodes>, NumHiddenNodes> inputWeights;
	std::array<std::array<float, NumHiddenNodes>, NumOutputNodes> outputWeights;
};

template<size_t NumInputNodes, size_t NumHiddenNodes, size_t NumOutputNodes>
auto Query(const NeuralNetwork<NumInputNodes, NumHiddenNodes, NumOutputNodes>& neuralNetwork, 
	gsl::span<float, NumInputNodes> input) -> std::array<float, NumOutputNodes>
{
	std::array<float, NumHiddenNodes> hiddenValues{};

	for (size_t i{}; i < NumHiddenNodes; ++i)
	{
		for (size_t j{}; j < NumInputNodes; ++j)
		{
			hiddenValues[i] += input[j] * neuralNetwork.inputWeights[i][j];
		}
	}

	for (auto& hiddenValue : hiddenValues)
	{
		hiddenValue = 1 / (1 + std::exp(-hiddenValue));
	}

	std::array<float, NumOutputNodes> outputValues{};

	for (size_t i{}; i < NumOutputNodes; ++i)
	{
		for (size_t j{}; j < NumHiddenNodes; ++j)
		{
			outputValues[i] += hiddenValues[j] * neuralNetwork.outputWeights[i][j];
		}
	}

	for (auto& outputValue : outputValues)
	{
		outputValue = 1 / (1 + std::exp(-outputValue));
	}

	return outputValues;
}

template<size_t NumInputNodes, size_t NumHiddenNodes, size_t NumOutputNodes>
void Train(NeuralNetwork<NumInputNodes, NumHiddenNodes, NumOutputNodes>& neuralNetwork,
	gsl::span<float, NumInputNodes> input,
	gsl::span<float, NumOutputNodes> target)
{
	std::array<float, NumHiddenNodes> hiddenValues{};

	for (size_t i{}; i < NumHiddenNodes; ++i)
	{
		for (size_t j{}; j < NumInputNodes; ++j)
		{
			hiddenValues[i] += input[j] * neuralNetwork.inputWeights[i][j];
		}
	}

	for (auto& hiddenValue : hiddenValues)
	{
		hiddenValue = 1 / (1 + std::exp(-hiddenValue));
	}

	std::array<float, NumOutputNodes> outputValues{};

	for (size_t i{}; i < NumOutputNodes; ++i)
	{
		for (size_t j{}; j < NumHiddenNodes; ++j)
		{
			outputValues[i] += hiddenValues[j] * neuralNetwork.outputWeights[i][j];
		}
	}

	for (auto& outputValue : outputValues)
	{
		outputValue = 1 / (1 + std::exp(-outputValue));
	}

	std::array<float, NumOutputNodes> outputErrorValues{};
	for (size_t i{}; i < NumOutputNodes; ++i)
	{
		outputErrorValues[i] = target[i] - outputValues[i];
	}

	std::array<float, NumHiddenNodes> hiddenErrorValues{};
	for (size_t i{}; i < NumHiddenNodes; ++i)
	{
		for (size_t j{}; j < NumOutputNodes; ++j)
		{
			hiddenErrorValues[i] += neuralNetwork.outputWeights[j][i] * outputErrorValues[j];
		}
	}

	for (size_t i{}; i < NumOutputNodes; ++i)
	{
		for (size_t j{}; j < NumHiddenNodes; ++j)
		{
			neuralNetwork.outputWeights[i][j] += neuralNetwork.learningRate * (outputErrorValues[i] * outputValues[i] * (1 - outputValues[i]) * hiddenValues[j]);
		}
	}

	for (size_t i{}; i < NumHiddenNodes; ++i)
	{
		for (size_t j{}; j < NumInputNodes; ++j)
		{
			neuralNetwork.inputWeights[i][j] += neuralNetwork.learningRate * (hiddenErrorValues[i] * hiddenValues[i] * (1 - hiddenValues[i]) * input[j]);
		}
	}
}