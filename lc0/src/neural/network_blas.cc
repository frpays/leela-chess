/*
 This file is part of Leela Chess Zero.
 Copyright (C) 2018 The LCZero Authors
 
 Leela Chess is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 Leela Chess is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "neural/factory.h"
#include "neural/network.h"
#include "neural/transforms.h"

#include <condition_variable>
#include <cassert>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <thread>

#include "utils/exception.h"
#include "utils/blas.h"

namespace lczero {

    class BlasNetwork;
    
    
    
    class BlasComputation : public NetworkComputation {
      
    public:
      
      BlasComputation(const BlasNetwork& network):
      network_(network),
      policy_data_(),
      q_value_(0) {
        
      }
      
      virtual ~BlasComputation() {
        
      }
      
      // Adds a sample to the batch.
      void AddInput(InputPlanes&& input) override {
        planes_.emplace_back(input);
      }

            
     // Do the computation.
      void ComputeBlocking() override {
        
        for (auto& sample : planes_) {
          auto value; auto policy;
          std::tie(value, policy) = network.ComputeBlocking(sample);
          q_value_.emplace_back(value);
          policy_data_.emplace_back(policy);
        }

      }
      
      // Returns how many times AddInput() was called.
      int GetBatchSize() const override {
         return planes_.size();
      }
      
      // Returns Q value of @sample.
      float GetQVal(int sample) const override {
        return q_value_[sample];
      }
      
      // Returns P value @move_id of @sample.
      float GetPVal(int sample, int move_id) const override {
        return policy_data_[sample][move_id];
      }
      
      
    private:
  
      const BlasNetwork& network_;
      
      std::vector<InputPlanes> planes_;
      std::vector<std::vector<float>> policy_data_;
      std::vector<float> q_value_;
      
    };
    
    class BlasNetwork : public Network {
    public:
      
      virtual ~BlasNetwork(){};

      
      BlasNetwork(const Weights& weights, const OptionsDict& /* options */):
      weights_(weights)
      {
        constexpr float EPSILON=1e-5;

        const size_t channels = weights.input.biases.size();
        const size_t residual_blocks = weights.residual.size();
        
        // input block
        weights_.input.weights = Transforms::winograd_transform_f(weights_.input.weights, channels, kInputPlanes);
        OffsetBatchNormMeans(weights_.input.bn_means, weights_.input.biases);
        InvertBatchNormStddev(weights_.input.bn_stddivs);
        
        // residual blocks
        for (auto& resblock : weights_.residual) {
          auto& conv1 = resblock.conv1;
          auto& conv2 = resblock.conv2;
          
          conv1.weights = Transforms::winograd_transform_f(conv1.weights, channels, channels);
          conv2.weights = Transforms::winograd_transform_f(conv2.weights, channels, channels);
          
          OffsetBatchNormMeans(conv1.bn_means, conv1.biases);
          OffsetBatchNormMeans(conv2.bn_means, conv2.biases);

          InvertBatchNormStddev(conv1.bn_stddivs);
          InvertBatchNormStddev(conv2.bn_stddivs);
        }
        
        OffsetBatchNormMeans(weights_.policy.bn_means, weights_.policy.biases);
        InvertBatchNormStddev(weights_.policy.bn_stddivs);
        
        OffsetBatchNormMeans(weights_.value.bn_means, weights_.value.biases);
        InvertBatchNormStddev(weights_.value.bn_stddivs);

#ifdef USE_OPENBLAS
        //openblas_set_num_threads(1);
        //printf("BLAS Core: %s\n", openblas_get_corename());
#endif
        
#ifdef USE_MKL
        //mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);
        mkl_set_num_threads(1);
        MKLVersion Version;
        mkl_get_version(&Version);
        printf("BLAS core: MKL %s\n", Version.Processor);
#endif

      }

      void BlasNetwork::forward(const std::vector<float>& input,
                                std::vector<float>& output_pol,
                                std::vector<float>& output_val) {
        
        // Input convolution
        constexpr int width = 8;
        constexpr int height = 8;
        constexpr int tiles = width * height / 4;

        // Calculate output channels
        const auto output_channels = weights_.input.biases.size();
        //input_channels is the maximum number of input channels of any convolution.
        //Residual blocks are identical, but the first convolution might be bigger
        //when the network has very few filters
        const auto input_channels = std::max(
                                    static_cast<size_t>(output_channels),
                                    static_cast<size_t>(kInputPlanes));
        auto conv_out = std::vector<float>(output_channels * width * height);
        
        auto V = std::vector<float>(WINOGRAD_TILE * input_channels * tiles);
        auto M = std::vector<float>(WINOGRAD_TILE * output_channels * tiles);
        
        std::vector<float> policy_data(weights_.policy.bn_means.size() * width * height); // NUM_POLICY_INPUT_PLANES*w*h
        std::vector<float> value_data(weights_.value.bn_means.size() * width * height); // NUM_VALUE_INPUT_PLANES*w*h
        
        Transforms::winograd_convolve3(output_channels, input,  weights_.input.weights, V, M, conv_out);
        Transforms::batchnorm(output_channels, conv_out,
                              weights_.input.bn_means,
                              weights_.input.bn_stddivs);
        
        // Residual tower
        auto conv_in = std::vector<float>(output_channels * width * height);
        auto res = std::vector<float>(output_channels * width * height);
        
        for (auto &residual : weights_.residual) {
          auto& conv1 = residual.conv1;
          auto output_channels = conv1.biases.size(); // really confusing overload of variable names.... gcp pls...
          std::swap(conv_out, conv_in);
          std::copy(begin(conv_in), end(conv_in), begin(res));
          
          Transforms::winograd_convolve3(output_channels, conv_in,
                                         conv1.weights, V, M, conv_out);
          Transforms::batchnorm(output_channels, conv_out,
                                conv1.bn_means,
                                conv1.bn_stddivs);
          
          auto& conv2 = residual.conv2;
          output_channels = conv2.biases.size();
          std::swap(conv_out, conv_in);
          Transforms::winograd_convolve3(output_channels, conv_in,
                                         conv2.weights, V, M, conv_out);
          Transforms::batchnorm<64>(output_channels, conv_out,
                                    conv2.bn_means,
                                    conv2.bn_stddivs,
                                    res.data());
        }

        Transforms::convolve(weights_.policy.bn_means.size(), conv_out, // NUM_POLICY_INPUT_PLANES
                             weights_.policy.weights, weights_.policy.biases, policy_data);
        Transforms::batchnorm(weights_.policy.bn_means.size(), policy_data, // NUM_POLICY_INPUT_PLANES
                              weights_.policy.bn_means, weights_.policy.bn_stddivs);
        Transforms::innerproduct(policy_data, weights_.ip_pol_w, weights_.ip_pol_b, output_pol);

        Transforms::convolve(weights_.value.bn_means.size(), conv_out, // NUM_VALUE_INPUT_PLANES
                             weights_.value.weights, weights_.value.biases, value_data);
        Transforms::batchnorm(weights_.value.bn_means.size(), value_data, // NUM_VALUE_INPUT_PLANES
                              weights_.value.bn_means, weights_.value.bn_stddivs);
        Transforms::innerproduct(value_data, weights_.ip1_val_w, weights_.ip1_val_b, output_val, true); // value head gets relu applied
      }

      std::pair<float, std::vector<float>> BlasNetwork::ComputeBlocking(const InputPlanes &sample) const {
        std::vector<float> input_data(kInputPlanes*64);
        int index=0;
        for (const InputPlane& plane : sample) {
          float value=plane.value;
          const uint64_t one=1;
          for (int i=0; i<64; i++)
            input_data_[index++]=((plane.mask&(one<<i))==0 ) ? 0 : value;
        }

        std::vector<float> policy_data(weights_.ip_pol_b.size());
        std::vector<float> value_data(weights_.ip1_val_b.size());

        forward(input_data, policy_data, value_data);
        
        // Get the moves
        Transforms::softmax(policy_data, policy_data);
        
        // Now get the score
        double winrate = Transforms::innerproduct(ip2_val_w_, value_data)+weights_.ip2_val_b_[0];
        return std::pair<float, std::vector<float>>(value, policy);
      }
    
      std::unique_ptr<NetworkComputation> NewComputation() override {
        return std::make_unique<BlasComputation>(weights_);
      }
      
      
    private:
      Weights weights_;
      
    };
      
    
    
  } // namespace
  
  REGISTER_NETWORK("blas", BlasNetwork, 50)
  
  
} // namespace lc0

