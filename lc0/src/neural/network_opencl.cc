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

#include "CL/OpenCLUtils.h"

#include "CL/OpenCLParams.h"
#include "CL/OpenCL.h"


#include "utils/blas.h"


namespace lczero {
  
    class OpenCLNetwork;

    class OpenCLComputation : public NetworkComputation {
      
    public:
      
      OpenCLComputation(const OpenCLNetwork& opencl_net):
      opencl_net_(opencl_net),
      policy_data_(),
      q_value_(0) {
        
      }
      
      virtual ~OpenCLComputation() {
        
      }
      
      // Adds a sample to the batch.
      void AddInput(InputPlanes&& input) override {
        planes_.emplace_back(input);
      }

      
    public:
      
     // Do the computation.
      void ComputeBlocking() override {
        for (auto& sample : planes_) {
          auto value; auto policy;
          std::tie(value, policy) = opencl_net_.ComputeBlocking(sample);
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
      
      const OpenCLNetwork& opencl_net_;
      
      std::vector<InputPlanes> planes_;
      std::vector<std::vector<float>> policy_data_;
      std::vector<float> q_value_;
      
    };
    
    class OpenCLNetwork : public Network {
    public:
      
      virtual ~OpenCLNetwork(){};

      OpenCLNetwork(const Weights& w, const OptionsDict& options):
      params_(),
      opencl_(),
      opencl_net_(opencl_)
      ip2_val_w_(w.ip2_val_w),
      ip2_val_b_(w.ip2_val_b),
      policy_data_size_(weights.ip_pol_b.size()),
      value_data_size_(weights.ip1_val_b.size())
      {
        Weights weights = w; // temporary local copy, to be released when ctor complete

        params_.gpuId=options.GetOrDefault<int>("gpu", -1);
        params_.verbose=options.GetOrDefault<bool>("verbose", false);
        params_.force_tune=options.GetOrDefault<int>("force_tune", false);
        params_.tune_only=options.GetOrDefault<int>("tune_only", false);
        params_.tune_exhaustive=options.GetOrDefault<int>("tune_exhaustive", false);
        

        const size_t channels = weights.input.biases.size();
        const size_t residual_blocks = weights.residual.size();
        
        
        opencl_.initialize(channels, params_);
        
        auto tuners = opencl_.get_sgemm_tuners();
        
        auto mwg = tuners[0];
        auto kwg = tuners[2];
        auto vwm = tuners[3];
        
        
        size_t m_ceil = ceilMultiple(ceilMultiple(channels, mwg), vwm);
        size_t k_ceil = ceilMultiple(ceilMultiple(kInputPlanes, kwg), vwm);
        
        // first, process the input block
        weights.input.weights = Transforms::winograd_transform_f(weights.input.weights, channels, inputChannels);
        auto Upad = Transforms::zeropad_U(weights.input.weights,
                                          channels, kInputPlanes,
                                          m_ceil, k_ceil);
        OffsetBatchNormMeans(input_batchnorm_means, weights.input.biases);
        InvertBatchNormStddev(weights.input.bn_stddivs);
        
        // Winograd filter transformation changes filter size to 4x4
        opencl_net_.push_input_convolution(WINOGRAD_ALPHA, kInputPlanes, channels,
                                           Upad, weights.input.bn_means, weights.input.bn_stddivs);
        
        // residual blocks
        for (auto& resblock : weights.residual) {
          
          auto& conv1 = resblock.conv1;
          auto& conv2 = resblock.conv2;
          
          conv1.weights = Transforms::winograd_transform_f(conv1.weights, channels, channels);
          conv2.weights = Transforms::winograd_transform_f(conv2.weights, channels, channels);
          
          auto Upad1 = Transforms::zeropad_U(conv1.weights,
                                             channels, channels,
                                             m_ceil, m_ceil);
          auto Upad2 = Transforms::zeropad_U(conv2.weights,
                                             channels, channels,
                                             m_ceil, m_ceil);
          
          OffsetBatchNormMeans(conv1.bn_means, conv1.biases);
          OffsetBatchNormMeans(conv2.bn_means, conv2.biases);

          InvertBatchNormStddev(conv1.bn_stddivs);
          InvertBatchNormStddev(conv2.bn_stddivs);
          
          opencl_net_.push_residual(WINOGRAD_ALPHA, channels, channels,
                                    Upad1, conv1.bn_means, conv1.bn_stddivs,
                                    Upad2, conv2.bn_means, conv2.bn_stddivs);
        }
        
        constexpr unsigned int width = 8;
        constexpr unsigned int height = 8;
        
        // policy head
        const auto num_p_inputs  = weights.policy.bn_means.size(); // NUM_POLICY_INPUT_PLANES
        OffsetBatchNormMeans(weights.policy.bn_means, weights.policy.biases);
        InvertBatchNormStddev(weights.policy.bn_stddivs);

        opencl_net_.push_policy(channels, num_p_inputs,
                                num_p_inputs*width*height,
                                weights.ip1_pol_b.size(),
                                weights.policy.weights,
                                weights.policy.bn_means,
                                weights.policy.bn_stddivs,
                                weights.ip_pol_w, weights.ip_pol_b);
        

        // value head
        const auto num_v_inputs  = weights.value.bn_means.size();  // NUM_VALUE_INPUT_PLANES
        OffsetBatchNormMeans(weights.value.bn_means, weights.value.biases);
        InvertBatchNormStddev(weights.value.bn_stddivs);

        opencl_net_.push_value(channels, num_v_inputs,
                               num_v_inputs*width*height,
                               weights.ip1_val_b.size(),
                               weights.value.weights,
                               weights.value.bn_means,
                               weights.value.bn_stddivs,
                               weights.ip1_val_w, weights.ip1_val_b);
      }
      
      std::unique_ptr<NetworkComputation> NewComputation() override {
        return std::make_unique<OpenCLComputation>(opencl_net_);
      }
      
      std::pair<float, std::vector<float>> OpenCLNetwork::ComputeBlocking(const InputPlanes &sample) const {
        std::vector<float> input_data(kInputPlanes*64);
        int index=0;
        for (const InputPlane& plane : sample) {
          float value=plane.value;
          const uint64_t one=1;
          for (int i=0; i<64; i++)
            input_data[index++]=((plane.mask&(one<<i))==0 ) ? 0 : value;
        }
        
        std::vector<float> policy_data(policy_data_size_);
        std::vector<float> value_data(value_data_size_);

        opencl_net_.forward(input_data, policy_data, value_data);
        
        // Get the moves
        Transforms::softmax(policy_data, policy_data);
        
        // Now get the score
        double winrate = Transforms::innerproduct(ip2_val_w_, value_data)+ip2_val_b_[0];
        return std::pair<float, std::vector<float>>(value, policy);        
      }
      
    private:

      OpenCLParams params_;
      OpenCL opencl_;
      OpenCL_Network opencl_net_;
      std::vector<float> ip2_val_w_;
      std::vector<float> ip2_val_b_;
      size_t policy_data_size_, value_data_size_;
      
    };
      
    
    
  } // namespace
  
  REGISTER_NETWORK("opencl", OpenCLNetwork, 100)
  
  
} // namespace lc0

