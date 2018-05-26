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

#pragma once

#include <vector>
#include <array>

namespace lczero {

    static constexpr auto WINOGRAD_ALPHA = 4;
    static constexpr auto WINOGRAD_TILE = WINOGRAD_ALPHA * WINOGRAD_ALPHA;
  
  class Transforms {
    
  public:
    
    static std::vector<float> zeropad_U(const std::vector<float>& U,
                                        const int outputs, const int channels,
                                        const int outputs_pad,
                                        const int channels_pad);
    
    static std::vector<float> winograd_transform_f(const std::vector<float>& f,
                                            const int outputs,
                                            const int channels);
    
    static void winograd_transform_in(const std::vector<float>& in,
                               std::vector<float>& V,
                               const int C);
    
    static void winograd_sgemm(const std::vector<float>& U,
                        std::vector<float>& V,
                        std::vector<float>& M,
                        const int C, const int K);
    
    static void winograd_transform_out(const std::vector<float>& M,
                                std::vector<float>& Y,
                                const int K);
    
    static void winograd_convolve3(const int outputs,
                            const std::vector<float>& input,
                            const std::vector<float>& U,
                            std::vector<float>& V,
                            std::vector<float>& M,
                            std::vector<float>& output);
    
    static void convolve(size_t outputs,
                  const std::vector<float>& input,
                  const std::vector<float>& weights,
                  const std::vector<float>& biases,
                  std::vector<float>& output);
    
    static void innerproduct(const std::vector<float>& input,
                             const std::vector<float>& weights,
                             const std::vector<float>& biases,
                             std::vector<float>& output,
                             bool apply_relu=false);
    
    static void batchnorm(size_t channels,
                   std::vector<float>& data,
                   const std::vector<float>& means,
                   const std::vector<float>& stddivs,
                   const float* eltwise = nullptr);
    
    static void softmax(const std::vector<float>& input,
                        std::vector<float>& output);
    
    static float innerproduct(const std::vector<float>& x, const std::vector<float>& y);

    static void OffsetBatchNormMeans(std::vector<float>& bn_means, const std::vector<float>& biases);

    static void InvertBatchNormStddev(std::vector<float>& weights);


  };
  
} // lczero



