// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include <string>
#include <type_traits>

#include "ck_tile/ops/fmha/print_utils.hpp"

// S[seqlen_q, seqlen_k] = Q[seqlen_q, hdim_q] @ K[seqlen_k, hdim_q]
// S'[seqlen_q, seqlen_k] = S[seqlen_q, seqlen_k] * Scale[1]
// S''[seqlen_q, seqlen_k] = S'[seqlen_q, seqlen_k] + Bias[seqlen_q, seqlen_k]
// P[seqlen_q, seqlen_k] = Softmax(S''[seqlen_q, seqlen_k])
// O[seqlen_q, hdim_v] = P[seqlen_q, seqlen_k] @ V^T[hdim_v, seqlen_k]

namespace ck_tile {

template <typename TilePartitioner_, typename FmhaPipeline_, typename EpiloguePipeline_>
struct FmhaFwdSplitKVKernel
{
    using TilePartitioner                         = ck_tile::remove_cvref_t<TilePartitioner_>;
    using FmhaPipeline                            = ck_tile::remove_cvref_t<FmhaPipeline_>;
    using EpiloguePipeline                        = ck_tile::remove_cvref_t<EpiloguePipeline_>;
    static constexpr ck_tile::index_t kBlockSize  = FmhaPipeline::kBlockSize;
    static constexpr ck_tile::index_t kBlockPerCu = FmhaPipeline::kBlockPerCu;
    static_assert(kBlockPerCu > 0);
    static constexpr ck_tile::index_t kBlockPerCuInput = FmhaPipeline::Problem::kBlockPerCu;

    using QDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::QDataType>;
    using KDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::KDataType>;
    using VDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::VDataType>;
    using BiasDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::BiasDataType>;
    using LSEDataType  = ck_tile::remove_cvref_t<typename FmhaPipeline::LSEDataType>;
    using SaccDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::SaccDataType>;
    using OaccDataType = remove_cvref_t<typename FmhaPipeline::OaccDataType>;

    using VLayout = ck_tile::remove_cvref_t<typename FmhaPipeline::VLayout>;

    static constexpr bool kIsGroupMode      = FmhaPipeline::kIsGroupMode;
    static constexpr bool kPadSeqLenQ       = FmhaPipeline::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK       = FmhaPipeline::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ      = FmhaPipeline::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV      = FmhaPipeline::kPadHeadDimV;
    static constexpr auto BiasEnum          = FmhaPipeline::BiasEnum;
    static constexpr bool kDoFp8StaticQuant = FmhaPipeline::Problem::kDoFp8StaticQuant;
    static constexpr bool kIsPagedKV        = FmhaPipeline::Problem::kIsPagedKV;
    static constexpr bool kXQA_enabled      = FmhaPipeline::Problem::kXQA_enabled;
    static constexpr bool kXQA_ready        = FmhaPipeline::Problem::kXQA_ready;
    static_assert(!kIsGroupMode || (kIsGroupMode && !kIsPagedKV),
                  "paged-kvcache only supported by batch mode kernels");
    using FmhaMask                 = ck_tile::remove_cvref_t<typename FmhaPipeline::FmhaMask>;
    static constexpr bool kHasMask = FmhaMask::IsMasking;

    // clang-format off
    template <typename T> struct t2s;
    template <> struct t2s<float> { static constexpr const char * name = "fp32"; };
    template <> struct t2s<ck_tile::fp16_t> { static constexpr const char * name = "fp16"; };
    template <> struct t2s<ck_tile::bf16_t> { static constexpr const char * name = "bf16"; };
    template <> struct t2s<ck_tile::fp8_t> { static constexpr const char * name = "fp8"; };
    template <> struct t2s<ck_tile::bf8_t> { static constexpr const char * name = "bf8"; };
    // clang-format on

    __host__ static std::string GetName()
    {
        // sync with generate.py
        // clang-format off
        using bfs = typename FmhaPipeline::BlockFmhaShape;
        using gbr = typename bfs::Gemm0BlockWarps;
        using gwt = typename bfs::Gemm0WarpTile;
        #define _SS_  std::string
        #define _TS_  std::to_string
        auto pn = [&] () {
            std::string n;
            if (kPadSeqLenQ) n += "s";
            if (kPadSeqLenK) n += "sk";
            if (kPadHeadDimQ) n += "d";
            if (kPadHeadDimV) n += "dv";
            return n.empty() ? n : std::string("p") + n; }();
        return
            _SS_("fmha_fwd_splitkv_d") + _TS_(bfs::kK0BlockLength) + "_" + _SS_(t2s<QDataType>::name) +
            "_" + (kIsGroupMode ? "group" : "batch") + "_"
            "b" + _TS_(bfs::kM0) + "x" + _TS_(bfs::kN0) + "x" + _TS_(bfs::kK0) + "x" +
                    _TS_(bfs::kN1) + "x" + _TS_(bfs::kK1) + "x" + _TS_(bfs::kK0BlockLength) + "_" +
            "r" + _TS_(gbr::at(ck_tile::number<0>{})) + "x" + _TS_(gbr::at(ck_tile::number<1>{})) + "x" + _TS_(gbr::at(ck_tile::number<2>{})) + "_" +
            "w" + _TS_(gwt::at(ck_tile::number<0>{})) + "x" + _TS_(gwt::at(ck_tile::number<1>{})) + "x" + _TS_(gwt::at(ck_tile::number<2>{})) + "_" +
            (kBlockPerCuInput == -1 ? "" : ("o" + _TS_(kBlockPerCu) + "_")) + _SS_(FmhaPipeline::name) + "_" +
            "v" + (std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor> ? "r" : "c") + (pn.empty() ? "" : "_" + pn) +
            (BiasEnum == BlockAttentionBiasEnum::NO_BIAS ? _SS_("") : (_SS_("_") + BlockAttentionBiasEnumToStr<BiasEnum>::name)) + 
            (kHasMask ? "_" + _SS_(FmhaMask::name) : "") + (kDoFp8StaticQuant ? "_squant" : "") + (kIsPagedKV ? "_pagedkv" : "" );
        #undef _SS_
        #undef _TS_
        // clang-format on
    }

    template <ck_tile::index_t I> // to avoid duplicated base class prblem, introduce an template
                                  // arg
    struct EmptyKargs
    {
    };

    // kargs use aggregate initializer, so no constructor will provided
    // use inheritance to minimize karg size
    // user need to use MakeKargs() function to create kargs.
    struct CommonKargs
    {
        const void* q_ptr;
        const void* k_ptr;
        const void* v_ptr;
        void* lse_acc_ptr;
        void* o_acc_ptr;

        ck_tile::index_t batch;

        ck_tile::index_t seqlen_q;
        ck_tile::index_t seqlen_k;
        ck_tile::index_t hdim_q;
        ck_tile::index_t hdim_v;

        ck_tile::index_t num_head_q;
        ck_tile::index_t nhead_k;
        // for MQA/GQA, nhead could be different. This parameter is nhead_q / nhead_k
        // if this param is larger than 1, indicate MQA/GQA case
        ck_tile::index_t nhead_ratio_qk;
        ck_tile::index_t num_splits;

        float scale_s;

        ck_tile::index_t stride_q;
        ck_tile::index_t stride_k;
        ck_tile::index_t stride_v;
        ck_tile::index_t stride_o_acc;

        ck_tile::index_t nhead_stride_q;
        ck_tile::index_t nhead_stride_k;
        ck_tile::index_t nhead_stride_v;
        ck_tile::index_t nhead_stride_lse_acc;
        ck_tile::index_t nhead_stride_o_acc;

        ck_tile::index_t batch_stride_lse_acc;
        ck_tile::index_t batch_stride_o_acc;

        ck_tile::index_t split_stride_lse_acc;
        ck_tile::index_t split_stride_o_acc;
    };

    struct CommonBiasKargs
    {
        const void* bias_ptr               = nullptr;
        ck_tile::index_t stride_bias       = 0;
        ck_tile::index_t nhead_stride_bias = 0;
    };

    struct BatchModeBiasKargs : CommonBiasKargs
    {
        ck_tile::index_t batch_stride_bias = 0;
    };

    struct AlibiKargs
    {
        // alibi is batch*nhead*1, no matter in batch/group mode, they are the same
        const void* alibi_slope_ptr;
        ck_tile::index_t alibi_slope_stride; // stride in batch, or 0 for all batch share same slope
    };

    struct MaskKargs
    {
        // ck_tile::index_t window_size_left, window_size_right;
        ck_tile::index_t window_size_left, window_size_right;
        ck_tile::GenericAttentionMaskEnum mask_type;
    };

    struct Fp8StaticQuantKargs
    {
        float scale_p;
    };

    struct PageBlockTableKargs
    {
        const int32_t* block_table_ptr;
        ck_tile::index_t batch_stride_block_table;
        ck_tile::index_t page_block_size;

        // batched k and v's caches ptrs.
        void*** __restrict__ k_batched_ptr = nullptr;
        void*** __restrict__ v_batched_ptr = nullptr;
        ck_tile::index_t k_batched_offset;
        ck_tile::index_t v_batched_offset;
    };

    struct XQAKargs
    {
        ck_tile::index_t xqa_ratio = 1;
    };

    struct CacheBatchIdxKargs
    {
        const int32_t* cache_batch_idx;
    };

    struct BatchModeKargs
        : CommonKargs,
          std::conditional_t<BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS,
                             BatchModeBiasKargs,
                             std::conditional_t<BiasEnum == BlockAttentionBiasEnum::ALIBI,
                                                AlibiKargs,
                                                EmptyKargs<0>>>,
          std::conditional_t<kHasMask, MaskKargs, EmptyKargs<1>>,
          std::conditional_t<kDoFp8StaticQuant, Fp8StaticQuantKargs, EmptyKargs<2>>,
          std::conditional_t<kIsPagedKV, PageBlockTableKargs, CacheBatchIdxKargs>,
          std::conditional_t<kXQA_enabled, XQAKargs, EmptyKargs<3>>
    {
        const int32_t* seqstart_q_ptr;
        const int32_t* seqstart_k_ptr;
        const int32_t* seqlen_k_ptr;

        ck_tile::index_t batch_stride_q;
        ck_tile::index_t batch_stride_k;
        ck_tile::index_t batch_stride_v;
    };

    struct GroupModeKargs
        : CommonKargs,
          std::conditional_t<BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS,
                             CommonBiasKargs,
                             std::conditional_t<BiasEnum == BlockAttentionBiasEnum::ALIBI,
                                                AlibiKargs,
                                                EmptyKargs<0>>>,
          std::conditional_t<kHasMask, MaskKargs, EmptyKargs<1>>,
          std::conditional_t<kDoFp8StaticQuant, Fp8StaticQuantKargs, EmptyKargs<2>>,
          std::conditional_t<kIsPagedKV, PageBlockTableKargs, EmptyKargs<7>>
    {
        const int32_t* seqstart_q_ptr;
        const int32_t* seqstart_k_ptr;
        const int32_t* seqlen_k_ptr;

        ck_tile::index_t batch_stride_k;
        ck_tile::index_t batch_stride_v;
    };

    using Kargs = std::conditional_t<kIsGroupMode, GroupModeKargs, BatchModeKargs>;

    template <bool Cond = !kIsGroupMode>
    __host__ static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              const void* bias_ptr,
              void* lse_acc_ptr,
              void* o_acc_ptr,
              ck_tile::index_t batch,
              ck_tile::index_t seqlen_q,
              ck_tile::index_t seqlen_k, // only used if 'seqlen_k_ptr' is not specified
              const void* seqlen_k_ptr,  // only used for (paged-) kvcache
              ck_tile::index_t hdim_q,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_head_q,
              ck_tile::index_t num_head_k,
              ck_tile::index_t nhead_ratio_qk,
              ck_tile::index_t num_splits,
              const void* block_table_ptr,
              ck_tile::index_t batch_stride_block_table,
              ck_tile::index_t page_block_size,
              const void* cache_batch_idx,
              float scale_s,
              float scale_p,
              ck_tile::index_t stride_q,
              ck_tile::index_t stride_k,
              ck_tile::index_t stride_v,
              ck_tile::index_t stride_bias,
              ck_tile::index_t stride_o_acc,
              ck_tile::index_t nhead_stride_q,
              ck_tile::index_t nhead_stride_k,
              ck_tile::index_t nhead_stride_v,
              ck_tile::index_t nhead_stride_bias,
              ck_tile::index_t nhead_stride_lse_acc,
              ck_tile::index_t nhead_stride_o_acc,
              ck_tile::index_t batch_stride_q,
              ck_tile::index_t batch_stride_k,
              ck_tile::index_t batch_stride_v,
              ck_tile::index_t batch_stride_bias,
              ck_tile::index_t batch_stride_lse_acc,
              ck_tile::index_t batch_stride_o_acc,
              ck_tile::index_t split_stride_lse_acc,
              ck_tile::index_t split_stride_o_acc,
              ck_tile::index_t window_size_left,
              ck_tile::index_t window_size_right,
              ck_tile::index_t mask_type,
              void*** k_batched_ptr,
              void*** v_batched_ptr,
              ck_tile::index_t k_batched_offset,
              ck_tile::index_t v_batched_offset,
              const void* seqstart_q_ptr,
              const void* seqstart_k_ptr,
              ck_tile::index_t xqa_ratio)
    {
        Kargs kargs{{q_ptr,
                     k_ptr,
                     v_ptr,
                     lse_acc_ptr,
                     o_acc_ptr,
                     batch,
                     seqlen_q,
                     seqlen_k,
                     hdim_q,
                     hdim_v,
                     num_head_q,
                     num_head_k,
                     nhead_ratio_qk,
                     num_splits,
#if CK_TILE_FMHA_FWD_FAST_EXP2
                     static_cast<float>(scale_s * ck_tile::log2e_v<>),
#else
                     scale_s,
#endif
                     stride_q,
                     stride_k,
                     stride_v,
                     stride_o_acc,
                     nhead_stride_q,
                     nhead_stride_k,
                     nhead_stride_v,
                     nhead_stride_lse_acc,
                     nhead_stride_o_acc,
                     batch_stride_lse_acc,
                     batch_stride_o_acc,
                     split_stride_lse_acc,
                     split_stride_o_acc}, // args for common karg
                    {},                   // placeholder for bias
                    {},                   // placeholder for mask
                    {},                   // placeholder for fp8_static_quant args
                    {},                   // placeholder for paged-block table or cache_batch_idx
                    {},                   // placeholder for XQA
                    reinterpret_cast<const int32_t*>(seqstart_q_ptr),
                    reinterpret_cast<const int32_t*>(seqstart_k_ptr),
                    reinterpret_cast<const int32_t*>(seqlen_k_ptr),
                    batch_stride_q,
                    batch_stride_k,
                    batch_stride_v};

        if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
        {
            kargs.bias_ptr          = bias_ptr;
            kargs.stride_bias       = stride_bias;
            kargs.nhead_stride_bias = nhead_stride_bias;
            kargs.batch_stride_bias = batch_stride_bias;
        }
        else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
        {
            kargs.alibi_slope_ptr    = bias_ptr;
            kargs.alibi_slope_stride = stride_bias;
        }
        if constexpr(kHasMask)
        {
            kargs.window_size_left  = window_size_left;
            kargs.window_size_right = window_size_right;
            kargs.mask_type         = static_cast<ck_tile::GenericAttentionMaskEnum>(mask_type);
        }
        if constexpr(kDoFp8StaticQuant)
        {
            kargs.scale_p = scale_p;
        }
        if constexpr(kIsPagedKV)
        {
            kargs.block_table_ptr          = reinterpret_cast<const int32_t*>(block_table_ptr);
            kargs.batch_stride_block_table = batch_stride_block_table;
            kargs.page_block_size          = page_block_size;

            kargs.k_batched_ptr    = k_batched_ptr;
            kargs.v_batched_ptr    = v_batched_ptr;
            kargs.k_batched_offset = k_batched_offset;
            kargs.v_batched_offset = v_batched_offset;
        }
        else
        {
            kargs.cache_batch_idx = reinterpret_cast<const int32_t*>(cache_batch_idx);
        }

        if constexpr(kXQA_enabled)
        {
            kargs.xqa_ratio = xqa_ratio;
        }

        return kargs;
    }

    template <bool Cond = kIsGroupMode>
    __host__ static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              const void* bias_ptr,
              void* lse_acc_ptr,
              void* o_acc_ptr,
              ck_tile::index_t batch,
              const void* seqstart_q_ptr,
              const void* seqstart_k_ptr,
              const void* seqlen_k_ptr,
              ck_tile::index_t hdim_q,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_head_q,
              ck_tile::index_t num_head_k,
              ck_tile::index_t nhead_ratio_qk,
              ck_tile::index_t num_splits,
              float scale_s,
              float scale_p,
              ck_tile::index_t stride_q,
              ck_tile::index_t stride_k,
              ck_tile::index_t stride_v,
              ck_tile::index_t stride_bias,
              ck_tile::index_t stride_o_acc,
              ck_tile::index_t nhead_stride_q,
              ck_tile::index_t nhead_stride_k,
              ck_tile::index_t nhead_stride_v,
              ck_tile::index_t nhead_stride_bias,
              ck_tile::index_t nhead_stride_lse_acc,
              ck_tile::index_t nhead_stride_o_acc,
              ck_tile::index_t batch_stride_k,
              ck_tile::index_t batch_stride_v,
              ck_tile::index_t batch_stride_lse_acc,
              ck_tile::index_t batch_stride_o_acc,
              ck_tile::index_t split_stride_lse_acc,
              ck_tile::index_t split_stride_o_acc,
              ck_tile::index_t window_size_left,
              ck_tile::index_t window_size_right,
              ck_tile::index_t mask_type,
              void*** k_batched_ptr,
              void*** v_batched_ptr,
              ck_tile::index_t k_batched_offset,
              ck_tile::index_t v_batched_offset)
    {
        Kargs kargs{{q_ptr,
                     k_ptr,
                     v_ptr,
                     lse_acc_ptr,
                     o_acc_ptr,
                     batch,
                     -1, // seqlen_q will be updated by another pointer
                     -1, // seqlen_k will be updated by another pointer
                     hdim_q,
                     hdim_v,
                     num_head_q,
                     num_head_k,
                     nhead_ratio_qk,
                     num_splits,
#if CK_TILE_FMHA_FWD_FAST_EXP2
                     static_cast<float>(scale_s * ck_tile::log2e_v<>),
#else
                     scale_s,
#endif
                     stride_q,
                     stride_k,
                     stride_v,
                     stride_o_acc,
                     nhead_stride_q,
                     nhead_stride_k,
                     nhead_stride_v,
                     nhead_stride_lse_acc,
                     nhead_stride_o_acc,
                     batch_stride_lse_acc,
                     batch_stride_o_acc,
                     split_stride_lse_acc,
                     split_stride_o_acc}, // args for common karg
                    {},                   // placeholder for bias
                    {},                   // placeholder for mask
                    {},                   // placeholder for fp8_static_quant args
                    {},                   // placeholder for pagekv
                    reinterpret_cast<const int32_t*>(seqstart_q_ptr),
                    reinterpret_cast<const int32_t*>(seqstart_k_ptr),
                    reinterpret_cast<const int32_t*>(seqlen_k_ptr),
                    batch_stride_k,
                    batch_stride_v};

        if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
        {
            kargs.bias_ptr          = bias_ptr;
            kargs.stride_bias       = stride_bias;
            kargs.nhead_stride_bias = nhead_stride_bias;
        }
        else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
        {
            kargs.alibi_slope_ptr    = bias_ptr;
            kargs.alibi_slope_stride = stride_bias;
        }
        if constexpr(kHasMask)
        {
            kargs.window_size_left  = window_size_left;
            kargs.window_size_right = window_size_right;
            kargs.mask_type         = static_cast<ck_tile::GenericAttentionMaskEnum>(mask_type);
        }
        if constexpr(kDoFp8StaticQuant)
        {
            kargs.scale_p = scale_p;
        }

        if constexpr(kIsPagedKV)
        {
            kargs.k_batched_ptr    = k_batched_ptr;
            kargs.v_batched_ptr    = v_batched_ptr;
            kargs.k_batched_offset = k_batched_offset;
            kargs.v_batched_offset = v_batched_offset;
        }

        return kargs;
    }

    __host__ static constexpr auto GridSize(ck_tile::index_t batch_size,
                                            ck_tile::index_t nhead,
                                            ck_tile::index_t seqlen_q,
                                            ck_tile::index_t hdim_v,
                                            ck_tile::index_t num_splits)
    {
        return TilePartitioner::GridSize(batch_size, nhead, seqlen_q, hdim_v, num_splits);
    }

    __host__ static constexpr auto BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return ck_tile::max(FmhaPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        // lms: Kernel Entrance
        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];
        // divide problem
        // lms: from the tiling, if kN1 is equal to hdimv, so the i_tilen will always be 0.
        const auto [i_tile_m, i_tile_n, i_split, i_nhead, i_batch] =
            TilePartitioner{}(kargs.seqlen_q,
                              kargs.hdim_v,
                              kargs.num_splits); // lms: this is thread block level tile, not thread

        // PRINT_ONLY_IN_GRID("LMS: kargs.seqlen_q: %d, num_splits: %d\n", kargs.seqlen_q,
        // kargs.num_splits);

        const index_t i_m0 = __builtin_amdgcn_readfirstlane(i_tile_m * FmhaPipeline::kM0);
        const index_t i_n1 = __builtin_amdgcn_readfirstlane(i_tile_n * FmhaPipeline::kN1);

        long_index_t batch_offset_q    = 0;
        long_index_t batch_offset_k    = 0;
        long_index_t batch_offset_v    = 0;
        long_index_t batch_offset_bias = 0;
        const long_index_t batch_offset_lse_acc =
            static_cast<long_index_t>(i_batch) * kargs.batch_stride_lse_acc;
        const long_index_t batch_offset_o_acc =
            static_cast<long_index_t>(i_batch) * kargs.batch_stride_o_acc;

        bool use_batched_ptrs = false;
        if constexpr(kIsPagedKV)
        {
            use_batched_ptrs = (kargs.k_batched_ptr != nullptr) && (kargs.v_batched_ptr != nullptr);
            assert(use_batched_ptrs); // FIXME: remove this constraint, using template.
            if(use_batched_ptrs)
            {
                // FIXME: (lms) check the v_batched_ptr should also be non-null
            }
        }

        if constexpr(kIsGroupMode)
        {
            // get starting offset for each batch
            const long_index_t query_start = kargs.seqstart_q_ptr[i_batch];
            const long_index_t key_start   = kargs.seqstart_k_ptr[i_batch];

            batch_offset_q = query_start * kargs.stride_q;
            batch_offset_k = key_start * kargs.stride_k;
            if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            {
                batch_offset_v = key_start * kargs.stride_v;
            }
            else
            {
                batch_offset_v = key_start;
            }
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                batch_offset_bias = query_start * kargs.stride_bias + key_start;
            }

            // get real # queries & # keys under group mode
            kargs.seqlen_q = kargs.seqstart_q_ptr[i_batch + 1] - kargs.seqstart_q_ptr[i_batch];

            // # of required blocks is different in each groups, terminate unnecessary blocks
            // earlier
            if(kargs.seqlen_q <= i_m0)
            {
                return;
            }

            if(kargs.seqlen_k_ptr != nullptr)
            {
                kargs.seqlen_k = kargs.seqlen_k_ptr[i_batch];
            }
            else
            {
                kargs.seqlen_k = kargs.seqstart_k_ptr[i_batch + 1] - kargs.seqstart_k_ptr[i_batch];
            }
        }
        else
        {
            const index_t i_cache_batch = [&, i_batch_ = i_batch] {
                if constexpr(kIsPagedKV)
                {
                    return i_batch_;
                }
                else
                {
                    return (kargs.cache_batch_idx != nullptr ? kargs.cache_batch_idx[i_batch_]
                                                             : i_batch_);
                }
            }();

            if(use_batched_ptrs)
            {
                // get starting offset for each batch
                long_index_t query_start = kargs.seqstart_q_ptr[i_batch];
                // const long_index_t key_start   = kargs.seqstart_k_ptr[i_batch];

                // PRINT_ONLY_IN_GRID("LMS: query_start: %d, key_start: %d\n",
                //                    static_cast<int>(query_start),
                //                    static_cast<int>(key_start));

                // if constexpr(kXQA_enabled)
                // {
                //     query_start *= kargs.xqa_ratio;
                // }

                batch_offset_q = query_start * kargs.stride_q;

                // batch_offset_k = key_start * kargs.stride_k;

                // if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
                // {
                //     batch_offset_v = key_start * kargs.stride_v;
                // }
                // else
                // {
                //     batch_offset_v = key_start;
                // }

                // get real # queries & # keys under group mode
                kargs.seqlen_q = kargs.seqstart_q_ptr[i_batch + 1] - kargs.seqstart_q_ptr[i_batch];

                // # of required blocks is different in each groups, terminate unnecessary blocks
                // earlier
                if(kargs.seqlen_q <= i_m0)
                {
                    return;
                }
                if constexpr(kXQA_enabled)
                {
                    kargs.seqlen_q *= kargs.xqa_ratio;
                }

                if(kargs.seqlen_k_ptr != nullptr)
                {
                    kargs.seqlen_k = kargs.seqlen_k_ptr[i_batch];
                }
                else
                {
                    kargs.seqlen_k =
                        kargs.seqstart_k_ptr[i_batch + 1] - kargs.seqstart_k_ptr[i_batch];
                }
            }
            else
            {
                batch_offset_q = static_cast<long_index_t>(i_batch) * kargs.batch_stride_q;
                batch_offset_k = static_cast<long_index_t>(i_cache_batch) * kargs.batch_stride_k;
                batch_offset_v = static_cast<long_index_t>(i_cache_batch) * kargs.batch_stride_v;
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                {
                    batch_offset_bias =
                        static_cast<long_index_t>(i_batch) * kargs.batch_stride_bias;
                }

                if(kargs.seqlen_k_ptr != nullptr)
                {
                    kargs.seqlen_k = kargs.seqlen_k_ptr[i_batch];
                }
            }

            // end
        }
        ck_tile::index_t head_kv;
        if constexpr(kXQA_ready)
        {
            if constexpr(kXQA_enabled)
            {
                head_kv = i_nhead;
            }
            else
            {

                if(kargs.nhead_ratio_qk > 1)
                {
                    head_kv = i_nhead % kargs.nhead_k;
                }
                else
                {
                    head_kv =
                        i_nhead / kargs.nhead_k +
                        (i_nhead % kargs.nhead_k) * kargs.nhead_ratio_qk; // lms: really weird?
                }
            }
        }
        else
        {
            head_kv = i_nhead / kargs.nhead_ratio_qk;
        }

        // PRINT_ONLY_IN_GRID("LMS: kXQA_ready: %d, kXQA_enabled: %d, i_nhead: %d, head_kv: %d, "
        //                    "seqlen_q: %d, seqlen_k: %d, "
        //                    "kargs.nhead_ratio_qk: %d\n",
        //                    kXQA_ready,
        //                    kXQA_enabled,
        //                    i_nhead,
        //                    head_kv,
        //                    kargs.seqlen_q,
        //                    kargs.seqlen_k,
        //                    kargs.nhead_ratio_qk);

        auto k_page_block_navigator = [&, i_batch_ = i_batch, i_nhead_ = i_nhead]() {
            if constexpr(kIsPagedKV)
            {
                // lms: block_table has shape: [batch_size, num_blocks_of_per_seq]
                // kv cache has shape: [num_blocks, page_block_size, nhead, head_dim]
                // const index_t num_blocks =
                //     integer_divide_ceil(kargs.seqlen_k, kargs.page_block_size);
                // const auto* block_indices =
                //     reinterpret_cast<const int32_t*>(kargs.block_table_ptr) +
                //     i_batch_ * kargs.batch_stride_block_table;

                // const long_index_t fixed_offset =
                //     static_cast<long_index_t>(i_nhead_ / kargs.nhead_ratio_qk) *
                //     kargs.nhead_stride_k;

                // return PageBlockNavigator<const KDataType, 0>(kargs.k_ptr,
                //                                               kargs.batch_stride_k,
                //                                               fixed_offset,
                //                                               block_indices,
                //                                               num_blocks,
                //                                               kargs.page_block_size);

                const index_t num_blocks =
                    integer_divide_ceil(kargs.seqlen_k, kargs.page_block_size);
                // batched ptrs
                // kv cache has shape: [[n_layer, nheads, block_size, head_dim]
                // const long_index_t nhead_offset =
                //     static_cast<long_index_t>(i_nhead_ / kargs.nhead_ratio_qk) *
                const long_index_t nhead_offset =
                    static_cast<long_index_t>(head_kv) * kargs.nhead_stride_k;

                return PageBlockBatchedPtrNavigator<KDataType, 0>(
                    // reinterpret_cast<KDataType***>(kargs.k_batched_ptr)[i_batch_],
                    kargs.k_batched_ptr[i_batch_],
                    kargs.k_batched_offset,
                    nhead_offset,
                    kargs.page_block_size,
                    kargs.stride_k, // FIXME: (lms) check this
                    num_blocks);
            }
            else
            {
                return TrivialPageBlockNavigator();
            }
        }();

        auto v_page_block_navigator = [&, i_batch_ = i_batch, i_nhead_ = i_nhead]() {
            if constexpr(kIsPagedKV)
            {
                // const index_t num_blocks =
                //     integer_divide_ceil(kargs.seqlen_k, kargs.page_block_size);
                // const auto* block_indices =
                //     reinterpret_cast<const int32_t*>(kargs.block_table_ptr) +
                //     i_batch_ * kargs.batch_stride_block_table;

                // const long_index_t fixed_offset =
                //     static_cast<long_index_t>(i_nhead_ / kargs.nhead_ratio_qk) *
                //     kargs.nhead_stride_v;

                // return PageBlockNavigator<const VDataType, 1>(kargs.v_ptr,
                //                                               kargs.batch_stride_v,
                //                                               fixed_offset,
                //                                               block_indices,
                //                                               num_blocks,
                //                                               kargs.page_block_size);

                const index_t num_blocks =
                    integer_divide_ceil(kargs.seqlen_k, kargs.page_block_size);
                // batched ptrs
                // kv cache has shape: [[n_layer, nheads, block_size, head_dim]
                const long_index_t nhead_offset =
                    static_cast<long_index_t>(head_kv) * kargs.nhead_stride_v;
                // const long_index_t nhead_offset =
                //     static_cast<long_index_t>(i_nhead_ / kargs.nhead_ratio_qk) *
                //     kargs.nhead_stride_v;
                return PageBlockBatchedPtrNavigator<VDataType, 0>(
                    // reinterpret_cast<VDataType***>(kargs.v_batched_ptr)[i_batch_],
                    kargs.v_batched_ptr[i_batch_],
                    kargs.v_batched_offset,
                    nhead_offset,
                    kargs.page_block_size,
                    kargs.stride_v, // FIXME: (lms) check this
                    num_blocks);
            }
            else
            {
                return TrivialPageBlockNavigator();
            }
        }();

        // for simplicity, batch stride we just modify the pointer
        const QDataType* q_ptr = reinterpret_cast<const QDataType*>(kargs.q_ptr) +
                                 static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_q +
                                 batch_offset_q;
        // lms: K cache: [bs, num_pages, page_block_size, nhead, hd]
        // k_ptr points to the kcache of current head and current batch, not the pages id and seqlen
        // const KDataType* k_ptr =
        //     reinterpret_cast<const KDataType*>(kargs.k_ptr) +
        //     static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_k +
        //     batch_offset_k;
        // const VDataType* v_ptr =
        //     reinterpret_cast<const VDataType*>(kargs.v_ptr) +
        //     static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_v +
        //     batch_offset_v;

        // FIXME: TODO: Add the K V ptr for batched kv cache ptrs.
        // Note: lms: seems this pointer could be not set in batched ptrs. have a try.
        const KDataType* k_ptr =
            use_batched_ptrs ? nullptr
                             : reinterpret_cast<const KDataType*>(kargs.k_ptr) +
                                   static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) *
                                       kargs.nhead_stride_k +
                                   batch_offset_k;
        const VDataType* v_ptr =
            use_batched_ptrs ? nullptr
                             : reinterpret_cast<const VDataType*>(kargs.v_ptr) +
                                   static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) *
                                       kargs.nhead_stride_v +
                                   batch_offset_v;
        // lms: o_acc: [splits, bs, head_num, seqlen, hd]]
        OaccDataType* o_acc_ptr = reinterpret_cast<OaccDataType*>(kargs.o_acc_ptr) +
                                  static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_o_acc +
                                  batch_offset_o_acc + i_split * kargs.split_stride_o_acc;

        // Q/K/V DRAM and DRAM window
        const auto q_dram = [&]() {
            const auto q_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                q_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_q),
                make_tuple(kargs.stride_q, 1),
                number<FmhaPipeline::kAlignmentQ>{},
                number<1>{});                      // lms: Abstract of Q memory in global.
            if constexpr(FmhaPipeline::kQLoadOnce) // lms: seems this will be always true now.
            {
                return pad_tensor_view(
                    q_dram_naive,
                    make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kK0BlockLength>{}),
                    sequence<kPadSeqLenQ, kPadHeadDimQ>{}); //
            }
            else
            {
                // lms: kK0: tileSize along qk gemm unroll; kM0: tileSize along q seqlen
                return pad_tensor_view(
                    q_dram_naive,
                    make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kK0>{}),
                    sequence<kPadSeqLenQ, kPadHeadDimQ>{});
            }
        }();
        // lms: For this k v dram, batched ptrs and block table should be the same.
        const auto k_dram = [&]() {
            const auto lengths = [&]() {
                if constexpr(kIsPagedKV)
                {
                    return make_tuple(kargs.page_block_size, kargs.hdim_q); // lms: one page block
                }
                else
                {
                    return make_tuple(kargs.seqlen_k, kargs.hdim_q);
                }
            }();

            // we should use ** in l3m, may be batch size stride = 0 and lengths is 1?
            const auto k_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                k_ptr, // will update this pointer if using paged-kvcache
                lengths,
                make_tuple(kargs.stride_k, 1), // lms: yes, it is. [num_pages, block_size, nhead,
                                               // head_q]; 1 in hd_q, stride_k = nhead_k * hd_q
                number<FmhaPipeline::kAlignmentK>{},
                number<1>{});

            return pad_tensor_view(
                k_dram_naive,
                make_tuple(number<FmhaPipeline::kN0>{},
                           number<FmhaPipeline::kK0>{}), // lms: KN0: tileSize along k seqlen
                sequence<kPadSeqLenK, kPadHeadDimQ>{});
        }();
        const auto v_dram = [&]() {
            if constexpr(std::is_same_v<
                             VLayout,
                             ck_tile::tensor_layout::gemm::RowMajor>) // lms: Always true for l3m
                                                                      // [b, seqlenk, head_num,
                                                                      // head_dim]
            {
                const auto lengths = [&]() {
                    if constexpr(kIsPagedKV)
                    {
                        return make_tuple(kargs.page_block_size, kargs.hdim_v);
                    }
                    else
                    {
                        return make_tuple(kargs.seqlen_k, kargs.hdim_v);
                    }
                }();

                const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                    v_ptr, // will update this pointer if using paged-kvcache
                    lengths,
                    make_tuple(kargs.stride_v, 1),
                    number<FmhaPipeline::kAlignmentV>{},
                    number<1>{});

                // FIXME: (lms) why transpose from seq_len, hd -> hd, seq_len?
                const auto v_dram_transposed = transform_tensor_view(
                    v_dram_naive,
                    make_tuple(make_pass_through_transform(lengths.at(number<1>{})),
                               make_pass_through_transform(lengths.at(number<0>{}))),
                    make_tuple(sequence<1>{}, sequence<0>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));

                return pad_tensor_view(
                    v_dram_transposed,
                    make_tuple(number<FmhaPipeline::kN1>{}, number<FmhaPipeline::kK1>{}),
                    sequence<kPadHeadDimV, kPadSeqLenK>{});
            }
            else
            {
                const auto lengths = [&]() {
                    if constexpr(kIsPagedKV)
                    {
                        return make_tuple(kargs.hdim_v, kargs.page_block_size);
                    }
                    else
                    {
                        return make_tuple(kargs.hdim_v, kargs.seqlen_k);
                    }
                }();

                const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                    v_ptr, // will update this pointer if using paged-kvcache
                    lengths,
                    make_tuple(kargs.stride_v, 1),
                    number<FmhaPipeline::kAlignmentV>{},
                    number<1>{});

                return pad_tensor_view(
                    v_dram_naive,
                    make_tuple(number<FmhaPipeline::kN1>{},
                               number<FmhaPipeline::kK1>{}), // lms: kN1: tile size along v
                                                             // head_dim, kK1: tile size along kv
                                                             // gemm unroll, typically on kN0_
                    sequence<kPadHeadDimV, kPadSeqLenK>{});
            }
        }();

        auto q_dram_window = make_tile_window(
            q_dram,
            [&]() {
                if constexpr(FmhaPipeline::kQLoadOnce)
                    return make_tuple(number<FmhaPipeline::kM0>{},
                                      number<FmhaPipeline::kK0BlockLength>{});
                else
                    return make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kK0>{});
            }(),
            {i_m0, 0}); // lms: tile start origin

        auto k_dram_window = make_tile_window(
            k_dram, make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kK0>{}), {0, 0});

        auto v_dram_window =
            make_tile_window(v_dram,
                             make_tuple(number<FmhaPipeline::kN1>{}, number<FmhaPipeline::kK1>{}),
                             {i_n1, 0}); // FIXME: (lms) why start from i_n1?
        /// FIXME: Before C++20, capturing structured binding variables are not supported. Remove
        /// following copy capture of the 'i_nhead' if in C++20
        const auto bias_dram_window = [&, i_nhead_ = i_nhead]() {
            constexpr auto bias_dram_window_lengths =
                make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN0>{});
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                const BiasDataType* bias_ptr =
                    reinterpret_cast<const BiasDataType*>(kargs.bias_ptr) +
                    static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_bias +
                    batch_offset_bias;

                const auto bias_dram = [&]() {
                    const auto bias_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                        bias_ptr,
                        make_tuple(kargs.seqlen_q, kargs.seqlen_k),
                        make_tuple(kargs.stride_bias, 1),
                        number<FmhaPipeline::kAlignmentBias>{},
                        number<1>{});

                    return pad_tensor_view(bias_dram_naive,
                                           bias_dram_window_lengths,
                                           sequence<kPadSeqLenQ, kPadSeqLenK>{});
                }();

                return make_tile_window(bias_dram, bias_dram_window_lengths, {i_m0, 0});
            }
            else
            {
                return make_null_tile_window(bias_dram_window_lengths);
            }
        }();

        // lse acc
        auto lse_acc_dram_window = [&, i_nhead_ = i_nhead, i_split_ = i_split]() {
            constexpr auto lse_acc_dram_window_lengths = make_tuple(number<FmhaPipeline::kM0>{});
            LSEDataType* lse_acc_ptr =
                reinterpret_cast<LSEDataType*>(kargs.lse_acc_ptr) +
                static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_lse_acc +
                batch_offset_lse_acc + i_split_ * kargs.split_stride_lse_acc;

            const auto lse_acc_dram = [&]() {
                const auto lse_acc_dram_naive =
                    make_naive_tensor_view<address_space_enum::global>(lse_acc_ptr,
                                                                       make_tuple(kargs.seqlen_q),
                                                                       make_tuple(1),
                                                                       number<1>{},
                                                                       number<1>{});

                return pad_tensor_view(
                    lse_acc_dram_naive, lse_acc_dram_window_lengths, sequence<kPadSeqLenQ>{});
            }();

            return make_tile_window(lse_acc_dram, lse_acc_dram_window_lengths, {i_m0});
        }();

        FmhaMask mask = [&]() {
            if constexpr(kHasMask)
            {
                ck_tile::index_t mask_x_total = kargs.seqlen_k;
                // if constexpr(kXQA_enabled)
                // {
                //     mask_x_total += kargs.xqa_ratio - 1; // FIXME: (lms) check this
                // }

                return ck_tile::make_generic_attention_mask_from_lr_window<FmhaMask>(
                    kargs.window_size_left,
                    kargs.window_size_right,
                    kargs.seqlen_q,
                    mask_x_total,
                    kargs.mask_type == GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT);
            }
            else
                return FmhaMask{kargs.seqlen_q, kargs.seqlen_k};
        }();

        // WA i_batch capture structure binding before c++20
        auto position_encoding = [&, i_batch_ = i_batch, i_nhead_ = i_nhead]() {
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
            {
                // data loading, shared by entire wg
                // TODO: how to use s_read?
                SaccDataType slope =
                    *(reinterpret_cast<const SaccDataType*>(kargs.alibi_slope_ptr) +
                      i_batch_ * kargs.alibi_slope_stride + i_nhead_);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                slope *= ck_tile::log2e_v<>;
#endif
                if constexpr(kHasMask)
                {
                    return make_alibi_from_lr_mask<SaccDataType, true>(slope,
                                                                       kargs.window_size_left,
                                                                       kargs.window_size_right,
                                                                       kargs.seqlen_q,
                                                                       kargs.seqlen_k,
                                                                       kargs.mask_type);
                }
                else
                {
                    return Alibi<SaccDataType, true>{
                        slope, kargs.seqlen_q, kargs.seqlen_k, AlibiMode::FROM_BOTTOM_RIGHT};
                }
            }
            else
            {
                return EmptyPositionEncoding<SaccDataType>{};
            }
        }();

        auto o_acc_tile = [&, i_split_ = i_split]() {
            if constexpr(kDoFp8StaticQuant)
            {
                return FmhaPipeline{}(q_dram_window,
                                      identity{}, // q_element_func
                                      k_dram_window,
                                      k_page_block_navigator,
                                      identity{}, // k_element_func
                                      v_dram_window,
                                      v_page_block_navigator,
                                      identity{}, // v_element_func
                                      bias_dram_window,
                                      identity{}, // bias_element_func
                                      lse_acc_dram_window,
                                      identity{},            // lse_element_func
                                      identity{},            // s_acc_element_func
                                      scales{kargs.scale_p}, // p_compute_element_func
                                      identity{},            // o_acc_element_func
                                      kargs.num_splits,
                                      i_split_,
                                      mask,
                                      position_encoding,
                                      kargs.scale_s,
                                      smem_ptr);
            }
            else
            {
                // lms: FP16 will enter this branch.
                return FmhaPipeline{}(q_dram_window,
                                      k_dram_window,
                                      k_page_block_navigator,
                                      v_dram_window,
                                      v_page_block_navigator,
                                      bias_dram_window,
                                      lse_acc_dram_window,
                                      kargs.num_splits,
                                      i_split_,
                                      mask,
                                      position_encoding,
                                      kargs.scale_s,
                                      smem_ptr);
            }
        }();

        // Oacc DRAM and Oacc DRAM window
        auto o_acc_dram = [&]() {
            const auto o_acc_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                o_acc_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_v),
                make_tuple(kargs.hdim_v, 1),
                number<FmhaPipeline::kAlignmentO>{},
                number<1>{});

            return pad_tensor_view(
                o_acc_dram_naive,
                make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN1>{}),
                sequence<kPadSeqLenQ, kPadHeadDimV>{});
        }();

        auto o_acc_dram_window =
            make_tile_window(o_acc_dram,
                             make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN1>{}),
                             {i_m0, i_n1});

        EpiloguePipeline{}(o_acc_dram_window, o_acc_tile);
    }
};

} // namespace ck_tile
