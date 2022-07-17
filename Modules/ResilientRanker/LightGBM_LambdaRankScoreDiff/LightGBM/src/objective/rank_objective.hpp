/*!
 * Copyright (c) 2020 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_OBJECTIVE_RANK_OBJECTIVE_HPP_
#define LIGHTGBM_OBJECTIVE_RANK_OBJECTIVE_HPP_

#include <LightGBM/metric.h>
#include <LightGBM/objective_function.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace LightGBM {

/*! \brief Debug purpose: sum gradient of original lambdarank */
double* sum_gradients_original_lambdarank_ = nullptr;
/*! \brief Debug purpose: sum gradient of scorediff lambdarank */
double* sum_gradients_scorediff_lambdarank_ = nullptr;
/*! \brief Debug purpose: sum gradient of score diff */
double* sum_gradients_scorediff_ = nullptr;

/*!
 * \brief Objective function for Ranking
 */
class RankingObjective : public ObjectiveFunction {
 public:
  explicit RankingObjective(const Config& config)
      : seed_(config.objective_seed) {}

  explicit RankingObjective(const std::vector<std::string>&) : seed_(0) {}

  ~RankingObjective() {
    if (sum_gradients_original_lambdarank_ != nullptr) {
      delete[] sum_gradients_original_lambdarank_;
      sum_gradients_original_lambdarank_ = nullptr;
    }
    if (sum_gradients_scorediff_lambdarank_ != nullptr) {
      delete[] sum_gradients_scorediff_lambdarank_;
      sum_gradients_scorediff_lambdarank_ = nullptr;
    }
    if (sum_gradients_scorediff_ != nullptr) {
      delete[] sum_gradients_scorediff_;
      sum_gradients_scorediff_ = nullptr;
    }
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    // get label
    label_ = metadata.label();
    // get weights
    weights_ = metadata.weights();
    // get boundries
    query_boundaries_ = metadata.query_boundaries();
    if (query_boundaries_ == nullptr) {
      Log::Fatal("Ranking tasks require query information");
    }
    num_queries_ = metadata.num_queries();

    if (sum_gradients_original_lambdarank_ == nullptr) {
      sum_gradients_original_lambdarank_ = new double[num_queries_];
    }
    if (sum_gradients_scorediff_lambdarank_ == nullptr) {
      sum_gradients_scorediff_lambdarank_ = new double[num_queries_];
    }
    if (sum_gradients_scorediff_ == nullptr) {
      sum_gradients_scorediff_ = new double[num_queries_];
    }
  }

  void GetGradients(const double* score, score_t* gradients,
                    score_t* hessians) const override {

#pragma omp parallel for schedule(guided)
    for (data_size_t i = 0; i < num_queries_; ++i) {
      sum_gradients_original_lambdarank_[i] = 0;
      sum_gradients_scorediff_lambdarank_[i] = 0;
      sum_gradients_scorediff_[i] = 0;

      const data_size_t start = query_boundaries_[i];
      const data_size_t cnt = query_boundaries_[i + 1] - query_boundaries_[i];
      GetGradientsForOneQuery(i, cnt, label_ + start, score + start,
                              gradients + start, hessians + start);
      if (weights_ != nullptr) {
        for (data_size_t j = 0; j < cnt; ++j) {
          gradients[start + j] =
              static_cast<score_t>(gradients[start + j] * weights_[start + j]);
          hessians[start + j] =
              static_cast<score_t>(hessians[start + j] * weights_[start + j]);
        }
      }
    }

    double sum_gradient_all_query_original_lambdarank = 0;
    double sum_gradient_all_query_scorediff_lambdarank = 0;
    double sum_gradient_all_query_scorediff = 0;
    for (data_size_t i = 0; i < num_queries_; ++i) {
      sum_gradient_all_query_original_lambdarank += sum_gradients_original_lambdarank_[i];
      sum_gradient_all_query_scorediff_lambdarank += sum_gradients_scorediff_lambdarank_[i];
      sum_gradient_all_query_scorediff += sum_gradients_scorediff_[i];
    }

    Log::Info("*-*-*-* %f %f %f", sum_gradient_all_query_original_lambdarank, sum_gradient_all_query_scorediff_lambdarank, sum_gradient_all_query_scorediff);
  }

  virtual void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt,
                                       const label_t* label,
                                       const double* score, score_t* lambdas,
                                       score_t* hessians) const = 0;

  const char* GetName() const override = 0;

  std::string ToString() const override {
    std::stringstream str_buf;
    str_buf << GetName();
    return str_buf.str();
  }

  bool NeedAccuratePrediction() const override { return false; }

 protected:
  int seed_;
  data_size_t num_queries_;
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const label_t* label_;
  /*! \brief Pointer of weights */
  const label_t* weights_;
  /*! \brief Query boundries */
  const data_size_t* query_boundaries_;
};

/*!
 * \brief Objective function for Lambdrank with NDCG
 */
class LambdarankNDCG : public RankingObjective {
 public:
  explicit LambdarankNDCG(const Config& config)
      : RankingObjective(config),
        sigmoid_(config.sigmoid),
        norm_(config.lambdarank_norm),
        truncation_level_(config.lambdarank_truncation_level) {
    label_gain_ = config.label_gain;
    // initialize DCG calculator
    DCGCalculator::DefaultLabelGain(&label_gain_);
    DCGCalculator::Init(label_gain_);
    sigmoid_table_.clear();
    inverse_max_dcgs_.clear();
    if (sigmoid_ <= 0.0) {
      Log::Fatal("Sigmoid param %f should be greater than zero", sigmoid_);
    }
  }

  explicit LambdarankNDCG(const std::vector<std::string>& strs)
      : RankingObjective(strs) {}

  ~LambdarankNDCG() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    RankingObjective::Init(metadata, num_data);
    DCGCalculator::CheckMetadata(metadata, num_queries_);
    DCGCalculator::CheckLabel(label_, num_data_);
    inverse_max_dcgs_.resize(num_queries_);
#pragma omp parallel for schedule(static)
    for (data_size_t i = 0; i < num_queries_; ++i) {
      inverse_max_dcgs_[i] = DCGCalculator::CalMaxDCGAtK(
          truncation_level_, label_ + query_boundaries_[i],
          query_boundaries_[i + 1] - query_boundaries_[i]);

      if (inverse_max_dcgs_[i] > 0.0) {
        inverse_max_dcgs_[i] = 1.0f / inverse_max_dcgs_[i];
      }
    }
    // construct sigmoid table to speed up sigmoid transform
    ConstructSigmoidTable();
  }

  inline void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt,
                                      const label_t* label, const double* score,
                                      score_t* lambdas,
                                      score_t* hessians) const override {
    // get max DCG on current query
    const double inverse_max_dcg = inverse_max_dcgs_[query_id];
    // initialize with zero
    for (data_size_t i = 0; i < cnt; ++i) {
      lambdas[i] = 0.0f;
      hessians[i] = 0.0f;
    }
    // get sorted indices for scores
    std::vector<data_size_t> sorted_idx(cnt);
    for (data_size_t i = 0; i < cnt; ++i) {
      sorted_idx[i] = i;
    }
    std::stable_sort(
        sorted_idx.begin(), sorted_idx.end(),
        [score](data_size_t a, data_size_t b) { return score[a] > score[b]; });
    // get best and worst score
    const double best_score = score[sorted_idx[0]];
    data_size_t worst_idx = cnt - 1;
    if (worst_idx > 0 && score[sorted_idx[worst_idx]] == kMinScore) {
      worst_idx -= 1;
    }
    const double worst_score = score[sorted_idx[worst_idx]];
    double sum_lambdas = 0.0;
    // start accmulate lambdas by pairs that contain at least one document above truncation level
    for (data_size_t i = 0; i < cnt - 1 && i < truncation_level_; ++i) {
      if (score[sorted_idx[i]] == kMinScore) { continue; }
      for (data_size_t j = i + 1; j < cnt; ++j) {
        if (score[sorted_idx[j]] == kMinScore) { continue; }
        // skip pairs with the same labels
        if (label[sorted_idx[i]] == label[sorted_idx[j]]) { continue; }
        data_size_t high_rank, low_rank;
        if (label[sorted_idx[i]] > label[sorted_idx[j]]) {
          high_rank = i;
          low_rank = j;
        } else {
          high_rank = j;
          low_rank = i;
        }
        const data_size_t high = sorted_idx[high_rank];
        const int high_label = static_cast<int>(label[high]);
        const double high_score = score[high];
        const double high_label_gain = label_gain_[high_label];
        const double high_discount = DCGCalculator::GetDiscount(high_rank);
        const data_size_t low = sorted_idx[low_rank];
        const int low_label = static_cast<int>(label[low]);
        const double low_score = score[low];
        const double low_label_gain = label_gain_[low_label];
        const double low_discount = DCGCalculator::GetDiscount(low_rank);

        const double delta_score = high_score - low_score;

        // get dcg gap
        const double dcg_gap = high_label_gain - low_label_gain;
        // get discount of this pair
        const double paired_discount = fabs(high_discount - low_discount);
        // get delta NDCG
        double delta_pair_NDCG = dcg_gap * paired_discount * inverse_max_dcg;
        // regular the delta_pair_NDCG by score distance
        if (norm_ && best_score != worst_score) {
          delta_pair_NDCG /= (0.01f + fabs(delta_score));
        }
        // calculate lambda for this pair
        double p_lambda = GetSigmoid(delta_score);
        double p_hessian = p_lambda * (1.0f - p_lambda);
        // update
        p_lambda *= -sigmoid_ * delta_pair_NDCG;
        p_hessian *= sigmoid_ * sigmoid_ * delta_pair_NDCG;
        lambdas[low] -= static_cast<score_t>(p_lambda);
        hessians[low] += static_cast<score_t>(p_hessian);
        lambdas[high] += static_cast<score_t>(p_lambda);
        hessians[high] += static_cast<score_t>(p_hessian);
        // lambda is negative, so use minus to accumulate
        sum_lambdas -= 2 * p_lambda;
      }
    }
    if (norm_ && sum_lambdas > 0) {
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = static_cast<score_t>(lambdas[i] * norm_factor);
        hessians[i] = static_cast<score_t>(hessians[i] * norm_factor);
      }
    }
  }

  inline double GetSigmoid(double score) const {
    if (score <= min_sigmoid_input_) {
      // too small, use lower bound
      return sigmoid_table_[0];
    } else if (score >= max_sigmoid_input_) {
      // too large, use upper bound
      return sigmoid_table_[_sigmoid_bins - 1];
    } else {
      return sigmoid_table_[static_cast<size_t>((score - min_sigmoid_input_) *
                                                sigmoid_table_idx_factor_)];
    }
  }

  void ConstructSigmoidTable() {
    // get boundary
    min_sigmoid_input_ = min_sigmoid_input_ / sigmoid_ / 2;
    max_sigmoid_input_ = -min_sigmoid_input_;
    sigmoid_table_.resize(_sigmoid_bins);
    // get score to bin factor
    sigmoid_table_idx_factor_ =
        _sigmoid_bins / (max_sigmoid_input_ - min_sigmoid_input_);
    // cache
    for (size_t i = 0; i < _sigmoid_bins; ++i) {
      const double score = i / sigmoid_table_idx_factor_ + min_sigmoid_input_;
      sigmoid_table_[i] = 1.0f / (1.0f + std::exp(score * sigmoid_));
    }
  }

  const char* GetName() const override { return "lambdarank"; }

 protected:
  /*! \brief Simgoid param */
  double sigmoid_;
  /*! \brief Normalize the lambdas or not */
  bool norm_;
  /*! \brief Truncation position for max DCG */
  int truncation_level_;
  /*! \brief Cache inverse max DCG, speed up calculation */
  std::vector<double> inverse_max_dcgs_;
  /*! \brief Cache result for sigmoid transform to speed up */
  std::vector<double> sigmoid_table_;
  /*! \brief Gains for labels */
  std::vector<double> label_gain_;
  /*! \brief Number of bins in simoid table */
  size_t _sigmoid_bins = 1024 * 1024;
  /*! \brief Minimal input of sigmoid table */
  double min_sigmoid_input_ = -50;
  /*! \brief Maximal input of sigmoid table */
  double max_sigmoid_input_ = 50;
  /*! \brief Factor that covert score to bin in sigmoid table */
  double sigmoid_table_idx_factor_;
};

/*!
 * \brief Objective function for Lambdrank with NDCG and ScoreDiff
 */
class LambdarankNDCGScoreDiff : public LambdarankNDCG {
  double lambda_scorediff_;
  const DocPairContainer* doc_pair_containers_;
public:
  explicit LambdarankNDCGScoreDiff(const Config& config)
    : LambdarankNDCG(config)
    , lambda_scorediff_(config.scorediff_alpha)
    , doc_pair_containers_(nullptr) {
    Log::Info("initializing LambdarankNDCGScoreDiff");
    Log::Info("scoredif lambda = %f", lambda_scorediff_);
  }

  explicit LambdarankNDCGScoreDiff(const std::vector<std::string>& strs)
    : LambdarankNDCG(strs)
    , lambda_scorediff_(1.0)
    , doc_pair_containers_(nullptr) {
    Log::Info("initializing LambdarankNDCGScoreDiff from string");
    for (const auto& s : strs) {
      Log::Info(s.c_str());
    }
    Log::Info("scoredif lambda = %f", lambda_scorediff_);
  }

  ~LambdarankNDCGScoreDiff() {
    Log::Info("uninitializing LambdarankNDCGScoreDiff");
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    Log::Info("initialize scordiff lambdarank");
    LambdarankNDCG::Init(metadata, num_data);

    doc_pair_containers_ = metadata.doc_pairs();
    if (doc_pair_containers_ == nullptr) {
      Log::Warning("no doc pair is found");
    }
  }

  inline void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt,
    const label_t* label, const double* score,
    score_t* lambdas,
    score_t* hessians) const override {
    // get max DCG on current query
    const double inverse_max_dcg = inverse_max_dcgs_[query_id];
    // initialize with zero
    for (data_size_t i = 0; i < cnt; ++i) {
      lambdas[i] = 0.0f;
      hessians[i] = 0.0f;
    }
    // get sorted indices for scores
    std::vector<data_size_t> sorted_idx(cnt);
    for (data_size_t i = 0; i < cnt; ++i) {
      sorted_idx[i] = i;
    }
    std::stable_sort(
      sorted_idx.begin(), sorted_idx.end(),
      [score](data_size_t a, data_size_t b) { return score[a] > score[b]; });
    // get best and worst score
    const double best_score = score[sorted_idx[0]];
    data_size_t worst_idx = cnt - 1;
    if (worst_idx > 0 && score[sorted_idx[worst_idx]] == kMinScore) {
      worst_idx -= 1;
    }
    const double worst_score = score[sorted_idx[worst_idx]];
    double sum_lambdas = 0.0;

    const DocPairContainer* doc_pair_container = nullptr;
    if (doc_pair_containers_ != nullptr) {
      doc_pair_container = doc_pair_containers_ + query_id;
    }
    // start accmulate lambdas by pairs that contain at least one document above truncation level
    for (data_size_t i = 0; i < cnt - 1 && i < truncation_level_; ++i) {
      if (score[sorted_idx[i]] == kMinScore) { continue; }
      for (data_size_t j = i + 1; j < cnt; ++j) {
        if (score[sorted_idx[j]] == kMinScore) { continue; }
        // skip pairs with the same labels
        if (label[sorted_idx[i]] == label[sorted_idx[j]]) { continue; }
        data_size_t high_rank, low_rank;
        if (label[sorted_idx[i]] > label[sorted_idx[j]]) {
          high_rank = i;
          low_rank = j;
        }
        else {
          high_rank = j;
          low_rank = i;
        }
        const data_size_t high = sorted_idx[high_rank];
        const int high_label = static_cast<int>(label[high]);
        const double high_score = score[high];
        const double high_label_gain = label_gain_[high_label];
        const double high_discount = DCGCalculator::GetDiscount(high_rank);
        const data_size_t low = sorted_idx[low_rank];
        const int low_label = static_cast<int>(label[low]);
        const double low_score = score[low];
        const double low_label_gain = label_gain_[low_label];
        const double low_discount = DCGCalculator::GetDiscount(low_rank);

        const double delta_score = high_score - low_score;

        // get dcg gap
        const double dcg_gap = high_label_gain - low_label_gain;
        // get discount of this pair
        const double paired_discount = fabs(high_discount - low_discount);
        // get delta NDCG
        double delta_pair_NDCG = dcg_gap * paired_discount * inverse_max_dcg;
        // regular the delta_pair_NDCG by score distance
        if (norm_ && best_score != worst_score) {
          delta_pair_NDCG /= (0.01f + fabs(delta_score));
        }

        // Pij = 1 if i and j are with same URL/doc id
        int Pij = static_cast<int>(doc_pair_container != nullptr && !doc_pair_container->empty() && doc_pair_container->find(std::make_pair(std::min(low, high), std::max(low, high))) != doc_pair_container->end());
        // calculate lambda for this pair
        double p_lambda = GetSigmoid(delta_score);
        double p_hessian = p_lambda * (1.0f - p_lambda);
        // update
        p_lambda *= -sigmoid_ * delta_pair_NDCG;
        p_hessian *= sigmoid_ * sigmoid_ * delta_pair_NDCG;
        // add score diff to gradient and hessian
        p_lambda += 2 * lambda_scorediff_ * delta_score * Pij;
        p_hessian += 2 * lambda_scorediff_ * Pij;
        lambdas[low] -= static_cast<score_t>(p_lambda);
        hessians[low] += static_cast<score_t>(p_hessian);
        lambdas[high] += static_cast<score_t>(p_lambda);
        hessians[high] += static_cast<score_t>(p_hessian);
        // lambda is negative, so use minus to accumulate
        //sum_lambdas -= 2 * p_lambda; //??
        sum_lambdas += 2 * std::abs(p_lambda); // change to abs because p_lambda might be positive
      }
    }
    if (norm_ && sum_lambdas > 0) {
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = static_cast<score_t>(lambdas[i] * norm_factor);
        hessians[i] = static_cast<score_t>(hessians[i] * norm_factor);
      }
    }
  }

  const char* GetName() const override { return "lambdarank_scorediff"; }
};

/*!
 * \brief Objective function for Lambdrank with NDCG and ScoreDiff
 */
class LambdarankNDCGScoreDiff2 : public LambdarankNDCG {
  double lambda_scorediff_;
  const data_size_t* impression_cnts_;
  std::vector<double> inverse_max_dcgs_2_;
  bool normalize_scorediff_gradient_;

public:
  explicit LambdarankNDCGScoreDiff2(const Config& config)
    : LambdarankNDCG(config)
    , lambda_scorediff_(config.scorediff_alpha)
    , impression_cnts_(nullptr)
    , normalize_scorediff_gradient_(false) {

    inverse_max_dcgs_2_.clear();
    normalize_scorediff_gradient_ = config.normalize_gradient_scorediff;

    Log::Info("initializing LambdarankNDCGScoreDiff2");
    Log::Info("scoredif lambda = %f", lambda_scorediff_);
    Log::Info("normalize_scorediff_gradient_ = %d", normalize_scorediff_gradient_);
  }

  explicit LambdarankNDCGScoreDiff2(const std::vector<std::string>& strs)
    : LambdarankNDCG(strs)
    , lambda_scorediff_(1.0)
    , impression_cnts_(nullptr)
    , normalize_scorediff_gradient_(false) {

    Log::Info("initializing LambdarankNDCGScoreDiff2 from string");
    for (const auto& s : strs) {
      Log::Info(s.c_str());
    }
    Log::Info("scoredif lambda = %f", lambda_scorediff_);
  }

  ~LambdarankNDCGScoreDiff2() {
    Log::Info("uninitializing LambdarankNDCGScoreDiff2");
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    Log::Info("initialize scordiff lambdarank 2");
    LambdarankNDCG::Init(metadata, num_data);

    impression_cnts_ = metadata.impression_cnts();
    if (nullptr == impression_cnts_) {
      Log::Warning("no impression cnt is found");
    }

    // re-calculate max dcg if there are more than one impressions
    if (nullptr != impression_cnts_) {
      inverse_max_dcgs_2_.resize(num_queries_);
      for (data_size_t i = 0; i < num_queries_; ++i) {
        if (impression_cnts_[i] > 1) {
          int len = query_boundaries_[i + 1] - query_boundaries_[i];
          if (impression_cnts_[i] > 2 || len % 2 != 0) {
            Log::Warning("For query %d, it has %d impressions and %d docs. It is not supported now. Only support 2 impressions. Will calculate original rank loss on it.", i, impression_cnts_[i], len);
          }
          inverse_max_dcgs_[i] = DCGCalculator::CalMaxDCGAtK(truncation_level_, label_ + query_boundaries_[i], len / 2);

          if (inverse_max_dcgs_[i] > 0.0) {
            inverse_max_dcgs_[i] = 1.0f / inverse_max_dcgs_[i];
          }

          inverse_max_dcgs_2_[i] = DCGCalculator::CalMaxDCGAtK(truncation_level_, label_ + query_boundaries_[i] + len / 2, len / 2);

          if (inverse_max_dcgs_2_[i] > 0.0) {
            inverse_max_dcgs_2_[i] = 1.0f / inverse_max_dcgs_2_[i];
          }
        }

      }
    }
  }

  inline double RankLossForOneQuery(data_size_t cnt, const label_t* label, const double* score,
    score_t* lambdas, score_t* hessians, const double inverse_max_dcg) const {
    // initialize with zero
    for (data_size_t i = 0; i < cnt; ++i) {
      lambdas[i] = 0.0f;
      hessians[i] = 0.0f;
    }
    // get sorted indices for scores
    std::vector<data_size_t> sorted_idx(cnt);
    for (data_size_t i = 0; i < cnt; ++i) {
      sorted_idx[i] = i;
    }
    std::stable_sort(
      sorted_idx.begin(), sorted_idx.end(),
      [score](data_size_t a, data_size_t b) { return score[a] > score[b]; });
    // get best and worst score
    const double best_score = score[sorted_idx[0]];
    data_size_t worst_idx = cnt - 1;
    if (worst_idx > 0 && score[sorted_idx[worst_idx]] == kMinScore) {
      worst_idx -= 1;
    }
    const double worst_score = score[sorted_idx[worst_idx]];
    double sum_lambdas = 0.0;

    // start accmulate lambdas by pairs that contain at least one document above truncation level
    for (data_size_t i = 0; i < cnt - 1 && i < truncation_level_; ++i) {
      if (score[sorted_idx[i]] == kMinScore) { continue; }
      for (data_size_t j = i + 1; j < cnt; ++j) {
        if (score[sorted_idx[j]] == kMinScore) { continue; }
        // skip pairs with the same labels
        if (label[sorted_idx[i]] == label[sorted_idx[j]]) { continue; }
        data_size_t high_rank, low_rank;
        if (label[sorted_idx[i]] > label[sorted_idx[j]]) {
          high_rank = i;
          low_rank = j;
        }
        else {
          high_rank = j;
          low_rank = i;
        }
        const data_size_t high = sorted_idx[high_rank];
        const int high_label = static_cast<int>(label[high]);
        const double high_score = score[high];
        const double high_label_gain = label_gain_[high_label];
        const double high_discount = DCGCalculator::GetDiscount(high_rank);
        const data_size_t low = sorted_idx[low_rank];
        const int low_label = static_cast<int>(label[low]);
        const double low_score = score[low];
        const double low_label_gain = label_gain_[low_label];
        const double low_discount = DCGCalculator::GetDiscount(low_rank);

        const double delta_score = high_score - low_score;

        // get dcg gap
        const double dcg_gap = high_label_gain - low_label_gain;
        // get discount of this pair
        const double paired_discount = fabs(high_discount - low_discount);
        // get delta NDCG
        double delta_pair_NDCG = dcg_gap * paired_discount * inverse_max_dcg;
        // regular the delta_pair_NDCG by score distance
        if (norm_ && best_score != worst_score) {
          delta_pair_NDCG /= (0.01f + fabs(delta_score));
        }

        // calculate lambda for this pair
        double p_lambda = GetSigmoid(delta_score);
        double p_hessian = p_lambda * (1.0f - p_lambda);
        // update
        p_lambda *= -sigmoid_ * delta_pair_NDCG;
        p_hessian *= sigmoid_ * sigmoid_ * delta_pair_NDCG;
        lambdas[low] -= static_cast<score_t>(p_lambda);
        hessians[low] += static_cast<score_t>(p_hessian);
        lambdas[high] += static_cast<score_t>(p_lambda);
        hessians[high] += static_cast<score_t>(p_hessian);
        // lambda is negative, so use minus to accumulate
        sum_lambdas -= 2 * p_lambda;
      }
    }

    // when normalize_scorediff_gradient_, the gradients for rank loss and score diff loss will be normalized together
    if (norm_ && !normalize_scorediff_gradient_ && sum_lambdas > 0) { 
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = static_cast<score_t>(lambdas[i] * norm_factor);
        hessians[i] = static_cast<score_t>(hessians[i] * norm_factor);
      }
    }
    
    return sum_lambdas;
  }

  inline double ScoreDiffLossForOneQuery(data_size_t cnt, const double* score, score_t* lambdas, score_t* hessians) const {

    double sum_lambdas = 0;
    data_size_t half_len = cnt / 2;
    for (data_size_t i = 0; i < half_len; ++i) {
      if (score[i] == kMinScore || score[i + half_len] == kMinScore) { continue; }
        const double delta_score = score[i] - score[i + half_len];

        // calculate lambda for this pair
        double p_lambda = 2 * lambda_scorediff_ * delta_score;
        double p_hessian = 2 * lambda_scorediff_;

        lambdas[i] += static_cast<score_t>(p_lambda);
        hessians[i] += static_cast<score_t>(p_hessian);
        lambdas[i + half_len] -= static_cast<score_t>(p_lambda);
        hessians[i + half_len] += static_cast<score_t>(p_hessian); // identical 2nd order gradient
        sum_lambdas += 2 * std::abs(p_lambda);
    }

    return sum_lambdas;
  }

  inline void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt,
    const label_t* label, const double* score,
    score_t* lambdas,
    score_t* hessians) const override {

    // for this loss, we assume that the records are sorted by
    //  1. query id
    //  2. impression id
    //  3. doc id

    // loss for score diff
    // 1. remove skip for equal label
    // 2. lambdarank inside one impresion, and score diff between impression
    // 3. re-calculate max dcg

    //const DocPairContainer* doc_pair_container = nullptr;
    //if (doc_pair_containers_ != nullptr) {
    //  doc_pair_container = doc_pair_containers_ + query_id;
    //}

    data_size_t impression_cnt = impression_cnts_ == nullptr ? 1 : impression_cnts_[query_id]; // at least one impression in one query
    if (impression_cnt > 2) {
      Log::Warning("number of impression is %d > 2", impression_cnt);
    }

    //bool has_doc_pair = doc_pair_container != nullptr && !doc_pair_container->empty();
    //bool use_scorediff = impression_cnt > 1 && has_doc_pair;
    bool use_scorediff = (impression_cnt == 2) && (cnt % 2 == 0);

    double sum_lambdas = 0;

    if (!use_scorediff) {
      // get max DCG on current query
      const double inverse_max_dcg = inverse_max_dcgs_[query_id];

      sum_lambdas = RankLossForOneQuery(cnt, label, score, lambdas, hessians, inverse_max_dcg);

    }
    else {
      if (cnt % 2 != 0) {
        Log::Fatal("using score diff, but number of docs in query %d is %d, not an even number.", query_id, cnt);
      }
      // get max DCG on current query
      const double inverse_max_dcg1 = inverse_max_dcgs_[query_id];
      const double inverse_max_dcg2 = inverse_max_dcgs_2_[query_id];
      // rank loss for first impression
      double sum_lambdas1 = RankLossForOneQuery(cnt / 2, label, score, lambdas, hessians, inverse_max_dcg1);
      // rank loss for second impression
      double sum_lambdas2 = RankLossForOneQuery(cnt / 2, label + cnt / 2, score + cnt / 2, lambdas + cnt / 2, hessians + cnt / 2, inverse_max_dcg2);
      // score diff loss for doc pairs
      double sum_lambdas3 = ScoreDiffLossForOneQuery(cnt, score, lambdas, hessians);

      sum_lambdas = sum_lambdas1 + sum_lambdas2 + sum_lambdas3;
    }
    
    if (norm_ && normalize_scorediff_gradient_ && sum_lambdas > 0) {
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = static_cast<score_t>(lambdas[i] * norm_factor);
        hessians[i] = static_cast<score_t>(hessians[i] * norm_factor);
      }
    }

  }

  const char* GetName() const override { return "lambdarank_scorediff2"; }
};

/*!
 * \brief Objective function for Lambdrank with NDCG and ScoreDiff
 */
class LambdarankNDCGScoreDiff3 : public LambdarankNDCG {
  double lambda_scorediff_;
  const data_size_t* impression_cnts_;
  std::vector<double> inverse_max_dcgs_2_;
  bool normalize_scorediff_gradient_;

public:
  explicit LambdarankNDCGScoreDiff3(const Config& config)
    : LambdarankNDCG(config)
    , lambda_scorediff_(config.scorediff_alpha)
    , impression_cnts_(nullptr)
    , normalize_scorediff_gradient_(false) {

    inverse_max_dcgs_2_.clear();
    normalize_scorediff_gradient_ = config.normalize_gradient_scorediff;

    Log::Info("initializing LambdarankNDCGScoreDiff3");
    Log::Info("scoredif lambda = %f", lambda_scorediff_);
    Log::Info("normalize_scorediff_gradient_ = %d", normalize_scorediff_gradient_);
  }

  explicit LambdarankNDCGScoreDiff3(const std::vector<std::string>& strs)
    : LambdarankNDCG(strs)
    , lambda_scorediff_(0.0)
    , impression_cnts_(nullptr)
    , normalize_scorediff_gradient_(false) {

    Log::Info("initializing LambdarankNDCGScoreDiff3 from string");
    for (const auto& s : strs) {
      Log::Info(s.c_str());
    }
    Log::Info("scoredif lambda = %f", lambda_scorediff_);
  }

  ~LambdarankNDCGScoreDiff3() {
    Log::Info("uninitializing LambdarankNDCGScoreDiff3");
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    Log::Info("initialize scordiff lambdarank 3");
    LambdarankNDCG::Init(metadata, num_data);

    impression_cnts_ = metadata.impression_cnts();
    if (nullptr == impression_cnts_) {
      Log::Warning("no impression cnt is found");
    }

    // re-calculate max dcg if there are more than one impressions
    if (nullptr != impression_cnts_) {
      inverse_max_dcgs_2_.resize(num_queries_);
      for (data_size_t i = 0; i < num_queries_; ++i) {
        if (impression_cnts_[i] > 1) {
          int len = query_boundaries_[i + 1] - query_boundaries_[i];
          if (impression_cnts_[i] > 2 || len % 2 != 0) {
            Log::Warning("For query %d, it has %d impressions and %d docs. It is not supported now. Only support 2 impressions. Will calculate original rank loss on it.", i, impression_cnts_[i], len);
          }
          inverse_max_dcgs_[i] = DCGCalculator::CalMaxDCGAtK(truncation_level_, label_ + query_boundaries_[i], len / 2);

          if (inverse_max_dcgs_[i] > 0.0) {
            inverse_max_dcgs_[i] = 1.0f / inverse_max_dcgs_[i];
          }

          inverse_max_dcgs_2_[i] = DCGCalculator::CalMaxDCGAtK(truncation_level_, label_ + query_boundaries_[i] + len / 2, len / 2);

          if (inverse_max_dcgs_2_[i] > 0.0) {
            inverse_max_dcgs_2_[i] = 1.0f / inverse_max_dcgs_2_[i];
          }
        }

      }
    }
  }

  inline double RankLossForOneQuery(data_size_t cnt, const label_t* label, const double* score,
    score_t* lambdas, score_t* hessians, const double inverse_max_dcg, const double weight) const {
    // initialize with zero
    for (data_size_t i = 0; i < cnt; ++i) {
      lambdas[i] = 0.0f;
      hessians[i] = 0.0f;
    }
    // get sorted indices for scores
    std::vector<data_size_t> sorted_idx(cnt);
    for (data_size_t i = 0; i < cnt; ++i) {
      sorted_idx[i] = i;
    }
    std::stable_sort(
      sorted_idx.begin(), sorted_idx.end(),
      [score](data_size_t a, data_size_t b) { return score[a] > score[b]; });
    // get best and worst score
    const double best_score = score[sorted_idx[0]];
    data_size_t worst_idx = cnt - 1;
    if (worst_idx > 0 && score[sorted_idx[worst_idx]] == kMinScore) {
      worst_idx -= 1;
    }
    const double worst_score = score[sorted_idx[worst_idx]];
    double sum_lambdas = 0.0;

    // start accmulate lambdas by pairs that contain at least one document above truncation level
    for (data_size_t i = 0; i < cnt - 1 && i < truncation_level_; ++i) {
      if (score[sorted_idx[i]] == kMinScore) { continue; }
      for (data_size_t j = i + 1; j < cnt; ++j) {
        if (score[sorted_idx[j]] == kMinScore) { continue; }
        // skip pairs with the same labels
        if (label[sorted_idx[i]] == label[sorted_idx[j]]) { continue; }
        data_size_t high_rank, low_rank;
        if (label[sorted_idx[i]] > label[sorted_idx[j]]) {
          high_rank = i;
          low_rank = j;
        }
        else {
          high_rank = j;
          low_rank = i;
        }
        const data_size_t high = sorted_idx[high_rank];
        const int high_label = static_cast<int>(label[high]);
        const double high_score = score[high];
        const double high_label_gain = label_gain_[high_label];
        const double high_discount = DCGCalculator::GetDiscount(high_rank);
        const data_size_t low = sorted_idx[low_rank];
        const int low_label = static_cast<int>(label[low]);
        const double low_score = score[low];
        const double low_label_gain = label_gain_[low_label];
        const double low_discount = DCGCalculator::GetDiscount(low_rank);

        const double delta_score = high_score - low_score;

        // get dcg gap
        const double dcg_gap = high_label_gain - low_label_gain;
        // get discount of this pair
        const double paired_discount = fabs(high_discount - low_discount);
        // get delta NDCG
        double delta_pair_NDCG = dcg_gap * paired_discount * inverse_max_dcg;
        // regular the delta_pair_NDCG by score distance
        if (norm_ && best_score != worst_score) {
          delta_pair_NDCG /= (0.01f + fabs(delta_score));
        }

        // calculate lambda for this pair
        double p_lambda = GetSigmoid(delta_score);
        double p_hessian = p_lambda * (1.0f - p_lambda);
        // update
        p_lambda *= -sigmoid_ * delta_pair_NDCG * weight;
        p_hessian *= sigmoid_ * sigmoid_ * delta_pair_NDCG * weight;
        lambdas[low] -= static_cast<score_t>(p_lambda);
        hessians[low] += static_cast<score_t>(p_hessian);
        lambdas[high] += static_cast<score_t>(p_lambda);
        hessians[high] += static_cast<score_t>(p_hessian);
        // lambda is negative, so use minus to accumulate
        sum_lambdas -= 2 * p_lambda;
      }
    }

    // when normalize_scorediff_gradient_, the gradients for rank loss and score diff loss will be normalized together
    if (norm_ && !normalize_scorediff_gradient_ && sum_lambdas > 0) {
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = static_cast<score_t>(lambdas[i] * norm_factor);
        hessians[i] = static_cast<score_t>(hessians[i] * norm_factor);
      }
    }

    return sum_lambdas;
  }

  inline double ScoreDiffLossForOneQuery(data_size_t cnt, const double* score, score_t* lambdas, score_t* hessians) const {

    double sum_lambdas = 0;
    data_size_t half_len = cnt / 2;
    for (data_size_t i = 0; i < half_len; ++i) {
      if (score[i] == kMinScore || score[i + half_len] == kMinScore) { continue; }
      const double delta_score = score[i] - score[i + half_len];

      // calculate lambda for this pair
      double p_lambda = 2 * lambda_scorediff_ * delta_score;
      double p_hessian = 2 * lambda_scorediff_;

      lambdas[i] += static_cast<score_t>(p_lambda);
      hessians[i] += static_cast<score_t>(p_hessian);
      lambdas[i + half_len] -= static_cast<score_t>(p_lambda);
      hessians[i + half_len] += static_cast<score_t>(p_hessian); // identical 2nd order gradient
      sum_lambdas += 2 * std::abs(p_lambda);
    }

    return sum_lambdas;
  }

  inline void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt,
    const label_t* label, const double* score,
    score_t* lambdas,
    score_t* hessians) const override {

    // for this loss, we assume that the records are sorted by
    //  1. query id
    //  2. impression id
    //  3. doc id

    // loss for score diff
    // 1. remove skip for equal label
    // 2. lambdarank inside one impresion, and score diff between impression
    // 3. re-calculate max dcg

    //const DocPairContainer* doc_pair_container = nullptr;
    //if (doc_pair_containers_ != nullptr) {
    //  doc_pair_container = doc_pair_containers_ + query_id;
    //}

    data_size_t impression_cnt = impression_cnts_ == nullptr ? 1 : impression_cnts_[query_id]; // at least one impression in one query
    if (impression_cnt > 2) {
      Log::Warning("number of impression is %d > 2", impression_cnt);
    }

    //bool has_doc_pair = doc_pair_container != nullptr && !doc_pair_container->empty();
    //bool use_scorediff = impression_cnt > 1 && has_doc_pair;
    bool use_scorediff = (impression_cnt == 2) && (cnt % 2 == 0);

    double sum_lambdas = 0;

    if (!use_scorediff) {
      // get max DCG on current query
      const double inverse_max_dcg = inverse_max_dcgs_[query_id];

      const double lambda_rank_loss_weight = 1.0;
      sum_lambdas = RankLossForOneQuery(cnt, label, score, lambdas, hessians, inverse_max_dcg, lambda_rank_loss_weight);
      
      if (norm_ && normalize_scorediff_gradient_ && sum_lambdas > 0) {
        sum_gradients_original_lambdarank_[query_id] = std::log2(1 + sum_lambdas);
      }
      else {
        sum_gradients_original_lambdarank_[query_id] = sum_lambdas;
      }
    }
    else {
      if (cnt % 2 != 0) {
        Log::Fatal("using score diff, but number of docs in query %d is %d, not an even number.", query_id, cnt);
      }

      const double lambda_rank_loss_weight = 0.5 * (1- lambda_scorediff_);
      // get max DCG on current query
      const double inverse_max_dcg1 = inverse_max_dcgs_[query_id];
      const double inverse_max_dcg2 = inverse_max_dcgs_2_[query_id];
      // rank loss for first impression
      double sum_lambdas1 = RankLossForOneQuery(cnt / 2, label, score, lambdas, hessians, inverse_max_dcg1, lambda_rank_loss_weight);
      // rank loss for second impression
      double sum_lambdas2 = RankLossForOneQuery(cnt / 2, label + cnt / 2, score + cnt / 2, lambdas + cnt / 2, hessians + cnt / 2, inverse_max_dcg2, lambda_rank_loss_weight);
      // score diff loss for doc pairs
      double sum_lambdas3 = ScoreDiffLossForOneQuery(cnt, score, lambdas, hessians);

      sum_lambdas = sum_lambdas1 + sum_lambdas2 + sum_lambdas3;

      if (norm_ && normalize_scorediff_gradient_ && sum_lambdas1 + sum_lambdas2 > 0) {
        sum_gradients_scorediff_lambdarank_[query_id] = std::log2(1 + sum_lambdas1 + sum_lambdas2);
      }
      else {
        sum_gradients_scorediff_lambdarank_[query_id] = sum_lambdas1 + sum_lambdas2;
      }

      if (norm_ && normalize_scorediff_gradient_ && sum_lambdas3 > 0) {
        sum_gradients_scorediff_[query_id] = std::log2(1 + sum_lambdas3);
      }
      else {
        sum_gradients_scorediff_[query_id] = sum_lambdas3;
      }
    }

    if (norm_ && normalize_scorediff_gradient_ && sum_lambdas > 0) {
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = static_cast<score_t>(lambdas[i] * norm_factor);
        hessians[i] = static_cast<score_t>(hessians[i] * norm_factor);
      }
    }

  }

  const char* GetName() const override { return "lambdarank_scorediff3"; }
};

/*!
 * \brief Objective function for Lambdrank with NDCG and ScoreDiff
 */
class LambdarankNDCGScoreDiff4 : public LambdarankNDCG {
  double lambda_scorediff_;
  double beta_scorediff_;
  const data_size_t* impression_cnts_;
  std::vector<double> inverse_max_dcgs_2_;
  bool normalize_scorediff_gradient_;

public:
  explicit LambdarankNDCGScoreDiff4(const Config& config)
    : LambdarankNDCG(config)
    , lambda_scorediff_(config.scorediff_alpha)
    , beta_scorediff_(config.scorediff_beta)
    , impression_cnts_(nullptr)
    , normalize_scorediff_gradient_(false) {

    inverse_max_dcgs_2_.clear();
    normalize_scorediff_gradient_ = config.normalize_gradient_scorediff;

    Log::Info("initializing LambdarankNDCGScoreDiff4");
    Log::Info("scoredif alpha = %f", lambda_scorediff_);
    Log::Info("scoredif beta = %f", beta_scorediff_);
    Log::Info("normalize_scorediff_gradient_ = %d", normalize_scorediff_gradient_);
  }

  explicit LambdarankNDCGScoreDiff4(const std::vector<std::string>& strs)
    : LambdarankNDCG(strs)
    , lambda_scorediff_(0.0)
    , beta_scorediff_(0.0)
    , impression_cnts_(nullptr)
    , normalize_scorediff_gradient_(false) {

    Log::Info("initializing LambdarankNDCGScoreDiff4 from string");
    for (const auto& s : strs) {
      Log::Info(s.c_str());
    }
    Log::Info("scoredif alpha = %f", lambda_scorediff_);
    Log::Info("scoredif beta = %f", beta_scorediff_);
  }

  ~LambdarankNDCGScoreDiff4() {
    Log::Info("uninitializing LambdarankNDCGScoreDiff4");
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    Log::Info("initialize scordiff lambdarank 4");
    LambdarankNDCG::Init(metadata, num_data);

    impression_cnts_ = metadata.impression_cnts();
    if (nullptr == impression_cnts_) {
      Log::Warning("no impression cnt is found");
    }

    // re-calculate max dcg if there are more than one impressions
    if (nullptr != impression_cnts_) {
      inverse_max_dcgs_2_.resize(num_queries_);
      for (data_size_t i = 0; i < num_queries_; ++i) {
        if (impression_cnts_[i] > 1) {
          int len = query_boundaries_[i + 1] - query_boundaries_[i];
          if (impression_cnts_[i] > 2 || len % 2 != 0) {
            Log::Warning("For query %d, it has %d impressions and %d docs. It is not supported now. Only support 2 impressions. Will calculate original rank loss on it.", i, impression_cnts_[i], len);
          }
          inverse_max_dcgs_[i] = DCGCalculator::CalMaxDCGAtK(truncation_level_, label_ + query_boundaries_[i], len / 2);

          if (inverse_max_dcgs_[i] > 0.0) {
            inverse_max_dcgs_[i] = 1.0f / inverse_max_dcgs_[i];
          }

          inverse_max_dcgs_2_[i] = DCGCalculator::CalMaxDCGAtK(truncation_level_, label_ + query_boundaries_[i] + len / 2, len / 2);

          if (inverse_max_dcgs_2_[i] > 0.0) {
            inverse_max_dcgs_2_[i] = 1.0f / inverse_max_dcgs_2_[i];
          }
        }

      }
    }
  }

  inline double RankLossForOneQuery(data_size_t cnt, const label_t* label, const double* score,
    score_t* lambdas, score_t* hessians, const double inverse_max_dcg, const double weight) const {
    // initialize with zero
    for (data_size_t i = 0; i < cnt; ++i) {
      lambdas[i] = 0.0f;
      hessians[i] = 0.0f;
    }
    // get sorted indices for scores
    std::vector<data_size_t> sorted_idx(cnt);
    for (data_size_t i = 0; i < cnt; ++i) {
      sorted_idx[i] = i;
    }
    std::stable_sort(
      sorted_idx.begin(), sorted_idx.end(),
      [score](data_size_t a, data_size_t b) { return score[a] > score[b]; });
    // get best and worst score
    const double best_score = score[sorted_idx[0]];
    data_size_t worst_idx = cnt - 1;
    if (worst_idx > 0 && score[sorted_idx[worst_idx]] == kMinScore) {
      worst_idx -= 1;
    }
    const double worst_score = score[sorted_idx[worst_idx]];
    double sum_lambdas = 0.0;

    // start accmulate lambdas by pairs that contain at least one document above truncation level
    for (data_size_t i = 0; i < cnt - 1 && i < truncation_level_; ++i) {
      if (score[sorted_idx[i]] == kMinScore) { continue; }
      for (data_size_t j = i + 1; j < cnt; ++j) {
        if (score[sorted_idx[j]] == kMinScore) { continue; }
        // skip pairs with the same labels
        if (label[sorted_idx[i]] == label[sorted_idx[j]]) { continue; }
        data_size_t high_rank, low_rank;
        if (label[sorted_idx[i]] > label[sorted_idx[j]]) {
          high_rank = i;
          low_rank = j;
        }
        else {
          high_rank = j;
          low_rank = i;
        }
        const data_size_t high = sorted_idx[high_rank];
        const int high_label = static_cast<int>(label[high]);
        const double high_score = score[high];
        const double high_label_gain = label_gain_[high_label];
        const double high_discount = DCGCalculator::GetDiscount(high_rank);
        const data_size_t low = sorted_idx[low_rank];
        const int low_label = static_cast<int>(label[low]);
        const double low_score = score[low];
        const double low_label_gain = label_gain_[low_label];
        const double low_discount = DCGCalculator::GetDiscount(low_rank);

        const double delta_score = high_score - low_score;

        // get dcg gap
        const double dcg_gap = high_label_gain - low_label_gain;
        // get discount of this pair
        const double paired_discount = fabs(high_discount - low_discount);
        // get delta NDCG
        double delta_pair_NDCG = dcg_gap * paired_discount * inverse_max_dcg;
        // regular the delta_pair_NDCG by score distance
        if (norm_ && best_score != worst_score) {
          delta_pair_NDCG /= (0.01f + fabs(delta_score));
        }

        // calculate lambda for this pair
        double p_lambda = GetSigmoid(delta_score);
        double p_hessian = p_lambda * (1.0f - p_lambda);
        // update
        p_lambda *= -sigmoid_ * delta_pair_NDCG * weight;
        p_hessian *= sigmoid_ * sigmoid_ * delta_pair_NDCG * weight;
        lambdas[low] -= static_cast<score_t>(p_lambda);
        hessians[low] += static_cast<score_t>(p_hessian);
        lambdas[high] += static_cast<score_t>(p_lambda);
        hessians[high] += static_cast<score_t>(p_hessian);
        // lambda is negative, so use minus to accumulate
        sum_lambdas -= 2 * p_lambda;
      }
    }

    // when normalize_scorediff_gradient_, the gradients for rank loss and score diff loss will be normalized together
    if (norm_ && !normalize_scorediff_gradient_ && sum_lambdas > 0) {
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = static_cast<score_t>(lambdas[i] * norm_factor);
        hessians[i] = static_cast<score_t>(hessians[i] * norm_factor);
      }
    }

    return sum_lambdas;
  }

  inline double ScoreDiffLossForOneQuery(data_size_t cnt, const double* score, score_t* lambdas, score_t* hessians) const {

    double sum_lambdas = 0;
    data_size_t half_len = cnt / 2;
    for (data_size_t i = 0; i < half_len; ++i) {
      if (score[i] == kMinScore || score[i + half_len] == kMinScore) { continue; }
      const double delta_score = score[i] - score[i + half_len];
      const double diff = beta_scorediff_ - delta_score * delta_score;
      // NOTE: it only works when sigma = 1
      const double s = GetSigmoid(diff);

      // calculate lambda for this pair
      const double p_lambda = 2 * lambda_scorediff_ * delta_score * s;
      const double p_hessian = lambda_scorediff_ * (2 * s + 4 * delta_score * delta_score * s * (1 - s));

      lambdas[i] += static_cast<score_t>(p_lambda);
      hessians[i] += static_cast<score_t>(p_hessian);
      lambdas[i + half_len] -= static_cast<score_t>(p_lambda);
      hessians[i + half_len] += static_cast<score_t>(p_hessian); // identical 2nd order gradient
      sum_lambdas += 2 * std::abs(p_lambda);
    }

    return sum_lambdas;
  }

  inline void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt,
    const label_t* label, const double* score,
    score_t* lambdas,
    score_t* hessians) const override {

    // for this loss, we assume that the records are sorted by
    //  1. query id
    //  2. impression id
    //  3. doc id


    data_size_t impression_cnt = impression_cnts_ == nullptr ? 1 : impression_cnts_[query_id]; // at least one impression in one query
    if (impression_cnt > 2) {
      Log::Warning("number of impression is %d > 2", impression_cnt);
    }

    bool use_scorediff = (impression_cnt == 2) && (cnt % 2 == 0);

    double sum_lambdas = 0;

    if (!use_scorediff) {
      // get max DCG on current query
      const double inverse_max_dcg = inverse_max_dcgs_[query_id];
      const double lambda_rank_loss_weight = 1.0;
      sum_lambdas = RankLossForOneQuery(cnt, label, score, lambdas, hessians, inverse_max_dcg, lambda_rank_loss_weight);

      if (norm_ && normalize_scorediff_gradient_ && sum_lambdas > 0) {
        sum_gradients_original_lambdarank_[query_id] = std::log2(1 + sum_lambdas);
      }
      else {
        sum_gradients_original_lambdarank_[query_id] = sum_lambdas;
      }
    }
    else {
      if (cnt % 2 != 0) {
        Log::Fatal("using score diff, but number of docs in query %d is %d, not an even number.", query_id, cnt);
      }

      const double lambda_rank_loss_weight = 0.5 * (1 - lambda_scorediff_);
      // get max DCG on current query
      const double inverse_max_dcg1 = inverse_max_dcgs_[query_id];
      const double inverse_max_dcg2 = inverse_max_dcgs_2_[query_id];
      // rank loss for first impression
      double sum_lambdas1 = RankLossForOneQuery(cnt / 2, label, score, lambdas, hessians, inverse_max_dcg1, lambda_rank_loss_weight);
      // rank loss for second impression
      double sum_lambdas2 = RankLossForOneQuery(cnt / 2, label + cnt / 2, score + cnt / 2, lambdas + cnt / 2, hessians + cnt / 2, inverse_max_dcg2, lambda_rank_loss_weight);
      // score diff loss for doc pairs
      double sum_lambdas3 = ScoreDiffLossForOneQuery(cnt, score, lambdas, hessians);

      sum_lambdas = sum_lambdas1 + sum_lambdas2 + sum_lambdas3;

      if (norm_ && normalize_scorediff_gradient_ && sum_lambdas1 + sum_lambdas2 > 0) {
        sum_gradients_scorediff_lambdarank_[query_id] = std::log2(1 + sum_lambdas1 + sum_lambdas2);
      }
      else {
        sum_gradients_scorediff_lambdarank_[query_id] = sum_lambdas1 + sum_lambdas2;
      }

      if (norm_ && normalize_scorediff_gradient_ && sum_lambdas3 > 0) {
        sum_gradients_scorediff_[query_id] = std::log2(1 + sum_lambdas3);
      }
      else {
        sum_gradients_scorediff_[query_id] = sum_lambdas3;
      }
    }

    if (norm_ && normalize_scorediff_gradient_ && sum_lambdas > 0) {
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = static_cast<score_t>(lambdas[i] * norm_factor);
        hessians[i] = static_cast<score_t>(hessians[i] * norm_factor);
      }
    }

  }

  const char* GetName() const override { return "lambdarank_scorediff4"; }
};

/*!
 * \brief Objective function for Lambdrank with NDCG and ScoreDiff
 */
class LambdarankNDCGScoreDiff5 : public LambdarankNDCG {
  double lambda_scorediff_;
  double beta_scorediff_;
  const data_size_t* impression_cnts_;
  std::vector<double> inverse_max_dcgs_2_;
  bool normalize_scorediff_gradient_;

public:
  explicit LambdarankNDCGScoreDiff5(const Config& config)
    : LambdarankNDCG(config)
    , lambda_scorediff_(config.scorediff_alpha)
    , beta_scorediff_(config.scorediff_beta)
    , impression_cnts_(nullptr)
    , normalize_scorediff_gradient_(false) {

    inverse_max_dcgs_2_.clear();
    normalize_scorediff_gradient_ = config.normalize_gradient_scorediff;

    Log::Info("initializing LambdarankNDCGScoreDiff5");
    Log::Info("scoredif alpha = %f", lambda_scorediff_);
    Log::Info("scoredif beta = %f", beta_scorediff_);
    Log::Info("normalize_scorediff_gradient_ = %d", normalize_scorediff_gradient_);
  }

  explicit LambdarankNDCGScoreDiff5(const std::vector<std::string>& strs)
    : LambdarankNDCG(strs)
    , lambda_scorediff_(0.0)
    , beta_scorediff_(0.0)
    , impression_cnts_(nullptr)
    , normalize_scorediff_gradient_(false) {

    Log::Info("initializing LambdarankNDCGScoreDiff5 from string");
    for (const auto& s : strs) {
      Log::Info(s.c_str());
    }
    Log::Info("scorediff alpha = %f", lambda_scorediff_);
    Log::Info("scorediff beta = %f", beta_scorediff_);
  }

  ~LambdarankNDCGScoreDiff5() {
    Log::Info("uninitializing LambdarankNDCGScoreDiff5");
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    Log::Info("initialize scordiff lambdarank 5");
    LambdarankNDCG::Init(metadata, num_data);

    impression_cnts_ = metadata.impression_cnts();
    if (nullptr == impression_cnts_) {
      Log::Warning("no impression cnt is found");
    }

    // re-calculate max dcg if there are more than one impressions
    if (nullptr != impression_cnts_) {
      inverse_max_dcgs_2_.resize(num_queries_);
      for (data_size_t i = 0; i < num_queries_; ++i) {
        if (impression_cnts_[i] > 1) {
          int len = query_boundaries_[i + 1] - query_boundaries_[i];
          if (impression_cnts_[i] > 2 || len % 2 != 0) {
            Log::Warning("For query %d, it has %d impressions and %d docs. It is not supported now. Only support 2 impressions. Will calculate original rank loss on it.", i, impression_cnts_[i], len);
          }
          inverse_max_dcgs_[i] = DCGCalculator::CalMaxDCGAtK(truncation_level_, label_ + query_boundaries_[i], len / 2);

          if (inverse_max_dcgs_[i] > 0.0) {
            inverse_max_dcgs_[i] = 1.0f / inverse_max_dcgs_[i];
          }

          inverse_max_dcgs_2_[i] = DCGCalculator::CalMaxDCGAtK(truncation_level_, label_ + query_boundaries_[i] + len / 2, len / 2);

          if (inverse_max_dcgs_2_[i] > 0.0) {
            inverse_max_dcgs_2_[i] = 1.0f / inverse_max_dcgs_2_[i];
          }
        }

      }
    }
  }

  inline double RankLossForOneQuery(data_size_t cnt, const label_t* label, const double* score,
    score_t* lambdas, score_t* hessians, const double inverse_max_dcg, const double weight) const {
    // initialize with zero
    for (data_size_t i = 0; i < cnt; ++i) {
      lambdas[i] = 0.0f;
      hessians[i] = 0.0f;
    }
    // get sorted indices for scores
    std::vector<data_size_t> sorted_idx(cnt);
    for (data_size_t i = 0; i < cnt; ++i) {
      sorted_idx[i] = i;
    }
    std::stable_sort(
      sorted_idx.begin(), sorted_idx.end(),
      [score](data_size_t a, data_size_t b) { return score[a] > score[b]; });
    // get best and worst score
    const double best_score = score[sorted_idx[0]];
    data_size_t worst_idx = cnt - 1;
    if (worst_idx > 0 && score[sorted_idx[worst_idx]] == kMinScore) {
      worst_idx -= 1;
    }
    const double worst_score = score[sorted_idx[worst_idx]];
    double sum_lambdas = 0.0;

    // start accmulate lambdas by pairs that contain at least one document above truncation level
    for (data_size_t i = 0; i < cnt - 1 && i < truncation_level_; ++i) {
      if (score[sorted_idx[i]] == kMinScore) { continue; }
      for (data_size_t j = i + 1; j < cnt; ++j) {
        if (score[sorted_idx[j]] == kMinScore) { continue; }
        // skip pairs with the same labels
        if (label[sorted_idx[i]] == label[sorted_idx[j]]) { continue; }
        data_size_t high_rank, low_rank;
        if (label[sorted_idx[i]] > label[sorted_idx[j]]) {
          high_rank = i;
          low_rank = j;
        }
        else {
          high_rank = j;
          low_rank = i;
        }
        const data_size_t high = sorted_idx[high_rank];
        const int high_label = static_cast<int>(label[high]);
        const double high_score = score[high];
        const double high_label_gain = label_gain_[high_label];
        const double high_discount = DCGCalculator::GetDiscount(high_rank);
        const data_size_t low = sorted_idx[low_rank];
        const int low_label = static_cast<int>(label[low]);
        const double low_score = score[low];
        const double low_label_gain = label_gain_[low_label];
        const double low_discount = DCGCalculator::GetDiscount(low_rank);

        const double delta_score = high_score - low_score;

        // get dcg gap
        const double dcg_gap = high_label_gain - low_label_gain;
        // get discount of this pair
        const double paired_discount = fabs(high_discount - low_discount);
        // get delta NDCG
        double delta_pair_NDCG = dcg_gap * paired_discount * inverse_max_dcg;
        // regular the delta_pair_NDCG by score distance
        if (norm_ && best_score != worst_score) {
          delta_pair_NDCG /= (0.01f + fabs(delta_score));
        }

        // calculate lambda for this pair
        double p_lambda = GetSigmoid(delta_score);
        double p_hessian = p_lambda * (1.0f - p_lambda);
        // update
        p_lambda *= -sigmoid_ * delta_pair_NDCG * weight;
        p_hessian *= sigmoid_ * sigmoid_ * delta_pair_NDCG * weight;
        lambdas[low] -= static_cast<score_t>(p_lambda);
        hessians[low] += static_cast<score_t>(p_hessian);
        lambdas[high] += static_cast<score_t>(p_lambda);
        hessians[high] += static_cast<score_t>(p_hessian);
        // lambda is negative, so use minus to accumulate
        sum_lambdas -= 2 * p_lambda;
      }
    }

    // when normalize_scorediff_gradient_, the gradients for rank loss and score diff loss will be normalized together
    if (norm_ && !normalize_scorediff_gradient_ && sum_lambdas > 0) {
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = static_cast<score_t>(lambdas[i] * norm_factor);
        hessians[i] = static_cast<score_t>(hessians[i] * norm_factor);
      }
    }

    return sum_lambdas;
  }

  inline double ScoreDiffLossForOneQuery(data_size_t cnt, const label_t* label, const double* score, score_t* lambdas, score_t* hessians) const {
    data_size_t half_len = cnt / 2;

    // get sorted indices for scores
    std::vector<data_size_t> sorted_idx(half_len);
    for (data_size_t i = 0; i < half_len; ++i) {
      sorted_idx[i] = i;
    }
    std::stable_sort(
      sorted_idx.begin(), sorted_idx.end(),
      [label](data_size_t a, data_size_t b) { return label[a] > label[b]; });

    double sum_lambdas = 0;
    for (data_size_t i = 0; i < half_len; ++i) {
      int idx = sorted_idx[i];
      if (score[idx] == kMinScore || score[idx + half_len] == kMinScore) { continue; }
      const double delta_score = score[idx] - score[idx + half_len];
      const double diff = beta_scorediff_ - delta_score * delta_score;
      // NOTE: it only works when sigma = 1
      const double s = GetSigmoid(diff);

      // calculate lambda for this pair
      const double p_lambda = 2 * lambda_scorediff_ * delta_score * s;
      const double p_hessian = lambda_scorediff_ * (2 * s + 4 * delta_score * delta_score * s * (1 - s));

      //bool lower_than_top_1 = true, higher_than_bottom_1 = true, lower_than_top_2 = true, higher_than_bottom_2 = true;
      //if (i > 0 && score[idx] > score[sorted_idx[i - 1]]) {
      //  lower_than_top_1 = false;
      //}
      //if (i < half_len - 1 && score[idx] < score[sorted_idx[i + 1]]) {
      //  higher_than_bottom_1 = false;
      //}
      //if (i > 0 && score[idx + half_len] > score[sorted_idx[i - 1] + half_len]) {
      //  lower_than_top_2 = false;
      //}
      //if (i < half_len - 1 && score[idx + half_len] < score[sorted_idx[i + 1] + half_len]) {
      //  higher_than_bottom_2 = false;
      //}
      //bool update1 = true, update2 = true;
      //if (lower_than_top_1 == higher_than_bottom_1) {
      //  update1 = false;
      //}
      //if (lower_than_top_2 == higher_than_bottom_2) {
      //  update2 = false;
      //}

      bool need_update_1 = (delta_score < 0 && i < half_len - 1 && score[idx] < score[sorted_idx[i + 1]]) ||
        (delta_score > 0 && i >= 1 && score[idx] > score[sorted_idx[i - 1]]);
      bool need_update_2 = (delta_score < 0 && i >= 1 && score[idx + half_len] > score[sorted_idx[i - 1] + half_len]) ||
        (delta_score > 0 && i < half_len - 1 && score[idx + half_len] < score[sorted_idx[i + 1] + half_len]);

      if (need_update_1) {
        lambdas[idx] += static_cast<score_t>(p_lambda);
        hessians[idx] += static_cast<score_t>(p_hessian);
        sum_lambdas += std::abs(p_lambda);
      }
      if (need_update_2) {
        lambdas[idx + half_len] -= static_cast<score_t>(p_lambda);
        hessians[idx + half_len] += static_cast<score_t>(p_hessian); // identical 2nd order gradient
        sum_lambdas += std::abs(p_lambda);
      }
    }

    return sum_lambdas;
  }

  inline void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt,
    const label_t* label, const double* score,
    score_t* lambdas,
    score_t* hessians) const override {

    // for this loss, we assume that the records are sorted by
    //  1. query id
    //  2. impression id
    //  3. doc id


    data_size_t impression_cnt = impression_cnts_ == nullptr ? 1 : impression_cnts_[query_id]; // at least one impression in one query
    if (impression_cnt > 2) {
      Log::Warning("number of impression is %d > 2", impression_cnt);
    }

    bool use_scorediff = (impression_cnt == 2) && (cnt % 2 == 0);

    double sum_lambdas = 0;

    if (!use_scorediff) {
      // get max DCG on current query
      const double inverse_max_dcg = inverse_max_dcgs_[query_id];
      const double lambda_rank_loss_weight = 1.0;
      sum_lambdas = RankLossForOneQuery(cnt, label, score, lambdas, hessians, inverse_max_dcg, lambda_rank_loss_weight);

      if (norm_ && normalize_scorediff_gradient_ && sum_lambdas > 0) {
        sum_gradients_original_lambdarank_[query_id] = std::log2(1 + sum_lambdas);
      }
      else {
        sum_gradients_original_lambdarank_[query_id] = sum_lambdas;
      }
    }
    else {
      if (cnt % 2 != 0) {
        Log::Fatal("using score diff, but number of docs in query %d is %d, not an even number.", query_id, cnt);
      }

      const double lambda_rank_loss_weight = 0.5 * (1 - lambda_scorediff_);
      // get max DCG on current query
      const double inverse_max_dcg1 = inverse_max_dcgs_[query_id];
      const double inverse_max_dcg2 = inverse_max_dcgs_2_[query_id];
      // rank loss for first impression
      double sum_lambdas1 = RankLossForOneQuery(cnt / 2, label, score, lambdas, hessians, inverse_max_dcg1, lambda_rank_loss_weight);
      // rank loss for second impression
      double sum_lambdas2 = RankLossForOneQuery(cnt / 2, label + cnt / 2, score + cnt / 2, lambdas + cnt / 2, hessians + cnt / 2, inverse_max_dcg2, lambda_rank_loss_weight);
      // score diff loss for doc pairs
      double sum_lambdas3 = ScoreDiffLossForOneQuery(cnt, label, score, lambdas, hessians);

      sum_lambdas = sum_lambdas1 + sum_lambdas2 + sum_lambdas3;

      if (norm_ && normalize_scorediff_gradient_ && sum_lambdas1 + sum_lambdas2 > 0) {
        sum_gradients_scorediff_lambdarank_[query_id] = std::log2(1 + sum_lambdas1 + sum_lambdas2);
      }
      else {
        sum_gradients_scorediff_lambdarank_[query_id] = sum_lambdas1 + sum_lambdas2;
      }

      if (norm_ && normalize_scorediff_gradient_ && sum_lambdas3 > 0) {
        sum_gradients_scorediff_[query_id] = std::log2(1 + sum_lambdas3);
      }
      else {
        sum_gradients_scorediff_[query_id] = sum_lambdas3;
      }
    }

    if (norm_ && normalize_scorediff_gradient_ && sum_lambdas > 0) {
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = static_cast<score_t>(lambdas[i] * norm_factor);
        hessians[i] = static_cast<score_t>(hessians[i] * norm_factor);
      }
    }

  }

  const char* GetName() const override { return "lambdarank_scorediff5"; }
};


/*!
 * \brief Objective function for Lambdrank with NDCG and ScoreDiff
 */
class LambdarankNDCGScoreDiff6 : public LambdarankNDCG {
  double lambda_scorediff_;
  double beta_scorediff_;
  const data_size_t* impression_cnts_;
  std::vector<double> inverse_max_dcgs_2_;
  bool normalize_scorediff_gradient_;

public:
  explicit LambdarankNDCGScoreDiff6(const Config& config)
    : LambdarankNDCG(config)
    , lambda_scorediff_(config.scorediff_alpha)
    , beta_scorediff_(config.scorediff_beta)
    , impression_cnts_(nullptr)
    , normalize_scorediff_gradient_(false) {

    inverse_max_dcgs_2_.clear();
    normalize_scorediff_gradient_ = config.normalize_gradient_scorediff;

    Log::Info("initializing LambdarankNDCGScoreDiff6");
    Log::Info("scorediff alpha = %f", lambda_scorediff_);
    Log::Info("scorediff beta = %f", beta_scorediff_);
    Log::Info("normalize_scorediff_gradient_ = %d", normalize_scorediff_gradient_);
  }

  explicit LambdarankNDCGScoreDiff6(const std::vector<std::string>& strs)
    : LambdarankNDCG(strs)
    , lambda_scorediff_(0.0)
    , beta_scorediff_(0.0)
    , impression_cnts_(nullptr)
    , normalize_scorediff_gradient_(false) {

    Log::Info("initializing LambdarankNDCGScoreDiff6 from string");
    for (const auto& s : strs) {
      Log::Info(s.c_str());
    }
    Log::Info("scorediff alpha = %f", lambda_scorediff_);
    Log::Info("scorediff beta = %f", beta_scorediff_);
  }

  ~LambdarankNDCGScoreDiff6() {
    Log::Info("uninitializing LambdarankNDCGScoreDiff6");
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    Log::Info("initialize scordiff lambdarank 6");
    LambdarankNDCG::Init(metadata, num_data);

    impression_cnts_ = metadata.impression_cnts();
    if (nullptr == impression_cnts_) {
      Log::Warning("no impression cnt is found");
    }

    // re-calculate max dcg if there are more than one impressions
    if (nullptr != impression_cnts_) {
      inverse_max_dcgs_2_.resize(num_queries_);
      for (data_size_t i = 0; i < num_queries_; ++i) {
        if (impression_cnts_[i] > 1) {
          int len = query_boundaries_[i + 1] - query_boundaries_[i];
          if (impression_cnts_[i] > 2 || len % 2 != 0) {
            Log::Warning("For query %d, it has %d impressions and %d docs. It is not supported now. Only support 2 impressions. Will calculate original rank loss on it.", i, impression_cnts_[i], len);
          }
          inverse_max_dcgs_[i] = DCGCalculator::CalMaxDCGAtK(truncation_level_, label_ + query_boundaries_[i], len / 2);

          if (inverse_max_dcgs_[i] > 0.0) {
            inverse_max_dcgs_[i] = 1.0f / inverse_max_dcgs_[i];
          }

          inverse_max_dcgs_2_[i] = DCGCalculator::CalMaxDCGAtK(truncation_level_, label_ + query_boundaries_[i] + len / 2, len / 2);

          if (inverse_max_dcgs_2_[i] > 0.0) {
            inverse_max_dcgs_2_[i] = 1.0f / inverse_max_dcgs_2_[i];
          }
        }

      }
    }
  }

  inline double RankLossForOneQuery(data_size_t cnt, const label_t* label, const double* score,
    score_t* lambdas, score_t* hessians, const double inverse_max_dcg, const double weight) const {
    // initialize with zero
    for (data_size_t i = 0; i < cnt; ++i) {
      lambdas[i] = 0.0f;
      hessians[i] = 0.0f;
    }
    // get sorted indices for scores
    std::vector<data_size_t> sorted_idx(cnt);
    for (data_size_t i = 0; i < cnt; ++i) {
      sorted_idx[i] = i;
    }
    std::stable_sort(
      sorted_idx.begin(), sorted_idx.end(),
      [score](data_size_t a, data_size_t b) { return score[a] > score[b]; });
    // get best and worst score
    const double best_score = score[sorted_idx[0]];
    data_size_t worst_idx = cnt - 1;
    if (worst_idx > 0 && score[sorted_idx[worst_idx]] == kMinScore) {
      worst_idx -= 1;
    }
    const double worst_score = score[sorted_idx[worst_idx]];
    double sum_lambdas = 0.0;

    // start accmulate lambdas by pairs that contain at least one document above truncation level
    for (data_size_t i = 0; i < cnt - 1 && i < truncation_level_; ++i) {
      if (score[sorted_idx[i]] == kMinScore) { continue; }
      for (data_size_t j = i + 1; j < cnt; ++j) {
        if (score[sorted_idx[j]] == kMinScore) { continue; }
        // skip pairs with the same labels
        if (label[sorted_idx[i]] == label[sorted_idx[j]]) { continue; }
        data_size_t high_rank, low_rank;
        if (label[sorted_idx[i]] > label[sorted_idx[j]]) {
          high_rank = i;
          low_rank = j;
        }
        else {
          high_rank = j;
          low_rank = i;
        }
        const data_size_t high = sorted_idx[high_rank];
        const int high_label = static_cast<int>(label[high]);
        const double high_score = score[high];
        const double high_label_gain = label_gain_[high_label];
        const double high_discount = DCGCalculator::GetDiscount(high_rank);
        const data_size_t low = sorted_idx[low_rank];
        const int low_label = static_cast<int>(label[low]);
        const double low_score = score[low];
        const double low_label_gain = label_gain_[low_label];
        const double low_discount = DCGCalculator::GetDiscount(low_rank);

        const double delta_score = high_score - low_score;

        // get dcg gap
        const double dcg_gap = high_label_gain - low_label_gain;
        // get discount of this pair
        const double paired_discount = fabs(high_discount - low_discount);
        // get delta NDCG
        double delta_pair_NDCG = dcg_gap * paired_discount * inverse_max_dcg;
        // regular the delta_pair_NDCG by score distance
        if (norm_ && best_score != worst_score) {
          delta_pair_NDCG /= (0.01f + fabs(delta_score));
        }

        // calculate lambda for this pair
        double p_lambda = GetSigmoid(delta_score);
        double p_hessian = p_lambda * (1.0f - p_lambda);
        // update
        p_lambda *= -sigmoid_ * delta_pair_NDCG * weight;
        p_hessian *= sigmoid_ * sigmoid_ * delta_pair_NDCG * weight;
        lambdas[low] -= static_cast<score_t>(p_lambda);
        hessians[low] += static_cast<score_t>(p_hessian);
        lambdas[high] += static_cast<score_t>(p_lambda);
        hessians[high] += static_cast<score_t>(p_hessian);
        // lambda is negative, so use minus to accumulate
        sum_lambdas -= 2 * p_lambda;
      }
    }

    // when normalize_scorediff_gradient_, the gradients for rank loss and score diff loss will be normalized together
    if (norm_ && !normalize_scorediff_gradient_ && sum_lambdas > 0) {
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = static_cast<score_t>(lambdas[i] * norm_factor);
        hessians[i] = static_cast<score_t>(hessians[i] * norm_factor);
      }
    }

    return sum_lambdas;
  }

  inline double ScoreDiffLossForOneQuery(data_size_t cnt, const label_t* label, const double* score, score_t* lambdas, score_t* hessians) const {
    data_size_t half_len = cnt / 2;

    // get sorted indices for scores
    std::vector<data_size_t> sorted_idx(half_len);
    for (data_size_t i = 0; i < half_len; ++i) {
      sorted_idx[i] = i;
    }
    std::stable_sort(
      sorted_idx.begin(), sorted_idx.end(),
      [label](data_size_t a, data_size_t b) { return label[a] > label[b]; });

    double sum_lambdas = 0;
    for (data_size_t i = 0; i < half_len; ++i) {
      int idx = sorted_idx[i];
      if (score[idx] == kMinScore || score[idx + half_len] == kMinScore) { continue; }
      const double delta_score = score[idx] - score[idx + half_len];
      const double diff = beta_scorediff_ - delta_score * delta_score;
      // NOTE: it only works when sigma = 1
      const double s = GetSigmoid(diff);

      // calculate lambda for this pair
      const double p_lambda = 2 * lambda_scorediff_ * delta_score * s;
      const double p_hessian = lambda_scorediff_ * (2 * s + 4 * delta_score * delta_score * s * (1 - s));

      bool need_update_1 = (delta_score < 0 && i < half_len - 1 && score[idx] < score[sorted_idx[i + 1]]) ||
        (delta_score > 0 && i >= 1 && score[idx] > score[sorted_idx[i - 1]]);
      bool need_update_2 = (delta_score < 0 && i >= 1 && score[idx + half_len] > score[sorted_idx[i - 1] + half_len]) ||
        (delta_score > 0 && i < half_len - 1 && score[idx + half_len] < score[sorted_idx[i + 1] + half_len]);

      if (need_update_1) {
        lambdas[idx] += static_cast<score_t>(p_lambda);
        hessians[idx] += static_cast<score_t>(p_hessian);
        sum_lambdas += std::abs(p_lambda);
      }
      if (need_update_2) {
        lambdas[idx + half_len] -= static_cast<score_t>(p_lambda);
        hessians[idx + half_len] += static_cast<score_t>(p_hessian); // identical 2nd order gradient
        sum_lambdas += std::abs(p_lambda);
      }
    }

    return sum_lambdas;
  }

  inline void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt,
    const label_t* label, const double* score,
    score_t* lambdas,
    score_t* hessians) const override {

    // for this loss, we assume that the records are sorted by
    //  1. query id
    //  2. impression id
    //  3. doc id


    data_size_t impression_cnt = impression_cnts_ == nullptr ? 1 : impression_cnts_[query_id]; // at least one impression in one query
    if (impression_cnt > 2) {
      Log::Warning("number of impression is %d > 2", impression_cnt);
    }

    bool use_scorediff = (impression_cnt == 2) && (cnt % 2 == 0);

    double sum_lambdas = 0;

    if (!use_scorediff) {
      // get max DCG on current query
      const double inverse_max_dcg = inverse_max_dcgs_[query_id];
      const double lambda_rank_loss_weight = 1.0;
      sum_lambdas = RankLossForOneQuery(cnt, label, score, lambdas, hessians, inverse_max_dcg, lambda_rank_loss_weight);

      if (norm_ && normalize_scorediff_gradient_ && sum_lambdas > 0) {
        sum_gradients_original_lambdarank_[query_id] = std::log2(1 + sum_lambdas);
      }
      else {
        sum_gradients_original_lambdarank_[query_id] = sum_lambdas;
      }
    }
    else {
      if (cnt % 2 != 0) {
        Log::Fatal("using score diff, but number of docs in query %d is %d, not an even number.", query_id, cnt);
      }

      const double lambda_rank_loss_weight = 1.0;
      // get max DCG on current query
      const double inverse_max_dcg1 = inverse_max_dcgs_[query_id];
      const double inverse_max_dcg2 = inverse_max_dcgs_2_[query_id];
      // rank loss for first impression
      double sum_lambdas1 = RankLossForOneQuery(cnt / 2, label, score, lambdas, hessians, inverse_max_dcg1, lambda_rank_loss_weight);
      // rank loss for second impression
      double sum_lambdas2 = RankLossForOneQuery(cnt / 2, label + cnt / 2, score + cnt / 2, lambdas + cnt / 2, hessians + cnt / 2, inverse_max_dcg2, lambda_rank_loss_weight);
      // score diff loss for doc pairs
      double sum_lambdas3 = ScoreDiffLossForOneQuery(cnt, label, score, lambdas, hessians);

      sum_lambdas = sum_lambdas1 + sum_lambdas2 + sum_lambdas3;

      if (norm_ && normalize_scorediff_gradient_ && sum_lambdas1 + sum_lambdas2 > 0) {
        sum_gradients_scorediff_lambdarank_[query_id] = std::log2(1 + sum_lambdas1 + sum_lambdas2);
      }
      else {
        sum_gradients_scorediff_lambdarank_[query_id] = sum_lambdas1 + sum_lambdas2;
      }

      if (norm_ && normalize_scorediff_gradient_ && sum_lambdas3 > 0) {
        sum_gradients_scorediff_[query_id] = std::log2(1 + sum_lambdas3);
      }
      else {
        sum_gradients_scorediff_[query_id] = sum_lambdas3;
      }
    }

    if (norm_ && normalize_scorediff_gradient_ && sum_lambdas > 0) {
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = static_cast<score_t>(lambdas[i] * norm_factor);
        hessians[i] = static_cast<score_t>(hessians[i] * norm_factor);
      }
    }

  }

  const char* GetName() const override { return "lambdarank_scorediff6"; }
};


/*!
 * \brief Objective function for Lambdrank with NDCG and RankDiff
 */
class LambdarankNDCGRankDiff : public LambdarankNDCG {
  double lambda_scorediff_;
  const data_size_t* impression_cnts_;
  std::vector<double> inverse_max_dcgs_2_;

public:
  explicit LambdarankNDCGRankDiff(const Config& config)
    : LambdarankNDCG(config)
    , lambda_scorediff_(config.scorediff_alpha)
    , impression_cnts_(nullptr) {

    inverse_max_dcgs_2_.clear();

    Log::Info("initializing LambdarankNDCGRankDiff");
    Log::Info("scoredif lambda = %f", lambda_scorediff_);
  }

  explicit LambdarankNDCGRankDiff(const std::vector<std::string>& strs)
    : LambdarankNDCG(strs)
    , lambda_scorediff_(1.0)
    , impression_cnts_(nullptr) {

    Log::Info("initializing LambdarankNDCGRankDiff from string");
    for (const auto& s : strs) {
      Log::Info(s.c_str());
    }
    Log::Info("scoredif lambda = %f", lambda_scorediff_);
  }

  ~LambdarankNDCGRankDiff() {
    Log::Info("uninitializing LambdarankNDCGRankDiff");
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    Log::Info("initialize rankdiff lambdarank");
    LambdarankNDCG::Init(metadata, num_data);

    impression_cnts_ = metadata.impression_cnts();
    if (nullptr == impression_cnts_) {
      Log::Warning("no impression cnt is found");
    }

    // re-calculate max dcg if there are more than one impressions
    if (nullptr != impression_cnts_) {
      inverse_max_dcgs_2_.resize(num_queries_);
      for (data_size_t i = 0; i < num_queries_; ++i) {
        if (impression_cnts_[i] > 1) {
          int len = query_boundaries_[i + 1] - query_boundaries_[i];
          if (impression_cnts_[i] > 2 || len % 2 != 0) {
            Log::Warning("For query %d, it has %d impressions and %d docs. It is not supported now. Only support 2 impressions. Will calculate original rank loss on it.", i, impression_cnts_[i], len);
          }
          inverse_max_dcgs_[i] = DCGCalculator::CalMaxDCGAtK(truncation_level_, label_ + query_boundaries_[i], len / 2);

          if (inverse_max_dcgs_[i] > 0.0) {
            inverse_max_dcgs_[i] = 1.0f / inverse_max_dcgs_[i];
          }

          inverse_max_dcgs_2_[i] = DCGCalculator::CalMaxDCGAtK(truncation_level_, label_ + query_boundaries_[i] + len / 2, len / 2);

          if (inverse_max_dcgs_2_[i] > 0.0) {
            inverse_max_dcgs_2_[i] = 1.0f / inverse_max_dcgs_2_[i];
          }
        }

      }
    }
  }

  inline double RankLossForOneQuery(data_size_t cnt, const label_t* label, const double* score,
    score_t* lambdas, score_t* hessians, const double inverse_max_dcg) const {
    // initialize with zero
    for (data_size_t i = 0; i < cnt; ++i) {
      lambdas[i] = 0.0f;
      hessians[i] = 0.0f;
    }
    // get sorted indices for scores
    std::vector<data_size_t> sorted_idx(cnt);
    for (data_size_t i = 0; i < cnt; ++i) {
      sorted_idx[i] = i;
    }
    std::stable_sort(
      sorted_idx.begin(), sorted_idx.end(),
      [score](data_size_t a, data_size_t b) { return score[a] > score[b]; });
    // get best and worst score
    const double best_score = score[sorted_idx[0]];
    data_size_t worst_idx = cnt - 1;
    if (worst_idx > 0 && score[sorted_idx[worst_idx]] == kMinScore) {
      worst_idx -= 1;
    }
    const double worst_score = score[sorted_idx[worst_idx]];
    double sum_lambdas = 0.0;

    // start accmulate lambdas by pairs that contain at least one document above truncation level
    for (data_size_t i = 0; i < cnt - 1 && i < truncation_level_; ++i) {
      if (score[sorted_idx[i]] == kMinScore) { continue; }
      for (data_size_t j = i + 1; j < cnt; ++j) {
        if (score[sorted_idx[j]] == kMinScore) { continue; }
        // skip pairs with the same labels
        if (label[sorted_idx[i]] == label[sorted_idx[j]]) { continue; }
        data_size_t high_rank, low_rank;
        if (label[sorted_idx[i]] > label[sorted_idx[j]]) {
          high_rank = i;
          low_rank = j;
        }
        else {
          high_rank = j;
          low_rank = i;
        }
        const data_size_t high = sorted_idx[high_rank];
        const int high_label = static_cast<int>(label[high]);
        const double high_score = score[high];
        const double high_label_gain = label_gain_[high_label];
        const double high_discount = DCGCalculator::GetDiscount(high_rank);
        const data_size_t low = sorted_idx[low_rank];
        const int low_label = static_cast<int>(label[low]);
        const double low_score = score[low];
        const double low_label_gain = label_gain_[low_label];
        const double low_discount = DCGCalculator::GetDiscount(low_rank);

        const double delta_score = high_score - low_score;

        // get dcg gap
        const double dcg_gap = high_label_gain - low_label_gain;
        // get discount of this pair
        const double paired_discount = fabs(high_discount - low_discount);
        // get delta NDCG
        double delta_pair_NDCG = dcg_gap * paired_discount * inverse_max_dcg;
        // regular the delta_pair_NDCG by score distance
        if (norm_ && best_score != worst_score) {
          delta_pair_NDCG /= (0.01f + fabs(delta_score));
        }

        // calculate lambda for this pair
        double p_lambda = GetSigmoid(delta_score);
        double p_hessian = p_lambda * (1.0f - p_lambda);
        // update
        p_lambda *= -sigmoid_ * delta_pair_NDCG;
        p_hessian *= sigmoid_ * sigmoid_ * delta_pair_NDCG;
        lambdas[low] -= static_cast<score_t>(p_lambda);
        hessians[low] += static_cast<score_t>(p_hessian);
        lambdas[high] += static_cast<score_t>(p_lambda);
        hessians[high] += static_cast<score_t>(p_hessian);
        // lambda is negative, so use minus to accumulate
        sum_lambdas -= 2 * p_lambda;
      }
    }

    return sum_lambdas;
  }

  inline double RankDiffLossForOneQuery(data_size_t cnt, const double* score, score_t* lambdas, score_t* hessians) const {

    double sum_lambdas = 0;
    data_size_t half_len = cnt / 2;
    for (data_size_t i = 0; i < half_len - 1; ++i) {
      for (data_size_t j = i + 1; j < half_len; ++j) {
        if (score[i] == kMinScore ||
          score[i + half_len] == kMinScore ||
          score[j] == kMinScore ||
          score[j + half_len] == kMinScore) { continue; }

        const double delta_score1 = score[i] - score[j];
        const double delta_score2 = score[i + half_len] - score[j + half_len];


        lambdas[i] -= static_cast<score_t>(delta_score2);
        lambdas[j] += static_cast<score_t>(delta_score2);
        lambdas[i + half_len] -= static_cast<score_t>(delta_score1);
        lambdas[j + half_len] += static_cast<score_t>(delta_score1);

        sum_lambdas += 2 * std::abs(delta_score1);
        sum_lambdas += 2 * std::abs(delta_score2);

        // 2nd order gradient is zero
      }

    }

    return sum_lambdas;
  }

  inline void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt,
    const label_t* label, const double* score,
    score_t* lambdas,
    score_t* hessians) const override {

    // for this loss, we assume that the records are sorted by
    //  1. query id
    //  2. impression id
    //  3. doc id

    // loss for score diff
    // 1. remove skip for equal label
    // 2. lambdarank inside one impresion, and rank diff between impression
    // 3. re-calculate max dcg

    data_size_t impression_cnt = impression_cnts_ == nullptr ? 1 : impression_cnts_[query_id]; // at least one impression in one query
    if (impression_cnt > 2) {
      Log::Warning("number of impression is %d > 2", impression_cnt);
    }

    bool use_scorediff = (impression_cnt == 2) && (cnt % 2 == 0);

    double sum_lambdas = 0;

    if (!use_scorediff) {
      // get max DCG on current query
      const double inverse_max_dcg = inverse_max_dcgs_[query_id];

      sum_lambdas = RankLossForOneQuery(cnt, label, score, lambdas, hessians, inverse_max_dcg);

    }
    else {
      if (cnt % 2 != 0) {
        Log::Fatal("using score diff, but number of docs in query %d is %d, not an even number.", query_id, cnt);
      }
      // get max DCG on current query
      const double inverse_max_dcg1 = inverse_max_dcgs_[query_id];
      const double inverse_max_dcg2 = inverse_max_dcgs_2_[query_id];
      // rank loss for first impression
      double sum_lambdas1 = RankLossForOneQuery(cnt / 2, label, score, lambdas, hessians, inverse_max_dcg1);
      // rank loss for second impression
      double sum_lambdas2 = RankLossForOneQuery(cnt / 2, label + cnt / 2, score + cnt / 2, lambdas + cnt / 2, hessians + cnt / 2, inverse_max_dcg2);
      // score diff loss for doc pairs
      double sum_lambdas3 = RankDiffLossForOneQuery(cnt, score, lambdas, hessians);

      sum_lambdas = sum_lambdas1 + sum_lambdas2 + sum_lambdas3;
    }

    if (norm_ && sum_lambdas > 0) {
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = static_cast<score_t>(lambdas[i] * norm_factor);
        hessians[i] = static_cast<score_t>(hessians[i] * norm_factor);
      }
    }

  }

  const char* GetName() const override { return "lambdarank_rankdiff"; }
};

/*!
 * \brief Objective function for Lambdrank with NDCG and RankDiff 2
 */
class LambdarankNDCGRankDiff2 : public LambdarankNDCG {
  double lambda_scorediff_;
  const data_size_t* impression_cnts_;
  std::vector<double> inverse_max_dcgs_2_;
  bool normalize_scorediff_gradient_;

public:
  explicit LambdarankNDCGRankDiff2(const Config& config)
    : LambdarankNDCG(config)
    , lambda_scorediff_(config.scorediff_alpha)
    , impression_cnts_(nullptr)
    , normalize_scorediff_gradient_(false) {

    inverse_max_dcgs_2_.clear();

    normalize_scorediff_gradient_ = config.normalize_gradient_scorediff;

    Log::Info("initializing LambdarankNDCGRankDiff2");
    Log::Info("scoredif lambda = %f", lambda_scorediff_);
    Log::Info("normalize_scorediff_gradient_ = %d", normalize_scorediff_gradient_);
  }

  explicit LambdarankNDCGRankDiff2(const std::vector<std::string>& strs)
    : LambdarankNDCG(strs)
    , lambda_scorediff_(1.0)
    , impression_cnts_(nullptr)
    , normalize_scorediff_gradient_(false) {

    Log::Info("initializing LambdarankNDCGRankDiff2 from string");
    for (const auto& s : strs) {
      Log::Info(s.c_str());
    }
    Log::Info("scoredif lambda = %f", lambda_scorediff_);
    Log::Info("normalize_scorediff_gradient_ = %d", normalize_scorediff_gradient_);
  }

  ~LambdarankNDCGRankDiff2() {
    Log::Info("uninitializing LambdarankNDCGRankDiff2");
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    Log::Info("initialize rankdiff lambdarank 2");
    LambdarankNDCG::Init(metadata, num_data);

    impression_cnts_ = metadata.impression_cnts();
    if (nullptr == impression_cnts_) {
      Log::Warning("no impression cnt is found");
    }

    // re-calculate max dcg if there are more than one impressions
    if (nullptr != impression_cnts_) {
      inverse_max_dcgs_2_.resize(num_queries_);
      for (data_size_t i = 0; i < num_queries_; ++i) {
        if (impression_cnts_[i] > 1) {
          int len = query_boundaries_[i + 1] - query_boundaries_[i];
          if (impression_cnts_[i] > 2 || len % 2 != 0) {
            Log::Warning("For query %d, it has %d impressions and %d docs. It is not supported now. Only support 2 impressions. Will calculate original rank loss on it.", i, impression_cnts_[i], len);
          }
          inverse_max_dcgs_[i] = DCGCalculator::CalMaxDCGAtK(truncation_level_, label_ + query_boundaries_[i], len / 2);

          if (inverse_max_dcgs_[i] > 0.0) {
            inverse_max_dcgs_[i] = 1.0f / inverse_max_dcgs_[i];
          }

          inverse_max_dcgs_2_[i] = DCGCalculator::CalMaxDCGAtK(truncation_level_, label_ + query_boundaries_[i] + len / 2, len / 2);

          if (inverse_max_dcgs_2_[i] > 0.0) {
            inverse_max_dcgs_2_[i] = 1.0f / inverse_max_dcgs_2_[i];
          }
        }

      }
    }
  }

  inline double RankLossForOneQuery(data_size_t cnt, const label_t* label, const double* score,
    score_t* lambdas, score_t* hessians, const double inverse_max_dcg) const {
    // initialize with zero
    for (data_size_t i = 0; i < cnt; ++i) {
      lambdas[i] = 0.0f;
      hessians[i] = 0.0f;
    }
    // get sorted indices for scores
    std::vector<data_size_t> sorted_idx(cnt);
    for (data_size_t i = 0; i < cnt; ++i) {
      sorted_idx[i] = i;
    }
    std::stable_sort(
      sorted_idx.begin(), sorted_idx.end(),
      [score](data_size_t a, data_size_t b) { return score[a] > score[b]; });
    // get best and worst score
    const double best_score = score[sorted_idx[0]];
    data_size_t worst_idx = cnt - 1;
    if (worst_idx > 0 && score[sorted_idx[worst_idx]] == kMinScore) {
      worst_idx -= 1;
    }
    const double worst_score = score[sorted_idx[worst_idx]];
    double sum_lambdas = 0.0;

    // start accmulate lambdas by pairs that contain at least one document above truncation level
    for (data_size_t i = 0; i < cnt - 1 && i < truncation_level_; ++i) {
      if (score[sorted_idx[i]] == kMinScore) { continue; }
      for (data_size_t j = i + 1; j < cnt; ++j) {
        if (score[sorted_idx[j]] == kMinScore) { continue; }
        // skip pairs with the same labels
        if (label[sorted_idx[i]] == label[sorted_idx[j]]) { continue; }
        data_size_t high_rank, low_rank;
        if (label[sorted_idx[i]] > label[sorted_idx[j]]) {
          high_rank = i;
          low_rank = j;
        }
        else {
          high_rank = j;
          low_rank = i;
        }
        const data_size_t high = sorted_idx[high_rank];
        const int high_label = static_cast<int>(label[high]);
        const double high_score = score[high];
        const double high_label_gain = label_gain_[high_label];
        const double high_discount = DCGCalculator::GetDiscount(high_rank);
        const data_size_t low = sorted_idx[low_rank];
        const int low_label = static_cast<int>(label[low]);
        const double low_score = score[low];
        const double low_label_gain = label_gain_[low_label];
        const double low_discount = DCGCalculator::GetDiscount(low_rank);

        const double delta_score = high_score - low_score;

        // get dcg gap
        const double dcg_gap = high_label_gain - low_label_gain;
        // get discount of this pair
        const double paired_discount = fabs(high_discount - low_discount);
        // get delta NDCG
        double delta_pair_NDCG = dcg_gap * paired_discount * inverse_max_dcg;
        // regular the delta_pair_NDCG by score distance
        if (norm_ && best_score != worst_score) {
          delta_pair_NDCG /= (0.01f + fabs(delta_score));
        }

        // calculate lambda for this pair
        double p_lambda = GetSigmoid(delta_score);
        double p_hessian = p_lambda * (1.0f - p_lambda);
        // update
        p_lambda *= -sigmoid_ * delta_pair_NDCG;
        p_hessian *= sigmoid_ * sigmoid_ * delta_pair_NDCG;
        lambdas[low] -= static_cast<score_t>(p_lambda);
        hessians[low] += static_cast<score_t>(p_hessian);
        lambdas[high] += static_cast<score_t>(p_lambda);
        hessians[high] += static_cast<score_t>(p_hessian);
        // lambda is negative, so use minus to accumulate
        sum_lambdas -= 2 * p_lambda;
      }
    }

    // when normalize_scorediff_gradient_, the gradients for rank loss and score diff loss will be normalized together
    if (norm_ && !normalize_scorediff_gradient_ && sum_lambdas > 0) {
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = static_cast<score_t>(lambdas[i] * norm_factor);
        hessians[i] = static_cast<score_t>(hessians[i] * norm_factor);
      }
    }

    return sum_lambdas;
  }

  inline double RankDiffLossForOneQuery(data_size_t cnt, const double* score, score_t* lambdas, score_t* hessians) const {

    double sum_lambdas = 0;
    data_size_t half_len = cnt / 2;
    for (data_size_t i = 0; i < half_len - 1; ++i) {
      for (data_size_t j = i + 1; j < half_len; ++j) {
        if (score[i] == kMinScore ||
          score[i + half_len] == kMinScore ||
          score[j] == kMinScore ||
          score[j + half_len] == kMinScore) {
          continue;
        }

        const double delta_score1 = score[i] - score[j];
        const double delta_score2 = score[i + half_len] - score[j + half_len];
        const double s = delta_score1 * delta_score2 < 0 ? 1 : 0;
        const double lambda2 = lambda_scorediff_ * s * delta_score2;
        const double lambda1 = lambda_scorediff_ * s * delta_score1;


        lambdas[i] -= static_cast<score_t>(lambda2);
        lambdas[j] += static_cast<score_t>(lambda2);
        lambdas[i + half_len] -= static_cast<score_t>(lambda1);
        lambdas[j + half_len] += static_cast<score_t>(lambda1);

        sum_lambdas += 2 * std::abs(lambda1);
        sum_lambdas += 2 * std::abs(lambda2);

        // 2nd order gradient is zero
      }

    }

    return sum_lambdas;
  }

  inline void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt,
    const label_t* label, const double* score,
    score_t* lambdas,
    score_t* hessians) const override {

    // for this loss, we assume that the records are sorted by
    //  1. query id
    //  2. impression id
    //  3. doc id

    // loss for score diff
    // 1. remove skip for equal label
    // 2. lambdarank inside one impresion, and rank diff between impression
    // 3. re-calculate max dcg

    data_size_t impression_cnt = impression_cnts_ == nullptr ? 1 : impression_cnts_[query_id]; // at least one impression in one query
    if (impression_cnt > 2) {
      Log::Warning("number of impression is %d > 2", impression_cnt);
    }

    bool use_scorediff = (impression_cnt == 2) && (cnt % 2 == 0);

    double sum_lambdas = 0;

    if (!use_scorediff) {
      // get max DCG on current query
      const double inverse_max_dcg = inverse_max_dcgs_[query_id];

      sum_lambdas = RankLossForOneQuery(cnt, label, score, lambdas, hessians, inverse_max_dcg);

    }
    else {
      if (cnt % 2 != 0) {
        Log::Fatal("using score diff, but number of docs in query %d is %d, not an even number.", query_id, cnt);
      }
      // get max DCG on current query
      const double inverse_max_dcg1 = inverse_max_dcgs_[query_id];
      const double inverse_max_dcg2 = inverse_max_dcgs_2_[query_id];
      // rank loss for first impression
      double sum_lambdas1 = RankLossForOneQuery(cnt / 2, label, score, lambdas, hessians, inverse_max_dcg1);
      // rank loss for second impression
      double sum_lambdas2 = RankLossForOneQuery(cnt / 2, label + cnt / 2, score + cnt / 2, lambdas + cnt / 2, hessians + cnt / 2, inverse_max_dcg2);
      // score diff loss for doc pairs
      double sum_lambdas3 = RankDiffLossForOneQuery(cnt, score, lambdas, hessians);

      sum_lambdas = sum_lambdas1 + sum_lambdas2 + sum_lambdas3;
    }

    if (norm_ && normalize_scorediff_gradient_ && sum_lambdas > 0) {
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = static_cast<score_t>(lambdas[i] * norm_factor);
        hessians[i] = static_cast<score_t>(hessians[i] * norm_factor);
      }
    }

  }

  const char* GetName() const override { return "lambdarank_rankdiff2"; }
};

/*!
 * \brief Objective function for Lambdrank with NDCG and ScoreDiff
 */
class LambdarankNDCGRankDiff3 : public LambdarankNDCG {
  double diff_alpha_;
  const data_size_t* impression_cnts_;
  std::vector<double> inverse_max_dcgs_2_;
  bool normalize_scorediff_gradient_;

public:
  explicit LambdarankNDCGRankDiff3(const Config& config)
    : LambdarankNDCG(config)
    , diff_alpha_(config.rankdiff_alpha)
    , impression_cnts_(nullptr)
    , normalize_scorediff_gradient_(false) {

    inverse_max_dcgs_2_.clear();
    normalize_scorediff_gradient_ = config.normalize_gradient_scorediff;

    Log::Info("initializing LambdarankNDCGRankDiff3");
    Log::Info("scoredif lambda = %f", diff_alpha_);
    Log::Info("normalize_scorediff_gradient_ = %d", normalize_scorediff_gradient_);
  }

  explicit LambdarankNDCGRankDiff3(const std::vector<std::string>& strs)
    : LambdarankNDCG(strs)
    , diff_alpha_(0.0)
    , impression_cnts_(nullptr)
    , normalize_scorediff_gradient_(false) {

    Log::Info("initializing LambdarankNDCGRankDiff3 from string");
    for (const auto& s : strs) {
      Log::Info(s.c_str());
    }
    Log::Info("scoredif lambda = %f", diff_alpha_);
    Log::Info("normalize_scorediff_gradient_ = %d", normalize_scorediff_gradient_);
  }

  ~LambdarankNDCGRankDiff3() {
    Log::Info("uninitializing LambdarankNDCGRankDiff3");
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    Log::Info("initialize rankdiff lambdarank 3");
    LambdarankNDCG::Init(metadata, num_data);

    impression_cnts_ = metadata.impression_cnts();
    if (nullptr == impression_cnts_) {
      Log::Warning("no impression cnt is found");
    }

    // re-calculate max dcg if there are more than one impressions
    if (nullptr != impression_cnts_) {
      inverse_max_dcgs_2_.resize(num_queries_);
      for (data_size_t i = 0; i < num_queries_; ++i) {
        if (impression_cnts_[i] > 1) {
          int len = query_boundaries_[i + 1] - query_boundaries_[i];
          if (impression_cnts_[i] > 2 || len % 2 != 0) {
            Log::Warning("For query %d, it has %d impressions and %d docs. It is not supported now. Only support 2 impressions. Will calculate original rank loss on it.", i, impression_cnts_[i], len);
          }
          inverse_max_dcgs_[i] = DCGCalculator::CalMaxDCGAtK(truncation_level_, label_ + query_boundaries_[i], len / 2);

          if (inverse_max_dcgs_[i] > 0.0) {
            inverse_max_dcgs_[i] = 1.0f / inverse_max_dcgs_[i];
          }

          inverse_max_dcgs_2_[i] = DCGCalculator::CalMaxDCGAtK(truncation_level_, label_ + query_boundaries_[i] + len / 2, len / 2);

          if (inverse_max_dcgs_2_[i] > 0.0) {
            inverse_max_dcgs_2_[i] = 1.0f / inverse_max_dcgs_2_[i];
          }
        }

      }
    }
  }

  inline double RankLossForOneQuery(data_size_t cnt, const label_t* label, const double* score,
    score_t* lambdas, score_t* hessians, const double inverse_max_dcg, const double weight) const {
    // initialize with zero
    for (data_size_t i = 0; i < cnt; ++i) {
      lambdas[i] = 0.0f;
      hessians[i] = 0.0f;
    }
    // get sorted indices for scores
    std::vector<data_size_t> sorted_idx(cnt);
    for (data_size_t i = 0; i < cnt; ++i) {
      sorted_idx[i] = i;
    }
    std::stable_sort(
      sorted_idx.begin(), sorted_idx.end(),
      [score](data_size_t a, data_size_t b) { return score[a] > score[b]; });
    // get best and worst score
    const double best_score = score[sorted_idx[0]];
    data_size_t worst_idx = cnt - 1;
    if (worst_idx > 0 && score[sorted_idx[worst_idx]] == kMinScore) {
      worst_idx -= 1;
    }
    const double worst_score = score[sorted_idx[worst_idx]];
    double sum_lambdas = 0.0;

    // start accmulate lambdas by pairs that contain at least one document above truncation level
    for (data_size_t i = 0; i < cnt - 1 && i < truncation_level_; ++i) {
      if (score[sorted_idx[i]] == kMinScore) { continue; }
      for (data_size_t j = i + 1; j < cnt; ++j) {
        if (score[sorted_idx[j]] == kMinScore) { continue; }
        // skip pairs with the same labels
        if (label[sorted_idx[i]] == label[sorted_idx[j]]) { continue; }
        data_size_t high_rank, low_rank;
        if (label[sorted_idx[i]] > label[sorted_idx[j]]) {
          high_rank = i;
          low_rank = j;
        }
        else {
          high_rank = j;
          low_rank = i;
        }
        const data_size_t high = sorted_idx[high_rank];
        const int high_label = static_cast<int>(label[high]);
        const double high_score = score[high];
        const double high_label_gain = label_gain_[high_label];
        const double high_discount = DCGCalculator::GetDiscount(high_rank);
        const data_size_t low = sorted_idx[low_rank];
        const int low_label = static_cast<int>(label[low]);
        const double low_score = score[low];
        const double low_label_gain = label_gain_[low_label];
        const double low_discount = DCGCalculator::GetDiscount(low_rank);

        const double delta_score = high_score - low_score;

        // get dcg gap
        const double dcg_gap = high_label_gain - low_label_gain;
        // get discount of this pair
        const double paired_discount = fabs(high_discount - low_discount);
        // get delta NDCG
        double delta_pair_NDCG = dcg_gap * paired_discount * inverse_max_dcg;
        // regular the delta_pair_NDCG by score distance
        if (norm_ && best_score != worst_score) {
          delta_pair_NDCG /= (0.01f + fabs(delta_score));
        }

        // calculate lambda for this pair
        double p_lambda = GetSigmoid(delta_score);
        double p_hessian = p_lambda * (1.0f - p_lambda);
        // update
        p_lambda *= -sigmoid_ * delta_pair_NDCG * weight;
        p_hessian *= sigmoid_ * sigmoid_ * delta_pair_NDCG * weight;
        lambdas[low] -= static_cast<score_t>(p_lambda);
        hessians[low] += static_cast<score_t>(p_hessian);
        lambdas[high] += static_cast<score_t>(p_lambda);
        hessians[high] += static_cast<score_t>(p_hessian);
        // lambda is negative, so use minus to accumulate
        sum_lambdas -= 2 * p_lambda;
      }
    }

    // when normalize_scorediff_gradient_, the gradients for rank loss and score diff loss will be normalized together
    if (norm_ && !normalize_scorediff_gradient_ && sum_lambdas > 0) {
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = static_cast<score_t>(lambdas[i] * norm_factor);
        hessians[i] = static_cast<score_t>(hessians[i] * norm_factor);
      }
    }

    return sum_lambdas;
  }

  // calculate lambdarank loss across two impressions
  inline double CrossRankLossForOneQuery(data_size_t cnt, const label_t* label, const double* score,
    score_t* lambdas, score_t* hessians, const double inverse_max_dcg, const double weight) const {
    // initialize with zero
    for (data_size_t i = 0; i < cnt; ++i) {
      lambdas[i] = 0.0f;
      hessians[i] = 0.0f;
    }
    // get sorted indices for scores
    std::vector<data_size_t> sorted_idx(cnt);
    for (data_size_t i = 0; i < cnt; ++i) {
      sorted_idx[i] = i;
    }
    std::stable_sort(
      sorted_idx.begin(), sorted_idx.end(),
      [score](data_size_t a, data_size_t b) { return score[a] > score[b]; });
    // get best and worst score
    double max_score = kMinScore, min_score = kMinScore;
    for (int i = 0; i < cnt * 2; ++i) {
      if (score[i] > max_score) {
        max_score = score[i];
      }
      if (score[i] > kMinScore && (min_score == kMinScore || score[i] < min_score)) {
        min_score = score[i];
      }
    }
    const double best_score = max_score;
    const double worst_score = min_score;
    double sum_lambdas = 0.0;
    

    // TODO: fix truncation_level_ in the future.
    // For now it is OK because truncation_level_ is 30 and it will not trigger.
    // We assume that label[i] == label[i + cnt].
    // We assuem cnt > 1.
    for (data_size_t i = 0; i < cnt && i < truncation_level_; ++i) {
      if (score[sorted_idx[i] + cnt] == kMinScore) { continue; }
      for (data_size_t j = 0; j != i && j < cnt; ++j) {
        if (score[sorted_idx[j]] == kMinScore) { continue; }
        // skip pairs with the same labels
        if (label[sorted_idx[i]] == label[sorted_idx[j]]) { continue; }
        data_size_t high_rank, low_rank;
        if (label[sorted_idx[i]] > label[sorted_idx[j]]) {
          high_rank = i;
          low_rank = j;
        }
        else {
          high_rank = j;
          low_rank = i;
        }
        const data_size_t high = sorted_idx[high_rank];
        const int high_label = static_cast<int>(label[high]);
        const double high_score = high_rank != i ? score[high] : score[sorted_idx[i] + cnt];
        const double high_label_gain = label_gain_[high_label];
        const double high_discount = DCGCalculator::GetDiscount(high_rank);
        const data_size_t low = sorted_idx[low_rank];
        const int low_label = static_cast<int>(label[low]);
        const double low_score = low_rank != i ? score[low] : score[sorted_idx[i] + cnt];
        const double low_label_gain = label_gain_[low_label];
        const double low_discount = DCGCalculator::GetDiscount(low_rank);

        const double delta_score = high_score - low_score;

        // get dcg gap
        const double dcg_gap = high_label_gain - low_label_gain;
        // get discount of this pair
        const double paired_discount = fabs(high_discount - low_discount);
        // get delta NDCG
        double delta_pair_NDCG = dcg_gap * paired_discount * inverse_max_dcg;
        // regular the delta_pair_NDCG by score distance
        if (norm_ && best_score != worst_score) {
          delta_pair_NDCG /= (0.01f + fabs(delta_score));
        }

        // calculate lambda for this pair
        double p_lambda = GetSigmoid(delta_score);
        double p_hessian = p_lambda * (1.0f - p_lambda);
        // update
        p_lambda *= -sigmoid_ * delta_pair_NDCG * weight;
        p_hessian *= sigmoid_ * sigmoid_ * delta_pair_NDCG * weight;
        lambdas[low] -= static_cast<score_t>(p_lambda);
        hessians[low] += static_cast<score_t>(p_hessian);
        lambdas[high] += static_cast<score_t>(p_lambda);
        hessians[high] += static_cast<score_t>(p_hessian);
        // lambda is negative, so use minus to accumulate
        sum_lambdas -= 2 * p_lambda;
      }
    }

    // when normalize_scorediff_gradient_, the gradients for rank loss and score diff loss will be normalized together
    if (norm_ && !normalize_scorediff_gradient_ && sum_lambdas > 0) {
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = static_cast<score_t>(lambdas[i] * norm_factor);
        hessians[i] = static_cast<score_t>(hessians[i] * norm_factor);
      }
    }

    return sum_lambdas;
  }

  inline void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt,
    const label_t* label, const double* score,
    score_t* lambdas,
    score_t* hessians) const override {

    // for this loss, we assume that the records are sorted by
    //  1. query id
    //  2. impression id
    //  3. doc id

    data_size_t impression_cnt = impression_cnts_ == nullptr ? 1 : impression_cnts_[query_id]; // at least one impression in one query
    if (impression_cnt > 2) {
      Log::Warning("number of impression is %d > 2", impression_cnt);
    }

    bool use_scorediff = (impression_cnt == 2) && (cnt % 2 == 0);

    double sum_lambdas = 0;

    if (!use_scorediff) {
      // get max DCG on current query
      const double inverse_max_dcg = inverse_max_dcgs_[query_id];

      const double lambda_rank_loss_weight = 1.0;
      sum_lambdas = RankLossForOneQuery(cnt, label, score, lambdas, hessians, inverse_max_dcg, lambda_rank_loss_weight);

    }
    else {
      if (cnt % 2 != 0) {
        Log::Fatal("using score diff, but number of docs in query %d is %d, not an even number.", query_id, cnt);
      }

      const double lambda_rank_loss_weight = 0.5 * (1 - diff_alpha_);
      const double cross_rank_loss_weight = 0.5 * diff_alpha_;
      // get max DCG on current query
      const double inverse_max_dcg1 = inverse_max_dcgs_[query_id];
      const double inverse_max_dcg2 = inverse_max_dcgs_2_[query_id];
      // rank loss for first impression
      double sum_lambdas1 = RankLossForOneQuery(cnt / 2, label, score, lambdas, hessians, inverse_max_dcg1, lambda_rank_loss_weight);
      // rank loss for second impression
      double sum_lambdas2 = RankLossForOneQuery(cnt / 2, label + cnt / 2, score + cnt / 2, lambdas + cnt / 2, hessians + cnt / 2, inverse_max_dcg2, lambda_rank_loss_weight);
      // score diff loss for doc pairs
      double sum_lambdas3 = CrossRankLossForOneQuery(cnt / 2, label, score, lambdas, hessians, inverse_max_dcg1, cross_rank_loss_weight);

      sum_lambdas = sum_lambdas1 + sum_lambdas2 + sum_lambdas3;
    }

    if (norm_ && normalize_scorediff_gradient_ && sum_lambdas > 0) {
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = static_cast<score_t>(lambdas[i] * norm_factor);
        hessians[i] = static_cast<score_t>(hessians[i] * norm_factor);
      }
    }

  }

  const char* GetName() const override { return "lambdarank_rankdiff3"; }
};

/*!
 * \brief Objective function for Lambdrank with NDCG and ScoreDiff
 */
class LambdarankNDCGRankDiff4 : public LambdarankNDCG {
  double lambda_scorediff_;
  const data_size_t* impression_cnts_;
  std::vector<double> inverse_max_dcgs_2_;
  bool normalize_scorediff_gradient_;

public:
  explicit LambdarankNDCGRankDiff4(const Config& config)
    : LambdarankNDCG(config)
    , lambda_scorediff_(config.scorediff_alpha)
    , impression_cnts_(nullptr)
    , normalize_scorediff_gradient_(false) {

    inverse_max_dcgs_2_.clear();
    normalize_scorediff_gradient_ = config.normalize_gradient_scorediff;

    Log::Info("initializing LambdarankNDCGRankDiff4");
    Log::Info("scoredif lambda = %f", lambda_scorediff_);
    Log::Info("normalize_scorediff_gradient_ = %d", normalize_scorediff_gradient_);
  }

  explicit LambdarankNDCGRankDiff4(const std::vector<std::string>& strs)
    : LambdarankNDCG(strs)
    , lambda_scorediff_(0.0)
    , impression_cnts_(nullptr)
    , normalize_scorediff_gradient_(false) {

    Log::Info("initializing LambdarankNDCGRankDiff4 from string");
    for (const auto& s : strs) {
      Log::Info(s.c_str());
    }
    Log::Info("scoredif lambda = %f", lambda_scorediff_);
  }

  ~LambdarankNDCGRankDiff4() {
    Log::Info("uninitializing LambdarankNDCGRankDiff4");
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    Log::Info("initialize rankdiff lambdarank 4");
    LambdarankNDCG::Init(metadata, num_data);

    impression_cnts_ = metadata.impression_cnts();
    if (nullptr == impression_cnts_) {
      Log::Warning("no impression cnt is found");
    }

    // re-calculate max dcg if there are more than one impressions
    if (nullptr != impression_cnts_) {
      inverse_max_dcgs_2_.resize(num_queries_);
      for (data_size_t i = 0; i < num_queries_; ++i) {
        if (impression_cnts_[i] > 1) {
          int len = query_boundaries_[i + 1] - query_boundaries_[i];
          if (impression_cnts_[i] > 2 || len % 2 != 0) {
            Log::Warning("For query %d, it has %d impressions and %d docs. It is not supported now. Only support 2 impressions. Will calculate original rank loss on it.", i, impression_cnts_[i], len);
          }
          inverse_max_dcgs_[i] = DCGCalculator::CalMaxDCGAtK(truncation_level_, label_ + query_boundaries_[i], len / 2);

          if (inverse_max_dcgs_[i] > 0.0) {
            inverse_max_dcgs_[i] = 1.0f / inverse_max_dcgs_[i];
          }

          inverse_max_dcgs_2_[i] = DCGCalculator::CalMaxDCGAtK(truncation_level_, label_ + query_boundaries_[i] + len / 2, len / 2);

          if (inverse_max_dcgs_2_[i] > 0.0) {
            inverse_max_dcgs_2_[i] = 1.0f / inverse_max_dcgs_2_[i];
          }
        }

      }
    }
  }

  inline double RankLossForOneQuery(data_size_t cnt, const label_t* label, const double* score,
    score_t* lambdas, score_t* hessians, const double inverse_max_dcg, const double weight) const {
    // initialize with zero
    for (data_size_t i = 0; i < cnt; ++i) {
      lambdas[i] = 0.0f;
      hessians[i] = 0.0f;
    }
    // get sorted indices for scores
    std::vector<data_size_t> sorted_idx(cnt);
    for (data_size_t i = 0; i < cnt; ++i) {
      sorted_idx[i] = i;
    }
    std::stable_sort(
      sorted_idx.begin(), sorted_idx.end(),
      [score](data_size_t a, data_size_t b) { return score[a] > score[b]; });
    // get best and worst score
    const double best_score = score[sorted_idx[0]];
    data_size_t worst_idx = cnt - 1;
    if (worst_idx > 0 && score[sorted_idx[worst_idx]] == kMinScore) {
      worst_idx -= 1;
    }
    const double worst_score = score[sorted_idx[worst_idx]];
    double sum_lambdas = 0.0;

    // start accmulate lambdas by pairs that contain at least one document above truncation level
    for (data_size_t i = 0; i < cnt - 1 && i < truncation_level_; ++i) {
      if (score[sorted_idx[i]] == kMinScore) { continue; }
      for (data_size_t j = i + 1; j < cnt; ++j) {
        if (score[sorted_idx[j]] == kMinScore) { continue; }
        // skip pairs with the same labels
        if (label[sorted_idx[i]] == label[sorted_idx[j]]) { continue; }
        data_size_t high_rank, low_rank;
        if (label[sorted_idx[i]] > label[sorted_idx[j]]) {
          high_rank = i;
          low_rank = j;
        }
        else {
          high_rank = j;
          low_rank = i;
        }
        const data_size_t high = sorted_idx[high_rank];
        const int high_label = static_cast<int>(label[high]);
        const double high_score = score[high];
        const double high_label_gain = label_gain_[high_label];
        const double high_discount = DCGCalculator::GetDiscount(high_rank);
        const data_size_t low = sorted_idx[low_rank];
        const int low_label = static_cast<int>(label[low]);
        const double low_score = score[low];
        const double low_label_gain = label_gain_[low_label];
        const double low_discount = DCGCalculator::GetDiscount(low_rank);

        const double delta_score = high_score - low_score;

        // get dcg gap
        const double dcg_gap = high_label_gain - low_label_gain;
        // get discount of this pair
        const double paired_discount = fabs(high_discount - low_discount);
        // get delta NDCG
        double delta_pair_NDCG = dcg_gap * paired_discount * inverse_max_dcg;
        // regular the delta_pair_NDCG by score distance
        if (norm_ && best_score != worst_score) {
          delta_pair_NDCG /= (0.01f + fabs(delta_score));
        }

        // calculate lambda for this pair
        double p_lambda = GetSigmoid(delta_score);
        double p_hessian = p_lambda * (1.0f - p_lambda);
        // update
        p_lambda *= -sigmoid_ * delta_pair_NDCG * weight;
        p_hessian *= sigmoid_ * sigmoid_ * delta_pair_NDCG * weight;
        lambdas[low] -= static_cast<score_t>(p_lambda);
        hessians[low] += static_cast<score_t>(p_hessian);
        lambdas[high] += static_cast<score_t>(p_lambda);
        hessians[high] += static_cast<score_t>(p_hessian);
        // lambda is negative, so use minus to accumulate
        sum_lambdas -= 2 * p_lambda;
      }
    }

    // when normalize_scorediff_gradient_, the gradients for rank loss and score diff loss will be normalized together
    if (norm_ && !normalize_scorediff_gradient_ && sum_lambdas > 0) {
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = static_cast<score_t>(lambdas[i] * norm_factor);
        hessians[i] = static_cast<score_t>(hessians[i] * norm_factor);
      }
    }

    return sum_lambdas;
  }

  inline double RankDiffLossForOneQuery(data_size_t cnt, const double* score, score_t* lambdas, score_t* hessians) const {

    double sum_lambdas = 0;


    data_size_t half_len = cnt / 2;
    for (data_size_t i = 0; i < half_len - 1; ++i) {
      for (data_size_t j = i + 1; j < half_len; ++j) {
        if (score[i] == kMinScore ||
          score[i + half_len] == kMinScore ||
          score[j] == kMinScore ||
          score[j + half_len] == kMinScore) {
          continue;
        }

        const double delta_score1 = score[i] - score[j];
        const double delta_score2 = score[i + half_len] - score[j + half_len];
        const double s = GetSigmoid(delta_score1 * delta_score2);
        const double lambda2 = lambda_scorediff_ * s * delta_score2;
        const double lambda1 = lambda_scorediff_ * s * delta_score1;

        const double hessians2 = lambda_scorediff_ * s * (1 - s) * delta_score2 * delta_score2;
        const double hessians1 = lambda_scorediff_ * s * (1 - s) * delta_score1 * delta_score1;


        lambdas[i] -= static_cast<score_t>(lambda2);
        lambdas[j] += static_cast<score_t>(lambda2);
        lambdas[i + half_len] -= static_cast<score_t>(lambda1);
        lambdas[j + half_len] += static_cast<score_t>(lambda1);

        hessians[i] += static_cast<score_t>(hessians2);
        hessians[j] += static_cast<score_t>(hessians2);
        hessians[i + half_len] += static_cast<score_t>(hessians1);
        hessians[j + half_len] += static_cast<score_t>(hessians1);

        sum_lambdas += 2 * std::abs(lambda1);
        sum_lambdas += 2 * std::abs(lambda2);

        // 2nd order gradient is zero
      }

    }

    return sum_lambdas;
  }

  inline void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt,
    const label_t* label, const double* score,
    score_t* lambdas,
    score_t* hessians) const override {

    // for this loss, we assume that the records are sorted by
    //  1. query id
    //  2. impression id
    //  3. doc id


    data_size_t impression_cnt = impression_cnts_ == nullptr ? 1 : impression_cnts_[query_id]; // at least one impression in one query
    if (impression_cnt > 2) {
      Log::Warning("number of impression is %d > 2", impression_cnt);
    }

    bool use_scorediff = (impression_cnt == 2) && (cnt % 2 == 0);

    double sum_lambdas = 0;

    if (!use_scorediff) {
      // get max DCG on current query
      const double inverse_max_dcg = inverse_max_dcgs_[query_id];
      const double lambda_rank_loss_weight = 1.0;
      sum_lambdas = RankLossForOneQuery(cnt, label, score, lambdas, hessians, inverse_max_dcg, lambda_rank_loss_weight);

      if (norm_ && normalize_scorediff_gradient_ && sum_lambdas > 0) {
        sum_gradients_original_lambdarank_[query_id] = std::log2(1 + sum_lambdas);
      }
      else {
        sum_gradients_original_lambdarank_[query_id] = sum_lambdas;
      }
    }
    else {
      if (cnt % 2 != 0) {
        Log::Fatal("using score diff, but number of docs in query %d is %d, not an even number.", query_id, cnt);
      }

      const double lambda_rank_loss_weight = 0.5 * (1 - lambda_scorediff_);
      // get max DCG on current query
      const double inverse_max_dcg1 = inverse_max_dcgs_[query_id];
      const double inverse_max_dcg2 = inverse_max_dcgs_2_[query_id];
      // rank loss for first impression
      double sum_lambdas1 = RankLossForOneQuery(cnt / 2, label, score, lambdas, hessians, inverse_max_dcg1, lambda_rank_loss_weight);
      // rank loss for second impression
      double sum_lambdas2 = RankLossForOneQuery(cnt / 2, label + cnt / 2, score + cnt / 2, lambdas + cnt / 2, hessians + cnt / 2, inverse_max_dcg2, lambda_rank_loss_weight);
      // score diff loss for doc pairs
      double sum_lambdas3 = RankDiffLossForOneQuery(cnt, score, lambdas, hessians);

      sum_lambdas = sum_lambdas1 + sum_lambdas2 + sum_lambdas3;

      if (norm_ && normalize_scorediff_gradient_ && sum_lambdas1 + sum_lambdas2 > 0) {
        sum_gradients_scorediff_lambdarank_[query_id] = std::log2(1 + sum_lambdas1 + sum_lambdas2);
      }
      else {
        sum_gradients_scorediff_lambdarank_[query_id] = sum_lambdas1 + sum_lambdas2;
      }

      if (norm_ && normalize_scorediff_gradient_ && sum_lambdas3 > 0) {
        sum_gradients_scorediff_[query_id] = std::log2(1 + sum_lambdas3);
      }
      else {
        sum_gradients_scorediff_[query_id] = sum_lambdas3;
      }
    }

    if (norm_ && normalize_scorediff_gradient_ && sum_lambdas > 0) {
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = static_cast<score_t>(lambdas[i] * norm_factor);
        hessians[i] = static_cast<score_t>(hessians[i] * norm_factor);
      }
    }

  }

  const char* GetName() const override { return "lambdarank_rankdiff4"; }
};

/*!
 * \brief Objective function for Lambdrank with NDCG and ScoreDiff
 */
class LambdarankNDCGRankDiff5 : public LambdarankNDCG {
  double lambda_scorediff_;
  const data_size_t* impression_cnts_;
  std::vector<double> inverse_max_dcgs_2_;
  bool normalize_scorediff_gradient_;

public:
  explicit LambdarankNDCGRankDiff5(const Config& config)
    : LambdarankNDCG(config)
    , lambda_scorediff_(config.scorediff_alpha)
    , impression_cnts_(nullptr)
    , normalize_scorediff_gradient_(false) {

    inverse_max_dcgs_2_.clear();
    normalize_scorediff_gradient_ = config.normalize_gradient_scorediff;

    Log::Info("initializing LambdarankNDCGRankDiff5");
    Log::Info("scoredif lambda = %f", lambda_scorediff_);
    Log::Info("normalize_scorediff_gradient_ = %d", normalize_scorediff_gradient_);
  }

  explicit LambdarankNDCGRankDiff5(const std::vector<std::string>& strs)
    : LambdarankNDCG(strs)
    , lambda_scorediff_(0.0)
    , impression_cnts_(nullptr)
    , normalize_scorediff_gradient_(false) {

    Log::Info("initializing LambdarankNDCGRankDiff5 from string");
    for (const auto& s : strs) {
      Log::Info(s.c_str());
    }
    Log::Info("scoredif lambda = %f", lambda_scorediff_);
  }

  ~LambdarankNDCGRankDiff5() {
    Log::Info("uninitializing LambdarankNDCGRankDiff5");
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    Log::Info("initialize rankdiff lambdarank 5");
    LambdarankNDCG::Init(metadata, num_data);

    impression_cnts_ = metadata.impression_cnts();
    if (nullptr == impression_cnts_) {
      Log::Warning("no impression cnt is found");
    }

    // re-calculate max dcg if there are more than one impressions
    if (nullptr != impression_cnts_) {
      inverse_max_dcgs_2_.resize(num_queries_);
      for (data_size_t i = 0; i < num_queries_; ++i) {
        if (impression_cnts_[i] > 1) {
          int len = query_boundaries_[i + 1] - query_boundaries_[i];
          if (impression_cnts_[i] > 2 || len % 2 != 0) {
            Log::Warning("For query %d, it has %d impressions and %d docs. It is not supported now. Only support 2 impressions. Will calculate original rank loss on it.", i, impression_cnts_[i], len);
          }
          inverse_max_dcgs_[i] = DCGCalculator::CalMaxDCGAtK(truncation_level_, label_ + query_boundaries_[i], len / 2);

          if (inverse_max_dcgs_[i] > 0.0) {
            inverse_max_dcgs_[i] = 1.0f / inverse_max_dcgs_[i];
          }

          inverse_max_dcgs_2_[i] = DCGCalculator::CalMaxDCGAtK(truncation_level_, label_ + query_boundaries_[i] + len / 2, len / 2);

          if (inverse_max_dcgs_2_[i] > 0.0) {
            inverse_max_dcgs_2_[i] = 1.0f / inverse_max_dcgs_2_[i];
          }
        }

      }
    }
  }

  inline double RankLossForOneQuery(data_size_t cnt, const label_t* label, const double* score,
    score_t* lambdas, score_t* hessians, const double inverse_max_dcg, const double weight) const {
    // initialize with zero
    for (data_size_t i = 0; i < cnt; ++i) {
      lambdas[i] = 0.0f;
      hessians[i] = 0.0f;
    }
    // get sorted indices for scores
    std::vector<data_size_t> sorted_idx(cnt);
    for (data_size_t i = 0; i < cnt; ++i) {
      sorted_idx[i] = i;
    }
    std::stable_sort(
      sorted_idx.begin(), sorted_idx.end(),
      [score](data_size_t a, data_size_t b) { return score[a] > score[b]; });
    // get best and worst score
    const double best_score = score[sorted_idx[0]];
    data_size_t worst_idx = cnt - 1;
    if (worst_idx > 0 && score[sorted_idx[worst_idx]] == kMinScore) {
      worst_idx -= 1;
    }
    const double worst_score = score[sorted_idx[worst_idx]];
    double sum_lambdas = 0.0;

    // start accmulate lambdas by pairs that contain at least one document above truncation level
    for (data_size_t i = 0; i < cnt - 1 && i < truncation_level_; ++i) {
      if (score[sorted_idx[i]] == kMinScore) { continue; }
      for (data_size_t j = i + 1; j < cnt; ++j) {
        if (score[sorted_idx[j]] == kMinScore) { continue; }
        // skip pairs with the same labels
        if (label[sorted_idx[i]] == label[sorted_idx[j]]) { continue; }
        data_size_t high_rank, low_rank;
        if (label[sorted_idx[i]] > label[sorted_idx[j]]) {
          high_rank = i;
          low_rank = j;
        }
        else {
          high_rank = j;
          low_rank = i;
        }
        const data_size_t high = sorted_idx[high_rank];
        const int high_label = static_cast<int>(label[high]);
        const double high_score = score[high];
        const double high_label_gain = label_gain_[high_label];
        const double high_discount = DCGCalculator::GetDiscount(high_rank);
        const data_size_t low = sorted_idx[low_rank];
        const int low_label = static_cast<int>(label[low]);
        const double low_score = score[low];
        const double low_label_gain = label_gain_[low_label];
        const double low_discount = DCGCalculator::GetDiscount(low_rank);

        const double delta_score = high_score - low_score;

        // get dcg gap
        const double dcg_gap = high_label_gain - low_label_gain;
        // get discount of this pair
        const double paired_discount = fabs(high_discount - low_discount);
        // get delta NDCG
        double delta_pair_NDCG = dcg_gap * paired_discount * inverse_max_dcg;
        // regular the delta_pair_NDCG by score distance
        if (norm_ && best_score != worst_score) {
          delta_pair_NDCG /= (0.01f + fabs(delta_score));
        }

        // calculate lambda for this pair
        double p_lambda = GetSigmoid(delta_score);
        double p_hessian = p_lambda * (1.0f - p_lambda);
        // update
        p_lambda *= -sigmoid_ * delta_pair_NDCG * weight;
        p_hessian *= sigmoid_ * sigmoid_ * delta_pair_NDCG * weight;
        lambdas[low] -= static_cast<score_t>(p_lambda);
        hessians[low] += static_cast<score_t>(p_hessian);
        lambdas[high] += static_cast<score_t>(p_lambda);
        hessians[high] += static_cast<score_t>(p_hessian);
        // lambda is negative, so use minus to accumulate
        sum_lambdas -= 2 * p_lambda;
      }
    }

    // when normalize_scorediff_gradient_, the gradients for rank loss and score diff loss will be normalized together
    if (norm_ && !normalize_scorediff_gradient_ && sum_lambdas > 0) {
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = static_cast<score_t>(lambdas[i] * norm_factor);
        hessians[i] = static_cast<score_t>(hessians[i] * norm_factor);
      }
    }

    return sum_lambdas;
  }

  inline double RankDiffLossForOneQuery(data_size_t cnt, const double* score, score_t* lambdas, score_t* hessians) const {

    double sum_lambdas = 0;


    data_size_t half_len = cnt / 2;
    for (data_size_t i = 0; i < half_len - 1; ++i) {
      for (data_size_t j = i + 1; j < half_len; ++j) {
        if (score[i] == kMinScore ||
          score[i + half_len] == kMinScore ||
          score[j] == kMinScore ||
          score[j + half_len] == kMinScore) {
          continue;
        }

        const double delta_score1 = score[i] - score[j];
        const double delta_score2 = score[i + half_len] - score[j + half_len];
        const double I = delta_score1 * delta_score2 < 0 ? 1 : 0;
        const double s = GetSigmoid(delta_score1 * delta_score2 * I);
        const double lambda2 = lambda_scorediff_ * s * delta_score2 * I;
        const double lambda1 = lambda_scorediff_ * s * delta_score1 * I;

        const double hessians2 = lambda_scorediff_ * s * (1 - s) * delta_score2 * delta_score2;
        const double hessians1 = lambda_scorediff_ * s * (1 - s) * delta_score1 * delta_score1;


        lambdas[i] -= static_cast<score_t>(lambda2);
        lambdas[j] += static_cast<score_t>(lambda2);
        lambdas[i + half_len] -= static_cast<score_t>(lambda1);
        lambdas[j + half_len] += static_cast<score_t>(lambda1);

        hessians[i] += static_cast<score_t>(hessians2);
        hessians[j] += static_cast<score_t>(hessians2);
        hessians[i + half_len] += static_cast<score_t>(hessians1);
        hessians[j + half_len] += static_cast<score_t>(hessians1);

        sum_lambdas += 2 * std::abs(lambda1);
        sum_lambdas += 2 * std::abs(lambda2);

        // 2nd order gradient is zero
      }

    }

    return sum_lambdas;
  }

  inline void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt,
    const label_t* label, const double* score,
    score_t* lambdas,
    score_t* hessians) const override {

    // for this loss, we assume that the records are sorted by
    //  1. query id
    //  2. impression id
    //  3. doc id


    data_size_t impression_cnt = impression_cnts_ == nullptr ? 1 : impression_cnts_[query_id]; // at least one impression in one query
    if (impression_cnt > 2) {
      Log::Warning("number of impression is %d > 2", impression_cnt);
    }

    bool use_scorediff = (impression_cnt == 2) && (cnt % 2 == 0);

    double sum_lambdas = 0;

    if (!use_scorediff) {
      // get max DCG on current query
      const double inverse_max_dcg = inverse_max_dcgs_[query_id];
      const double lambda_rank_loss_weight = 1.0;
      sum_lambdas = RankLossForOneQuery(cnt, label, score, lambdas, hessians, inverse_max_dcg, lambda_rank_loss_weight);

      if (norm_ && normalize_scorediff_gradient_ && sum_lambdas > 0) {
        sum_gradients_original_lambdarank_[query_id] = std::log2(1 + sum_lambdas);
      }
      else {
        sum_gradients_original_lambdarank_[query_id] = sum_lambdas;
      }
    }
    else {
      if (cnt % 2 != 0) {
        Log::Fatal("using score diff, but number of docs in query %d is %d, not an even number.", query_id, cnt);
      }

      const double lambda_rank_loss_weight = 0.5 * (1 - lambda_scorediff_);
      // get max DCG on current query
      const double inverse_max_dcg1 = inverse_max_dcgs_[query_id];
      const double inverse_max_dcg2 = inverse_max_dcgs_2_[query_id];
      // rank loss for first impression
      double sum_lambdas1 = RankLossForOneQuery(cnt / 2, label, score, lambdas, hessians, inverse_max_dcg1, lambda_rank_loss_weight);
      // rank loss for second impression
      double sum_lambdas2 = RankLossForOneQuery(cnt / 2, label + cnt / 2, score + cnt / 2, lambdas + cnt / 2, hessians + cnt / 2, inverse_max_dcg2, lambda_rank_loss_weight);
      // score diff loss for doc pairs
      double sum_lambdas3 = RankDiffLossForOneQuery(cnt, score, lambdas, hessians);

      sum_lambdas = sum_lambdas1 + sum_lambdas2 + sum_lambdas3;

      if (norm_ && normalize_scorediff_gradient_ && sum_lambdas1 + sum_lambdas2 > 0) {
        sum_gradients_scorediff_lambdarank_[query_id] = std::log2(1 + sum_lambdas1 + sum_lambdas2);
      }
      else {
        sum_gradients_scorediff_lambdarank_[query_id] = sum_lambdas1 + sum_lambdas2;
      }

      if (norm_ && normalize_scorediff_gradient_ && sum_lambdas3 > 0) {
        sum_gradients_scorediff_[query_id] = std::log2(1 + sum_lambdas3);
      }
      else {
        sum_gradients_scorediff_[query_id] = sum_lambdas3;
      }
    }

    if (norm_ && normalize_scorediff_gradient_ && sum_lambdas > 0) {
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = static_cast<score_t>(lambdas[i] * norm_factor);
        hessians[i] = static_cast<score_t>(hessians[i] * norm_factor);
      }
    }

  }

  const char* GetName() const override { return "lambdarank_rankdiff5"; }
};

/*!
 * \brief Objective function for Lambdrank with NDCG and RankDiff 2
 */
class LambdarankNDCGScoreRankDiff : public LambdarankNDCG {
  double lambda_scorediff_;
  double lambda_rankdiff_;
  const data_size_t* impression_cnts_;
  std::vector<double> inverse_max_dcgs_2_;
  bool normalize_scorediff_gradient_;

public:
  explicit LambdarankNDCGScoreRankDiff(const Config& config)
    : LambdarankNDCG(config)
    , lambda_scorediff_(config.scorediff_alpha)
    , lambda_rankdiff_(config.rankdiff_alpha)
    , impression_cnts_(nullptr)
    , normalize_scorediff_gradient_(false) {

    inverse_max_dcgs_2_.clear();

    normalize_scorediff_gradient_ = config.normalize_gradient_scorediff;

    Log::Info("initializing LambdarankNDCGScoreRankDiff");
    Log::Info("scorediff lambda = %f", lambda_scorediff_);
    Log::Info("rankdiff lambda = %f", lambda_rankdiff_);
    Log::Info("normalize_scorediff_gradient_ = %d", normalize_scorediff_gradient_);
  }

  explicit LambdarankNDCGScoreRankDiff(const std::vector<std::string>& strs)
    : LambdarankNDCG(strs)
    , lambda_scorediff_(1.0)
    , lambda_rankdiff_(1.0)
    , impression_cnts_(nullptr)
    , normalize_scorediff_gradient_(false) {

    Log::Info("initializing LambdarankNDCGScoreRankDiff from string");
    for (const auto& s : strs) {
      Log::Info(s.c_str());
    }
    Log::Info("scoredif lambda = %f", lambda_scorediff_);
    Log::Info("rankdiff lambda = %f", lambda_rankdiff_);
    Log::Info("normalize_scorediff_gradient_ = %d", normalize_scorediff_gradient_);
  }

  ~LambdarankNDCGScoreRankDiff() {
    Log::Info("uninitializing LambdarankNDCGScoreRankDiff");
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    Log::Info("initialize score rankdiff lambdarank");
    LambdarankNDCG::Init(metadata, num_data);

    impression_cnts_ = metadata.impression_cnts();
    if (nullptr == impression_cnts_) {
      Log::Warning("no impression cnt is found");
    }

    // re-calculate max dcg if there are more than one impressions
    if (nullptr != impression_cnts_) {
      inverse_max_dcgs_2_.resize(num_queries_);
      for (data_size_t i = 0; i < num_queries_; ++i) {
        if (impression_cnts_[i] > 1) {
          int len = query_boundaries_[i + 1] - query_boundaries_[i];
          if (impression_cnts_[i] > 2 || len % 2 != 0) {
            Log::Warning("For query %d, it has %d impressions and %d docs. It is not supported now. Only support 2 impressions. Will calculate original rank loss on it.", i, impression_cnts_[i], len);
          }
          inverse_max_dcgs_[i] = DCGCalculator::CalMaxDCGAtK(truncation_level_, label_ + query_boundaries_[i], len / 2);

          if (inverse_max_dcgs_[i] > 0.0) {
            inverse_max_dcgs_[i] = 1.0f / inverse_max_dcgs_[i];
          }

          inverse_max_dcgs_2_[i] = DCGCalculator::CalMaxDCGAtK(truncation_level_, label_ + query_boundaries_[i] + len / 2, len / 2);

          if (inverse_max_dcgs_2_[i] > 0.0) {
            inverse_max_dcgs_2_[i] = 1.0f / inverse_max_dcgs_2_[i];
          }
        }

      }
    }
  }

  inline double RankLossForOneQuery(data_size_t cnt, const label_t* label, const double* score,
    score_t* lambdas, score_t* hessians, const double inverse_max_dcg) const {
    // initialize with zero
    for (data_size_t i = 0; i < cnt; ++i) {
      lambdas[i] = 0.0f;
      hessians[i] = 0.0f;
    }
    // get sorted indices for scores
    std::vector<data_size_t> sorted_idx(cnt);
    for (data_size_t i = 0; i < cnt; ++i) {
      sorted_idx[i] = i;
    }
    std::stable_sort(
      sorted_idx.begin(), sorted_idx.end(),
      [score](data_size_t a, data_size_t b) { return score[a] > score[b]; });
    // get best and worst score
    const double best_score = score[sorted_idx[0]];
    data_size_t worst_idx = cnt - 1;
    if (worst_idx > 0 && score[sorted_idx[worst_idx]] == kMinScore) {
      worst_idx -= 1;
    }
    const double worst_score = score[sorted_idx[worst_idx]];
    double sum_lambdas = 0.0;

    // start accmulate lambdas by pairs that contain at least one document above truncation level
    for (data_size_t i = 0; i < cnt - 1 && i < truncation_level_; ++i) {
      if (score[sorted_idx[i]] == kMinScore) { continue; }
      for (data_size_t j = i + 1; j < cnt; ++j) {
        if (score[sorted_idx[j]] == kMinScore) { continue; }
        // skip pairs with the same labels
        if (label[sorted_idx[i]] == label[sorted_idx[j]]) { continue; }
        data_size_t high_rank, low_rank;
        if (label[sorted_idx[i]] > label[sorted_idx[j]]) {
          high_rank = i;
          low_rank = j;
        }
        else {
          high_rank = j;
          low_rank = i;
        }
        const data_size_t high = sorted_idx[high_rank];
        const int high_label = static_cast<int>(label[high]);
        const double high_score = score[high];
        const double high_label_gain = label_gain_[high_label];
        const double high_discount = DCGCalculator::GetDiscount(high_rank);
        const data_size_t low = sorted_idx[low_rank];
        const int low_label = static_cast<int>(label[low]);
        const double low_score = score[low];
        const double low_label_gain = label_gain_[low_label];
        const double low_discount = DCGCalculator::GetDiscount(low_rank);

        const double delta_score = high_score - low_score;

        // get dcg gap
        const double dcg_gap = high_label_gain - low_label_gain;
        // get discount of this pair
        const double paired_discount = fabs(high_discount - low_discount);
        // get delta NDCG
        double delta_pair_NDCG = dcg_gap * paired_discount * inverse_max_dcg;
        // regular the delta_pair_NDCG by score distance
        if (norm_ && best_score != worst_score) {
          delta_pair_NDCG /= (0.01f + fabs(delta_score));
        }

        // calculate lambda for this pair
        double p_lambda = GetSigmoid(delta_score);
        double p_hessian = p_lambda * (1.0f - p_lambda);
        // update
        p_lambda *= -sigmoid_ * delta_pair_NDCG;
        p_hessian *= sigmoid_ * sigmoid_ * delta_pair_NDCG;
        lambdas[low] -= static_cast<score_t>(p_lambda);
        hessians[low] += static_cast<score_t>(p_hessian);
        lambdas[high] += static_cast<score_t>(p_lambda);
        hessians[high] += static_cast<score_t>(p_hessian);
        // lambda is negative, so use minus to accumulate
        sum_lambdas -= 2 * p_lambda;
      }
    }

    // when normalize_scorediff_gradient_, the gradients for rank loss and score diff loss will be normalized together
    if (norm_ && !normalize_scorediff_gradient_ && sum_lambdas > 0) {
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = static_cast<score_t>(lambdas[i] * norm_factor);
        hessians[i] = static_cast<score_t>(hessians[i] * norm_factor);
      }
    }

    return sum_lambdas;
  }

  inline double RankDiffLossForOneQuery(data_size_t cnt, const double* score, score_t* lambdas, score_t* hessians) const {

    double sum_lambdas = 0;
    data_size_t half_len = cnt / 2;
    for (data_size_t i = 0; i < half_len - 1; ++i) {
      for (data_size_t j = i + 1; j < half_len; ++j) {
        if (score[i] == kMinScore ||
          score[i + half_len] == kMinScore ||
          score[j] == kMinScore ||
          score[j + half_len] == kMinScore) {
          continue;
        }

        const double delta_score1 = score[i] - score[j];
        const double delta_score2 = score[i + half_len] - score[j + half_len];
        const double s = delta_score1 * delta_score2 < 0 ? 1 : 0;
        const double lambda2 = lambda_rankdiff_ * s * delta_score2;
        const double lambda1 = lambda_rankdiff_ * s * delta_score1;


        lambdas[i] -= static_cast<score_t>(lambda2);
        lambdas[j] += static_cast<score_t>(lambda2);
        lambdas[i + half_len] -= static_cast<score_t>(lambda1);
        lambdas[j + half_len] += static_cast<score_t>(lambda1);

        sum_lambdas += 2 * std::abs(lambda1);
        sum_lambdas += 2 * std::abs(lambda2);

        // 2nd order gradient is zero
      }

    }

    return sum_lambdas;
  }

  inline double ScoreDiffLossForOneQuery(data_size_t cnt, const double* score, score_t* lambdas, score_t* hessians) const {

    double sum_lambdas = 0;
    data_size_t half_len = cnt / 2;
    for (data_size_t i = 0; i < half_len; ++i) {
      if (score[i] == kMinScore || score[i + half_len] == kMinScore) { continue; }
      const double delta_score = score[i] - score[i + half_len];

      // calculate lambda for this pair
      double p_lambda = 2 * lambda_scorediff_ * delta_score;
      double p_hessian = 2 * lambda_scorediff_;

      lambdas[i] += static_cast<score_t>(p_lambda);
      hessians[i] += static_cast<score_t>(p_hessian);
      lambdas[i + half_len] -= static_cast<score_t>(p_lambda);
      hessians[i + half_len] += static_cast<score_t>(p_hessian); // identical 2nd order gradient
      sum_lambdas += 2 * std::abs(p_lambda);
    }

    return sum_lambdas;
  }

  inline void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt,
    const label_t* label, const double* score,
    score_t* lambdas,
    score_t* hessians) const override {

    // for this loss, we assume that the records are sorted by
    //  1. query id
    //  2. impression id
    //  3. doc id

    // loss for score diff
    // 1. remove skip for equal label
    // 2. lambdarank inside one impresion, and rank diff between impression
    // 3. re-calculate max dcg

    data_size_t impression_cnt = impression_cnts_ == nullptr ? 1 : impression_cnts_[query_id]; // at least one impression in one query
    if (impression_cnt > 2) {
      Log::Warning("number of impression is %d > 2", impression_cnt);
    }

    bool use_scorediff = (impression_cnt == 2) && (cnt % 2 == 0);

    double sum_lambdas = 0;

    if (!use_scorediff) {
      // get max DCG on current query
      const double inverse_max_dcg = inverse_max_dcgs_[query_id];

      sum_lambdas = RankLossForOneQuery(cnt, label, score, lambdas, hessians, inverse_max_dcg);

    }
    else {
      if (cnt % 2 != 0) {
        Log::Fatal("using score diff, but number of docs in query %d is %d, not an even number.", query_id, cnt);
      }
      // get max DCG on current query
      const double inverse_max_dcg1 = inverse_max_dcgs_[query_id];
      const double inverse_max_dcg2 = inverse_max_dcgs_2_[query_id];
      // rank loss for first impression
      double sum_lambdas1 = RankLossForOneQuery(cnt / 2, label, score, lambdas, hessians, inverse_max_dcg1);
      // rank loss for second impression
      double sum_lambdas2 = RankLossForOneQuery(cnt / 2, label + cnt / 2, score + cnt / 2, lambdas + cnt / 2, hessians + cnt / 2, inverse_max_dcg2);
      // score diff loss for doc pairs
      double sum_lambdas3 = ScoreDiffLossForOneQuery(cnt, score, lambdas, hessians);
      // rank diff loss for doc pairs
      double sum_lambdas4 = RankDiffLossForOneQuery(cnt, score, lambdas, hessians);

      sum_lambdas = sum_lambdas1 + sum_lambdas2 + sum_lambdas3 + sum_lambdas4;
    }

    if (norm_ && normalize_scorediff_gradient_ && sum_lambdas > 0) {
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = static_cast<score_t>(lambdas[i] * norm_factor);
        hessians[i] = static_cast<score_t>(hessians[i] * norm_factor);
      }
    }

  }

  const char* GetName() const override { return "lambdarank_scorerankdiff"; }
};

/*!
 * \brief Implementation of the learning-to-rank objective function, XE_NDCG
 * [arxiv.org/abs/1911.09798].
 */
class RankXENDCG : public RankingObjective {
 public:
  explicit RankXENDCG(const Config& config) : RankingObjective(config) {}

  explicit RankXENDCG(const std::vector<std::string>& strs)
      : RankingObjective(strs) {}

  ~RankXENDCG() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    RankingObjective::Init(metadata, num_data);
    for (data_size_t i = 0; i < num_queries_; ++i) {
      rands_.emplace_back(seed_ + i);
    }
  }

  inline void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt,
                                      const label_t* label, const double* score,
                                      score_t* lambdas,
                                      score_t* hessians) const override {
    // Skip groups with too few items.
    if (cnt <= 1) {
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = 0.0f;
        hessians[i] = 0.0f;
      }
      return;
    }

    // Turn scores into a probability distribution using Softmax.
    std::vector<double> rho(cnt, 0.0);
    Common::Softmax(score, rho.data(), cnt);

    // An auxiliary buffer of parameters used to form the ground-truth
    // distribution and compute the loss.
    std::vector<double> params(cnt);

    double inv_denominator = 0;
    for (data_size_t i = 0; i < cnt; ++i) {
      params[i] = Phi(label[i], rands_[query_id].NextFloat());
      inv_denominator += params[i];
    }
    // sum_labels will always be positive number
    inv_denominator = 1. / std::max<double>(kEpsilon, inv_denominator);

    // Approximate gradients and inverse Hessian.
    // First order terms.
    double sum_l1 = 0.0;
    for (data_size_t i = 0; i < cnt; ++i) {
      double term = -params[i] * inv_denominator + rho[i];
      lambdas[i] = static_cast<score_t>(term);
      // Params will now store terms needed to compute second-order terms.
      params[i] = term / (1. - rho[i]);
      sum_l1 += params[i];
    }
    // Second order terms.
    double sum_l2 = 0.0;
    for (data_size_t i = 0; i < cnt; ++i) {
      double term = rho[i] * (sum_l1 - params[i]);
      lambdas[i] += static_cast<score_t>(term);
      // Params will now store terms needed to compute third-order terms.
      params[i] = term / (1. - rho[i]);
      sum_l2 += params[i];
    }
    for (data_size_t i = 0; i < cnt; ++i) {
      lambdas[i] += static_cast<score_t>(rho[i] * (sum_l2 - params[i]));
      hessians[i] = static_cast<score_t>(rho[i] * (1.0 - rho[i]));
    }
  }

  double Phi(const label_t l, double g) const {
    return Common::Pow(2, static_cast<int>(l)) - g;
  }

  const char* GetName() const override { return "rank_xendcg"; }

 private:
  mutable std::vector<Random> rands_;
};

}  // namespace LightGBM
#endif  // LightGBM_OBJECTIVE_RANK_OBJECTIVE_HPP_
