/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/dataset.h>
#include <LightGBM/utils/common.h>

#include <string>
#include <vector>
#include <exception>

namespace LightGBM {

  const doc_id_t DEFAULT_DOC_ID = 0;

Metadata::Metadata() {
  num_weights_ = 0;
  num_init_score_ = 0;
  num_data_ = 0;
  num_queries_ = 0;
  num_docs_ = 0;
  num_secondary_labels_ = 0;
  weight_load_from_file_ = false;
  query_load_from_file_ = false;
  init_score_load_from_file_ = false;
  doc_load_from_file_ = false;
  impression_load_from_file_ = false;
  secondary_labels_load_from_file_ = false;
  last_impression_id_ = -1;
  total_set_impression_ = 0;
  last_secondary_label_ = -1;
  total_set_secondary_label_ = 0;
}

void Metadata::Init(const char* data_filename) {
  Log::Info("Metadata::init");
  data_filename_ = data_filename;
  // for lambdarank, it needs query data for partition data in distributed learning
  LoadQueryBoundaries();
  LoadSecondaryLabels();
  LoadWeights();
  LoadQueryWeights();
  LoadInitialScore();
  LoadDocIDs();
  LoadDocPairs();
  LoadImpressionIDs();
  LoadNumImpressions();

  Log::Info("INit:: Totally set secondary label for %d times, and last index is %d.", total_set_secondary_label_, last_secondary_label_);
}

Metadata::~Metadata() {
}

void Metadata::Init(data_size_t num_data, int weight_idx, int query_idx, int doc_idx, int imp_idx, int secondary_label_idx = -1) {
  num_data_ = num_data;
  label_ = std::vector<label_t>(num_data_);
  if (weight_idx >= 0) {
    if (!weights_.empty()) {
      Log::Info("Using weights in data file, ignoring the additional weights file");
      weights_.clear();
    }
    weights_ = std::vector<label_t>(num_data_, 0.0f);
    num_weights_ = num_data_;
    weight_load_from_file_ = false;
  }
  if (secondary_label_idx >= 0) {
    if (!secondary_label_.empty()) {
      Log::Info("Using secondary_label in data file, ignoring the additional secondary_label file");
      secondary_label_.clear();
    }
    secondary_label_ = std::vector<label_t>(num_data_, 0.0f);
    num_secondary_labels_ = num_data_;
    secondary_labels_load_from_file_ = false;
  }
  if (query_idx >= 0) {
    if (!query_boundaries_.empty()) {
      Log::Info("Using query id in data file, ignoring the additional query file");
      query_boundaries_.clear();
    }
    if (!query_weights_.empty()) { query_weights_.clear(); }
    queries_ = std::vector<data_size_t>(num_data_, 0);
    query_load_from_file_ = false;
  }
  if (doc_idx >= 0) {
    if (!doc_pairs_.empty()) {
      Log::Info("Using doc id in data file, ignoring the additional doc file");
      doc_pairs_.clear();
    }
    docs_ = std::vector<doc_id_t>(num_data_, DEFAULT_DOC_ID);
    num_docs_ = num_data_;
    doc_load_from_file_ = false;
  }

  Log::Info("intializing with imp_idx = %d", imp_idx);
  if (imp_idx >= 0) {
    if (!num_impressions_.empty()) {
      Log::Info("Using impression id in data file, ignoring the additional impression file");
      num_impressions_.clear();
    }
    if (!impressions_.empty()) {
      Log::Info("Using impression id in data file, ignoring... clear impressions");
      impressions_.clear();
    }
    impressions_ = std::vector<data_size_t>(num_data_, 0);
    impression_load_from_file_ = false;
  }
}

void Metadata::Init(const Metadata& fullset, const data_size_t* used_indices, data_size_t num_used_indices) {
  num_data_ = num_used_indices;

  label_ = std::vector<label_t>(num_used_indices);
#pragma omp parallel for schedule(static, 512) if (num_used_indices >= 1024)
  for (data_size_t i = 0; i < num_used_indices; ++i) {
    label_[i] = fullset.label_[used_indices[i]];
  }

  if (!fullset.weights_.empty()) {
    weights_ = std::vector<label_t>(num_used_indices);
    num_weights_ = num_used_indices;
#pragma omp parallel for schedule(static, 512) if (num_used_indices >= 1024)
    for (data_size_t i = 0; i < num_used_indices; ++i) {
      weights_[i] = fullset.weights_[used_indices[i]];
    }
  } else {
    num_weights_ = 0;
  }

  if (!fullset.docs_.empty()) {
    docs_ = std::vector<doc_id_t>(num_used_indices);
    num_docs_ = num_used_indices;
#pragma omp parallel for schedule(static, 512) if (num_used_indices >= 1024)
    for (data_size_t i = 0; i < num_used_indices; ++i) {
      docs_[i] = fullset.docs_[used_indices[i]];
    }
  }
  else {
    num_docs_ = 0;
  }

  if (!fullset.impressions_.empty()) {
    impressions_ = std::vector<data_size_t>(num_used_indices);
#pragma omp parallel for schedule(static, 512) if (num_used_indices >= 1024)
    for (data_size_t i = 0; i < num_used_indices; ++i) {
      impressions_[i] = fullset.impressions_[used_indices[i]];
    }
  }

  if (!fullset.secondary_label_.empty()) {
    secondary_label_ = std::vector<label_t>(num_used_indices);
    num_secondary_labels_ = num_used_indices;
#pragma omp parallel for schedule(static, 512) if (num_used_indices >= 1024)
    for (data_size_t i = 0; i < num_used_indices; ++i) {
      secondary_label_[i] = fullset.secondary_label_[used_indices[i]];
    }
  }
  else {
    num_secondary_labels_ = 0;
  }

  if (!fullset.init_score_.empty()) {
    int num_class = static_cast<int>(fullset.num_init_score_ / fullset.num_data_);
    init_score_ = std::vector<double>(static_cast<size_t>(num_used_indices) * num_class);
    num_init_score_ = static_cast<int64_t>(num_used_indices) * num_class;
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < num_class; ++k) {
      const size_t offset_dest = static_cast<size_t>(k) * num_data_;
      const size_t offset_src = static_cast<size_t>(k) * fullset.num_data_;
      for (data_size_t i = 0; i < num_used_indices; ++i) {
        init_score_[offset_dest + i] = fullset.init_score_[offset_src + used_indices[i]];
      }
    }
  } else {
    num_init_score_ = 0;
  }

  if (!fullset.query_boundaries_.empty()) {
    Log::Info("Updating query boundaries from fullset");
    std::vector<data_size_t> used_query;
    data_size_t data_idx = 0;
    for (data_size_t qid = 0; qid < num_queries_ && data_idx < num_used_indices; ++qid) {
      data_size_t start = fullset.query_boundaries_[qid];
      data_size_t end = fullset.query_boundaries_[qid + 1];
      data_size_t len = end - start;
      if (used_indices[data_idx] > start) {
        continue;
      } else if (used_indices[data_idx] == start) {
        if (num_used_indices >= data_idx + len && used_indices[data_idx + len - 1] == end - 1) {
          used_query.push_back(qid);
          data_idx += len;
        } else {
          Log::Fatal("Data partition error, data didn't match queries");
        }
      } else {
        Log::Fatal("Data partition error, data didn't match queries");
      }
    }
    query_boundaries_ = std::vector<data_size_t>(used_query.size() + 1);
    num_queries_ = static_cast<data_size_t>(used_query.size());
    query_boundaries_[0] = 0;
    for (data_size_t i = 0; i < num_queries_; ++i) {
      data_size_t qid = used_query[i];
      data_size_t len = fullset.query_boundaries_[qid + 1] - fullset.query_boundaries_[qid];
      query_boundaries_[i + 1] = query_boundaries_[i] + len;
    }
  } else {
    num_queries_ = 0;
  }
}

void Metadata::PartitionLabel(const std::vector<data_size_t>& used_indices) {
  if (used_indices.empty()) {
    return;
  }
  auto old_label = label_;
  num_data_ = static_cast<data_size_t>(used_indices.size());
  label_ = std::vector<label_t>(num_data_);
#pragma omp parallel for schedule(static, 512) if (num_data_ >= 1024)
  for (data_size_t i = 0; i < num_data_; ++i) {
    label_[i] = old_label[used_indices[i]];
  }
  old_label.clear();
}

void Metadata::CheckOrPartition(data_size_t num_all_data, const std::vector<data_size_t>& used_data_indices) {
  if (used_data_indices.empty()) {
    Log::Info("CheckOrPartition, num_all_data=%d", num_all_data);
    if (!queries_.empty()) {
      Log::Info("converting queries to boundaries, queries size = %d", queries_.size());
      // need convert query_id to boundaries
      std::vector<data_size_t> tmp_buffer;
      data_size_t last_qid = -1;
      data_size_t cur_cnt = 0;
      for (data_size_t i = 0; i < num_data_; ++i) {
        if (last_qid != queries_[i]) {
          if (cur_cnt > 0) {
            tmp_buffer.push_back(cur_cnt);
          }
          cur_cnt = 0;
          last_qid = queries_[i];
        }
        ++cur_cnt;
      }
      tmp_buffer.push_back(cur_cnt);
      query_boundaries_ = std::vector<data_size_t>(tmp_buffer.size() + 1);
      num_queries_ = static_cast<data_size_t>(tmp_buffer.size());
      query_boundaries_[0] = 0;
      for (size_t i = 0; i < tmp_buffer.size(); ++i) {
        query_boundaries_[i + 1] = query_boundaries_[i] + tmp_buffer[i];
      }
      LoadQueryWeights();
      LoadNumImpressions();
      queries_.clear();
    }
    // check weights
    if (!weights_.empty() && num_weights_ != num_data_) {
      weights_.clear();
      num_weights_ = 0;
      Log::Fatal("Weights size doesn't match data size");
    }

    // check query boundries
    if (!query_boundaries_.empty() && query_boundaries_[num_queries_] != num_data_) {
      query_boundaries_.clear();
      num_queries_ = 0;
      Log::Fatal("Query size doesn't match data size");
    }

    // contain initial score file
    if (!init_score_.empty() && (num_init_score_ % num_data_) != 0) {
      init_score_.clear();
      num_init_score_ = 0;
      Log::Fatal("Initial score size doesn't match data size");
    }

    // check doc id and doc pairs
    Log::Info("checking docs in CheckOrPartition");
    if (!docs_.empty() && num_docs_ != num_data_) {
      docs_.clear();
      num_docs_ = 0;
      Log::Fatal("Docs size doesn't match data size");
    }

    if (!docs_.empty()) {
      LoadDocPairs();
      docs_.clear();
    }

    // check impression id and number of impressions
    Log::Info("checking impressions in CheckOrPartition");
    if (!impressions_.empty() && impressions_.size() != num_data_) {
      impressions_.clear();
      Log::Fatal("Impression size doesn't match data size");
    }

    //if (!impressions_.empty()) {
    //  LoadNumImpressions();
    //  //impressions_.clear();
    //}
  } else {
    if (!queries_.empty()) {
      Log::Fatal("Cannot used query_id for distributed training");
    }
    Log::Warning("Running into a branch I didn't handled.......");
    data_size_t num_used_data = static_cast<data_size_t>(used_data_indices.size());
    // check weights
    if (weight_load_from_file_) {
      if (weights_.size() > 0 && num_weights_ != num_all_data) {
        weights_.clear();
        num_weights_ = 0;
        Log::Fatal("Weights size doesn't match data size");
      }
      // get local weights
      if (!weights_.empty()) {
        auto old_weights = weights_;
        num_weights_ = num_data_;
        weights_ = std::vector<label_t>(num_data_);
#pragma omp parallel for schedule(static, 512)
        for (int i = 0; i < static_cast<int>(used_data_indices.size()); ++i) {
          weights_[i] = old_weights[used_data_indices[i]];
        }
        old_weights.clear();
      }
    }
    if (query_load_from_file_) {
      // check query boundries
      if (!query_boundaries_.empty() && query_boundaries_[num_queries_] != num_all_data) {
        query_boundaries_.clear();
        num_queries_ = 0;
        Log::Fatal("Query size doesn't match data size");
      }
      // get local query boundaries
      if (!query_boundaries_.empty()) {
        std::vector<data_size_t> used_query;
        data_size_t data_idx = 0;
        for (data_size_t qid = 0; qid < num_queries_ && data_idx < num_used_data; ++qid) {
          data_size_t start = query_boundaries_[qid];
          data_size_t end = query_boundaries_[qid + 1];
          data_size_t len = end - start;
          if (used_data_indices[data_idx] > start) {
            continue;
          } else if (used_data_indices[data_idx] == start) {
            if (num_used_data >= data_idx + len && used_data_indices[data_idx + len - 1] == end - 1) {
              used_query.push_back(qid);
              data_idx += len;
            } else {
              Log::Fatal("Data partition error, data didn't match queries");
            }
          } else {
            Log::Fatal("Data partition error, data didn't match queries");
          }
        }
        auto old_query_boundaries = query_boundaries_;
        query_boundaries_ = std::vector<data_size_t>(used_query.size() + 1);
        num_queries_ = static_cast<data_size_t>(used_query.size());
        query_boundaries_[0] = 0;
        for (data_size_t i = 0; i < num_queries_; ++i) {
          data_size_t qid = used_query[i];
          data_size_t len = old_query_boundaries[qid + 1] - old_query_boundaries[qid];
          query_boundaries_[i + 1] = query_boundaries_[i] + len;
        }
        old_query_boundaries.clear();
      }
    }
    if (init_score_load_from_file_) {
      // contain initial score file
      if (!init_score_.empty() && (num_init_score_ % num_all_data) != 0) {
        init_score_.clear();
        num_init_score_ = 0;
        Log::Fatal("Initial score size doesn't match data size");
      }

      // get local initial scores
      if (!init_score_.empty()) {
        auto old_scores = init_score_;
        int num_class = static_cast<int>(num_init_score_ / num_all_data);
        num_init_score_ = static_cast<int64_t>(num_data_) * num_class;
        init_score_ = std::vector<double>(num_init_score_);
#pragma omp parallel for schedule(static)
        for (int k = 0; k < num_class; ++k) {
          const size_t offset_dest = static_cast<size_t>(k) * num_data_;
          const size_t offset_src = static_cast<size_t>(k) * num_all_data;
          for (size_t i = 0; i < used_data_indices.size(); ++i) {
            init_score_[offset_dest + i] = old_scores[offset_src + used_data_indices[i]];
          }
        }
        old_scores.clear();
      }
    }
    // re-load query weight
    LoadQueryWeights();
    LoadNumImpressions();

    Log::Fatal("distributed training is called, but we don't handle this case for lambdarank loss with scorediff");
  }

  Log::Info("Totally set secondary label for %d times, and last index is %d.", total_set_secondary_label_, last_secondary_label_);


  if (num_queries_ > 0) {
    Log::Debug("Number of queries in %s: %i. Average number of rows per query: %f.",
      data_filename_.c_str(), static_cast<int>(num_queries_), static_cast<double>(num_data_) / num_queries_);
  }
}

void Metadata::SetInitScore(const double* init_score, data_size_t len) {
  std::lock_guard<std::mutex> lock(mutex_);
  // save to nullptr
  if (init_score == nullptr || len == 0) {
    init_score_.clear();
    num_init_score_ = 0;
    return;
  }
  if ((len % num_data_) != 0) {
    Log::Fatal("Initial score size doesn't match data size");
  }
  if (init_score_.empty()) { init_score_.resize(len); }
  num_init_score_ = len;

  #pragma omp parallel for schedule(static, 512) if (num_init_score_ >= 1024)
  for (int64_t i = 0; i < num_init_score_; ++i) {
    init_score_[i] = Common::AvoidInf(init_score[i]);
  }
  init_score_load_from_file_ = false;
}

void Metadata::SetLabel(const label_t* label, data_size_t len) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (label == nullptr) {
    Log::Fatal("label cannot be nullptr");
  }
  if (num_data_ != len) {
    Log::Fatal("Length of label is not same with #data");
  }
  if (label_.empty()) { label_.resize(num_data_); }

  #pragma omp parallel for schedule(static, 512) if (num_data_ >= 1024)
  for (data_size_t i = 0; i < num_data_; ++i) {
    label_[i] = Common::AvoidInf(label[i]);
  }
}

void Metadata::SetSecondaryLabel(const label_t* label, data_size_t len) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (label == nullptr) {
    Log::Fatal("secondary label cannot be nullptr");
  }
  if (num_data_ != len) {
    Log::Fatal("Length of secondary label is not same with #data");
  }
  if (secondary_label_.empty()) { secondary_label_.resize(num_data_); }

#pragma omp parallel for schedule(static, 512) if (num_data_ >= 1024)
  for (data_size_t i = 0; i < num_data_; ++i) {
    secondary_label_[i] = Common::AvoidInf(label[i]);
  }
}

void Metadata::SetWeights(const label_t* weights, data_size_t len) {
  std::lock_guard<std::mutex> lock(mutex_);
  // save to nullptr
  if (weights == nullptr || len == 0) {
    weights_.clear();
    num_weights_ = 0;
    return;
  }
  if (num_data_ != len) {
    Log::Fatal("Length of weights is not same with #data");
  }
  if (weights_.empty()) { weights_.resize(num_data_); }
  num_weights_ = num_data_;

  #pragma omp parallel for schedule(static, 512) if (num_weights_ >= 1024)
  for (data_size_t i = 0; i < num_weights_; ++i) {
    weights_[i] = Common::AvoidInf(weights[i]);
  }
  LoadQueryWeights();
  weight_load_from_file_ = false;
}

void Metadata::SetQuery(const data_size_t* query, data_size_t len) {
  std::lock_guard<std::mutex> lock(mutex_);
  Log::Info("SetQuery........");
  // save to nullptr
  if (query == nullptr || len == 0) {
    query_boundaries_.clear();
    num_queries_ = 0;
    return;
  }
  data_size_t sum = 0;
  #pragma omp parallel for schedule(static) reduction(+:sum)
  for (data_size_t i = 0; i < len; ++i) {
    sum += query[i];
  }
  if (num_data_ != sum) {
    Log::Fatal("Sum of query counts is not same with #data");
  }
  num_queries_ = len;
  query_boundaries_.resize(num_queries_ + 1);
  query_boundaries_[0] = 0;
  for (data_size_t i = 0; i < num_queries_; ++i) {
    query_boundaries_[i + 1] = query_boundaries_[i] + query[i];
  }
  LoadQueryWeights();
  LoadNumImpressions();
  query_load_from_file_ = false;
}

void Metadata::SetDoc(const data_size_t* doc, data_size_t len) {
  std::lock_guard<std::mutex> lock(mutex_);
  // save to nullptr
  if (doc == nullptr || len == 0) {
    docs_.clear();
    return;
  }
  if (num_data_ != len) {
    Log::Fatal("Length of docs is not same with #data");
  }
  if (docs_.empty()) { docs_.resize(num_data_); }

  for (data_size_t i = 0; i < num_data_; ++i) {
    docs_[i] = doc[i];
  }
  LoadDocPairs();
  doc_load_from_file_ = false;
}

void Metadata::SetImpression(const data_size_t* impression, data_size_t len) {
  std::lock_guard<std::mutex> lock(mutex_);
  Log::Info("SetImpression......");
  // save to nullptr
  if (impression == nullptr || len == 0) {
    impressions_.clear();
    return;
  }
  if (num_data_ != len) {
    Log::Fatal("Length of impressions is not same with #data");
  }
  Log::Info("setting impressions...");
  if (impressions_.empty()) { impressions_.resize(num_data_); }

  for (data_size_t i = 0; i < num_data_; ++i) {
    impressions_[i] = impression[i];
  }
  LoadNumImpressions();
  impression_load_from_file_ = false;
}

void Metadata::LoadSecondaryLabels() {
  num_secondary_labels_ = 0;
  std::string secondary_label_filename(data_filename_);
  secondary_label_filename.append(".secondary_label");
  TextReader<size_t> reader(secondary_label_filename.c_str(), false);
  reader.ReadAllLines();
  if (reader.Lines().empty()) {
    return;
  }
  Log::Info("Loading secondary_labels...");
  num_secondary_labels_ = static_cast<data_size_t>(reader.Lines().size());
  secondary_label_ = std::vector<label_t>(num_secondary_labels_);
#pragma omp parallel for schedule(static)
  for (data_size_t i = 0; i < num_secondary_labels_; ++i) {
    double tmp_label = 0.0f;
    Common::Atof(reader.Lines()[i].c_str(), &tmp_label);
    secondary_label_[i] = Common::AvoidInf(static_cast<label_t>(tmp_label));
  }
  secondary_labels_load_from_file_ = true;
}

void Metadata::LoadWeights() {
  num_weights_ = 0;
  std::string weight_filename(data_filename_);
  // default weight file name
  weight_filename.append(".weight");
  TextReader<size_t> reader(weight_filename.c_str(), false);
  reader.ReadAllLines();
  if (reader.Lines().empty()) {
    return;
  }
  Log::Info("Loading weights...");
  num_weights_ = static_cast<data_size_t>(reader.Lines().size());
  weights_ = std::vector<label_t>(num_weights_);
  #pragma omp parallel for schedule(static)
  for (data_size_t i = 0; i < num_weights_; ++i) {
    double tmp_weight = 0.0f;
    Common::Atof(reader.Lines()[i].c_str(), &tmp_weight);
    weights_[i] = Common::AvoidInf(static_cast<label_t>(tmp_weight));
  }
  weight_load_from_file_ = true;
}

void Metadata::LoadDocIDs() {
  num_docs_ = 0;
  std::string doc_filename(data_filename_);
  // default doc id file name
  doc_filename.append(".doc_id");
  TextReader<size_t> reader(doc_filename.c_str(), false);
  reader.ReadAllLines();
  if (reader.Lines().empty()) {
    return;
  }
  Log::Info("Loading doc ids...");
  num_docs_ = static_cast<data_size_t>(reader.Lines().size());
  docs_ = std::vector<doc_id_t>(num_docs_);
#pragma omp parallel for schedule(static)
  for (data_size_t i = 0; i < num_docs_; ++i) {
    doc_id_t tmp_docid = 0;
    Common::Atoi(reader.Lines()[i].c_str(), &tmp_docid);
    docs_[i] = tmp_docid;
  }
  doc_load_from_file_ = true;
}

void Metadata::LoadImpressionIDs() {
  std::string imp_filename(data_filename_);
  // default imp id file name
  imp_filename.append(".imp_id");
  TextReader<size_t> reader(imp_filename.c_str(), false);
  reader.ReadAllLines();
  if (reader.Lines().empty()) {
    return;
  }
  Log::Info("Loading impression ids from %s...", imp_filename);
  size_t num_imp = reader.Lines().size();
  impressions_ = std::vector<data_size_t>(num_imp);
#pragma omp parallel for schedule(static)
  for (data_size_t i = 0; i < num_imp; ++i) {
    data_size_t tmp_impid = 0;
    Common::Atoi(reader.Lines()[i].c_str(), &tmp_impid);
    impressions_[i] = tmp_impid;
  }
  impression_load_from_file_ = true;
}

void Metadata::LoadInitialScore() {
  num_init_score_ = 0;
  std::string init_score_filename(data_filename_);
  init_score_filename = std::string(data_filename_);
  // default init_score file name
  init_score_filename.append(".init");
  TextReader<size_t> reader(init_score_filename.c_str(), false);
  reader.ReadAllLines();
  if (reader.Lines().empty()) {
    return;
  }
  Log::Info("Loading initial scores...");

  // use first line to count number class
  int num_class = static_cast<int>(Common::Split(reader.Lines()[0].c_str(), '\t').size());
  data_size_t num_line = static_cast<data_size_t>(reader.Lines().size());
  num_init_score_ = static_cast<int64_t>(num_line) * num_class;

  init_score_ = std::vector<double>(num_init_score_);
  if (num_class == 1) {
    #pragma omp parallel for schedule(static)
    for (data_size_t i = 0; i < num_line; ++i) {
      double tmp = 0.0f;
      Common::Atof(reader.Lines()[i].c_str(), &tmp);
      init_score_[i] = Common::AvoidInf(static_cast<double>(tmp));
    }
  } else {
    std::vector<std::string> oneline_init_score;
    #pragma omp parallel for schedule(static)
    for (data_size_t i = 0; i < num_line; ++i) {
      double tmp = 0.0f;
      oneline_init_score = Common::Split(reader.Lines()[i].c_str(), '\t');
      if (static_cast<int>(oneline_init_score.size()) != num_class) {
        Log::Fatal("Invalid initial score file. Redundant or insufficient columns");
      }
      for (int k = 0; k < num_class; ++k) {
        Common::Atof(oneline_init_score[k].c_str(), &tmp);
        init_score_[static_cast<size_t>(k) * num_line + i] = Common::AvoidInf(static_cast<double>(tmp));
      }
    }
  }
  init_score_load_from_file_ = true;
}

void Metadata::LoadQueryBoundaries() {
  num_queries_ = 0;
  std::string query_filename(data_filename_);
  // default query file name
  query_filename.append(".query");
  TextReader<size_t> reader(query_filename.c_str(), false);
  reader.ReadAllLines();
  if (reader.Lines().empty()) {
    return;
  }
  Log::Info("Loading query boundaries...");
  query_boundaries_ = std::vector<data_size_t>(reader.Lines().size() + 1);
  num_queries_ = static_cast<data_size_t>(reader.Lines().size());
  query_boundaries_[0] = 0;
  for (size_t i = 0; i < reader.Lines().size(); ++i) {
    int tmp_cnt;
    Common::Atoi(reader.Lines()[i].c_str(), &tmp_cnt);
    query_boundaries_[i + 1] = query_boundaries_[i] + static_cast<data_size_t>(tmp_cnt);
  }
  query_load_from_file_ = true;
}

void Metadata::LoadQueryWeights() {
  if (weights_.size() == 0 || query_boundaries_.size() == 0) {
    return;
  }
  query_weights_.clear();
  Log::Info("Loading query weights...");
  query_weights_ = std::vector<label_t>(num_queries_);
  for (data_size_t i = 0; i < num_queries_; ++i) {
    query_weights_[i] = 0.0f;
    for (data_size_t j = query_boundaries_[i]; j < query_boundaries_[i + 1]; ++j) {
      query_weights_[i] += weights_[j];
    }
    query_weights_[i] /= (query_boundaries_[i + 1] - query_boundaries_[i]);
  }

}

void Metadata::LoadDocPairs() {
  if (docs_.size() == 0 || query_boundaries_.size() == 0) {
    return;
  }
  doc_pairs_.clear();
  Log::Info("Loading doc pairs...");
  auto pair_cmp = [](const DocPair& a, const DocPair& b) {
    // we assume first < second, for both a and b.
    if (a.first != b.first) {
      return a.first < b.first;
    }
    else {
      return a.second < b.second;
    }
  };
  doc_pairs_ = std::vector<DocPairContainer>(num_queries_);

  data_size_t n_zero_doc_pair = 0;
  data_size_t n_more_doc_pair = 0;
  data_size_t max_doc_pair = 0;
  for (data_size_t i = 0; i < num_queries_; ++i) {
    doc_pairs_[i] = DocPairContainer(pair_cmp);

    for (data_size_t j = query_boundaries_[i]; j < query_boundaries_[i + 1] - 1; ++j) {
      for (data_size_t k = j + 1; k < query_boundaries_[i + 1]; ++k) {
        if (docs_[j] == docs_[k] && docs_[j] != 0) { // 0 is default value
          doc_pairs_[i].insert(std::make_pair(j - query_boundaries_[i], k - query_boundaries_[i]));  // normalize to one query
        }
      }
    }
    if (doc_pairs_[i].size() > 0) {
      n_more_doc_pair++;
    }
    else {
      n_zero_doc_pair++;
    }
    if (doc_pairs_[i].size() > max_doc_pair) {
      max_doc_pair = doc_pairs_[i].size();
    }
  }
  Log::Info("Totally %d queries, where %d (%d %%) queries has no doc pairs, %d (%d %%) has doc pairs. Max doc pair is %d, and average doc pair (excluded 0) is %d",
    num_queries_, n_zero_doc_pair, n_zero_doc_pair * 100 / num_queries_, n_more_doc_pair, n_more_doc_pair * 100 / num_queries_, max_doc_pair, n_more_doc_pair < 1 ? 0 : max_doc_pair / n_more_doc_pair);
}

void Metadata::LoadNumImpressions() {
  Log::Info("LoadNumImpressions");
  if (impressions_.size() == 0 || query_boundaries_.size() == 0) {
    return;
  }
  num_impressions_.clear();
  Log::Info("Totally set impression id for %d times, and last index is %d.", total_set_impression_, last_impression_id_);

  Log::Info("Loading num impressions, impressions_ size is %d", impressions_.size());
  num_impressions_ = std::vector<data_size_t>(num_queries_, 0);

  long impression_id_cnts[] = {0,0,0,0,0,0,0,0,0,0,0};

  int imp_1 = 0, imp_2 = 0, imp_gt2 = 0;
  for (data_size_t i = 0; i < num_queries_; ++i) {
    
    data_size_t last_impression_id = impressions_[query_boundaries_[i]];
    data_size_t impression_cnt = 1;
    for (data_size_t j = query_boundaries_[i]; j < query_boundaries_[i + 1]; ++j) {
      if (impressions_[j] != last_impression_id) {
        impression_cnt++;
        last_impression_id = impressions_[j];
      }
      if (impressions_[j] < 10) {
        impression_id_cnts[impressions_[j]]++;
      }
      else {
        impression_id_cnts[10]++;
      }
    }
    num_impressions_[i] = impression_cnt;

    if (impression_cnt == 1) {
      imp_1++;
    }
    else if (impression_cnt == 2) {
      imp_2++;
    }
    else {
      imp_gt2++;
    }
  }

  Log::Info("Loaded number of impressions. Totally %d queries, where %d has 1 impression, %d has 2 impressions, and %d has more than 2 impressions.", num_queries_, imp_1, imp_2, imp_gt2);
  for (int i = 0; i < 11; i++) {
    Log::Info("number of records impression id == %d, is %8d", i, impression_id_cnts[i]);
  }

}

// TODO: we need to load doc id in this function
void Metadata::LoadFromMemory(const void* memory) {
  Log::Info("Loading from memory......");
  const char* mem_ptr = reinterpret_cast<const char*>(memory);

  num_data_ = *(reinterpret_cast<const data_size_t*>(mem_ptr));
  mem_ptr += VirtualFileWriter::AlignedSize(sizeof(num_data_));
  num_weights_ = *(reinterpret_cast<const data_size_t*>(mem_ptr));
  mem_ptr += VirtualFileWriter::AlignedSize(sizeof(num_weights_));
  num_queries_ = *(reinterpret_cast<const data_size_t*>(mem_ptr));
  mem_ptr += VirtualFileWriter::AlignedSize(sizeof(num_queries_));

  if (!label_.empty()) { label_.clear(); }
  label_ = std::vector<label_t>(num_data_);
  std::memcpy(label_.data(), mem_ptr, sizeof(label_t) * num_data_);
  mem_ptr += VirtualFileWriter::AlignedSize(sizeof(label_t) * num_data_);

  if (num_weights_ > 0) {
    if (!weights_.empty()) { weights_.clear(); }
    weights_ = std::vector<label_t>(num_weights_);
    std::memcpy(weights_.data(), mem_ptr, sizeof(label_t) * num_weights_);
    mem_ptr += VirtualFileWriter::AlignedSize(sizeof(label_t) * num_weights_);
    weight_load_from_file_ = true;
  }
  if (num_queries_ > 0) {
    if (!query_boundaries_.empty()) { query_boundaries_.clear(); }
    query_boundaries_ = std::vector<data_size_t>(num_queries_ + 1);
    std::memcpy(query_boundaries_.data(), mem_ptr, sizeof(data_size_t) * (num_queries_ + 1));
    mem_ptr += VirtualFileWriter::AlignedSize(sizeof(data_size_t) *
                                              (num_queries_ + 1));
    query_load_from_file_ = true;
  }
  LoadQueryWeights();
  Log::Warning("Aha, this case [load from memory] we don't handle....");
  LoadNumImpressions();
}

void Metadata::SaveBinaryToFile(const VirtualFileWriter* writer) const {
  writer->AlignedWrite(&num_data_, sizeof(num_data_));
  writer->AlignedWrite(&num_weights_, sizeof(num_weights_));
  writer->AlignedWrite(&num_queries_, sizeof(num_queries_));
  writer->AlignedWrite(label_.data(), sizeof(label_t) * num_data_);
  if (!weights_.empty()) {
    writer->AlignedWrite(weights_.data(), sizeof(label_t) * num_weights_);
  }
  if (!query_boundaries_.empty()) {
    writer->AlignedWrite(query_boundaries_.data(),
                         sizeof(data_size_t) * (num_queries_ + 1));
  }
  if (num_init_score_ > 0) {
    Log::Warning("Please note that `init_score` is not saved in binary file.\n"
      "If you need it, please set it again after loading Dataset.");
  }
}

size_t Metadata::SizesInByte() const {
  size_t size = VirtualFileWriter::AlignedSize(sizeof(num_data_)) +
                VirtualFileWriter::AlignedSize(sizeof(num_weights_)) +
                VirtualFileWriter::AlignedSize(sizeof(num_queries_));
  size += VirtualFileWriter::AlignedSize(sizeof(label_t) * num_data_);
  if (!weights_.empty()) {
    size += VirtualFileWriter::AlignedSize(sizeof(label_t) * num_weights_);
  }
  if (!query_boundaries_.empty()) {
    size += VirtualFileWriter::AlignedSize(sizeof(data_size_t) *
                                           (num_queries_ + 1));
  }
  return size;
}


}  // namespace LightGBM
